#!/usr/bin/env python3
"""编排主号 + 观测账号的对称观察 probe 数据采集。

默认只做只读操作。会修改平台状态的子命令必须显式传入 ``--yes``：

- ``publish-main``: 主号直连调用 ``add_model_4``，上传供他人挑战的公开 probe。
- ``run``: 观测账号经 ``OBSERVER_PROXY`` 调用 ``add_model_1``，上传本局 probe 并创建对局。

平台接口依据 ``gamerules/platform.md``。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


BASE_URL = "http://47.103.127.219"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OBSERVER_ENV = REPO_ROOT / "mechanism" / "scripts" / "observer" / ".env"
PROBE_ROOT = REPO_ROOT / "mechanism" / "scripts" / "probes" / "symmetry_observer"
RAW_RUN_ROOT = REPO_ROOT / "mechanism" / "data" / "raw" / "observer_runs"
PY_LANG = "1"
DEFAULT_TIMEOUT = 60
READY_UPLOAD = 1
ERROR_UPLOAD = 2
READY_PARSE = 1


LAYOUT_FILES = {
    "a": PROBE_ROOT / "layout_a.py",
    "ca": PROBE_ROOT / "layout_ca.py",
    "m3a": PROBE_ROOT / "layout_m3_a.py",
    "m3b": PROBE_ROOT / "layout_m3_b.py",
}


def now_text() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_env(paths: list[Path]) -> dict[str, str]:
    values: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            values[key] = value.strip().strip("\"'")
    for key, value in os.environ.items():
        if key in {"LOGIN_KEY", "OBSERVER_KEY", "OBSERVER_PROXY"}:
            values[key] = value
    return values


def require_env(env: dict[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise SystemExit(f"缺少环境变量 {key}")
    return value


def ensure_yes(args: argparse.Namespace, message: str) -> None:
    if not getattr(args, "yes", False):
        raise SystemExit(message + "\n如确认执行，请追加 --yes。")


def require_layout(layout: str) -> Path:
    path = LAYOUT_FILES[layout]
    if not path.exists():
        raise SystemExit(f"找不到布局策略文件：{path}")
    return path


def normalize_model_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", value):
        raise SystemExit(f"非法模型名 {value!r}：必须字母开头且只包含字母数字")
    return value


def manifest_path(run_id: str) -> Path:
    root = RAW_RUN_ROOT / run_id
    root.mkdir(parents=True, exist_ok=True)
    return root / "manifest.jsonl"


def append_manifest(run_id: str, record: dict[str, Any]) -> None:
    path = manifest_path(run_id)
    record = {"time": iso_now(), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_manifest_games(run_id: str) -> list[int]:
    path = manifest_path(run_id)
    if not path.exists():
        raise SystemExit(f"manifest 不存在：{path}")
    game_ids: list[int] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        record = json.loads(raw)
        game_id = record.get("game_id")
        if isinstance(game_id, int) and game_id not in game_ids:
            game_ids.append(game_id)
    return game_ids


@dataclass
class Account:
    label: str
    key_name: str
    key: str
    proxy: str | None = None
    token: str | None = None
    user: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.session = requests.Session()
        if self.proxy:
            self.session.proxies.update({"http": self.proxy, "https": self.proxy})

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        data: Any = None,
        files: Any = None,
        raw: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> Any:
        headers = {}
        if self.token:
            headers["Authorization"] = self.token
        url = BASE_URL + path
        resp = self.session.request(
            method,
            url,
            headers=headers,
            json=json_body,
            data=data,
            files=files,
            timeout=timeout,
        )
        resp.raise_for_status()
        if raw:
            return resp.content
        try:
            return resp.json()
        except ValueError as exc:
            raise SystemExit(f"{self.label} 请求 {path} 返回非 JSON 响应") from exc

    def login(self) -> None:
        resp = self.request("POST", "/api/user/login", json_body={"key": self.key})
        token = ((resp.get("data") or {}).get("token")) if isinstance(resp, dict) else None
        user = ((resp.get("data") or {}).get("user")) if isinstance(resp, dict) else None
        if not token or not isinstance(user, dict):
            raise SystemExit(f"{self.label} 登录失败：code={resp.get('code')} message={resp.get('message')!r}")
        self.token = token
        self.user = user

    def user_info(self) -> dict[str, Any]:
        resp = self.request("GET", "/api/user/get_user_info")
        data = resp.get("data") if isinstance(resp, dict) else None
        return data if isinstance(data, dict) else {}


def env_paths(args: argparse.Namespace) -> list[Path]:
    return [DEFAULT_ENV, DEFAULT_OBSERVER_ENV] + list(args.env or [])


def build_accounts(args: argparse.Namespace) -> tuple[Account, Account]:
    env = read_env(env_paths(args))
    main = Account("主号", "LOGIN_KEY", require_env(env, "LOGIN_KEY"))
    observer_proxy = require_env(env, "OBSERVER_PROXY")
    if not observer_proxy.startswith("socks5h://"):
        raise SystemExit("OBSERVER_PROXY 必须使用 socks5h://，避免 DNS 从本地泄漏")
    observer = Account("观测账号", "OBSERVER_KEY", require_env(env, "OBSERVER_KEY"), observer_proxy)
    return main, observer


def public_model_list(account: Account) -> list[dict[str, Any]]:
    resp = account.request("GET", "/api/user/get_model_list_4")
    return list(((resp.get("data") or {}).get("list")) or [])


def select_public_model(
    models: list[dict[str, Any]],
    *,
    user_id: int | None = None,
    user_name: str | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    candidates = models
    if user_id is not None:
        candidates = [m for m in candidates if int(m.get("user_id", -1)) == user_id]
    if user_name:
        candidates = [m for m in candidates if m.get("user_name_cn") == user_name]
    if model_name:
        candidates = [m for m in candidates if m.get("name") == model_name]
    if not candidates:
        raise SystemExit("未在可挑战模型列表中找到主号公开模型")
    candidates.sort(key=lambda m: str(m.get("updated_at") or m.get("created_at") or ""), reverse=True)
    return candidates[0]


def publish_public_model(account: Account, layout: str) -> dict[str, Any]:
    source = require_layout(layout)
    with source.open("rb") as f:
        files = {"model_file": ("player.py", f, "text/x-python")}
        data = {"model_lang": PY_LANG}
        resp = account.request("POST", "/api/user/add_model_4", data=data, files=files)
    if not isinstance(resp, dict) or resp.get("code") != 0:
        raise SystemExit(f"主号公开上传失败：code={resp.get('code')} message={resp.get('message')!r}")
    return resp


def submit_challenge(
    account: Account,
    *,
    layout: str,
    map_id: int,
    model_name: str,
    opponent_model_id: int,
) -> dict[str, Any]:
    source = require_layout(layout)
    normalize_model_name(model_name)
    with source.open("rb") as f:
        files = [("model_files", ("player.py", f, "text/x-python"))]
        data = [
            ("map_id", str(map_id)),
            ("model_langs", PY_LANG),
            ("model_names", model_name),
            ("model_id", str(opponent_model_id)),
        ]
        resp = account.request("POST", "/api/user/add_model_1", data=data, files=files, timeout=90)
    if not isinstance(resp, dict) or resp.get("code") != 0 or not ((resp.get("data") or {}).get("game_id")):
        raise SystemExit(f"创建对局失败：code={resp.get('code')} message={resp.get('message')!r}")
    return resp


def game_info(account: Account, game_id: int) -> dict[str, Any]:
    resp = account.request("GET", f"/api/user/get_game_info?id={game_id}")
    if not isinstance(resp, dict):
        return {}
    data = resp.get("data")
    return data if isinstance(data, dict) else resp


def game_ready(info: dict[str, Any]) -> bool:
    return info.get("is_upload_log") == READY_UPLOAD and info.get("is_parse_log") == READY_PARSE


def game_failed(info: dict[str, Any]) -> bool:
    return info.get("is_upload_log") == ERROR_UPLOAD


def wait_for_game(account: Account, game_id: int, timeout_s: int, interval_s: int) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    while True:
        last = game_info(account, game_id)
        if game_ready(last) or game_failed(last):
            return last
        if time.time() >= deadline:
            raise SystemExit(f"等待对局 {game_id} 解析超时；最后状态：{redact_game_info(last)}")
        time.sleep(interval_s)


def redact_game_info(info: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "id",
        "stage",
        "user_id",
        "user_id2",
        "map_id",
        "is_upload_log",
        "is_parse_log",
        "error_msg",
        "viewer_side",
        "is_admin",
    }
    return {k: v for k, v in info.items() if k in keep}


def download_log(account: Account, game_id: int) -> bytes:
    return account.request("GET", f"/api/user/get_game_log?id={game_id}", raw=True, timeout=90)


def default_run_id(args: argparse.Namespace) -> str:
    if getattr(args, "run_id", None):
        return args.run_id
    return f"symobs-{args.layout}-{now_text()}"


def cmd_doctor(args: argparse.Namespace) -> None:
    main, observer = build_accounts(args)
    main.login()
    observer.login()
    main_info = main.user_info()
    observer_info = observer.user_info()
    root_resp = observer.session.get(BASE_URL + "/", timeout=DEFAULT_TIMEOUT)
    root_resp.raise_for_status()
    models = public_model_list(observer)
    print("doctor ok")
    print(f"主号 user_id={main.user.get('id')} name={main.user.get('name_cn')}")
    print(f"观测账号 user_id={observer.user.get('id')} name={observer.user.get('name_cn')}")
    print(f"主号今日提交 {main_info.get('today_initiated')} / {main_info.get('daily_initiate_limit')}")
    print(f"观测账号今日提交 {observer_info.get('today_initiated')} / {observer_info.get('daily_initiate_limit')}")
    print(f"观测账号代理 {observer.proxy}")
    print(f"可挑战公开模型数量 {len(models)}")


def cmd_publish_main(args: argparse.Namespace) -> None:
    ensure_yes(args, "publish-main 会修改主号供他人挑战的公开代码。")
    run_id = default_run_id(args)
    main, _observer = build_accounts(args)
    main.login()
    source = require_layout(args.layout)
    print("即将执行：主号直连上传公开 probe")
    print(f"run_id={run_id}")
    print(f"layout={args.layout}")
    print(f"source={source}")
    resp = publish_public_model(main, args.layout)
    expected_name = f"player{main.user.get('id')}"
    append_manifest(
        run_id,
        {
            "action": "publish-main",
            "account": "main",
            "layout": args.layout,
            "source": str(source.relative_to(REPO_ROOT)),
            "expected_public_name": expected_name,
            "response": resp,
            "proxy": None,
        },
    )
    print("publish-main ok")
    print(f"公开模型预计显示为 {expected_name}")
    print(f"manifest={manifest_path(run_id)}")


def cmd_resolve_main(args: argparse.Namespace) -> None:
    main, observer = build_accounts(args)
    main.login()
    observer.login()
    models = public_model_list(observer)
    user_id = args.main_user_id if args.main_user_id is not None else int(main.user["id"])
    user_name = args.main_user_name
    model_name = args.main_public_name
    selected = select_public_model(models, user_id=user_id, user_name=user_name, model_name=model_name)
    print(json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True))


def make_model_name(prefix: str, layout: str, index: int) -> str:
    safe_prefix = normalize_model_name(prefix)
    layout_tags = {
        "a": "A",
        "ca": "C",
        "m3a": "M",
        "m3b": "N",
    }
    layout_tag = layout_tags[layout]
    return normalize_model_name(f"{safe_prefix}{layout_tag}{index:03d}")


def cmd_run(args: argparse.Namespace) -> None:
    ensure_yes(args, "run 会由观测账号经代理上传本局 probe 并创建对局。")
    run_id = default_run_id(args)
    main, observer = build_accounts(args)
    main.login()
    observer.login()
    models = public_model_list(observer)
    user_id = args.main_user_id if args.main_user_id is not None else int(main.user["id"])
    selected = select_public_model(
        models,
        user_id=user_id,
        user_name=args.main_user_name,
        model_name=args.main_public_name,
    )
    source = require_layout(args.layout)
    map_ids = args.map_id
    total_games = len(map_ids) * args.count_per_map
    print("即将执行：观测账号挑战主号公开 probe")
    print(f"run_id={run_id}")
    print(f"layout={args.layout}")
    print(f"observer_source={source}")
    print(f"main_public_model_id={selected.get('id')} name={selected.get('name')} user={selected.get('user_name_cn')}")
    print(f"observer_proxy={observer.proxy}")
    print(f"map_ids={map_ids}")
    print(f"count_per_map={args.count_per_map}")
    print(f"total_games={total_games}")
    append_manifest(
        run_id,
        {
            "action": "run-start",
            "direction": "observer-challenges-main-public",
            "layout": args.layout,
            "observer_source": str(source.relative_to(REPO_ROOT)),
            "main_public_model": selected,
            "map_ids": map_ids,
            "count_per_map": args.count_per_map,
            "observer_proxy": observer.proxy,
        },
    )
    seq = args.start_index
    for map_id in map_ids:
        for _ in range(args.count_per_map):
            model_name = make_model_name(args.model_prefix, args.layout, seq)
            seq += 1
            resp = submit_challenge(
                observer,
                layout=args.layout,
                map_id=map_id,
                model_name=model_name,
                opponent_model_id=int(selected["id"]),
            )
            game_id = int((resp.get("data") or {})["game_id"])
            append_manifest(
                run_id,
                {
                    "action": "create-game",
                    "direction": "observer-challenges-main-public",
                    "layout": args.layout,
                    "map_id": map_id,
                    "observer_model_name": model_name,
                    "main_public_model_id": int(selected["id"]),
                    "game_id": game_id,
                    "response": resp,
                },
            )
            print(f"created game_id={game_id} map_id={map_id} observer_model={model_name}")
            if args.sleep > 0:
                time.sleep(args.sleep)
    print(f"run ok manifest={manifest_path(run_id)}")


def cmd_fetch(args: argparse.Namespace) -> None:
    run_id = args.run_id
    game_ids = list(args.game_id or [])
    if not game_ids:
        game_ids = read_manifest_games(run_id)
    if not game_ids:
        raise SystemExit("没有可下载的 game_id")
    main, observer = build_accounts(args)
    main.login()
    observer.login()
    root = RAW_RUN_ROOT / run_id
    print(f"fetch run_id={run_id} game_count={len(game_ids)}")
    for game_id in game_ids:
        if args.wait:
            info = wait_for_game(observer, game_id, args.timeout, args.interval)
        else:
            info = game_info(observer, game_id)
        append_manifest(
            run_id,
            {
                "action": "game-info",
                "game_id": game_id,
                "observer_view_info": redact_game_info(info),
            },
        )
        if not game_ready(info):
            print(f"skip game_id={game_id} status={redact_game_info(info)}")
            continue
        for account, dirname in [(main, "main"), (observer, "observer")]:
            target = root / dirname / "replays" / f"{game_id}.ndjson"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size > 0 and not args.refresh:
                print(f"exists {target}")
                continue
            target.write_bytes(download_log(account, game_id))
            append_manifest(
                run_id,
                {
                    "action": "download-replay",
                    "game_id": game_id,
                    "account": dirname,
                    "path": str(target.relative_to(REPO_ROOT)),
                    "bytes": target.stat().st_size,
                },
            )
            print(f"downloaded {target}")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env",
        type=Path,
        action="append",
        help="额外 env 文件；缺省读取仓库根 .env 和 mechanism/scripts/observer/.env",
    )


def add_layout(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--layout", choices=sorted(LAYOUT_FILES), required=True, help="probe 布局")


def add_run_id(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", help="采集批次 ID；默认按布局和当前时间生成")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="只读检查账号、代理和可挑战列表")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("publish-main", help="主号上传供他人挑战的公开 probe")
    add_layout(p)
    add_run_id(p)
    p.add_argument("--yes", action="store_true", help="确认执行平台状态变更")
    p.set_defaults(func=cmd_publish_main)

    p = sub.add_parser("resolve-main", help="观测账号经代理解析主号公开模型 id")
    p.add_argument("--main-user-id", type=int, help="主号 user_id；默认用 LOGIN_KEY 登录结果")
    p.add_argument("--main-user-name", help="主号中文用户名过滤")
    p.add_argument("--main-public-name", help="公开模型名过滤，例如 player142")
    p.set_defaults(func=cmd_resolve_main)

    p = sub.add_parser("run", help="观测账号经代理挑战主号公开模型")
    add_layout(p)
    add_run_id(p)
    p.add_argument("--main-user-id", type=int, help="主号 user_id；默认用 LOGIN_KEY 登录结果")
    p.add_argument("--main-user-name", help="主号中文用户名过滤")
    p.add_argument("--main-public-name", help="公开模型名过滤，例如 player142")
    p.add_argument("--map-id", type=int, action="append", required=True, help="地图 ID，可重复")
    p.add_argument("--count-per-map", type=int, default=1, help="每张地图创建对局数")
    p.add_argument("--model-prefix", default="Obs", help="观测账号本局模型名前缀；必须字母开头且只含字母数字")
    p.add_argument("--start-index", type=int, default=1, help="观测账号模型名序号起点")
    p.add_argument("--sleep", type=float, default=0.0, help="每次创建对局后的暂停秒数")
    p.add_argument("--yes", action="store_true", help="确认执行平台状态变更")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("fetch", help="下载双方视角 replay")
    p.add_argument("--run-id", required=True, help="采集批次 ID")
    p.add_argument("--game-id", type=int, action="append", help="指定 game_id；可重复。缺省时从 manifest 读取")
    p.add_argument("--wait", action="store_true", help="轮询等待对局解析完成")
    p.add_argument("--timeout", type=int, default=600, help="单局等待超时秒数")
    p.add_argument("--interval", type=int, default=5, help="轮询间隔秒数")
    p.add_argument("--refresh", action="store_true", help="覆盖已下载 replay")
    p.set_defaults(func=cmd_fetch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except requests.RequestException as exc:
        raise SystemExit(f"网络请求失败：{exc}") from exc


if __name__ == "__main__":
    main()
