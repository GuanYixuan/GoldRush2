#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


BASE_URL = "http://47.103.127.219"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV = REPO_ROOT / ".env"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "temp" / "submission_runs" / "platform_selfplay"
CPP_LANG = "2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit one C++ .so self-play game and download replay.")
    parser.add_argument("--so", type=Path, required=True)
    parser.add_argument("--map-id", type=int, default=2)
    parser.add_argument("--model-a", default=None)
    parser.add_argument("--model-b", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--interval-s", type=int, default=10)
    args = parser.parse_args()

    so_path = args.so.resolve()
    if not so_path.exists():
        raise FileNotFoundError(so_path)
    if so_path.suffix != ".so":
        raise ValueError(f"expected .so file, got {so_path}")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_a = _model_name(args.model_a or f"A{timestamp[-10:]}A")
    model_b = _model_name(args.model_b or f"A{timestamp[-10:]}B")
    run_dir = args.output_root.resolve() / f"selfplay_{timestamp}_map{int(args.map_id)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    token = _login(session, _read_login_key(DEFAULT_ENV))
    create_resp = _submit_selfplay(
        session,
        token=token,
        so_path=so_path,
        map_id=int(args.map_id),
        model_a=model_a,
        model_b=model_b,
    )
    game_id = int((create_resp.get("data") or {})["game_id"])
    _write_json(run_dir / "create_response.json", _redact(create_resp))
    _write_json(
        run_dir / "submission.json",
        {
            "schema": "goldrush2_platform_selfplay_submission_v1",
            "game_id": game_id,
            "map_id": int(args.map_id),
            "model_a": model_a,
            "model_b": model_b,
            "so_path": str(so_path),
            "so_bytes": so_path.stat().st_size,
        },
    )

    info = _wait_for_game(session, token=token, game_id=game_id, timeout_s=int(args.timeout_s), interval_s=int(args.interval_s))
    _write_json(run_dir / "game_info.json", info)
    replay_path = run_dir / f"{game_id}.ndjson"
    replay_path.write_bytes(_download_log(session, token=token, game_id=game_id))
    summary = {
        "schema": "goldrush2_platform_selfplay_result_v1",
        "game_id": game_id,
        "map_id": int(args.map_id),
        "model_a": model_a,
        "model_b": model_b,
        "run_dir": str(run_dir),
        "replay_path": str(replay_path),
        "replay_bytes": replay_path.stat().st_size,
        "game_info": _redact_game_info(info),
        "forfeit_count": _forfeit_count(replay_path),
    }
    _write_json(run_dir / "result.json", summary)
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))


def _read_login_key(path: Path) -> str:
    if os.environ.get("LOGIN_KEY"):
        return os.environ["LOGIN_KEY"]
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "LOGIN_KEY":
            return value.strip().strip("\"'")
    raise RuntimeError("LOGIN_KEY not found")


def _login(session: requests.Session, key: str) -> str:
    resp = session.post(f"{BASE_URL}/api/user/login", json={"key": key}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    token = ((payload.get("data") or {}).get("token")) if isinstance(payload, dict) else None
    if not token:
        raise RuntimeError(f"login failed: code={payload.get('code')} message={payload.get('message')!r}")
    return str(token)


def _submit_selfplay(
    session: requests.Session,
    *,
    token: str,
    so_path: Path,
    map_id: int,
    model_a: str,
    model_b: str,
) -> dict[str, Any]:
    with so_path.open("rb") as file_a, so_path.open("rb") as file_b:
        files = [
            ("model_files", (f"{model_a}.so", file_a, "application/octet-stream")),
            ("model_files", (f"{model_b}.so", file_b, "application/octet-stream")),
        ]
        data = [
            ("map_id", str(map_id)),
            ("model_langs", CPP_LANG),
            ("model_langs", CPP_LANG),
            ("model_names", model_a),
            ("model_names", model_b),
        ]
        resp = session.post(f"{BASE_URL}/api/user/add_model_1", headers={"Authorization": token}, data=data, files=files, timeout=120)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict) or payload.get("code") != 0 or not ((payload.get("data") or {}).get("game_id")):
        raise RuntimeError(f"create game failed: code={payload.get('code')} message={payload.get('message')!r}")
    return payload


def _game_info(session: requests.Session, *, token: str, game_id: int) -> dict[str, Any]:
    resp = session.get(f"{BASE_URL}/api/user/get_game_info", headers={"Authorization": token}, params={"id": game_id}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    return data if isinstance(data, dict) else payload


def _wait_for_game(
    session: requests.Session,
    *,
    token: str,
    game_id: int,
    timeout_s: int,
    interval_s: int,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    while True:
        last = _game_info(session, token=token, game_id=game_id)
        if int(last.get("is_parse_log") or 0) == 1:
            return last
        if int(last.get("is_upload_log") or 0) == 2:
            return last
        if time.time() >= deadline:
            raise TimeoutError(f"game {game_id} timed out; last status={_redact_game_info(last)}")
        time.sleep(max(1, interval_s))


def _download_log(session: requests.Session, *, token: str, game_id: int) -> bytes:
    resp = session.get(f"{BASE_URL}/api/user/get_game_log", headers={"Authorization": token}, params={"id": game_id}, timeout=120)
    resp.raise_for_status()
    return resp.content


def _model_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", value):
        raise ValueError(f"model name must start with a letter and contain only letters/digits: {value!r}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ("<redacted>" if key.lower() in {"token", "authorization"} else _redact(item)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _redact_game_info(info: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "id",
        "stage",
        "user_id",
        "user_id2",
        "map_id",
        "map_name",
        "is_upload_log",
        "is_parse_log",
        "error_msg",
        "created_at",
        "updated_at",
        "players",
    }
    return {key: value for key, value in info.items() if key in keep}


def _forfeit_count(path: Path) -> int:
    count = 0
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "\"forfeit\"" in raw:
            count += 1
    return count


if __name__ == "__main__":
    main()
