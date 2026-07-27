#!/usr/bin/env python3
"""Read-only helpers for GoldRush platform game lists and replay stats."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


BASE_URL = "http://47.103.127.219"
LOCAL_TZ = ZoneInfo("Asia/Shanghai")
DEFAULT_CACHE_DIR = Path("/tmp/goldrush_logs")


@dataclass
class PlayerStats:
    player_id: int
    model: str
    gross: int
    vision_spent: int
    net: int
    p50: int
    p90: int
    p95: int
    p99: int
    avg: float
    max_cost: int
    first_moves: int
    first_ratio: float


def read_env_key(env_path: Path) -> str:
    if "LOGIN_KEY" in os.environ:
        return os.environ["LOGIN_KEY"]
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "LOGIN_KEY":
                return value.strip().strip("\"'")
    except FileNotFoundError:
        pass
    raise SystemExit("LOGIN_KEY not found in environment or .env")


def request_json(url: str, token: str | None = None, data: bytes | None = None) -> Any:
    headers = {}
    if token is not None:
        headers["Authorization"] = token
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} for {url}: {body}") from exc


def login(env_path: Path) -> str:
    key = read_env_key(env_path)
    payload = json.dumps({"key": key}).encode("utf-8")
    resp = request_json(f"{BASE_URL}/api/user/login", data=payload)
    token = ((resp.get("data") or {}).get("token")) if isinstance(resp, dict) else None
    if not token:
        raise SystemExit(f"login failed: code={resp.get('code')} message={resp.get('message')!r}")
    return token


def get_game_list(token: str, page_size: int) -> list[dict[str, Any]]:
    url = f"{BASE_URL}/api/user/get_game_list_1?{urllib.parse.urlencode({'page_size': page_size})}"
    resp = request_json(url, token=token)
    return list(((resp.get("data") or {}).get("list")) or [])


def parse_since(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        dt = datetime.fromisoformat(text[:-1] + "+00:00")
    elif "T" in text and ("+" in text[10:] or text.endswith("+00:00")):
        dt = datetime.fromisoformat(text)
    else:
        if len(text) == 10:
            text += " 00:00:00"
        dt = datetime.fromisoformat(text).replace(tzinfo=LOCAL_TZ)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def format_local_time(utc_text: str) -> str:
    try:
        dt = datetime.fromisoformat(utc_text.replace("Z", "+00:00"))
    except ValueError:
        return utc_text
    return dt.astimezone(LOCAL_TZ).strftime("%m-%d %H:%M:%S")


def game_has_model(game: dict[str, Any], model: str | None) -> bool:
    if not model:
        return True
    return any((p.get("model_name") == model) for p in game.get("players", []))


def game_has_user(game: dict[str, Any], user: str | None) -> bool:
    if not user:
        return True
    return any((p.get("user_name_cn") == user) for p in game.get("players", []))


def result_for_model(game: dict[str, Any], model: str | None) -> str:
    if not model:
        return ""
    for p in game.get("players", []):
        if p.get("model_name") == model:
            return "胜" if p.get("is_win") else "负"
    return ""


def coins_for_model(game: dict[str, Any], model: str | None) -> str:
    if not model:
        return " / ".join(f"{p.get('model_name')}={p.get('coin_num')}" for p in game.get("players", []))
    ours = None
    others: list[str] = []
    for p in game.get("players", []):
        if p.get("model_name") == model:
            ours = p.get("coin_num")
        else:
            others.append(str(p.get("coin_num")))
    if ours is None:
        return ""
    return f"{ours} vs {'/'.join(others)}"


def print_markdown(headers: list[str], rows: list[list[Any]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(str(x) for x in row) + " |")


def cmd_list(args: argparse.Namespace) -> None:
    token = login(args.env)
    since = parse_since(args.since)
    games = get_game_list(token, args.page_size)
    rows: list[list[Any]] = []
    for game in games:
        if since and game.get("created_at", "") < since:
            continue
        if not game_has_model(game, args.model):
            continue
        if not game_has_user(game, args.user):
            continue
        opponent = ""
        if args.model:
            opponents = [
                f"{p.get('user_name_cn')} / {p.get('model_name')}"
                for p in game.get("players", [])
                if p.get("model_name") != args.model
            ]
            opponent = "; ".join(opponents)
        rows.append(
            [
                game.get("id", ""),
                format_local_time(str(game.get("created_at", ""))),
                game.get("map_name", ""),
                result_for_model(game, args.model),
                opponent,
                coins_for_model(game, args.model),
                game.get("is_upload_log", ""),
                game.get("is_parse_log", ""),
                game.get("error_msg", ""),
            ]
        )
    print_markdown(
        ["game_id", "时间", "地图", "结果", "对手", "金币", "upload", "parse", "error"],
        rows,
    )


def load_log(game_id: int, token: str, cache_dir: Path, refresh: bool) -> list[Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{game_id}.ndjson"
    if refresh or not path.exists() or path.stat().st_size == 0:
        url = f"{BASE_URL}/api/user/get_game_log?{urllib.parse.urlencode({'id': game_id})}"
        req = urllib.request.Request(url, headers={"Authorization": token})
        with urllib.request.urlopen(req, timeout=60) as resp:
            path.write_bytes(resp.read())
    items: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def percentile(values: list[int], numerator: int, denominator: int = 100) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = ((len(ordered) * numerator + (denominator - 1)) // denominator) - 1
    return ordered[max(0, min(idx, len(ordered) - 1))]


def fmt_ns(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}".rstrip("0").rstrip(".") + "ms"
    if value >= 1_000:
        return f"{value / 1_000:.2f}".rstrip("0").rstrip(".") + "us"
    return f"{value:.2f}".rstrip("0").rstrip(".") + "ns"


def first_player(round_obj: dict[str, Any]) -> int:
    order = ((round_obj.get("end") or {}).get("dispatch_order")) or []
    for item in order:
        if isinstance(item, int) and item > 0:
            return item
    players = ((round_obj.get("end") or {}).get("players")) or []
    if len(players) >= 2:
        return int(players[0]["id"] if players[0].get("cost", 0) <= players[1].get("cost", 0) else players[1]["id"])
    return 0


def stats_for_items(items: list[Any]) -> tuple[dict[int, PlayerStats], int, int]:
    if not items or not isinstance(items[0], dict):
        raise ValueError("invalid replay log: missing player-name object")
    names = {1: items[0].get("player1", "player1"), 2: items[0].get("player2", "player2")}
    rounds = [x for x in items if isinstance(x, dict) and "round" in x and "end" in x]
    forfeit_count = sum(1 for x in items if isinstance(x, dict) and "forfeit" in x)
    first_counts = {1: 0, 2: 0}
    costs = {1: [], 2: []}
    final_gold = {1: 0, 2: 0}
    final_spent = {1: 0, 2: 0}
    for rnd in rounds:
        fp = first_player(rnd)
        if fp in first_counts:
            first_counts[fp] += 1
        for player in (rnd.get("end") or {}).get("players", []):
            pid = int(player.get("id", 0))
            if pid not in costs:
                continue
            costs[pid].append(int(player.get("cost", 0)))
            final_gold[pid] = int(player.get("gold", 0))
            final_spent[pid] = int(player.get("vision_spent", 0))
    result: dict[int, PlayerStats] = {}
    total = len(rounds)
    for pid in (1, 2):
        vals = costs[pid]
        avg = sum(vals) / len(vals) if vals else 0.0
        result[pid] = PlayerStats(
            player_id=pid,
            model=str(names[pid]),
            gross=final_gold[pid],
            vision_spent=final_spent[pid],
            net=final_gold[pid] - final_spent[pid],
            p50=percentile(vals, 50),
            p90=percentile(vals, 90),
            p95=percentile(vals, 95),
            p99=percentile(vals, 99),
            avg=avg,
            max_cost=max(vals) if vals else 0,
            first_moves=first_counts[pid],
            first_ratio=(first_counts[pid] / total * 100.0) if total else 0.0,
        )
    return result, total, forfeit_count


def cmd_stats(args: argparse.Namespace) -> None:
    token = login(args.env)
    rows: list[list[Any]] = []
    for raw_id in args.game_ids:
        game_id = int(raw_id)
        stats, rounds, forfeit_count = stats_for_items(load_log(game_id, token, args.cache_dir, args.refresh))
        for pid in (1, 2):
            s = stats[pid]
            if args.focus and s.model != args.focus:
                continue
            rows.append(
                [
                    game_id,
                    s.player_id,
                    s.model,
                    rounds,
                    forfeit_count,
                    s.gross,
                    s.vision_spent,
                    s.net,
                    fmt_ns(s.p50),
                    fmt_ns(s.p90),
                    fmt_ns(s.p99),
                    fmt_ns(s.avg),
                    fmt_ns(s.max_cost),
                    s.first_moves,
                    f"{s.first_ratio:.1f}%",
                ]
            )
    print_markdown(
        [
            "game_id",
            "pid",
            "模型",
            "rounds",
            "forfeit",
            "毛金币",
            "视野花费",
            "净金币",
            "P50",
            "P90",
            "P99",
            "Avg",
            "Max",
            "先手回合",
            "先手比",
        ],
        rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=Path(".env"), help="path to .env containing LOGIN_KEY")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list public games")
    p_list.add_argument("--since", help="local time like '2026-07-26 18:58:48' or UTC ISO")
    p_list.add_argument("--model", help="filter by model_name, e.g. player142")
    p_list.add_argument("--user", help="filter by user_name_cn")
    p_list.add_argument("--page-size", type=int, default=1000)
    p_list.set_defaults(func=cmd_list)

    p_stats = sub.add_parser("stats", help="compute replay statistics for game ids")
    p_stats.add_argument("game_ids", nargs="+")
    p_stats.add_argument("--focus", help="only show this model name")
    p_stats.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p_stats.add_argument("--refresh", action="store_true", help="redownload logs even if cached")
    p_stats.set_defaults(func=cmd_stats)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
