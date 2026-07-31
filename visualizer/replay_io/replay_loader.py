from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .replay_types import Grid, RawObject, ReplayDocument, ReplayKind, ReplayLoadError, ReplayRoundRecord

GRID_SIZE = 17
MERGED_FORMAT = "goldrush2_merged_replay"
SIMULATOR_FORMAT = "goldrush2_simulator_full_replay"


class ReplayLoader:
    """Load official NDJSON or merged JSON replay files."""

    @staticmethod
    def load(path: str | Path) -> ReplayDocument:
        source_path = Path(path).expanduser()
        try:
            text = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ReplayLoadError(f"读取 replay 失败: {source_path}: {exc}") from exc

        if not text.strip():
            raise ReplayLoadError(f"replay 文件为空: {source_path}")
        return _load_auto(source_path, text)


def _load_auto(source_path: Path, text: str) -> ReplayDocument:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        if exc.msg == "Extra data":
            return _load_official_ndjson(source_path, text)
        stripped = text.lstrip()
        if stripped.startswith("{"):
            raise ReplayLoadError(f"JSON 解析失败: line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
        return _load_official_ndjson(source_path, text)
    if isinstance(payload, dict) and payload.get("format") == MERGED_FORMAT:
        return _load_merged_document(source_path, payload)
    if isinstance(payload, dict) and payload.get("format") == SIMULATOR_FORMAT:
        return _load_simulator_document(source_path, payload)
    raise ReplayLoadError("JSON replay 目前只支持 goldrush2_merged_replay 或 goldrush2_simulator_full_replay 格式")


def _load_json_document(source_path: Path, text: str) -> ReplayDocument:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReplayLoadError(f"JSON 解析失败: line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    if isinstance(payload, dict) and payload.get("format") == MERGED_FORMAT:
        return _load_merged_document(source_path, payload)
    if isinstance(payload, dict) and payload.get("format") == SIMULATOR_FORMAT:
        return _load_simulator_document(source_path, payload)
    raise ReplayLoadError("JSON replay 目前只支持 goldrush2_merged_replay 或 goldrush2_simulator_full_replay 格式")


def _load_merged_document(source_path: Path, payload: dict[str, Any]) -> ReplayDocument:
    players = _players(payload.get("players"))
    static_map = _grid(payload.get("maps"), "maps")
    rounds_payload = payload.get("rounds")
    if not isinstance(rounds_payload, list):
        raise ReplayLoadError("merged replay 缺少 rounds 数组")

    rounds: list[ReplayRoundRecord] = []
    for index, item in enumerate(rounds_payload):
        if not isinstance(item, dict):
            raise ReplayLoadError(f"rounds[{index}] 不是对象")
        round_index = _int(item.get("round"), f"rounds[{index}].round")
        merged = item.get("merged")
        if not isinstance(merged, dict):
            raise ReplayLoadError(f"rounds[{index}].merged 不是对象")
        rounds.append(
            ReplayRoundRecord(
                round_index=round_index,
                merged=merged,
                raw_round=None,
                viewer_side=None,
            )
        )

    return ReplayDocument(
        kind=ReplayKind.MERGED,
        source_path=source_path.resolve(),
        players=players,
        static_map=static_map,
        rounds=tuple(rounds),
        forfeit=_optional_object(payload.get("forfeit")),
    )


def _load_simulator_document(source_path: Path, payload: dict[str, Any]) -> ReplayDocument:
    players = _players(payload.get("players"))
    static_map = _grid(payload.get("maps"), "maps")
    rounds_payload = payload.get("rounds")
    if not isinstance(rounds_payload, list):
        raise ReplayLoadError("simulator replay 缺少 rounds 数组")

    rounds: list[ReplayRoundRecord] = []
    for index, item in enumerate(rounds_payload):
        if not isinstance(item, dict):
            raise ReplayLoadError(f"rounds[{index}] 不是对象")
        round_index = _int(item.get("round"), f"rounds[{index}].round")
        if not isinstance(item.get("start"), dict) or not isinstance(item.get("end"), dict):
            raise ReplayLoadError(f"rounds[{index}] 缺少 start/end 对象")
        rounds.append(
            ReplayRoundRecord(
                round_index=round_index,
                merged=None,
                raw_round=item,
                viewer_side=None,
            )
        )

    return ReplayDocument(
        kind=ReplayKind.SIMULATOR_FULL,
        source_path=source_path.resolve(),
        players=players,
        static_map=static_map,
        rounds=tuple(rounds),
        forfeit=_optional_object(payload.get("forfeit")),
    )


def _load_official_ndjson(source_path: Path, text: str) -> ReplayDocument:
    items: list[Any] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            items.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ReplayLoadError(f"NDJSON 第 {lineno} 行解析失败: {exc.msg}") from exc

    players_payload = next((item for item in items if _looks_like_players(item)), None)
    if players_payload is None:
        raise ReplayLoadError("official replay 缺少玩家名对象")
    players = _players(players_payload)

    map_payload = next((item for item in items if _looks_like_grid(item)), None)
    if map_payload is None:
        raise ReplayLoadError("official replay 缺少 17x17 静态地图")
    static_map = _grid(map_payload, "static map")

    rounds: list[ReplayRoundRecord] = []
    for item in items:
        if not isinstance(item, dict) or "round" not in item or "start" not in item or "end" not in item:
            continue
        round_index = _int(item.get("round"), "round.round")
        rounds.append(
            ReplayRoundRecord(
                round_index=round_index,
                merged=None,
                raw_round=item,
                viewer_side=None,
            )
        )
    if not rounds:
        raise ReplayLoadError("official replay 不包含 round/start/end 对象")

    forfeit = None
    for item in items:
        if isinstance(item, dict) and "forfeit" in item:
            forfeit = item
            break

    return ReplayDocument(
        kind=ReplayKind.OFFICIAL,
        source_path=source_path.resolve(),
        players=players,
        static_map=static_map,
        rounds=tuple(rounds),
        forfeit=forfeit,
    )


def _players(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ReplayLoadError("players 不是对象")
    return {
        "player1": str(value.get("player1", "player1")),
        "player2": str(value.get("player2", "player2")),
    }


def _grid(value: Any, path: str) -> Grid:
    if not _looks_like_grid(value):
        raise ReplayLoadError(f"{path} 不是 17x17 网格")
    return tuple(tuple(_int(cell, f"{path}[{row_index}][{col_index}]") for col_index, cell in enumerate(row)) for row_index, row in enumerate(value))


def _looks_like_players(value: Any) -> bool:
    return isinstance(value, dict) and ("player1" in value or "player2" in value)


def _looks_like_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == GRID_SIZE
        and all(isinstance(row, list) and len(row) == GRID_SIZE for row in value)
    )


def _optional_object(value: Any) -> RawObject | None:
    return value if isinstance(value, dict) else None


def _int(value: Any, path: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ReplayLoadError(f"{path} 不是整数: {value!r}") from exc


__all__ = ["ReplayLoader", "ReplayLoadError"]
