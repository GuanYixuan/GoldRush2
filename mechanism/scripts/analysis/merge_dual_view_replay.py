#!/usr/bin/env python3
"""离线合并同一对局的 main/observer 双视角 replay。

输入输出格式贴合 ``mechanism/data/raw/observer_runs``：

输入：
  mechanism/data/raw/observer_runs/<run_id>/main/replays/<game_id>.ndjson
  mechanism/data/raw/observer_runs/<run_id>/observer/replays/<game_id>.ndjson

输出：
  mechanism/data/processed/merged_replays/<run_id>/<game_id>.json

本脚本不访问评测平台，不读取 key。
输出只保存派生合并层；原始双视角 replay 通过顶层 source.views[*].log_ref 引用。
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_RUN_ROOT = REPO_ROOT / "mechanism" / "data" / "raw" / "observer_runs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "mechanism" / "data" / "processed" / "merged_replays"
ACCOUNT_DIRS = ("main", "observer")
GRID_SIZE = 17
FOG = -5
FORMAT = "goldrush2_merged_replay"
FORMAT_VERSION = 1


class MergeError(RuntimeError):
    """Raised when two view replays cannot be merged without guessing."""


@dataclass(frozen=True)
class ParsedReplay:
    account_dir: str
    path: Path
    players: dict[str, Any]
    maps: list[list[int]]
    rounds: list[dict[str, Any]]
    forfeit: dict[str, Any] | None
    viewer_side: int


def load_ndjson_replay(path: Path, account_dir: str) -> ParsedReplay:
    players: dict[str, Any] | None = None
    maps: list[list[int]] | None = None
    rounds: list[dict[str, Any]] = []
    forfeit: dict[str, Any] | None = None

    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MergeError(f"{path}:{line_no}: 非法 JSON 行：{exc}") from exc

            if is_map_row(obj):
                if maps is not None:
                    raise MergeError(f"{path}:{line_no}: 重复静态地图行")
                maps = normalize_grid(obj, f"{path}:{line_no}:maps")
            elif is_round(obj):
                rounds.append(obj)
            elif isinstance(obj, dict) and "forfeit" in obj:
                if forfeit is not None:
                    raise MergeError(f"{path}:{line_no}: 重复 forfeit 行")
                forfeit = obj
            elif isinstance(obj, dict) and players is None:
                players = obj
            else:
                raise MergeError(f"{path}:{line_no}: 无法识别 replay 行结构")

    if players is None:
        raise MergeError(f"{path}: 缺少玩家名行")
    if maps is None:
        raise MergeError(f"{path}: 缺少静态地图行")
    if not rounds:
        raise MergeError(f"{path}: 缺少轮次对象")

    viewer_side = infer_viewer_side(rounds, path)
    return ParsedReplay(
        account_dir=account_dir,
        path=path,
        players=players,
        maps=maps,
        rounds=rounds,
        forfeit=forfeit,
        viewer_side=viewer_side,
    )


def is_map_row(obj: Any) -> bool:
    return isinstance(obj, list) and len(obj) == GRID_SIZE and all(isinstance(row, list) for row in obj)


def is_round(obj: Any) -> bool:
    return isinstance(obj, dict) and {"round", "start", "end"}.issubset(obj.keys())


def normalize_grid(obj: Any, path: str) -> list[list[int]]:
    if not isinstance(obj, list) or len(obj) != GRID_SIZE:
        raise MergeError(f"{path}: 期望 {GRID_SIZE}x{GRID_SIZE} grid")
    out: list[list[int]] = []
    for r, row in enumerate(obj):
        if not isinstance(row, list) or len(row) != GRID_SIZE:
            raise MergeError(f"{path}[{r}]: 期望长度 {GRID_SIZE}")
        try:
            out.append([int(v) for v in row])
        except (TypeError, ValueError) as exc:
            raise MergeError(f"{path}[{r}]: grid 值无法转为 int") from exc
    return out


def infer_viewer_side(rounds: list[dict[str, Any]], path: Path) -> int:
    keys: set[str] = set()
    for round_obj in rounds:
        vision_r = round_obj.get("start", {}).get("vision_r")
        if isinstance(vision_r, dict):
            keys.update(str(k) for k in vision_r.keys())
    if keys != {"1"} and keys != {"2"}:
        raise MergeError(f"{path}: 无法从 start.vision_r 唯一推断 viewer_side，keys={sorted(keys)}")
    return int(next(iter(keys)))


def read_manifest_map_ids(run_id: str, raw_run_root: Path) -> dict[int, int]:
    path = raw_run_root / run_id / "manifest.jsonl"
    if not path.exists():
        return {}
    out: dict[int, int] = {}
    with path.open(encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            game_id = record.get("game_id")
            map_id = record.get("map_id")
            if isinstance(game_id, int) and isinstance(map_id, int):
                out[game_id] = map_id
    return out


def paired_game_ids(run_id: str, raw_run_root: Path) -> list[int]:
    root = raw_run_root / run_id
    id_sets: list[set[int]] = []
    for account_dir in ACCOUNT_DIRS:
        replay_dir = root / account_dir / "replays"
        if not replay_dir.exists():
            raise MergeError(f"缺少 replay 目录：{replay_dir}")
        ids: set[int] = set()
        for path in replay_dir.glob("*.ndjson"):
            try:
                ids.add(int(path.stem))
            except ValueError:
                continue
        id_sets.append(ids)
    return sorted(id_sets[0] & id_sets[1])


def merge_game(run_id: str, game_id: int, raw_run_root: Path = RAW_RUN_ROOT) -> dict[str, Any]:
    root = raw_run_root / run_id
    parsed_by_account = {
        account_dir: load_ndjson_replay(root / account_dir / "replays" / f"{game_id}.ndjson", account_dir)
        for account_dir in ACCOUNT_DIRS
    }
    parsed_by_side = index_by_viewer_side(parsed_by_account)
    view1 = parsed_by_side[1]
    view2 = parsed_by_side[2]

    assert_equal("players", view1.players, view2.players)
    assert_equal("maps", view1.maps, view2.maps)
    assert_rounds_compatible(view1.rounds, view2.rounds)
    assert_forfeit_compatible(view1.forfeit, view2.forfeit)

    map_ids = read_manifest_map_ids(run_id, raw_run_root)
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "source": build_source(run_id, game_id, map_ids.get(game_id), parsed_by_side),
        "players": copy.deepcopy(view1.players),
        "maps": copy.deepcopy(view1.maps),
        "rounds": [
            merge_round(game_id, round1, round2)
            for round1, round2 in zip(view1.rounds, view2.rounds)
        ],
        "forfeit": copy.deepcopy(view1.forfeit),
    }


def index_by_viewer_side(parsed_by_account: dict[str, ParsedReplay]) -> dict[int, ParsedReplay]:
    parsed_by_side: dict[int, ParsedReplay] = {}
    for replay in parsed_by_account.values():
        if replay.viewer_side in parsed_by_side:
            other = parsed_by_side[replay.viewer_side]
            raise MergeError(
                f"viewer_side={replay.viewer_side} 同时出现于 {other.account_dir} 和 {replay.account_dir}"
            )
        parsed_by_side[replay.viewer_side] = replay
    if set(parsed_by_side) != {1, 2}:
        raise MergeError(f"双视角不完整：实际 viewer_side={sorted(parsed_by_side)}")
    return parsed_by_side


def build_source(
    run_id: str,
    game_id: int,
    map_id: int | None,
    parsed_by_side: dict[int, ParsedReplay],
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "game_id": game_id,
        "run_id": run_id,
        "views": {},
    }
    if map_id is not None:
        source["map_id"] = map_id
    for side in (1, 2):
        replay = parsed_by_side[side]
        source["views"][str(side)] = {
            "viewer_side": side,
            "source_type": "platform_user_replay",
            "account_dir": replay.account_dir,
            "log_ref": relative_path(replay.path),
        }
    return source


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def assert_rounds_compatible(rounds1: list[dict[str, Any]], rounds2: list[dict[str, Any]]) -> None:
    if len(rounds1) != len(rounds2):
        raise MergeError(f"轮次数不一致：view1={len(rounds1)} view2={len(rounds2)}")
    for index, (round1, round2) in enumerate(zip(rounds1, rounds2)):
        if round1.get("round") != round2.get("round"):
            raise MergeError(
                f"round 编号不一致：index={index} view1={round1.get('round')} view2={round2.get('round')}"
            )


def assert_forfeit_compatible(forfeit1: dict[str, Any] | None, forfeit2: dict[str, Any] | None) -> None:
    assert_equal("forfeit", forfeit1, forfeit2)


def merge_round(game_id: int, round1: dict[str, Any], round2: dict[str, Any]) -> dict[str, Any]:
    round_no = require_int_round(round1)
    merged: dict[str, Any] = {
        "round": round_no,
        "merged": {
            "start": merge_frame(game_id, round_no, "start", round1["start"], round2["start"]),
            "end": merge_frame(game_id, round_no, "end", round1["end"], round2["end"]),
        },
    }
    if "snapshot" in round1 or "snapshot" in round2:
        if "snapshot" not in round1 or "snapshot" not in round2:
            raise MergeError(f"game_id={game_id} round={round_no} snapshot 单侧存在")
        assert_equal(f"game_id={game_id} round={round_no} snapshot", round1["snapshot"], round2["snapshot"])
        merged["merged"]["snapshot"] = copy.deepcopy(round1["snapshot"])
    return merged


def require_int_round(round_obj: dict[str, Any]) -> int:
    value = round_obj.get("round")
    if not isinstance(value, int):
        raise MergeError(f"非法 round 编号：{value!r}")
    return value


def merge_frame(
    game_id: int,
    round_no: int,
    phase: str,
    frame1: dict[str, Any],
    frame2: dict[str, Any],
) -> dict[str, Any]:
    grid1 = normalize_grid(frame1.get("grid"), f"game_id={game_id} round={round_no} {phase}.grid view1")
    grid2 = normalize_grid(frame2.get("grid"), f"game_id={game_id} round={round_no} {phase}.grid view2")
    merged_grid, visible_by = merge_grid(game_id, round_no, phase, grid1, grid2)
    merged: dict[str, Any] = {
        "grid": merged_grid,
        "visible_by": visible_by,
        "players": merge_players(game_id, round_no, phase, frame1.get("players", []), frame2.get("players", [])),
        "npcs": merge_npcs(game_id, round_no, phase, frame1.get("npcs", []), frame2.get("npcs", [])),
    }

    vision_r = merge_vision_r(game_id, round_no, phase, frame1.get("vision_r"), frame2.get("vision_r"))
    if vision_r:
        merged["vision_r"] = vision_r

    for key in ("dispatch_order", "burned"):
        copy_equal_optional_field(merged, key, frame1, frame2, f"game_id={game_id} round={round_no} {phase}.{key}")

    for key in ("overlap_events", "trample_events"):
        events = merge_events(frame1.get(key), frame2.get(key))
        if events:
            merged[key] = events
        elif key in frame1 or key in frame2:
            merged[key] = []
    return merged


def merge_grid(
    game_id: int,
    round_no: int,
    phase: str,
    grid1: list[list[int]],
    grid2: list[list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    merged_grid: list[list[int]] = []
    visible_by: list[list[int]] = []
    for r in range(GRID_SIZE):
        grid_row: list[int] = []
        visible_row: list[int] = []
        for c in range(GRID_SIZE):
            value1 = grid1[r][c]
            value2 = grid2[r][c]
            mask = visible_mask(value1 != FOG, value2 != FOG)
            if mask == 0:
                value = FOG
            elif mask == 1:
                value = value1
            elif mask == 2:
                value = value2
            else:
                if value1 != value2:
                    raise MergeError(
                        "共同可见 grid 冲突："
                        f"game_id={game_id} round={round_no} phase={phase} "
                        f"cell=({r},{c}) view1={value1} view2={value2}"
                    )
                value = value1
            grid_row.append(value)
            visible_row.append(mask)
        merged_grid.append(grid_row)
        visible_by.append(visible_row)
    return merged_grid, visible_by


def visible_mask(side1_visible: bool, side2_visible: bool) -> int:
    return (1 if side1_visible else 0) + (2 if side2_visible else 0)


def merge_vision_r(
    game_id: int,
    round_no: int,
    phase: str,
    vision1: Any,
    vision2: Any,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for side, value in ((1, vision1), (2, vision2)):
        if value is None:
            continue
        if not isinstance(value, dict):
            raise MergeError(f"game_id={game_id} round={round_no} {phase}.vision_r view{side} 不是对象")
        for key, radius in value.items():
            key = str(key)
            if key in merged and merged[key] != radius:
                raise MergeError(
                    f"game_id={game_id} round={round_no} {phase}.vision_r[{key}] 冲突："
                    f"{merged[key]!r} vs {radius!r}"
                )
            merged[key] = radius
    return merged


PLAYER_SCALAR_KEYS = ("cost", "gold", "vision_spent", "order")
UNIT_KEYS = ("position", "gold", "actions", "pickup")
NPC_KEYS = ("position", "gold", "actions", "pickup", "cost")


def merge_players(
    game_id: int,
    round_no: int,
    phase: str,
    players1: Any,
    players2: Any,
) -> list[dict[str, Any]]:
    by_side = {
        1: index_entities(players1, "players", f"game_id={game_id} round={round_no} {phase}.players view1"),
        2: index_entities(players2, "players", f"game_id={game_id} round={round_no} {phase}.players view2"),
    }
    player_ids = sorted(set(by_side[1]) | set(by_side[2]))
    merged_players: list[dict[str, Any]] = []
    for player_id in player_ids:
        if player_id not in (1, 2):
            raise MergeError(f"game_id={game_id} round={round_no} {phase}: 非法玩家 id={player_id!r}")
        preferred_side = player_id
        preferred = by_side[preferred_side].get(player_id)
        fallback = by_side[3 - preferred_side].get(player_id)
        base = preferred or fallback
        if base is None:
            continue
        merged: dict[str, Any] = {"id": player_id}
        for key in PLAYER_SCALAR_KEYS:
            value, _source = choose_equal_field(
                by_side[1].get(player_id),
                by_side[2].get(player_id),
                key,
                f"game_id={game_id} round={round_no} {phase}.players[{player_id}].{key}",
                preferred_side=preferred_side,
            )
            if value is not None:
                merged[key] = value
        merged["units"] = merge_units(
            game_id,
            round_no,
            phase,
            player_id,
            by_side[1].get(player_id),
            by_side[2].get(player_id),
            preferred_side,
        )
        merged_players.append(merged)
    return merged_players


def merge_units(
    game_id: int,
    round_no: int,
    phase: str,
    player_id: int,
    player1: dict[str, Any] | None,
    player2: dict[str, Any] | None,
    preferred_side: int,
) -> list[dict[str, Any]]:
    units_by_side = {
        1: list_or_empty(player1.get("units") if player1 else None),
        2: list_or_empty(player2.get("units") if player2 else None),
    }
    unit_count = max(len(units_by_side[1]), len(units_by_side[2]))
    merged_units: list[dict[str, Any]] = []
    for index in range(unit_count):
        unit1 = units_by_side[1][index] if index < len(units_by_side[1]) else None
        unit2 = units_by_side[2][index] if index < len(units_by_side[2]) else None
        if unit1 is not None and not isinstance(unit1, dict):
            raise MergeError(f"game_id={game_id} round={round_no} {phase}.player{player_id}.units[{index}] view1 非对象")
        if unit2 is not None and not isinstance(unit2, dict):
            raise MergeError(f"game_id={game_id} round={round_no} {phase}.player{player_id}.units[{index}] view2 非对象")
        merged: dict[str, Any] = {}
        field_sources: dict[str, int] = {}
        for key in UNIT_KEYS:
            value, source = choose_equal_field(
                unit1,
                unit2,
                key,
                f"game_id={game_id} round={round_no} {phase}.player{player_id}.units[{index}].{key}",
                preferred_side=preferred_side,
            )
            if value is not None:
                merged[key] = value
            field_sources[key] = source

        merged["visible_by"] = visible_mask(position_visible(unit1), position_visible(unit2))
        merged["position_source"] = field_sources["position"]
        merged["actions_source"] = field_sources["actions"]
        merged["is_complete"] = preferred_side == player_id and position_visible(unit1 if player_id == 1 else unit2)
        merged_units.append(merged)
    return merged_units


def merge_npcs(
    game_id: int,
    round_no: int,
    phase: str,
    npcs1: Any,
    npcs2: Any,
) -> list[dict[str, Any]]:
    by_side = {
        1: index_entities(npcs1, "npcs", f"game_id={game_id} round={round_no} {phase}.npcs view1"),
        2: index_entities(npcs2, "npcs", f"game_id={game_id} round={round_no} {phase}.npcs view2"),
    }
    npc_ids = sorted(set(by_side[1]) | set(by_side[2]))
    merged_npcs: list[dict[str, Any]] = []
    for npc_id in npc_ids:
        npc1 = by_side[1].get(npc_id)
        npc2 = by_side[2].get(npc_id)
        preferred_side = 1 if completeness_score(npc1) >= completeness_score(npc2) else 2
        merged: dict[str, Any] = {"id": npc_id}
        field_sources: dict[str, int] = {}
        for key in NPC_KEYS:
            value, source = choose_equal_field(
                npc1,
                npc2,
                key,
                f"game_id={game_id} round={round_no} {phase}.npcs[{npc_id}].{key}",
                preferred_side=preferred_side,
            )
            if value is not None:
                merged[key] = value
            field_sources[key] = source
        merged["visible_by"] = visible_mask(npc1 is not None, npc2 is not None)
        merged["position_source"] = field_sources["position"]
        merged["actions_source"] = field_sources["actions"]
        merged["is_complete"] = False
        merged_npcs.append(merged)
    return merged_npcs


def index_entities(value: Any, label: str, path: str) -> dict[int, dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, list):
        raise MergeError(f"{path}: {label} 不是数组")
    out: dict[int, dict[str, Any]] = {}
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise MergeError(f"{path}[{index}]: 不是对象")
        entity_id = item.get("id")
        if not isinstance(entity_id, int):
            raise MergeError(f"{path}[{index}].id: 非法 id={entity_id!r}")
        if entity_id in out:
            raise MergeError(f"{path}: 重复 id={entity_id}")
        out[entity_id] = item
    return out


def list_or_empty(value: Any) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise MergeError(f"units 不是数组：{value!r}")
    return value


def choose_equal_field(
    side1: dict[str, Any] | None,
    side2: dict[str, Any] | None,
    key: str,
    path: str,
    preferred_side: int,
) -> tuple[Any | None, int]:
    has1 = side1 is not None and key in side1 and side1[key] is not None
    has2 = side2 is not None and key in side2 and side2[key] is not None
    if has1 and has2:
        value1 = side1[key]  # type: ignore[index]
        value2 = side2[key]  # type: ignore[index]
        if value1 != value2:
            raise MergeError(f"{path} 冲突：view1={value1!r} view2={value2!r}")
        return copy.deepcopy(value1), 3
    if has1 and (preferred_side == 1 or not has2):
        return copy.deepcopy(side1[key]), 1  # type: ignore[index]
    if has2:
        return copy.deepcopy(side2[key]), 2  # type: ignore[index]
    return None, 0


def position_visible(entity: dict[str, Any] | None) -> bool:
    return entity is not None and entity.get("position") is not None


def completeness_score(entity: dict[str, Any] | None) -> int:
    if entity is None:
        return -1
    return sum(1 for key in NPC_KEYS if key in entity and entity[key] is not None)


def copy_equal_optional_field(
    out: dict[str, Any],
    key: str,
    frame1: dict[str, Any],
    frame2: dict[str, Any],
    path: str,
) -> None:
    if key not in frame1 and key not in frame2:
        return
    if key not in frame1 or key not in frame2:
        raise MergeError(f"{path}: 单侧存在")
    assert_equal(path, frame1[key], frame2[key])
    out[key] = copy.deepcopy(frame1[key])


def merge_events(events1: Any, events2: Any) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    source_by_key: dict[str, int] = {}
    for side, events in ((1, events1), (2, events2)):
        if events is None:
            continue
        if not isinstance(events, list):
            raise MergeError(f"事件字段不是数组：{events!r}")
        for event in events:
            if not isinstance(event, dict):
                raise MergeError(f"事件不是对象：{event!r}")
            key = canonical_json(event)
            by_key.setdefault(key, copy.deepcopy(event))
            source_by_key[key] = source_by_key.get(key, 0) | side
    merged: list[dict[str, Any]] = []
    for key in sorted(by_key):
        event = by_key[key]
        event["visible_by"] = source_by_key[key]
        merged.append(event)
    return merged


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def assert_equal(path: str, value1: Any, value2: Any) -> None:
    if value1 != value2:
        raise MergeError(f"{path} 不一致：view1={value1!r} view2={value2!r}")


def process_run(
    run_id: str,
    game_ids: list[int] | None,
    raw_run_root: Path = RAW_RUN_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    dry_run: bool = False,
) -> list[Path]:
    ids = game_ids or paired_game_ids(run_id, raw_run_root)
    if not ids:
        raise MergeError(f"run_id={run_id}: 未找到 main/observer 双视角 replay 交集")

    written: list[Path] = []
    for game_id in ids:
        merged = merge_game(run_id, game_id, raw_run_root)
        output_path = output_root / run_id / f"{game_id}.json"
        if dry_run:
            print(f"ok game_id={game_id} output={relative_path(output_path)} dry_run=1")
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            f.write("\n")
        written.append(output_path)
        print(f"wrote {relative_path(output_path)}")
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="离线合并 main/observer 双视角 GoldRush2 replay")
    parser.add_argument("--run-id", required=True, help="mechanism/data/raw/observer_runs 下的 run_id")
    parser.add_argument("--game-id", type=int, action="append", help="只合并指定 game_id；可重复传入")
    parser.add_argument("--raw-run-root", type=Path, default=RAW_RUN_ROOT, help="observer_runs 根目录")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="merged replay 输出根目录")
    parser.add_argument("--dry-run", action="store_true", help="只校验和合并，不写出文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    process_run(
        run_id=args.run_id,
        game_ids=args.game_id,
        raw_run_root=args.raw_run_root,
        output_root=args.output_root,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
