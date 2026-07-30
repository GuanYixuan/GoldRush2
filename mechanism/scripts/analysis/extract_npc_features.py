#!/usr/bin/env python3
"""从双视角合并 replay 抽取 NPC 行为建模特征表。

默认输入：
  mechanism/data/processed/merged_replays/<run_id>/*.json

默认输出：
  mechanism/data/processed/npc_features/<run_id>/

本脚本只做离线处理，不访问评测平台。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MERGED_ROOT = REPO_ROOT / "mechanism" / "data" / "processed" / "merged_replays"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "mechanism" / "data" / "processed" / "npc_features"
GRID_SIZE = 17
FOG = -5
BOMB = -3
OBSTACLE = -1
ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
    4: (0, 0),
}


class NpcFeatureError(RuntimeError):
    pass


def region_of(row: int, col: int) -> int:
    if 4 <= row <= 12 and 4 <= col <= 12:
        return 1
    if 0 <= row <= 3 and 0 <= col <= 12:
        return 2
    if 4 <= row <= 16 and 0 <= col <= 3:
        return 3
    if 13 <= row <= 16 and 4 <= col <= 16:
        return 4
    if 0 <= row <= 12 and 13 <= col <= 16:
        return 5
    raise NpcFeatureError(f"坐标不属于任何区域：({row},{col})")


def read_merged(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != "goldrush2_merged_replay":
        raise NpcFeatureError(f"{path}: 非 merged replay")
    return data


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def validate_static_map(static_map: Any, path: Path) -> None:
    if not isinstance(static_map, list) or len(static_map) != GRID_SIZE:
        raise NpcFeatureError(f"{path}: 静态地图不是 {GRID_SIZE}x{GRID_SIZE}")
    for row, values in enumerate(static_map):
        if not isinstance(values, list) or len(values) != GRID_SIZE:
            raise NpcFeatureError(f"{path}: 静态地图第 {row} 行长度非法")


def npc_by_id(frame: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for npc in frame.get("npcs", []):
        pos = npc.get("position")
        if pos is None:
            continue
        out[int(npc["id"])] = npc
    return out


def player_units(frame: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for player in frame.get("players", []):
        player_id = int(player["id"])
        order = player.get("order", 0)
        for unit_index, unit in enumerate(player.get("units", [])):
            pos = unit.get("position")
            if pos is None:
                continue
            out[(player_id, unit_index)] = {
                "player_id": player_id,
                "unit_index": unit_index,
                "order": order,
                "position": tuple(pos),
                "actions": unit.get("actions", []),
            }
    return out


def player_positions(frame: dict[str, Any]) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for player in frame.get("players", []):
        for unit in player.get("units", []):
            pos = unit.get("position")
            if pos is not None:
                positions.append(tuple(pos))
    return positions


def npc_positions(frame: dict[str, Any]) -> list[tuple[int, int]]:
    return [tuple(npc["position"]) for npc in frame.get("npcs", []) if npc.get("position") is not None]


def visible_cells(grid: list[list[int]], value: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if grid[r][c] == value]


def nearest_distance(pos: tuple[int, int], targets: list[tuple[int, int]]) -> int | None:
    if not targets:
        return None
    row, col = pos
    return min(abs(row - tr) + abs(col - tc) for tr, tc in targets)


def gold_context(grid: list[list[int]], pos: tuple[int, int], radius: int = 3) -> dict[str, int | None]:
    row, col = pos
    nearest: int | None = None
    count = 0
    total = 0
    max_gold = 0
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            value = grid[r][c]
            if value <= 0:
                continue
            dist = abs(row - r) + abs(col - c)
            if nearest is None or dist < nearest:
                nearest = dist
            if dist <= radius:
                count += 1
                total += value
                max_gold = max(max_gold, value)
    return {
        "nearest_gold_dist": nearest,
        "gold_count_r3": count,
        "gold_sum_r3": total,
        "max_gold_r3": max_gold,
    }


def blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def apply_npc_action(pos: tuple[int, int], action: int, static_map: list[list[int]]) -> tuple[tuple[int, int], bool, str]:
    if action not in ACTION_DELTAS:
        raise NpcFeatureError(f"非法 action code：{action}")
    row, col = pos
    dr, dc = ACTION_DELTAS[action]
    target = (row + dr, col + dc)
    if not (0 <= target[0] < GRID_SIZE and 0 <= target[1] < GRID_SIZE):
        return pos, False, "boundary"
    if static_map[target[0]][target[1]] == 1:
        return pos, False, "obstacle"
    return target, target != pos, ""


def apply_player_action(
    pos: tuple[int, int],
    action: int,
    static_map: list[list[int]],
    occupied_players: set[tuple[int, int]],
) -> tuple[tuple[int, int], bool]:
    if action not in ACTION_DELTAS:
        raise NpcFeatureError(f"非法 player action code：{action}")
    row, col = pos
    dr, dc = ACTION_DELTAS[action]
    target = (row + dr, col + dc)
    if not (0 <= target[0] < GRID_SIZE and 0 <= target[1] < GRID_SIZE):
        return pos, False
    if static_map[target[0]][target[1]] == 1:
        return pos, False
    other_players = set(occupied_players)
    other_players.discard(pos)
    if target in other_players:
        return pos, False
    return target, target != pos


def player_unit_execution_order(player: dict[str, Any]) -> list[int]:
    order = int(player.get("order", 0))
    return [0, 1] if order == 0 else [1, 0]


def write_header(writer: csv.writer, columns: list[str]) -> None:
    writer.writerow(columns)


def extract_run(
    run_id: str,
    merged_root: Path = DEFAULT_MERGED_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    input_dir = merged_root / run_id
    if not input_dir.exists():
        raise NpcFeatureError(f"merged replay 目录不存在：{input_dir}")
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise NpcFeatureError(f"merged replay 目录为空：{input_dir}")

    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    round_path = output_dir / "npc_round_observations.csv"
    step_path = output_dir / "npc_step_events.csv"
    dispatch_path = output_dir / "npc_dispatch_orders.csv"
    interaction_path = output_dir / "npc_interactions.csv"
    summary_path = output_dir / "summary.json"

    summary_counter: Counter[str] = Counter()
    action_counts: Counter[int] = Counter()
    interaction_counts: Counter[str] = Counter()

    with (
        round_path.open("w", encoding="utf-8", newline="") as round_file,
        step_path.open("w", encoding="utf-8", newline="") as step_file,
        dispatch_path.open("w", encoding="utf-8", newline="") as dispatch_file,
        interaction_path.open("w", encoding="utf-8", newline="") as interaction_file,
    ):
        round_writer = csv.writer(round_file)
        step_writer = csv.writer(step_file)
        dispatch_writer = csv.writer(dispatch_file)
        interaction_writer = csv.writer(interaction_file)

        write_header(
            round_writer,
            [
                "game_id",
                "map_id",
                "round",
                "npc_id",
                "start_row",
                "start_col",
                "end_row",
                "end_col",
                "start_region",
                "end_region",
                "actions_json",
                "action_len",
                "nonstay_actions",
                "actual_moves",
                "displacement",
                "dispatch_index",
                "visible_by_start",
                "visible_by_end",
                "actions_source",
                "position_source_start",
                "position_source_end",
                "pickup",
                "gold",
                "cost",
                "start_cell_grid",
                "end_cell_grid",
                "start_static_map",
                "end_static_map",
                "nearest_gold_dist",
                "nearest_bomb_dist",
                "nearest_player_dist",
                "gold_count_r3",
                "gold_sum_r3",
                "max_gold_r3",
            ],
        )
        write_header(
            step_writer,
            [
                "game_id",
                "map_id",
                "round",
                "npc_id",
                "dispatch_index",
                "step_idx",
                "action",
                "before_row",
                "before_col",
                "after_row",
                "after_col",
                "moved",
                "invalid_reason",
                "target_region",
                "target_static_map",
                "target_grid_at_round_start",
                "target_has_gold_at_round_start",
                "target_gold_at_round_start",
                "target_has_bomb_at_round_start",
                "nearest_gold_dist_before",
                "nearest_gold_dist_after",
                "nearest_bomb_dist_before",
                "nearest_bomb_dist_after",
                "nearest_player_dist_before",
                "nearest_player_dist_after",
            ],
        )
        write_header(
            dispatch_writer,
            [
                "game_id",
                "map_id",
                "round",
                "first_player",
                "slow_player",
                "npc_order_json",
                "npc_count",
                "is_complete_7npc_block",
            ],
        )
        write_header(
            interaction_writer,
            [
                "game_id",
                "map_id",
                "round",
                "event_type",
                "npc_id",
                "dispatch_index",
                "step_idx",
                "row",
                "col",
                "region",
                "static_map",
                "amount",
                "details_json",
            ],
        )

        for path in files:
            data = read_merged(path)
            summary_counter["game_count"] += 1
            game_id = data["source"]["game_id"]
            map_id = data["source"].get("map_id", "")
            static_map = data["maps"]
            validate_static_map(static_map, path)

            for round_obj in data["rounds"]:
                summary_counter["rounds"] += 1
                round_no = round_obj["round"]
                start_frame = round_obj["merged"]["start"]
                end_frame = round_obj["merged"]["end"]
                start_grid = start_frame["grid"]
                start_npcs = npc_by_id(start_frame)
                end_npcs = npc_by_id(end_frame)
                order = end_frame.get("dispatch_order") or []
                npc_order = [int(item) for item in order if int(item) < 0]
                dispatch_index = {npc_id: idx for idx, npc_id in enumerate(npc_order)}
                first_player = next((int(item) for item in order if int(item) > 0), "")
                slow_player = next((int(item) for item in reversed(order) if int(item) > 0), "")
                is_complete = int(
                    len(order) == 9
                    and len(npc_order) == 7
                    and all(int(item) < 0 for item in order[1:8])
                    and int(order[0]) > 0
                    and int(order[-1]) > 0
                )
                dispatch_writer.writerow(
                    [
                        game_id,
                        map_id,
                        round_no,
                        first_player,
                        slow_player,
                        json.dumps(npc_order, ensure_ascii=False, separators=(",", ":")),
                        len(npc_order),
                        is_complete,
                    ]
                )

                bomb_cells = visible_cells(start_grid, BOMB)
                player_cells = player_positions(start_frame)
                end_player_cell_counts = Counter(player_positions(end_frame))
                end_npc_cell_counts = Counter(npc_positions(end_frame))

                round_paths: dict[int, list[tuple[int, int]]] = {}
                round_actions: dict[int, list[int]] = {}

                for npc_id, end_npc in sorted(end_npcs.items()):
                    summary_counter["end_npc_entries"] += 1
                    if npc_id not in start_npcs:
                        continue
                    start_npc = start_npcs[npc_id]
                    actions = end_npc.get("actions")
                    if actions is None:
                        continue
                    start_pos = tuple(start_npc["position"])
                    end_pos = tuple(end_npc["position"])
                    pos = start_pos
                    actual_moves = 0
                    path_positions = [pos]
                    nearest_gold_targets = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE) if start_grid[r][c] > 0]
                    for action in actions:
                        action_int = int(action)
                        pos, moved, invalid_reason = apply_npc_action(pos, action_int, static_map)
                        if invalid_reason:
                            summary_counter[f"invalid_{invalid_reason}"] += 1
                        actual_moves += int(moved)
                        path_positions.append(pos)
                    if pos != end_pos:
                        raise NpcFeatureError(
                            f"{path}: NPC action replay mismatch "
                            f"game_id={game_id} round={round_no} npc_id={npc_id} "
                            f"start={start_pos} replay_end={pos} end={end_pos} actions={actions}"
                        )

                    summary_counter["continuous_trajectories"] += 1
                    round_paths[npc_id] = path_positions
                    round_actions[npc_id] = [int(action) for action in actions]
                    for action in actions:
                        action_counts[int(action)] += 1

                    gold_ctx = gold_context(start_grid, start_pos)
                    nearest_bomb = nearest_distance(start_pos, bomb_cells)
                    nearest_player = nearest_distance(start_pos, player_cells)
                    pickup = int(end_npc.get("pickup") or 0)
                    if pickup > 0:
                        interaction_counts["pickup"] += 1
                        interaction_writer.writerow(
                            [
                                game_id,
                                map_id,
                                round_no,
                                "pickup",
                                npc_id,
                                blank_if_none(dispatch_index.get(npc_id)),
                                "",
                                end_pos[0],
                                end_pos[1],
                                region_of(end_pos[0], end_pos[1]),
                                int(static_map[end_pos[0]][end_pos[1]]),
                                pickup,
                                "{}",
                            ]
                        )

                    if end_npc_cell_counts[end_pos] > 1:
                        interaction_counts["npc_overlap_end"] += 1
                        interaction_writer.writerow(
                            [
                                game_id,
                                map_id,
                                round_no,
                                "npc_overlap_end",
                                npc_id,
                                blank_if_none(dispatch_index.get(npc_id)),
                                "",
                                end_pos[0],
                                end_pos[1],
                                region_of(end_pos[0], end_pos[1]),
                                int(static_map[end_pos[0]][end_pos[1]]),
                                end_npc_cell_counts[end_pos],
                                "{}",
                            ]
                        )
                    if end_player_cell_counts[end_pos] > 0:
                        interaction_counts["player_overlap_end"] += 1
                        interaction_writer.writerow(
                            [
                                game_id,
                                map_id,
                                round_no,
                                "player_overlap_end",
                                npc_id,
                                blank_if_none(dispatch_index.get(npc_id)),
                                "",
                                end_pos[0],
                                end_pos[1],
                                region_of(end_pos[0], end_pos[1]),
                                int(static_map[end_pos[0]][end_pos[1]]),
                                end_player_cell_counts[end_pos],
                                "{}",
                            ]
                        )

                    round_writer.writerow(
                        [
                            game_id,
                            map_id,
                            round_no,
                            npc_id,
                            start_pos[0],
                            start_pos[1],
                            end_pos[0],
                            end_pos[1],
                            region_of(start_pos[0], start_pos[1]),
                            region_of(end_pos[0], end_pos[1]),
                            json.dumps([int(action) for action in actions], separators=(",", ":")),
                            len(actions),
                            sum(1 for action in actions if int(action) != 4),
                            actual_moves,
                            abs(end_pos[0] - start_pos[0]) + abs(end_pos[1] - start_pos[1]),
                            blank_if_none(dispatch_index.get(npc_id)),
                            start_npc.get("visible_by", ""),
                            end_npc.get("visible_by", ""),
                            end_npc.get("actions_source", ""),
                            start_npc.get("position_source", ""),
                            end_npc.get("position_source", ""),
                            pickup,
                            end_npc.get("gold", ""),
                            end_npc.get("cost", ""),
                            start_grid[start_pos[0]][start_pos[1]],
                            start_grid[end_pos[0]][end_pos[1]],
                            int(static_map[start_pos[0]][start_pos[1]]),
                            int(static_map[end_pos[0]][end_pos[1]]),
                            blank_if_none(gold_ctx["nearest_gold_dist"]),
                            blank_if_none(nearest_bomb),
                            blank_if_none(nearest_player),
                            gold_ctx["gold_count_r3"],
                            gold_ctx["gold_sum_r3"],
                            gold_ctx["max_gold_r3"],
                        ]
                    )

                    pos = start_pos
                    for step_idx, action in enumerate(actions):
                        action_int = int(action)
                        before = pos
                        after, moved, invalid_reason = apply_npc_action(before, action_int, static_map)
                        before_gold = gold_context(start_grid, before)["nearest_gold_dist"]
                        after_gold = gold_context(start_grid, after)["nearest_gold_dist"]
                        before_bomb = nearest_distance(before, bomb_cells)
                        after_bomb = nearest_distance(after, bomb_cells)
                        before_player = nearest_distance(before, player_cells)
                        after_player = nearest_distance(after, player_cells)
                        target_grid = start_grid[after[0]][after[1]]
                        step_writer.writerow(
                            [
                                game_id,
                                map_id,
                                round_no,
                                npc_id,
                                blank_if_none(dispatch_index.get(npc_id)),
                                step_idx,
                                action_int,
                                before[0],
                                before[1],
                                after[0],
                                after[1],
                                int(moved),
                                invalid_reason,
                                region_of(after[0], after[1]),
                                int(static_map[after[0]][after[1]]),
                                target_grid,
                                int(target_grid > 0),
                                target_grid if target_grid > 0 else 0,
                                int(target_grid == BOMB),
                                blank_if_none(before_gold),
                                blank_if_none(after_gold),
                                blank_if_none(before_bomb),
                                blank_if_none(after_bomb),
                                blank_if_none(before_player),
                                blank_if_none(after_player),
                            ]
                        )
                        summary_counter["step_rows"] += 1
                        pos = after

                write_bomb_trigger_interactions(
                    interaction_writer=interaction_writer,
                    interaction_counts=interaction_counts,
                    game_id=game_id,
                    map_id=map_id,
                    round_no=round_no,
                    static_map=static_map,
                    start_grid=start_grid,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    npc_order=npc_order,
                    dispatch_index=dispatch_index,
                    round_paths=round_paths,
                    round_actions=round_actions,
                )

    summary = {
        "run_id": run_id,
        "game_count": summary_counter["game_count"],
        "rounds": summary_counter["rounds"],
        "end_npc_entries": summary_counter["end_npc_entries"],
        "continuous_trajectories": summary_counter["continuous_trajectories"],
        "step_rows": summary_counter["step_rows"],
        "action_counts": {str(k): v for k, v in sorted(action_counts.items())},
        "interaction_counts": dict(sorted(interaction_counts.items())),
        "outputs": {
            "npc_round_observations": display_path(round_path),
            "npc_step_events": display_path(step_path),
            "npc_dispatch_orders": display_path(dispatch_path),
            "npc_interactions": display_path(interaction_path),
            "summary": display_path(summary_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    return summary


def write_bomb_trigger_interactions(
    *,
    interaction_writer: csv.writer,
    interaction_counts: Counter[str],
    game_id: int,
    map_id: int | str,
    round_no: int,
    static_map: list[list[int]],
    start_grid: list[list[int]],
    start_frame: dict[str, Any],
    end_frame: dict[str, Any],
    npc_order: list[int],
    dispatch_index: dict[int, int],
    round_paths: dict[int, list[tuple[int, int]]],
    round_actions: dict[int, list[int]],
) -> None:
    bombs = set(visible_cells(start_grid, BOMB))
    if not bombs:
        return

    player_state = player_units(start_frame)
    end_players_by_id = {int(player["id"]): player for player in end_frame.get("players", [])}
    npc_position = {npc_id: positions[0] for npc_id, positions in round_paths.items()}
    occupied_players = {item["position"] for item in player_state.values()}

    order = end_frame.get("dispatch_order") or []
    for actor in order:
        actor_id = int(actor)
        if actor_id > 0:
            player = end_players_by_id.get(actor_id)
            if player is None:
                continue
            for unit_index in player_unit_execution_order(player):
                units = player.get("units", [])
                if unit_index >= len(units):
                    continue
                key = (actor_id, unit_index)
                state = player_state.get(key)
                if state is None:
                    continue
                actions = units[unit_index].get("actions", [])
                pos = state["position"]
                for action in actions:
                    new_pos, moved = apply_player_action(pos, int(action), static_map, occupied_players)
                    if moved:
                        occupied_players.discard(pos)
                        occupied_players.add(new_pos)
                        pos = new_pos
                        bombs.discard(pos)
                state["position"] = pos
        elif actor_id < 0:
            actions = round_actions.get(actor_id)
            if actions is None:
                continue
            pos = npc_position.get(actor_id)
            if pos is None:
                continue
            for step_idx, action in enumerate(actions):
                new_pos, moved, _ = apply_npc_action(pos, int(action), static_map)
                if moved and new_pos in bombs:
                    interaction_counts["bomb_trigger"] += 1
                    bombs.remove(new_pos)
                    details = {
                        "action": int(action),
                        "visible_start_bomb": True,
                    }
                    interaction_writer.writerow(
                        [
                            game_id,
                            map_id,
                            round_no,
                            "bomb_trigger",
                            actor_id,
                            blank_if_none(dispatch_index.get(actor_id)),
                            step_idx,
                            new_pos[0],
                            new_pos[1],
                            region_of(new_pos[0], new_pos[1]),
                            int(static_map[new_pos[0]][new_pos[1]]),
                            "",
                            json.dumps(details, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                        ]
                    )
                pos = new_pos
            npc_position[actor_id] = pos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抽取 NPC 行为建模特征表")
    parser.add_argument("--run-id", required=True, help="merged_replays 下的 run_id")
    parser.add_argument("--merged-root", type=Path, default=DEFAULT_MERGED_ROOT, help="merged_replays 根目录")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="npc_features 输出根目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = extract_run(run_id=args.run_id, merged_root=args.merged_root, output_root=args.output_root)
    print(
        "wrote npc features "
        f"run_id={summary['run_id']} games={summary['game_count']} "
        f"continuous={summary['continuous_trajectories']} steps={summary['step_rows']}"
    )
    for name, path in summary["outputs"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
