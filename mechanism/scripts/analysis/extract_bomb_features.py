#!/usr/bin/env python3
"""从双视角合并 replay 抽取炸弹生成建模特征表。

默认输入：
  mechanism/data/processed/merged_replays/<run_id>/*.json

默认输出：
  mechanism/data/processed/bomb_features/<run_id>/

本脚本只做离线处理，不访问评测平台。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MERGED_ROOT = REPO_ROOT / "mechanism" / "data" / "processed" / "merged_replays"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "mechanism" / "data" / "processed" / "bomb_features"
GRID_SIZE = 17
FOG = -5
BOMB = -3
OBSTACLE = -1


class FeatureError(RuntimeError):
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
    raise FeatureError(f"坐标不属于任何区域：({row},{col})")


def read_merged(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != "goldrush2_merged_replay":
        raise FeatureError(f"{path}: 非 merged replay")
    return data


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_header(writer: csv.writer, columns: list[str]) -> None:
    writer.writerow(columns)


def visible_dynamic_cell(value: int) -> bool:
    return value != FOG and value != OBSTACLE


def gold_value(value: int) -> int:
    return value if value > 0 else 0


def frame_occupied_cells(frame: dict[str, Any]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for player in frame.get("players", []):
        for unit in player.get("units", []):
            position = unit.get("position")
            if is_position(position):
                cells.add((int(position[0]), int(position[1])))
    for npc in frame.get("npcs", []):
        position = npc.get("position")
        if is_position(position):
            cells.add((int(position[0]), int(position[1])))
    return cells


def is_position(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], int)
        and isinstance(value[1], int)
        and 0 <= value[0] < GRID_SIZE
        and 0 <= value[1] < GRID_SIZE
    )


def extract_run(
    run_id: str,
    merged_root: Path = DEFAULT_MERGED_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    write_transitions: bool = True,
) -> dict[str, Any]:
    input_dir = merged_root / run_id
    if not input_dir.exists():
        raise FeatureError(f"merged replay 目录不存在：{input_dir}")
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise FeatureError(f"merged replay 目录为空：{input_dir}")

    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    transition_path = output_dir / "bomb_cell_transitions.csv"
    round_region_path = output_dir / "bomb_round_region.csv"
    cell_summary_path = output_dir / "bomb_cell_summary.csv"
    summary_path = output_dir / "summary.json"

    round_region: dict[tuple[int, int, int, int], dict[str, int]] = defaultdict(
        lambda: {
            "exposure": 0,
            "prev_bombs": 0,
            "curr_bombs": 0,
            "appeared": 0,
            "disappeared": 0,
            "appeared_on_prev_gold": 0,
            "appeared_on_prev_occupied": 0,
            "appeared_on_curr_occupied": 0,
        }
    )
    cell_summary: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {
            "exposure": 0,
            "prev_bombs": 0,
            "curr_bombs": 0,
            "appeared": 0,
            "disappeared": 0,
            "appeared_on_prev_gold": 0,
            "appeared_on_prev_occupied": 0,
            "appeared_on_curr_occupied": 0,
        }
    )

    game_count = 0
    total_round_pairs = 0
    total_transitions = 0
    total_appeared = 0
    total_disappeared = 0

    transition_file = transition_path.open("w", encoding="utf-8", newline="") if write_transitions else None
    try:
        transition_writer = csv.writer(transition_file) if transition_file else None
        if transition_writer:
            write_header(
                transition_writer,
                [
                    "game_id",
                    "map_id",
                    "round",
                    "row",
                    "col",
                    "region",
                    "static_map",
                    "visible_by_prev",
                    "visible_by_start",
                    "prev_grid",
                    "curr_grid",
                    "prev_bomb",
                    "curr_bomb",
                    "appeared",
                    "disappeared",
                    "prev_gold",
                    "curr_gold",
                    "prev_occupied",
                    "curr_occupied",
                    "round_mod20",
                ],
            )

        for path in files:
            data = read_merged(path)
            game_count += 1
            game_id = data["source"]["game_id"]
            map_id = data["source"].get("map_id", "")
            static_map = data["maps"]
            rounds = data["rounds"]
            validate_static_map(static_map, path)

            for index in range(1, len(rounds)):
                prev_round = rounds[index - 1]
                curr_round = rounds[index]
                round_no = curr_round["round"]
                if round_no != prev_round["round"] + 1:
                    raise FeatureError(f"{path}: round 不连续：prev={prev_round['round']} curr={round_no}")

                total_round_pairs += 1
                prev_frame = prev_round["merged"]["end"]
                curr_frame = curr_round["merged"]["start"]
                prev_grid = prev_frame["grid"]
                curr_grid = curr_frame["grid"]
                prev_visible = prev_frame["visible_by"]
                curr_visible = curr_frame["visible_by"]
                prev_occupied = frame_occupied_cells(prev_frame)
                curr_occupied = frame_occupied_cells(curr_frame)

                for row in range(GRID_SIZE):
                    for col in range(GRID_SIZE):
                        prev_value = prev_grid[row][col]
                        curr_value = curr_grid[row][col]
                        if not visible_dynamic_cell(prev_value) or not visible_dynamic_cell(curr_value):
                            continue

                        prev_bomb = int(prev_value == BOMB)
                        curr_bomb = int(curr_value == BOMB)
                        appeared = int(not prev_bomb and curr_bomb)
                        disappeared = int(prev_bomb and not curr_bomb)
                        region_id = region_of(row, col)
                        static_value = int(static_map[row][col])
                        prev_gold = gold_value(prev_value)
                        curr_gold = gold_value(curr_value)
                        prev_occ = int((row, col) in prev_occupied)
                        curr_occ = int((row, col) in curr_occupied)

                        total_transitions += 1
                        total_appeared += appeared
                        total_disappeared += disappeared

                        for target in (
                            round_region[(game_id, map_id if isinstance(map_id, int) else -1, round_no, region_id)],
                            cell_summary[(row, col)],
                        ):
                            target["exposure"] += 1
                            target["prev_bombs"] += prev_bomb
                            target["curr_bombs"] += curr_bomb
                            target["appeared"] += appeared
                            target["disappeared"] += disappeared
                            if appeared:
                                target["appeared_on_prev_gold"] += int(prev_gold > 0)
                                target["appeared_on_prev_occupied"] += prev_occ
                                target["appeared_on_curr_occupied"] += curr_occ

                        if transition_writer:
                            transition_writer.writerow(
                                [
                                    game_id,
                                    map_id,
                                    round_no,
                                    row,
                                    col,
                                    region_id,
                                    static_value,
                                    prev_visible[row][col],
                                    curr_visible[row][col],
                                    prev_value,
                                    curr_value,
                                    prev_bomb,
                                    curr_bomb,
                                    appeared,
                                    disappeared,
                                    prev_gold,
                                    curr_gold,
                                    prev_occ,
                                    curr_occ,
                                    round_no % 20,
                                ]
                            )
    finally:
        if transition_file:
            transition_file.close()

    with round_region_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        write_header(
            writer,
            [
                "game_id",
                "map_id",
                "round",
                "region",
                "round_mod20",
                "exposure",
                "prev_bombs",
                "curr_bombs",
                "appeared",
                "disappeared",
                "appeared_on_prev_gold",
                "appeared_on_prev_occupied",
                "appeared_on_curr_occupied",
            ],
        )
        for (game_id, map_id, round_no, region_id), row in sorted(round_region.items()):
            writer.writerow(
                [
                    game_id,
                    "" if map_id == -1 else map_id,
                    round_no,
                    region_id,
                    round_no % 20,
                    row["exposure"],
                    row["prev_bombs"],
                    row["curr_bombs"],
                    row["appeared"],
                    row["disappeared"],
                    row["appeared_on_prev_gold"],
                    row["appeared_on_prev_occupied"],
                    row["appeared_on_curr_occupied"],
                ]
            )

    first_map = read_merged(files[0])["maps"]
    with cell_summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        write_header(
            writer,
            [
                "row",
                "col",
                "region",
                "static_map",
                "exposure",
                "prev_bombs",
                "curr_bombs",
                "appeared",
                "disappeared",
                "appeared_on_prev_gold",
                "appeared_on_prev_occupied",
                "appeared_on_curr_occupied",
            ],
        )
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                item = cell_summary[(row, col)]
                writer.writerow(
                    [
                        row,
                        col,
                        region_of(row, col),
                        int(first_map[row][col]),
                        item["exposure"],
                        item["prev_bombs"],
                        item["curr_bombs"],
                        item["appeared"],
                        item["disappeared"],
                        item["appeared_on_prev_gold"],
                        item["appeared_on_prev_occupied"],
                        item["appeared_on_curr_occupied"],
                    ]
                )

    summary = {
        "run_id": run_id,
        "game_count": game_count,
        "round_pairs": total_round_pairs,
        "cell_transitions": total_transitions,
        "bomb_appeared": total_appeared,
        "bomb_disappeared": total_disappeared,
        "wrote_cell_transitions": write_transitions,
        "outputs": {
            "round_region": display_path(round_region_path),
            "cell_summary": display_path(cell_summary_path),
            "summary": display_path(summary_path),
        },
    }
    if write_transitions:
        summary["outputs"]["cell_transitions"] = display_path(transition_path)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    return summary


def validate_static_map(static_map: Any, path: Path) -> None:
    if not isinstance(static_map, list) or len(static_map) != GRID_SIZE:
        raise FeatureError(f"{path}: 静态地图不是 {GRID_SIZE}x{GRID_SIZE}")
    for row, values in enumerate(static_map):
        if not isinstance(values, list) or len(values) != GRID_SIZE:
            raise FeatureError(f"{path}: 静态地图第 {row} 行长度非法")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="抽取炸弹生成建模特征表")
    parser.add_argument("--run-id", required=True, help="merged_replays 下的 run_id")
    parser.add_argument("--merged-root", type=Path, default=DEFAULT_MERGED_ROOT, help="merged_replays 根目录")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="bomb_features 输出根目录")
    parser.add_argument("--skip-transitions", action="store_true", help="不写逐 cell transition 明细表")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = extract_run(
        run_id=args.run_id,
        merged_root=args.merged_root,
        output_root=args.output_root,
        write_transitions=not args.skip_transitions,
    )
    print(
        "wrote bomb features "
        f"run_id={summary['run_id']} games={summary['game_count']} "
        f"transitions={summary['cell_transitions']} appeared={summary['bomb_appeared']} "
        f"disappeared={summary['bomb_disappeared']}"
    )
    for name, path in summary["outputs"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
