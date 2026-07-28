#!/usr/bin/env python3
"""从双视角合并 replay 抽取金币生成建模特征表。

默认输入：
  mechanism/data/processed/merged_replays/<run_id>/*.json

默认输出：
  mechanism/data/processed/gold_features/<run_id>/

本脚本只做离线处理，不访问评测平台。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MERGED_ROOT = REPO_ROOT / "mechanism" / "data" / "processed" / "merged_replays"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "mechanism" / "data" / "processed" / "gold_features"
GRID_SIZE = 17
FOG = -5
EXCLUDED_VALUES = {FOG, -3, -1}


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


def gold_value(value: int) -> int:
    return value if value > 0 else 0


def clean_cell(value: int) -> bool:
    return value not in EXCLUDED_VALUES and value >= 0


def window_begin_for_transition(round_no: int) -> int:
    if round_no <= 0:
        raise FeatureError(f"transition round 必须大于 0：{round_no}")
    return ((round_no - 1) // 5) * 5


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

    snapshot_path = output_dir / "snapshot_windows.csv"
    observed_path = output_dir / "window_observed.csv"
    cell_summary_path = output_dir / "cell_summary.csv"
    transition_path = output_dir / "cell_transitions.csv"
    summary_path = output_dir / "summary.json"

    cell_summary: dict[tuple[int, int], dict[str, Any]] = defaultdict(
        lambda: {"exposure": 0, "events": 0, "delta_sum": 0, "amounts": Counter()}
    )
    window_summary: dict[tuple[int, int, int, int], dict[str, int]] = defaultdict(
        lambda: {"exposure": 0, "events": 0, "delta_sum": 0}
    )
    window_rounds: dict[tuple[int, int, int, int], set[int]] = defaultdict(set)

    total_transitions = 0
    total_events = 0
    total_snapshot_rows = 0
    total_round_pairs = 0
    game_count = 0

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
                    "prev_gold",
                    "curr_gold",
                    "delta",
                    "round_mod5",
                    "round_mod20",
                    "window_begin",
                    "window_end",
                ],
            )

        with snapshot_path.open("w", encoding="utf-8", newline="") as snapshot_file:
            snapshot_writer = csv.writer(snapshot_file)
            write_header(
                snapshot_writer,
                [
                    "game_id",
                    "map_id",
                    "snapshot_round",
                    "window_begin",
                    "window_end",
                    "region",
                    "enter",
                    "leave",
                    "gold_generated",
                    "gold_collected",
                    "gold_remaining",
                    "occupants",
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

                for round_obj in rounds:
                    snapshot = round_obj["merged"].get("snapshot")
                    if not snapshot:
                        continue
                    window_begin, window_end = snapshot["window"]
                    for region in snapshot["regions"]:
                        snapshot_writer.writerow(
                            [
                                game_id,
                                map_id,
                                snapshot["round"],
                                window_begin,
                                window_end,
                                region["id"],
                                region["enter"],
                                region["leave"],
                                region["gold_generated"],
                                region["gold_collected"],
                                region["gold_remaining"],
                                region["occupants"],
                            ]
                        )
                        total_snapshot_rows += 1

                for index in range(1, len(rounds)):
                    prev_round = rounds[index - 1]
                    curr_round = rounds[index]
                    round_no = curr_round["round"]
                    if round_no != prev_round["round"] + 1:
                        raise FeatureError(
                            f"{path}: round 不连续：prev={prev_round['round']} curr={round_no}"
                        )
                    total_round_pairs += 1
                    window_begin = window_begin_for_transition(round_no)
                    window_end = window_begin + 4
                    prev_frame = prev_round["merged"]["end"]
                    curr_frame = curr_round["merged"]["start"]
                    prev_grid = prev_frame["grid"]
                    curr_grid = curr_frame["grid"]
                    prev_visible = prev_frame["visible_by"]
                    curr_visible = curr_frame["visible_by"]

                    for row in range(GRID_SIZE):
                        for col in range(GRID_SIZE):
                            prev_value = prev_grid[row][col]
                            curr_value = curr_grid[row][col]
                            if not clean_cell(prev_value) or not clean_cell(curr_value):
                                continue
                            prev_gold = gold_value(prev_value)
                            curr_gold = gold_value(curr_value)
                            delta = curr_gold - prev_gold
                            if delta < 0:
                                raise FeatureError(
                                    f"{path}: 金币 transition 负增量："
                                    f"game_id={game_id} round={round_no} cell=({row},{col}) "
                                    f"prev={prev_value} curr={curr_value}"
                                )

                            region_id = region_of(row, col)
                            static_value = int(static_map[row][col])
                            total_transitions += 1
                            total_events += int(delta > 0)
                            cell_key = (row, col)
                            cell_row = cell_summary[cell_key]
                            cell_row["exposure"] += 1
                            window_key = (game_id, map_id if isinstance(map_id, int) else -1, window_begin, region_id)
                            window_row = window_summary[window_key]
                            window_rounds[window_key].add(round_no)
                            window_row["exposure"] += 1
                            if delta > 0:
                                cell_row["events"] += 1
                                cell_row["delta_sum"] += delta
                                cell_row["amounts"][delta] += 1
                                window_row["events"] += 1
                                window_row["delta_sum"] += delta

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
                                        prev_gold,
                                        curr_gold,
                                        delta,
                                        round_no % 5,
                                        round_no % 20,
                                        window_begin,
                                        window_end,
                                    ]
                                )
    finally:
        if transition_file:
            transition_file.close()

    with observed_path.open("w", encoding="utf-8", newline="") as observed_file:
        observed_writer = csv.writer(observed_file)
        write_header(
            observed_writer,
            [
                "game_id",
                "map_id",
                "window_begin",
                "window_end",
                "region",
                "transition_rounds",
                "exposure",
                "events",
                "delta_sum",
            ],
        )
        for (game_id, map_id, window_begin, region_id), row in sorted(window_summary.items()):
            observed_writer.writerow(
                [
                    game_id,
                    "" if map_id == -1 else map_id,
                    window_begin,
                    window_begin + 4,
                    region_id,
                    len(window_rounds[(game_id, map_id, window_begin, region_id)]),
                    row["exposure"],
                    row["events"],
                    row["delta_sum"],
                ]
            )

    with cell_summary_path.open("w", encoding="utf-8", newline="") as cell_file:
        cell_writer = csv.writer(cell_file)
        write_header(
            cell_writer,
            [
                "row",
                "col",
                "region",
                "static_map",
                "exposure",
                "events",
                "delta_sum",
                "amount_counts_json",
            ],
        )
        first_map = read_merged(files[0])["maps"]
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                item = cell_summary.get((row, col), {"exposure": 0, "events": 0, "delta_sum": 0, "amounts": Counter()})
                amount_counts = {str(k): v for k, v in sorted(item["amounts"].items())}
                cell_writer.writerow(
                    [
                        row,
                        col,
                        region_of(row, col),
                        int(first_map[row][col]),
                        item["exposure"],
                        item["events"],
                        item["delta_sum"],
                        json.dumps(amount_counts, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    ]
                )

    summary = {
        "run_id": run_id,
        "game_count": game_count,
        "round_pairs": total_round_pairs,
        "snapshot_rows": total_snapshot_rows,
        "cell_transitions": total_transitions,
        "cell_events": total_events,
        "wrote_cell_transitions": write_transitions,
        "outputs": {
            "snapshot_windows": display_path(snapshot_path),
            "window_observed": display_path(observed_path),
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
    parser = argparse.ArgumentParser(description="抽取金币生成建模特征表")
    parser.add_argument("--run-id", required=True, help="merged_replays 下的 run_id")
    parser.add_argument("--merged-root", type=Path, default=DEFAULT_MERGED_ROOT, help="merged_replays 根目录")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="gold_features 输出根目录")
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
        "wrote gold features "
        f"run_id={summary['run_id']} games={summary['game_count']} "
        f"transitions={summary['cell_transitions']} events={summary['cell_events']}"
    )
    for name, path in summary["outputs"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
