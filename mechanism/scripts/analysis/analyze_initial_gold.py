#!/usr/bin/env python3
"""分析初始金币生成。

输入：
  official_sdk/data/full/g*.txt
  mechanism/data/processed/gold_features/<run_id>/snapshot_windows.csv

输出：
  mechanism/data/processed/initial_gold_analysis/<analysis_id>/

本脚本只做离线处理，不访问评测平台。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FULL_ROOT = REPO_ROOT / "official_sdk" / "data" / "full"
DEFAULT_GOLD_ROOT = REPO_ROOT / "mechanism" / "data" / "processed" / "gold_features"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "mechanism" / "data" / "processed" / "initial_gold_analysis"
DEFAULT_ANALYSIS_ID = "full_and_snapshots"
DEFAULT_RUN_IDS = [
    "symobs-a-map1-100-20260727-231446",
    "symobs-a-map2-100-20260727-231704",
    "symobs-m3a-map3-100-20260728-200953",
    "symobs-m3b-map3-100-20260728-202647",
]
RUN_LABELS = {
    "symobs-a-map1-100-20260727-231446": "map1",
    "symobs-a-map2-100-20260727-231704": "map2",
    "symobs-m3a-map3-100-20260728-200953": "map3",
    "symobs-m3b-map3-100-20260728-202647": "map3",
}
CENTER_PARAMS = {
    "map1": (0.05962, 0.06745, 5.984),
    "map2": (0.05956, 0.06726, 6.016),
    "map3": (0.05641, 0.06260, 6.0020),
}
GRID_SIZE = 17
EXCLUDED_VALUES = {-5, -4, -3, -2, -1}


class InitialGoldError(RuntimeError):
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
    raise InitialGoldError(f"坐标不属于任何区域：({row},{col})")


def category(region: int, static_value: int) -> str:
    if region == 1 and static_value == 0:
        return "center_static0"
    if region != 1 and static_value == 2:
        return "outer_static2"
    if region != 1 and static_value == 0:
        return "outer_static0"
    return f"region{region}_static{static_value}"


def gold_value(value: Any) -> int:
    return value if isinstance(value, int) and value > 0 else 0


def clean_value(value: Any) -> bool:
    return isinstance(value, int) and value not in EXCLUDED_VALUES


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_full_replay(path: Path) -> tuple[dict[str, Any], list[list[int]], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as f:
        players = json.loads(next(f))
        static_map = [[int(value) for value in row] for row in json.loads(next(f))]
        rounds = [json.loads(line) for line in f if line.strip()]
    if not rounds or rounds[0]["round"] != 0:
        raise InitialGoldError(f"{path}: full replay 缺少 round=0")
    return players, static_map, rounds


def center_tick_expectation(label: str, static_map: list[list[int]]) -> float:
    center_a, center_b, amount_mean = CENTER_PARAMS[label]
    total = 0.0
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if region_of(row, col) != 1 or int(static_map[row][col]) != 0:
                continue
            dist2 = (row - 8) ** 2 + (col - 8) ** 2
            total += center_a * math.exp(-center_b * dist2) * amount_mean
    return total


def summarize_values(values: list[int | float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0, "sd": 0, "min": 0, "max": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(mean(values), 6),
        "sd": round(pstdev(values), 6),
        "min": ordered[0],
        "p10": ordered[int((len(ordered) - 1) * 0.1)],
        "median": ordered[int((len(ordered) - 1) * 0.5)],
        "p90": ordered[int((len(ordered) - 1) * 0.9)],
        "max": ordered[-1],
    }


def analyze_full_replays(full_root: Path) -> tuple[list[dict[str, Any]], list[tuple[int, int]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    initial_sets: list[set[tuple[int, int]]] = []
    tick_expectations: dict[str, float] = {}
    for index, path in enumerate(sorted(full_root.glob("g*.txt"))):
        _, static_map, rounds = read_full_replay(path)
        label = "map1"
        tick_expectations[path.stem] = center_tick_expectation(label, static_map)
        round0 = rounds[0]
        by_category: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"visible_cells": 0, "positive_cells": 0, "total": 0, "amounts": Counter(), "coords": []}
        )
        grid = round0["start"]["grid"]
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                value = grid[row][col]
                if not clean_value(value):
                    continue
                region = region_of(row, col)
                static_value = int(static_map[row][col])
                item = by_category[category(region, static_value)]
                item["visible_cells"] += 1
                amount = gold_value(value)
                if amount > 0:
                    item["positive_cells"] += 1
                    item["total"] += amount
                    item["amounts"][amount] += 1
                    item["coords"].append((row, col, amount))

        center_coords = {(row, col) for row, col, _ in by_category["center_static0"]["coords"]}
        initial_sets.append(center_coords)
        snapshot0 = next(item["snapshot"] for item in rounds if item.get("snapshot", {}).get("window") == [0, 4])
        snapshot_generated = {region["id"]: region["gold_generated"] for region in snapshot0["regions"]}
        transition_generated = {region_id: 0 for region_id in range(1, 6)}
        for round_index in range(1, 6):
            prev_grid = rounds[round_index - 1]["end"]["grid"]
            curr_grid = rounds[round_index]["start"]["grid"]
            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    if not clean_value(prev_grid[row][col]) or not clean_value(curr_grid[row][col]):
                        continue
                    delta = gold_value(curr_grid[row][col]) - gold_value(prev_grid[row][col])
                    if delta > 0:
                        transition_generated[region_of(row, col)] += delta

        for cat_name, item in sorted(by_category.items()):
            rows.append(
                {
                    "file": path.name,
                    "category": cat_name,
                    "visible_cells": item["visible_cells"],
                    "positive_cells": item["positive_cells"],
                    "total": item["total"],
                    "amount_counts_json": json.dumps(dict(sorted(item["amounts"].items())), sort_keys=True),
                    "coords_json": json.dumps(item["coords"], sort_keys=True),
                    "snapshot0_generated": snapshot_generated.get(1 if cat_name == "center_static0" else -1, ""),
                    "round1_5_generated": transition_generated[1] if cat_name == "center_static0" else "",
                }
            )

    common = set.intersection(*initial_sets) if initial_sets else set()
    union = set.union(*initial_sets) if initial_sets else set()
    if common != union:
        raise InitialGoldError("official full replay 的初始中心金币坐标集合不一致")
    return rows, sorted(common), tick_expectations


def analyze_snapshots(
    run_ids: list[str],
    gold_root: Path,
    center_tick_by_label: dict[str, float],
) -> list[dict[str, Any]]:
    by_label_window: dict[tuple[str, int], list[int]] = defaultdict(list)
    for run_id in run_ids:
        label = RUN_LABELS[run_id]
        path = gold_root / run_id / "snapshot_windows.csv"
        if not path.exists():
            raise InitialGoldError(f"snapshot_windows.csv 不存在：{path}")
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for item in reader:
                if int(item["region"]) != 1:
                    continue
                by_label_window[(label, int(item["window_begin"]))].append(int(item["gold_generated"]))

    rows: list[dict[str, Any]] = []
    for label in sorted({RUN_LABELS[run_id] for run_id in run_ids}):
        window0 = by_label_window[(label, 0)]
        later = [
            value
            for (item_label, window_begin), values in by_label_window.items()
            if item_label == label and window_begin > 0
            for value in values
        ]
        tick = center_tick_by_label[label]
        model_base_only = 50 + 5 * tick
        model_with_extra_tick = 50 + 6 * tick
        rows.append(
            {
                "label": label,
                "games": len(window0),
                "window0_center_mean": round(mean(window0), 6),
                "window0_center_sd": round(pstdev(window0), 6),
                "later_center_window_mean": round(mean(later), 6),
                "center_tick_expectation": round(tick, 6),
                "inferred_initial_mean": round(mean(window0) - 5 * tick, 6),
                "model_base50_plus_5ticks": round(model_base_only, 6),
                "model_base50_extra_tick_plus_5ticks": round(model_with_extra_tick, 6),
                "abs_error_base_only": round(abs(mean(window0) - model_base_only), 6),
                "abs_error_with_extra_tick": round(abs(mean(window0) - model_with_extra_tick), 6),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    analysis_id: str,
    full_root: Path = DEFAULT_FULL_ROOT,
    gold_root: Path = DEFAULT_GOLD_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_ids: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = output_root / analysis_id
    output_dir.mkdir(parents=True, exist_ok=True)

    full_rows, initial_cells, _ = analyze_full_replays(full_root)
    center_tick_by_label: dict[str, float] = {}
    for label, run_ids_for_label in {
        "map1": ["symobs-a-map1-100-20260727-231446"],
        "map2": ["symobs-a-map2-100-20260727-231704"],
        "map3": ["symobs-m3a-map3-100-20260728-200953", "symobs-m3b-map3-100-20260728-202647"],
    }.items():
        path = gold_root / run_ids_for_label[0] / "cell_summary.csv"
        cells: list[tuple[int, int, int, int]] = []
        with path.open(encoding="utf-8", newline="") as f:
            for item in csv.DictReader(f):
                cells.append((int(item["row"]), int(item["col"]), int(item["region"]), int(item["static_map"])))
        center_a, center_b, amount_mean = CENTER_PARAMS[label]
        center_tick_by_label[label] = sum(
            center_a * math.exp(-center_b * ((row - 8) ** 2 + (col - 8) ** 2)) * amount_mean
            for row, col, region, static_value in cells
            if region == 1 and static_value == 0
        )

    snapshot_rows = analyze_snapshots(run_ids or DEFAULT_RUN_IDS, gold_root, center_tick_by_label)

    full_path = output_dir / "official_full_initial_gold.csv"
    cells_path = output_dir / "initial_center_base_cells.csv"
    snapshot_path = output_dir / "snapshot_window0_model_comparison.csv"
    summary_path = output_dir / "summary.json"

    write_csv(
        full_path,
        full_rows,
        [
            "file",
            "category",
            "visible_cells",
            "positive_cells",
            "total",
            "amount_counts_json",
            "coords_json",
            "snapshot0_generated",
            "round1_5_generated",
        ],
    )
    write_csv(cells_path, [{"row": row, "col": col} for row, col in initial_cells], ["row", "col"])
    write_csv(
        snapshot_path,
        snapshot_rows,
        [
            "label",
            "games",
            "window0_center_mean",
            "window0_center_sd",
            "later_center_window_mean",
            "center_tick_expectation",
            "inferred_initial_mean",
            "model_base50_plus_5ticks",
            "model_base50_extra_tick_plus_5ticks",
            "abs_error_base_only",
            "abs_error_with_extra_tick",
        ],
    )

    center_totals = [
        row["total"] for row in full_rows if row["category"] == "center_static0"
    ]
    summary = {
        "analysis_id": analysis_id,
        "official_full_files": len(sorted(full_root.glob("g*.txt"))),
        "initial_center_base_cell_count": len(initial_cells),
        "official_full_center_initial_total": summarize_values(center_totals),
        "snapshot_model_comparison": snapshot_rows,
        "outputs": {
            "official_full_initial_gold": display_path(full_path),
            "initial_center_base_cells": display_path(cells_path),
            "snapshot_window0_model_comparison": display_path(snapshot_path),
            "summary": display_path(summary_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分析初始金币生成")
    parser.add_argument("--analysis-id", default=DEFAULT_ANALYSIS_ID, help="输出分析 ID")
    parser.add_argument("--full-root", type=Path, default=DEFAULT_FULL_ROOT, help="official full replay 目录")
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT, help="gold_features 根目录")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="输出根目录")
    parser.add_argument("--run-id", action="append", dest="run_ids", help="gold_features 下的 run_id，可重复")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = analyze(
        analysis_id=args.analysis_id,
        full_root=args.full_root,
        gold_root=args.gold_root,
        output_root=args.output_root,
        run_ids=args.run_ids,
    )
    print(
        "wrote initial gold analysis "
        f"analysis_id={summary['analysis_id']} "
        f"initial_center_base_cell_count={summary['initial_center_base_cell_count']}"
    )
    for name, path in summary["outputs"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
