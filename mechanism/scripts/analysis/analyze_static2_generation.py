#!/usr/bin/env python3
"""从金币特征表抽取外围 static_map=2 高额 batch 分析表。

默认输入：
  mechanism/data/processed/gold_features/<run_id>/cell_transitions.csv

默认输出：
  mechanism/data/processed/static2_analysis/<analysis_id>/

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
DEFAULT_GOLD_ROOT = REPO_ROOT / "mechanism" / "data" / "processed" / "gold_features"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "mechanism" / "data" / "processed" / "static2_analysis"
DEFAULT_RUN_IDS = [
    "symobs-a-map1-100-20260727-231446",
    "symobs-a-map2-100-20260727-231704",
    "symobs-m3a-map3-100-20260728-200953",
    "symobs-m3b-map3-100-20260728-202647",
]
DEFAULT_ANALYSIS_ID = "maps123_static2_generation"
DEFAULT_CELL_HIGH_THRESHOLD = 20


class AnalysisError(RuntimeError):
    pass


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_header(writer: csv.writer, columns: list[str]) -> None:
    writer.writerow(columns)


def load_run(run_id: str, gold_root: Path) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    path = gold_root / run_id / "cell_transitions.csv"
    if not path.exists():
        raise AnalysisError(f"cell_transitions.csv 不存在：{path}")

    rows: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "game_id",
            "map_id",
            "round",
            "row",
            "col",
            "region",
            "static_map",
            "delta",
            "round_mod5",
            "round_mod20",
            "window_begin",
        }
        if not required.issubset(reader.fieldnames or []):
            raise AnalysisError(f"{path}: 字段不完整")

        for item in reader:
            if item["window_begin"] == "0":
                continue
            region = int(item["region"])
            if region == 1:
                continue
            static_map = int(item["static_map"])
            if static_map not in (0, 2):
                continue

            key = (run_id, item["game_id"], int(item["round"]), region)
            row = rows.get(key)
            if row is None:
                row = {
                    "run_id": run_id,
                    "map_id": item["map_id"],
                    "game_id": item["game_id"],
                    "round": int(item["round"]),
                    "region": region,
                    "round_mod5": int(item["round_mod5"]),
                    "round_mod20": int(item["round_mod20"]),
                    "static2_exposure": 0,
                    "static2_events": 0,
                    "static2_delta_sum": 0,
                    "static2_max_delta": 0,
                    "static0_exposure": 0,
                    "static0_events": 0,
                    "static0_delta_sum": 0,
                    "static2_cells": [],
                }
                rows[key] = row

            delta = int(item["delta"])
            if static_map == 2:
                row["static2_exposure"] += 1
                if delta > 0:
                    row["static2_events"] += 1
                    row["static2_delta_sum"] += delta
                    row["static2_max_delta"] = max(row["static2_max_delta"], delta)
                    row["static2_cells"].append((int(item["row"]), int(item["col"]), delta))
            else:
                row["static0_exposure"] += 1
                if delta > 0:
                    row["static0_events"] += 1
                    row["static0_delta_sum"] += delta

    return rows


def analyze(
    run_ids: list[str],
    analysis_id: str,
    gold_root: Path = DEFAULT_GOLD_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    high_threshold: int = 50,
    cell_high_threshold: int = DEFAULT_CELL_HIGH_THRESHOLD,
) -> dict[str, Any]:
    output_dir = output_root / analysis_id
    output_dir.mkdir(parents=True, exist_ok=True)

    region_round_path = output_dir / "static2_region_rounds.csv"
    batch_path = output_dir / "static2_batches.csv"
    batch_cell_path = output_dir / "static2_batch_cells.csv"
    summary_path = output_dir / "summary.json"

    all_rows: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    run_summaries: dict[str, dict[str, int]] = {}
    for run_id in run_ids:
        rows = load_run(run_id, gold_root)
        all_rows.update(rows)
        run_summaries[run_id] = {
            "region_rounds": len(rows),
            "static2_events": sum(row["static2_events"] for row in rows.values()),
            "static2_delta_sum": sum(row["static2_delta_sum"] for row in rows.values()),
        }

    by_game_round: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows.values():
        by_game_round[(row["run_id"], row["game_id"], row["round"])].append(row)

    observed_high_rows: list[dict[str, Any]] = []
    full_high_rows: list[dict[str, Any]] = []
    observed_multi_high_rounds = 0
    full_multi_high_rounds = 0
    for rows in by_game_round.values():
        observed_highs = [
            row
            for row in rows
            if row["static2_delta_sum"] >= high_threshold or row["static2_max_delta"] >= cell_high_threshold
        ]
        full_highs = [row for row in rows if row["static2_delta_sum"] >= high_threshold]
        if len(observed_highs) > 1:
            observed_multi_high_rounds += 1
        if len(full_highs) > 1:
            full_multi_high_rounds += 1
        observed_high_rows.extend(observed_highs)
        full_high_rows.extend(full_highs)

    with region_round_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        write_header(
            writer,
            [
                "run_id",
                "map_id",
                "game_id",
                "round",
                "region",
                "round_mod5",
                "round_mod20",
                "static2_exposure",
                "static2_events",
                "static2_delta_sum",
                "static2_max_delta",
                "static2_positive_cells",
                "static0_exposure",
                "static0_events",
                "static0_delta_sum",
                "is_high_batch_observed",
                "is_full_high_batch_observed",
            ],
        )
        for row in sorted(all_rows.values(), key=lambda x: (x["run_id"], int(x["game_id"]), x["round"], x["region"])):
            writer.writerow(
                [
                    row["run_id"],
                    row["map_id"],
                    row["game_id"],
                    row["round"],
                    row["region"],
                    row["round_mod5"],
                    row["round_mod20"],
                    row["static2_exposure"],
                    row["static2_events"],
                    row["static2_delta_sum"],
                    row["static2_max_delta"],
                    len(row["static2_cells"]),
                    row["static0_exposure"],
                    row["static0_events"],
                    row["static0_delta_sum"],
                    int(row["static2_delta_sum"] >= high_threshold or row["static2_max_delta"] >= cell_high_threshold),
                    int(row["static2_delta_sum"] >= high_threshold),
                ]
            )

    with batch_path.open("w", encoding="utf-8", newline="") as batch_file, batch_cell_path.open(
        "w", encoding="utf-8", newline=""
    ) as cell_file:
        batch_writer = csv.writer(batch_file)
        cell_writer = csv.writer(cell_file)
        write_header(
            batch_writer,
            [
                "run_id",
                "map_id",
                "game_id",
                "round",
                "high_region",
                "round_mod5",
                "round_mod20",
                "total",
                "max_static2_delta",
                "positive_static2_cells",
                "visible_static2_cells",
                "is_full_high_batch_observed",
                "same_region_static0_events",
                "same_region_static0_delta",
                "other_region_static0_events",
                "other_region_static0_delta",
            ],
        )
        write_header(
            cell_writer,
            [
                "run_id",
                "map_id",
                "game_id",
                "round",
                "region",
                "row",
                "col",
                "delta",
                "batch_total",
                "batch_max_static2_delta",
                "positive_static2_cells",
                "visible_static2_cells",
                "is_full_high_batch_observed",
            ],
        )
        for row in sorted(observed_high_rows, key=lambda x: (x["run_id"], int(x["game_id"]), x["round"], x["region"])):
            peers = by_game_round[(row["run_id"], row["game_id"], row["round"])]
            other_static0_events = sum(peer["static0_events"] for peer in peers if peer["region"] != row["region"])
            other_static0_delta = sum(peer["static0_delta_sum"] for peer in peers if peer["region"] != row["region"])
            positive_cells = len(row["static2_cells"])
            is_full_high = int(row["static2_delta_sum"] >= high_threshold)
            batch_writer.writerow(
                [
                    row["run_id"],
                    row["map_id"],
                    row["game_id"],
                    row["round"],
                    row["region"],
                    row["round_mod5"],
                    row["round_mod20"],
                    row["static2_delta_sum"],
                    row["static2_max_delta"],
                    positive_cells,
                    row["static2_exposure"],
                    is_full_high,
                    row["static0_events"],
                    row["static0_delta_sum"],
                    other_static0_events,
                    other_static0_delta,
                ]
            )
            for cell_row, cell_col, delta in row["static2_cells"]:
                cell_writer.writerow(
                    [
                        row["run_id"],
                        row["map_id"],
                        row["game_id"],
                        row["round"],
                        row["region"],
                        cell_row,
                        cell_col,
                        delta,
                        row["static2_delta_sum"],
                        row["static2_max_delta"],
                        positive_cells,
                        row["static2_exposure"],
                        is_full_high,
                    ]
                )

    summary = {
        "analysis_id": analysis_id,
        "run_ids": run_ids,
        "high_threshold": high_threshold,
        "cell_high_threshold": cell_high_threshold,
        "game_rounds": len(by_game_round),
        "region_rounds": len(all_rows),
        "observed_high_batches": len(observed_high_rows),
        "full_high_batches": len(full_high_rows),
        "observed_multi_high_rounds": observed_multi_high_rounds,
        "full_multi_high_rounds": full_multi_high_rounds,
        "runs": run_summaries,
        "outputs": {
            "region_rounds": display_path(region_round_path),
            "batches": display_path(batch_path),
            "batch_cells": display_path(batch_cell_path),
            "summary": display_path(summary_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分析外围 static_map=2 高额 batch 生成")
    parser.add_argument("--run-id", action="append", dest="run_ids", help="gold_features 下的 run_id，可重复")
    parser.add_argument("--analysis-id", default=DEFAULT_ANALYSIS_ID, help="static2_analysis 输出子目录名")
    parser.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT, help="gold_features 根目录")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="static2_analysis 输出根目录")
    parser.add_argument("--high-threshold", type=int, default=50, help="高额 batch 的 static2 delta_sum 阈值")
    parser.add_argument(
        "--cell-high-threshold",
        type=int,
        default=DEFAULT_CELL_HIGH_THRESHOLD,
        help="单格 static2 delta 判定 partial high batch 的阈值",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = analyze(
        run_ids=args.run_ids or DEFAULT_RUN_IDS,
        analysis_id=args.analysis_id,
        gold_root=args.gold_root,
        output_root=args.output_root,
        high_threshold=args.high_threshold,
        cell_high_threshold=args.cell_high_threshold,
    )
    print(
        "wrote static2 analysis "
        f"analysis_id={summary['analysis_id']} region_rounds={summary['region_rounds']} "
        f"observed_high_batches={summary['observed_high_batches']} "
        f"full_high_batches={summary['full_high_batches']} "
        f"observed_multi_high_rounds={summary['observed_multi_high_rounds']}"
    )
    for name, path in summary["outputs"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
