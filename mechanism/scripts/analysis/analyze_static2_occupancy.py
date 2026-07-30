#!/usr/bin/env python3
"""回查 static2 high batch 触发时候选格的动态占用影响。

默认输入：
  mechanism/data/processed/static2_analysis/maps123_static2_generation/
  mechanism/data/processed/merged_replays/<run_id>/*.json

默认输出：
  mechanism/data/processed/static2_occupancy_analysis/<analysis_id>/

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
DEFAULT_STATIC2_DIR = (
    REPO_ROOT / "mechanism" / "data" / "processed" / "static2_analysis" / "maps123_static2_generation"
)
DEFAULT_MERGED_ROOT = REPO_ROOT / "mechanism" / "data" / "processed" / "merged_replays"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "mechanism" / "data" / "processed" / "static2_occupancy_analysis"
DEFAULT_ANALYSIS_ID = "maps123_static2_occupancy"
GRID_SIZE = 17
EXCLUDED_VALUES = {-5, -3, -1}


class AnalysisError(RuntimeError):
    pass


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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
    raise AnalysisError(f"坐标不属于任何区域：({row},{col})")


def clean_cell(value: int) -> bool:
    return value not in EXCLUDED_VALUES and value >= 0


def frame_dynamic_cells(frame: dict[str, Any]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for player in frame.get("players", []):
        for unit in player.get("units", []):
            cells.add(tuple(unit["position"]))
    for npc in frame.get("npcs", []):
        cells.add(tuple(npc["position"]))
    return cells


def frame_bomb_cells(frame: dict[str, Any]) -> set[tuple[int, int]]:
    grid = frame["grid"]
    return {(row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE) if grid[row][col] == -3}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise AnalysisError(f"输入文件不存在：{path}")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_replay(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AnalysisError(f"merged replay 不存在：{path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != "goldrush2_merged_replay":
        raise AnalysisError(f"{path}: 非 merged replay")
    return data


def candidate_static2_cells(static_map: list[list[int]], region: int) -> set[tuple[int, int]]:
    return {
        (row, col)
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
        if static_map[row][col] == 2 and region_of(row, col) == region
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_blocked: dict[str, dict[str, Any]] = {}
    for blocked in sorted({item["blocked_candidates"] for item in records}):
        rows = [item for item in records if item["blocked_candidates"] == blocked]
        totals = [item["total"] for item in rows]
        by_blocked[str(blocked)] = {
            "events": len(rows),
            "positive_cell_count_dist": dict(sorted(Counter(item["positive_cells"] for item in rows).items())),
            "mean_total": (sum(totals) / len(totals)) if totals else None,
        }

    return {
        "events": len(records),
        "blocked_candidate_dist": dict(sorted(Counter(item["blocked_candidates"] for item in records).items())),
        "dynamic_blocked_candidate_dist": dict(
            sorted(Counter(item["dynamic_blocked_candidates"] for item in records).items())
        ),
        "bomb_blocked_candidate_dist": dict(sorted(Counter(item["bomb_blocked_candidates"] for item in records).items())),
        "positive_on_blocked_events": sum(item["positive_on_blocked_candidates"] > 0 for item in records),
        "positive_on_blocked_cells": sum(item["positive_on_blocked_candidates"] for item in records),
        "positive_cells_gt_legal_candidates": sum(
            item["positive_cells"] > item["legal_candidates"] for item in records
        ),
        "by_blocked_candidates": by_blocked,
    }


def analyze(
    static2_dir: Path = DEFAULT_STATIC2_DIR,
    merged_root: Path = DEFAULT_MERGED_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    analysis_id: str = DEFAULT_ANALYSIS_ID,
) -> dict[str, Any]:
    output_dir = output_root / analysis_id
    output_dir.mkdir(parents=True, exist_ok=True)

    batches_path = static2_dir / "static2_batches.csv"
    cells_path = static2_dir / "static2_batch_cells.csv"
    detail_path = output_dir / "static2_occupancy_batches.csv"
    summary_path = output_dir / "summary.json"

    positive_cells: dict[tuple[str, str, int, int], set[tuple[int, int]]] = defaultdict(set)
    for row in read_csv_rows(cells_path):
        key = (row["run_id"], row["game_id"], int(row["round"]), int(row["region"]))
        positive_cells[key].add((int(row["row"]), int(row["col"])))

    replay_cache: dict[tuple[str, str], dict[str, Any]] = {}
    candidate_cache: dict[tuple[str, str, int], set[tuple[int, int]]] = {}

    def replay(run_id: str, game_id: str) -> dict[str, Any]:
        key = (run_id, game_id)
        if key not in replay_cache:
            replay_cache[key] = load_replay(merged_root / run_id / f"{game_id}.json")
        return replay_cache[key]

    def candidates(run_id: str, game_id: str, region: int) -> set[tuple[int, int]]:
        key = (run_id, game_id, region)
        if key not in candidate_cache:
            candidate_cache[key] = candidate_static2_cells(replay(run_id, game_id)["maps"], region)
        return candidate_cache[key]

    records: list[dict[str, Any]] = []
    for row in read_csv_rows(batches_path):
        run_id = row["run_id"]
        game_id = row["game_id"]
        round_no = int(row["round"])
        region = int(row["high_region"])
        data = replay(run_id, game_id)
        if round_no <= 0:
            raise AnalysisError(f"{run_id}/{game_id}: high batch round 非法：{round_no}")

        prev_frame = data["rounds"][round_no - 1]["merged"]["end"]
        curr_frame = data["rounds"][round_no]["merged"]["start"]
        candidate_cells = candidates(run_id, game_id, region)
        dynamic_blocked = frame_dynamic_cells(curr_frame) & candidate_cells
        bomb_blocked = frame_bomb_cells(curr_frame) & candidate_cells
        blocked = dynamic_blocked | bomb_blocked
        positives = positive_cells[(run_id, game_id, round_no, region)]
        clean_visible = {
            cell
            for cell in candidate_cells
            if clean_cell(prev_frame["grid"][cell[0]][cell[1]]) and clean_cell(curr_frame["grid"][cell[0]][cell[1]])
        }

        records.append(
            {
                "run_id": run_id,
                "map_id": row["map_id"],
                "game_id": game_id,
                "round": round_no,
                "region": region,
                "total": int(row["total"]),
                "is_full_high_batch_observed": int(row["is_full_high_batch_observed"]),
                "candidate_cells": len(candidate_cells),
                "clean_visible_candidates": len(clean_visible),
                "dynamic_blocked_candidates": len(dynamic_blocked),
                "bomb_blocked_candidates": len(bomb_blocked),
                "blocked_candidates": len(blocked),
                "legal_candidates": len(candidate_cells - blocked),
                "positive_cells": len(positives),
                "positive_on_blocked_candidates": len(positives & blocked),
            }
        )

    with detail_path.open("w", encoding="utf-8", newline="") as f:
        columns = [
            "run_id",
            "map_id",
            "game_id",
            "round",
            "region",
            "total",
            "is_full_high_batch_observed",
            "candidate_cells",
            "clean_visible_candidates",
            "dynamic_blocked_candidates",
            "bomb_blocked_candidates",
            "blocked_candidates",
            "legal_candidates",
            "positive_cells",
            "positive_on_blocked_candidates",
        ]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    groups = {
        "all_observed_high": records,
        "all_full_high": [item for item in records if item["is_full_high_batch_observed"]],
        "all_full_high_all_candidates_clean_visible": [
            item
            for item in records
            if item["is_full_high_batch_observed"] and item["clean_visible_candidates"] == item["candidate_cells"]
        ],
        "map12_full_high_all_5_candidates_clean_visible": [
            item
            for item in records
            if item["map_id"] in ("1", "2")
            and item["is_full_high_batch_observed"]
            and item["candidate_cells"] == 5
            and item["clean_visible_candidates"] == 5
        ],
    }
    summary = {
        "analysis_id": analysis_id,
        "inputs": {
            "static2_batches": display_path(batches_path),
            "static2_batch_cells": display_path(cells_path),
            "merged_root": display_path(merged_root),
        },
        "groups": {name: summarize_records(rows) for name, rows in groups.items()},
        "outputs": {
            "batches": display_path(detail_path),
            "summary": display_path(summary_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="分析 static2 high batch 候选格占用影响")
    parser.add_argument("--static2-dir", type=Path, default=DEFAULT_STATIC2_DIR, help="static2_generation 输出目录")
    parser.add_argument("--merged-root", type=Path, default=DEFAULT_MERGED_ROOT, help="merged_replays 根目录")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="输出根目录")
    parser.add_argument("--analysis-id", default=DEFAULT_ANALYSIS_ID, help="输出子目录名")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = analyze(
        static2_dir=args.static2_dir,
        merged_root=args.merged_root,
        output_root=args.output_root,
        analysis_id=args.analysis_id,
    )
    print(
        "wrote static2 occupancy analysis "
        f"analysis_id={summary['analysis_id']} "
        f"events={summary['groups']['all_observed_high']['events']}"
    )
    for name, path in summary["outputs"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
