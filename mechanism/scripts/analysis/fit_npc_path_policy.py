#!/usr/bin/env python3
"""拟合 NPC 3-step path-level softmax 策略。

默认读取：
  mechanism/data/processed/merged_replays/<run_id>/*.json

默认输出：
  mechanism/data/processed/npc_policy_fit/<fit_id>/

本脚本只做离线拟合，不访问评测平台。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MERGED_ROOT = REPO_ROOT / "mechanism" / "data" / "processed" / "merged_replays"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "mechanism" / "data" / "processed" / "npc_policy_fit"
DEFAULT_RUN_IDS = [
    "symobs-a-map1-100-20260727-231446",
    "symobs-a-map2-100-20260727-231704",
    "symobs-m3a-map3-100-20260728-200953",
    "symobs-m3b-map3-100-20260728-202647",
]
GRID_SIZE = 17
BOMB = -3
OBSTACLE_STATIC = 1
ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
    4: (0, 0),
}
FEATURE_NAMES = [
    "dynamic_reward_div10",
    "enter_bomb_count",
    "adjacent_bomb_count",
    "stay_count",
    "straight3",
    "backtrack",
    "bomb_trapped_stay",
    "first_two_same_nonstay",
    "last_two_same_nonstay",
    "sandwich",
]
MODEL_FEATURES = {
    "M1_dynamic_reward": [0],
    "M2a_dynamic_reward_enter_bomb": [0, 1],
    "M2_dynamic_reward_bomb": [0, 1, 2],
    "M3a_dynamic_enter_bomb_stay": [0, 1, 3],
    "M3b_dynamic_enter_bomb_stay_straight": [0, 1, 3, 4],
    "M3c_dynamic_enter_bomb_shape": [0, 1, 3, 4, 5],
    "M4a_dynamic_enter_bomb_shape_trapped_stay": [0, 1, 3, 4, 5, 6],
    "M5a_m4a_first_two_same": [0, 1, 3, 4, 5, 6, 7],
    "M5b_m5a_last_two_same": [0, 1, 3, 4, 5, 6, 7, 8],
    "M5c_m5b_sandwich": [0, 1, 3, 4, 5, 6, 7, 8, 9],
}
OPPOSITE_ACTIONS = {(0, 1), (1, 0), (2, 3), (3, 2)}


class FitError(RuntimeError):
    pass


@dataclass(frozen=True)
class PathCandidate:
    actions: tuple[int, int, int]
    positions: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass
class Sample:
    features: np.ndarray
    actual_index: int
    split: str


def read_merged(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("format") != "goldrush2_merged_replay":
        raise FitError(f"{path}: 非 merged replay")
    return data


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def validate_static_map(static_map: Any, path: Path) -> None:
    if not isinstance(static_map, list) or len(static_map) != GRID_SIZE:
        raise FitError(f"{path}: 静态地图不是 {GRID_SIZE}x{GRID_SIZE}")
    for row, values in enumerate(static_map):
        if not isinstance(values, list) or len(values) != GRID_SIZE:
            raise FitError(f"{path}: 静态地图第 {row} 行长度非法")


def npc_by_id(frame: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for npc in frame.get("npcs", []):
        pos = npc.get("position")
        if pos is None:
            continue
        out[int(npc["id"])] = npc
    return out


def stable_bucket(value: str, modulus: int) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulus


def legal_paths(start: tuple[int, int], static_map: tuple[tuple[int, ...], ...]) -> list[PathCandidate]:
    paths: list[PathCandidate] = []

    def rec(pos: tuple[int, int], actions: list[int], positions: list[tuple[int, int]]) -> None:
        if len(actions) == 3:
            paths.append(PathCandidate(tuple(actions), tuple(positions)))  # type: ignore[arg-type]
            return
        row, col = pos
        for action, (dr, dc) in ACTION_DELTAS.items():
            target = (row + dr, col + dc)
            if not (0 <= target[0] < GRID_SIZE and 0 <= target[1] < GRID_SIZE):
                continue
            if static_map[target[0]][target[1]] == OBSTACLE_STATIC:
                continue
            actions.append(action)
            positions.append(target)
            rec(target, actions, positions)
            positions.pop()
            actions.pop()

    rec(start, [], [])
    return paths


def all_legal_neighbors_are_bombs(
    start_pos: tuple[int, int],
    static_map: tuple[tuple[int, ...], ...],
    bombs_remaining: set[tuple[int, int]],
) -> bool:
    row, col = start_pos
    legal_neighbors = []
    for action, (dr, dc) in ACTION_DELTAS.items():
        if action == 4:
            continue
        target = (row + dr, col + dc)
        if not (0 <= target[0] < GRID_SIZE and 0 <= target[1] < GRID_SIZE):
            continue
        if static_map[target[0]][target[1]] == OBSTACLE_STATIC:
            continue
        legal_neighbors.append(target)
    return bool(legal_neighbors) and all(pos in bombs_remaining for pos in legal_neighbors)


def path_features(
    start_pos: tuple[int, int],
    candidate: PathCandidate,
    gold_remaining: dict[tuple[int, int], int],
    bombs_remaining: set[tuple[int, int]],
    bomb_trapped_start: bool,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    local_gold = dict(gold_remaining)
    local_bombs = set(bombs_remaining)
    reward = 0
    enter_bomb = 0
    adjacent_bomb = 0
    prev = start_pos
    for action, pos in zip(candidate.actions, candidate.positions):
        moved = action != 4 and pos != prev
        if moved:
            value = local_gold.get(pos, 0)
            if value > 0:
                gain = math.ceil(0.65 * value)
                reward += gain
                remaining = value - gain
                if remaining > 0:
                    local_gold[pos] = remaining
                else:
                    local_gold.pop(pos, None)
            if pos in local_bombs:
                enter_bomb += 1
                local_bombs.remove(pos)
        if any(abs(pos[0] - br) + abs(pos[1] - bc) == 1 for br, bc in local_bombs):
            adjacent_bomb += 1
        prev = pos

    actions = candidate.actions
    stay_count = sum(1 for action in actions if action == 4)
    straight3 = int(actions[0] == actions[1] == actions[2] and actions[0] != 4)
    backtrack = int(any((left, right) in OPPOSITE_ACTIONS for left, right in zip(actions, actions[1:])))
    bomb_trapped_stay = int(bomb_trapped_start and actions == (4, 4, 4))
    first_two_same = int(actions[0] == actions[1] and actions[0] != 4)
    last_two_same = int(actions[1] == actions[2] and actions[1] != 4)
    sandwich = int(actions[0] == actions[2] and actions[0] != actions[1] and actions[0] != 4 and actions[1] != 4)
    return (
        reward / 10.0,
        float(enter_bomb),
        float(adjacent_bomb),
        float(stay_count),
        float(straight3),
        float(backtrack),
        float(bomb_trapped_stay),
        float(first_two_same),
        float(last_two_same),
        float(sandwich),
    )


def consume_actual_path(
    start_pos: tuple[int, int],
    candidate: PathCandidate,
    gold_remaining: dict[tuple[int, int], int],
    bombs_remaining: set[tuple[int, int]],
) -> None:
    prev = start_pos
    for action, pos in zip(candidate.actions, candidate.positions):
        moved = action != 4 and pos != prev
        if moved:
            value = gold_remaining.get(pos, 0)
            if value > 0:
                gain = math.ceil(0.65 * value)
                remaining = value - gain
                if remaining > 0:
                    gold_remaining[pos] = remaining
                else:
                    gold_remaining.pop(pos, None)
            bombs_remaining.discard(pos)
        prev = pos


def visible_gold(grid: list[list[int]]) -> dict[tuple[int, int], int]:
    return {
        (r, c): int(grid[r][c])
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if int(grid[r][c]) > 0
    }


def visible_bombs(grid: list[list[int]]) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if int(grid[r][c]) == BOMB
    }


def collect_samples(
    run_ids: list[str],
    merged_root: Path,
    sample_mod: int,
    max_samples_per_run: int,
    valid_mod: int,
    valid_remainder: int,
) -> tuple[list[Sample], dict[str, Any]]:
    samples: list[Sample] = []
    path_cache: dict[tuple[tuple[tuple[int, ...], ...], tuple[int, int]], list[PathCandidate]] = {}
    counters = {
        "eligible_ordered_samples": 0,
        "used_samples": 0,
        "skipped_incomplete_ordered_rounds": 0,
        "by_run": {},
    }

    for run_id in run_ids:
        input_dir = merged_root / run_id
        if not input_dir.exists():
            raise FitError(f"merged replay 目录不存在：{input_dir}")
        files = sorted(input_dir.glob("*.json"))
        if not files:
            raise FitError(f"merged replay 目录为空：{input_dir}")

        run_seen = 0
        run_used = 0
        run_incomplete_rounds = 0
        for path in files:
            if max_samples_per_run and run_used >= max_samples_per_run:
                break
            data = read_merged(path)
            game_id = data["source"]["game_id"]
            validate_static_map(data["maps"], path)
            static_map = tuple(tuple(int(v) for v in row) for row in data["maps"])
            split = "valid" if stable_bucket(f"{run_id}:{game_id}", valid_mod) == valid_remainder else "train"

            for round_obj in data["rounds"]:
                if max_samples_per_run and run_used >= max_samples_per_run:
                    break
                start_frame = round_obj["merged"]["start"]
                end_frame = round_obj["merged"]["end"]
                start_npcs = npc_by_id(start_frame)
                end_npcs = npc_by_id(end_frame)
                order = end_frame.get("dispatch_order") or []
                npc_order = [int(item) for item in order if int(item) < 0]
                if len(npc_order) != 7:
                    run_incomplete_rounds += 1
                    continue

                round_paths: dict[int, tuple[tuple[int, int], PathCandidate]] = {}
                for npc_id in npc_order:
                    start_npc = start_npcs.get(npc_id)
                    end_npc = end_npcs.get(npc_id)
                    if start_npc is None or end_npc is None:
                        break
                    actions_obj = end_npc.get("actions")
                    if not actions_obj or len(actions_obj) != 3:
                        break
                    start_pos = tuple(int(x) for x in start_npc["position"])
                    end_pos = tuple(int(x) for x in end_npc["position"])
                    actions = tuple(int(a) for a in actions_obj)
                    key = (static_map, start_pos)
                    candidates = path_cache.get(key)
                    if candidates is None:
                        candidates = legal_paths(start_pos, static_map)
                        path_cache[key] = candidates
                    candidate_by_actions = {candidate.actions: candidate for candidate in candidates}
                    actual_candidate = candidate_by_actions.get(actions)
                    if actual_candidate is None:
                        raise FitError(
                            f"{path}: round={round_obj['round']} npc={npc_id} actual actions 不在合法候选集中：{actions}"
                        )
                    if actual_candidate.positions[-1] != end_pos:
                        raise FitError(
                            f"{path}: round={round_obj['round']} npc={npc_id} actual path 终点不一致："
                            f"candidate={actual_candidate.positions[-1]} end={end_pos}"
                        )
                    round_paths[npc_id] = (start_pos, actual_candidate)
                else:
                    gold_remaining = visible_gold(start_frame["grid"])
                    bombs_remaining = visible_bombs(start_frame["grid"])
                    for npc_id in npc_order:
                        start_pos, actual_candidate = round_paths[npc_id]
                        run_seen += 1
                        counters["eligible_ordered_samples"] += 1
                        if (run_seen - 1) % sample_mod == 0 and (
                            not max_samples_per_run or run_used < max_samples_per_run
                        ):
                            candidates = path_cache[(static_map, start_pos)]
                            actual_index = None
                            feature_rows = []
                            bomb_trapped_start = all_legal_neighbors_are_bombs(
                                start_pos,
                                static_map,
                                bombs_remaining,
                            )
                            for index, candidate in enumerate(candidates):
                                if candidate.actions == actual_candidate.actions:
                                    actual_index = index
                                feature_rows.append(
                                    path_features(
                                        start_pos,
                                        candidate,
                                        gold_remaining,
                                        bombs_remaining,
                                        bomb_trapped_start,
                                    )
                                )
                            if actual_index is None:
                                raise FitError("internal error: actual_index missing")
                            samples.append(
                                Sample(
                                    features=np.asarray(feature_rows, dtype=np.float64),
                                    actual_index=actual_index,
                                    split=split,
                                )
                            )
                            run_used += 1
                            counters["used_samples"] += 1
                        consume_actual_path(start_pos, actual_candidate, gold_remaining, bombs_remaining)
                    continue

                run_incomplete_rounds += 1

        counters["by_run"][run_id] = {
            "eligible_ordered_samples": run_seen,
            "used_samples": run_used,
            "skipped_incomplete_ordered_rounds": run_incomplete_rounds,
        }
        counters["skipped_incomplete_ordered_rounds"] += run_incomplete_rounds
        print(f"loaded run={run_id} eligible={run_seen} used={run_used}")

    return samples, counters


def loss_and_grad(samples: list[Sample], feature_indices: list[int], weights: np.ndarray, l2: float) -> tuple[float, np.ndarray]:
    loss = 0.0
    grad = np.zeros_like(weights)
    for sample in samples:
        x = sample.features[:, feature_indices]
        scores = x @ weights
        max_score = float(np.max(scores))
        exp_scores = np.exp(scores - max_score)
        probs = exp_scores / float(np.sum(exp_scores))
        actual = sample.actual_index
        loss += max_score + math.log(float(np.sum(exp_scores))) - float(scores[actual])
        grad += probs @ x - x[actual]
    n = max(len(samples), 1)
    return loss / n + 0.5 * l2 * float(weights @ weights), grad / n + l2 * weights


def fit_model(samples: list[Sample], feature_indices: list[int], l2: float, maxiter: int) -> dict[str, Any]:
    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        return loss_and_grad(samples, feature_indices, weights, l2)

    result = minimize(
        objective,
        np.zeros(len(feature_indices), dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-8, "gtol": 1e-6},
    )
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "loss": float(result.fun),
        "weights": [float(x) for x in result.x],
    }


def evaluate(samples: list[Sample], feature_indices: list[int], weights: np.ndarray) -> dict[str, Any]:
    if not samples:
        raise FitError("评估样本为空")
    total_nll = 0.0
    top1 = 0
    top3 = 0
    mrr = 0.0
    prob_sum = 0.0
    actual_features = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    expected_features = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    uniform_features = np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    for sample in samples:
        scores = sample.features[:, feature_indices] @ weights if feature_indices else np.zeros(len(sample.features))
        max_score = float(np.max(scores))
        exp_scores = np.exp(scores - max_score)
        probs = exp_scores / float(np.sum(exp_scores))
        actual = sample.actual_index
        actual_score = float(scores[actual])
        nll = max_score + math.log(float(np.sum(exp_scores))) - actual_score
        total_nll += nll
        prob_sum += math.exp(-nll)

        order = np.argsort(-scores, kind="stable")
        rank = int(np.where(order == actual)[0][0]) + 1
        top1 += int(rank == 1)
        top3 += int(rank <= 3)
        mrr += 1.0 / rank
        actual_features += sample.features[actual]
        expected_features += probs @ sample.features
        uniform_features += np.mean(sample.features, axis=0)

    n = len(samples)
    return {
        "samples": n,
        "nll": total_nll / n,
        "top1": top1 / n,
        "top3": top3 / n,
        "mrr": mrr / n,
        "actual_path_probability": prob_sum / n,
        "actual_feature_avg": dict(zip(FEATURE_NAMES, (actual_features / n).tolist())),
        "model_expected_feature_avg": dict(zip(FEATURE_NAMES, (expected_features / n).tolist())),
        "uniform_expected_feature_avg": dict(zip(FEATURE_NAMES, (uniform_features / n).tolist())),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", action="append", dest="run_ids", default=None)
    parser.add_argument("--merged-root", type=Path, default=DEFAULT_MERGED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fit-id", default=None)
    parser.add_argument("--sample-mod", type=int, default=5)
    parser.add_argument("--max-samples-per-run", type=int, default=50000)
    parser.add_argument("--valid-mod", type=int, default=5)
    parser.add_argument("--valid-remainder", type=int, default=0)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--maxiter", type=int, default=100)
    args = parser.parse_args()

    if args.sample_mod <= 0:
        raise FitError("--sample-mod 必须为正数")
    if args.valid_mod <= 1:
        raise FitError("--valid-mod 必须大于 1")
    if not (0 <= args.valid_remainder < args.valid_mod):
        raise FitError("--valid-remainder 必须在 [0, valid_mod) 内")

    run_ids = args.run_ids or DEFAULT_RUN_IDS
    fit_id = args.fit_id or datetime.now().strftime("npc_path_m1_m5_%Y%m%d_%H%M%S")
    output_dir = args.output_root / fit_id
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, counters = collect_samples(
        run_ids=run_ids,
        merged_root=args.merged_root,
        sample_mod=args.sample_mod,
        max_samples_per_run=args.max_samples_per_run,
        valid_mod=args.valid_mod,
        valid_remainder=args.valid_remainder,
    )
    train_samples = [sample for sample in samples if sample.split == "train"]
    valid_samples = [sample for sample in samples if sample.split == "valid"]
    if not train_samples or not valid_samples:
        raise FitError(f"train/valid 样本为空：train={len(train_samples)} valid={len(valid_samples)}")

    models: dict[str, Any] = {
        "M0_uniform": {
            "feature_names": [],
            "weights": [],
            "fit": {"success": True, "message": "baseline", "iterations": 0, "loss": None},
        }
    }
    for model_name, indices in MODEL_FEATURES.items():
        fit = fit_model(train_samples, indices, args.l2, args.maxiter)
        models[model_name] = {
            "feature_names": [FEATURE_NAMES[index] for index in indices],
            "weights": dict(zip((FEATURE_NAMES[index] for index in indices), fit["weights"])),
            "fit": {k: v for k, v in fit.items() if k != "weights"},
        }
        print(f"fit {model_name} success={fit['success']} loss={fit['loss']:.6f} weights={fit['weights']}")

    metrics_rows: list[dict[str, Any]] = []
    validation_summary_rows: list[dict[str, Any]] = []
    for model_name, info in models.items():
        indices = [FEATURE_NAMES.index(name) for name in info["feature_names"]]
        weights = np.asarray([info["weights"][name] for name in info["feature_names"]], dtype=np.float64)
        for split_name, split_samples in [("train", train_samples), ("valid", valid_samples)]:
            metric = evaluate(split_samples, indices, weights)
            info.setdefault("metrics", {})[split_name] = metric
            metrics_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "samples": metric["samples"],
                    "nll": f"{metric['nll']:.8f}",
                    "top1": f"{metric['top1']:.6f}",
                    "top3": f"{metric['top3']:.6f}",
                    "mrr": f"{metric['mrr']:.6f}",
                    "actual_path_probability": f"{metric['actual_path_probability']:.8f}",
                }
            )
            for feature_name in FEATURE_NAMES:
                validation_summary_rows.append(
                    {
                        "model": model_name,
                        "split": split_name,
                        "feature": feature_name,
                        "actual_avg": f"{metric['actual_feature_avg'][feature_name]:.8f}",
                        "model_expected_avg": f"{metric['model_expected_feature_avg'][feature_name]:.8f}",
                        "uniform_expected_avg": f"{metric['uniform_expected_feature_avg'][feature_name]:.8f}",
                    }
                )

    weights_path = output_dir / "weights.json"
    metrics_path = output_dir / "metrics.json"
    ablation_path = output_dir / "feature_ablation.csv"
    summary_path = output_dir / "validation_summary.csv"
    metadata = {
        "fit_id": fit_id,
        "run_ids": run_ids,
        "feature_names": FEATURE_NAMES,
        "sample_mod": args.sample_mod,
        "max_samples_per_run": args.max_samples_per_run,
        "valid_mod": args.valid_mod,
        "valid_remainder": args.valid_remainder,
        "l2": args.l2,
        "counters": counters,
        "split_counts": {"train": len(train_samples), "valid": len(valid_samples)},
        "models": models,
    }
    with weights_path.open("w", encoding="utf-8") as f:
        json.dump({name: info["weights"] for name, info in models.items()}, f, ensure_ascii=False, indent=2)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    write_csv(
        ablation_path,
        metrics_rows,
        ["model", "split", "samples", "nll", "top1", "top3", "mrr", "actual_path_probability"],
    )
    write_csv(
        summary_path,
        validation_summary_rows,
        ["model", "split", "feature", "actual_avg", "model_expected_avg", "uniform_expected_avg"],
    )

    print(f"weights: {display_path(weights_path)}")
    print(f"metrics: {display_path(metrics_path)}")
    print(f"feature_ablation: {display_path(ablation_path)}")
    print(f"validation_summary: {display_path(summary_path)}")


if __name__ == "__main__":
    main()
