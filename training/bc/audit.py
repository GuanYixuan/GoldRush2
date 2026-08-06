from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from simulator.errors import SimulatorRuleError
from training.models import GoldRushPolicyNetwork, PolicyNetworkConfig

from .dataset import load_manifest, load_shard
from .schema import write_json


@dataclass(frozen=True)
class BcAuditResult:
    summary: dict[str, Any]
    incompatibility_count: int


def audit_dataset(
    dataset_dir: Path,
    *,
    device: torch.device | str = "cpu",
    sample_limit: int | None = None,
    output_dir: Path | None = None,
    model_config: PolicyNetworkConfig | None = None,
) -> BcAuditResult:
    dataset_dir = Path(dataset_dir)
    output_dir = dataset_dir / "audit" if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(dataset_dir)
    model = GoldRushPolicyNetwork(model_config).to(device)
    model.eval()

    counters: Counter[str] = Counter()
    action_counts: Counter[int] = Counter()
    k_counts: Counter[int] = Counter()
    order_counts: Counter[int] = Counter()
    vp_counts: Counter[int] = Counter()
    ko_counts: Counter[int] = Counter()
    diagnostic_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    split_seeds: dict[str, set[int]] = defaultdict(set)
    margins: list[int] = []
    incompatible: list[dict[str, Any]] = []

    for shard_info in manifest["shards"]:
        shard = load_shard(dataset_dir / shard_info["path"], expected_config_hash=manifest["config_hash"])
        n = int(shard["k"].shape[0])
        indices = range(n if sample_limit is None or sample_limit < 0 else min(n, sample_limit))
        _validate_round_continuity(shard, shard_info["path"])
        for idx in indices:
            counters["transition_count"] += 1
            split = shard["split"][idx]
            split_counts[split] += 1
            split_seeds[split].add(int(shard["paired_seed"][idx].item()))
            k = int(shard["k"][idx].item())
            order = int(shard["order"][idx].item())
            vp = int(shard["vp"][idx].item())
            k_counts[k] += 1
            order_counts[order] += 1
            vp_counts[vp] += 1
            ko_counts[2 * k + order] += 1
            for action in shard["actions"][idx].tolist():
                action_counts[int(action)] += 1
            diagnostic_counts[_diagnostic_tag(shard, idx)] += 1
            margins.append(int(shard["margin"][idx].item()))

        incompatible.extend(_audit_mask_compatibility(model, shard, shard_info["path"], device=device, sample_limit=sample_limit))

    _validate_split_seeds(split_seeds)
    summary = {
        "schema": "goldrush2_bc_audit_v1",
        "transition_count": int(counters["transition_count"]),
        "split_counts": dict(sorted(split_counts.items())),
        "split_seed_counts": {split: len(seeds) for split, seeds in sorted(split_seeds.items())},
        "action_counts": _int_counter(action_counts),
        "k_counts": _int_counter(k_counts),
        "order_counts": _int_counter(order_counts),
        "vp_counts": _int_counter(vp_counts),
        "ko_counts": _int_counter(ko_counts),
        "teacher_diagnostic_counts": dict(sorted(diagnostic_counts.items())),
        "margin_mean": None if not margins else mean(margins),
        "hard_mask_incompatibility_count": len(incompatible),
        "hard_mask_incompatibility_rate": 0.0 if counters["transition_count"] == 0 else len(incompatible) / counters["transition_count"],
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "incompatibilities.jsonl").open("w", encoding="utf-8") as handle:
        for record in incompatible:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
    return BcAuditResult(summary=summary, incompatibility_count=len(incompatible))


def _audit_mask_compatibility(
    model: GoldRushPolicyNetwork,
    shard: dict[str, Any],
    shard_path: str,
    *,
    device: torch.device | str,
    sample_limit: int | None,
) -> list[dict[str, Any]]:
    n = int(shard["k"].shape[0])
    limit = n if sample_limit is None or sample_limit < 0 else min(n, sample_limit)
    incompatible: list[dict[str, Any]] = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, limit, batch_size):
            end = min(limit, start + batch_size)
            try:
                model.evaluate_actions(
                    shard["planes"][start:end].to(device),
                    shard["scalars"][start:end].to(device),
                    shard["actions"][start:end].to(device),
                    shard["k"][start:end].to(device),
                    shard["order"][start:end].to(device),
                    shard["vp"][start:end].to(device),
                )
            except ValueError:
                for idx in range(start, end):
                    try:
                        model.evaluate_actions(
                            shard["planes"][idx : idx + 1].to(device),
                            shard["scalars"][idx : idx + 1].to(device),
                            shard["actions"][idx : idx + 1].to(device),
                            shard["k"][idx : idx + 1].to(device),
                            shard["order"][idx : idx + 1].to(device),
                            shard["vp"][idx : idx + 1].to(device),
                        )
                    except ValueError as exc:
                        incompatible.append(
                            {
                                "shard": shard_path,
                                "index": idx,
                                "episode_id": shard["episode_id"][idx],
                                "round_index": int(shard["round_index"][idx].item()),
                                "player_id": int(shard["player_id"][idx].item()),
                                "first_player_id": int(shard["first_player_id"][idx].item()),
                                "actions": [int(value) for value in shard["actions"][idx].tolist()],
                                "k": int(shard["k"][idx].item()),
                                "order": int(shard["order"][idx].item()),
                                "vp": int(shard["vp"][idx].item()),
                                "error": str(exc),
                            }
                        )
    return incompatible


def _validate_round_continuity(shard: dict[str, Any], shard_path: str) -> None:
    by_episode: dict[str, list[int]] = defaultdict(list)
    for episode_id, round_index in zip(shard["episode_id"], shard["round_index"].tolist(), strict=True):
        by_episode[str(episode_id)].append(int(round_index))
    for episode_id, rounds in by_episode.items():
        expected = list(range(len(rounds)))
        if rounds != expected:
            raise SimulatorRuleError(f"round discontinuity in {shard_path} episode {episode_id}: first values {rounds[:10]}")


def _validate_split_seeds(split_seeds: dict[str, set[int]]) -> None:
    splits = sorted(split_seeds)
    for idx, left in enumerate(splits):
        for right in splits[idx + 1 :]:
            overlap = split_seeds[left] & split_seeds[right]
            if overlap:
                raise SimulatorRuleError(f"BC split seeds overlap between {left} and {right}: {sorted(overlap)[:5]}")


def _int_counter(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _diagnostic_tag(shard: dict[str, Any], idx: int) -> str:
    if "teacher_diagnostic_tag" in shard:
        return str(shard["teacher_diagnostic_tag"][idx])
    if "layer_tag" in shard:
        return str(shard["layer_tag"][idx])
    return "unknown"
