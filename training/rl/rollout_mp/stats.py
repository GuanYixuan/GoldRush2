from __future__ import annotations

import torch


def worker_profile_metrics(totals: dict[str, int], episode_count: int) -> dict[str, float | int]:
    steps = int(totals.get("steps", 0))
    metrics: dict[str, float | int] = {
        "worker_profile_episode_count": int(episode_count),
        "worker_profile_transition_count": steps,
    }
    for key, value in sorted(totals.items()):
        metrics[f"worker_sum_{key}"] = int(value)
        if key.endswith("_ns"):
            metrics[f"worker_sum_{key[:-3]}_ms"] = int(value) / 1_000_000
            if steps > 0:
                metrics[f"worker_per_transition_{key[:-3]}_ms"] = int(value) / steps / 1_000_000
    return metrics


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mean(values: list[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
