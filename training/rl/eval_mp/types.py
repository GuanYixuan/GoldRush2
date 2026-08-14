from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from training.opponents import OpponentSpec


@dataclass(frozen=True)
class ParallelEvalConfig:
    num_workers: int = 8
    max_inference_batch_size: int = 64
    inference_timeout_ms: float = 2.0
    worker_join_timeout_s: float = 5.0
    deterministic: bool = False
    enable_fast_runtime_features: bool = False
    fixed_threshold_int: int | None = None


@dataclass(frozen=True)
class EvalTask:
    task_id: str
    seed: int
    map_id: int | None
    opponent_spec: OpponentSpec | None
    agent_player_id: int
    round_count: int
    setting: str = "eval"
    tags: tuple[str, ...] = field(default_factory=tuple)
    policy_sample_key: str | None = None
    policy_sample_seed: int | None = None


@dataclass(frozen=True)
class EvalFeatureRequest:
    worker_id: int
    eval_id: str
    task_id: str
    request_id: str
    round_index: int
    feature_slot: int
    policy_sample_key: str | None = None
    policy_sample_seed: int | None = None


@dataclass(frozen=True)
class EvalEpisodeSummary:
    task_id: str
    setting: str
    seed: int
    map_id: int | None
    opponent: str
    agent_player_id: int
    round_count: int
    episode_length: int
    agent_net_gold: int
    opponent_net_gold: int
    margin: int
    agent_won: bool
    agent_pickup_gold: int
    agent_pickups: int
    agent_bomb_lost_gold: int
    agent_bomb_triggers: int
    agent_trample_penalty: int
    agent_tramples: int
    agent_vision_spent: int
    opponent_pickup_gold: int
    opponent_pickups: int
    opponent_bomb_lost_gold: int
    opponent_bomb_triggers: int
    opponent_trample_penalty: int
    opponent_tramples: int
    opponent_vision_spent: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PairedEvalResult:
    fast_off_summaries: tuple[EvalEpisodeSummary, ...]
    fast_on_summaries: tuple[EvalEpisodeSummary, ...]
    fast_off_stats: dict[str, Any]
    fast_on_stats: dict[str, Any]
    paired_stats: dict[str, Any]


def opponent_key(spec: OpponentSpec | None) -> str:
    if spec is None:
        return "none"
    params = ",".join(f"{key}={value!r}" for key, value in sorted(spec.params.items()))
    return f"{spec.kind}:{spec.name}:{params}"
