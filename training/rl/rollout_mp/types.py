from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from training.opponents import OpponentSpec


PairRole = Literal["first", "second"]


@dataclass(frozen=True)
class MultiprocessRolloutConfig:
    num_workers: int = 8
    max_inference_batch_size: int = 64
    inference_timeout_ms: float = 2.0
    worker_join_timeout_s: float = 5.0
    transition_info_mode: Literal["training", "debug"] = "training"
    enable_fast_runtime_features: bool = False
    reward_fold_gamma: float = 0.97


@dataclass(frozen=True)
class EpisodeTask:
    task_id: str
    pair_id: str
    pair_role: PairRole
    seed: int
    map_id: int | None
    agent_player_id: int
    opponent_spec: OpponentSpec | None
    transition_slot: int


@dataclass(frozen=True)
class FeatureRequest:
    worker_id: int
    rollout_id: str
    task_id: str
    request_id: str
    round_index: int
    feature_slot: int
