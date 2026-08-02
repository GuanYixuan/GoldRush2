from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from statistics import mean, median
from typing import Any

from simulator.envs.round_step import RoundStepMechanisms
from simulator.errors import SimulatorRuleError
from simulator.mechanisms.maps import MapPool, SpawnConfig
from simulator.observation.sdk import GameInput
from simulator.types import GameOutput
from training.opponents import OpponentSpec

from .env import SingleAgentEnvConfig
from .rewards import RewardFn
from .rollout import EpisodeBatch, Trajectory
from .sampler import BatchRolloutSampler


AgentPolicy = Callable[[GameInput], GameOutput | Sequence[int]]


@dataclass(frozen=True)
class EvaluationCase:
    seed: int
    map_id: int | None = None
    opponent_spec: OpponentSpec | None = None
    tag: str = ""


@dataclass(frozen=True)
class EvaluationConfig:
    env_config: SingleAgentEnvConfig
    cases: tuple[EvaluationCase, ...]
    mechanisms: RoundStepMechanisms | None = None
    map_pool: MapPool | None = None
    spawn: SpawnConfig | None = None
    reward_fn: RewardFn | None = None

    def __post_init__(self) -> None:
        if not self.cases:
            raise SimulatorRuleError("EvaluationConfig requires at least one case")

    @staticmethod
    def anchor_grid(
        *,
        env_config: SingleAgentEnvConfig,
        seeds: Sequence[int],
        map_ids: Sequence[int | None],
        opponent_specs: Sequence[OpponentSpec | None],
        tag: str = "anchor",
        mechanisms: RoundStepMechanisms | None = None,
        map_pool: MapPool | None = None,
        spawn: SpawnConfig | None = None,
        reward_fn: RewardFn | None = None,
    ) -> EvaluationConfig:
        _require_non_empty("seeds", seeds)
        _require_non_empty("map_ids", map_ids)
        _require_non_empty("opponent_specs", opponent_specs)
        cases = tuple(
            EvaluationCase(seed=int(seed), map_id=map_id, opponent_spec=opponent_spec, tag=tag)
            for seed in seeds
            for map_id in map_ids
            for opponent_spec in opponent_specs
        )
        return EvaluationConfig(env_config=env_config, cases=cases, mechanisms=mechanisms, map_pool=map_pool, spawn=spawn, reward_fn=reward_fn)

    @staticmethod
    def matrix_by_slice(
        *,
        env_config: SingleAgentEnvConfig,
        map_ids: Sequence[int | None],
        opponent_specs: Sequence[OpponentSpec | None],
        base_seed: int,
        seeds_per_slice: int,
        slice_seed_stride: int = 10_000,
        tag: str = "matrix",
        mechanisms: RoundStepMechanisms | None = None,
        map_pool: MapPool | None = None,
        spawn: SpawnConfig | None = None,
        reward_fn: RewardFn | None = None,
    ) -> EvaluationConfig:
        _require_non_empty("map_ids", map_ids)
        _require_non_empty("opponent_specs", opponent_specs)
        if seeds_per_slice <= 0:
            raise SimulatorRuleError(f"seeds_per_slice must be positive, got {seeds_per_slice}")
        if slice_seed_stride < seeds_per_slice:
            raise SimulatorRuleError(f"slice_seed_stride must be >= seeds_per_slice, got {slice_seed_stride}")

        cases: list[EvaluationCase] = []
        for map_index, map_id in enumerate(map_ids):
            for opponent_index, opponent_spec in enumerate(opponent_specs):
                slice_index = map_index * len(opponent_specs) + opponent_index
                slice_base_seed = int(base_seed) + slice_index * slice_seed_stride
                for offset in range(seeds_per_slice):
                    cases.append(EvaluationCase(seed=slice_base_seed + offset, map_id=map_id, opponent_spec=opponent_spec, tag=tag))
        return EvaluationConfig(env_config=env_config, cases=tuple(cases), mechanisms=mechanisms, map_pool=map_pool, spawn=spawn, reward_fn=reward_fn)


@dataclass(frozen=True)
class EvaluationResult:
    config: EvaluationConfig
    batch: EpisodeBatch
    metrics: dict[str, Any]


class EvaluationSuite:
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def evaluate(self, agent_policy: AgentPolicy) -> EvaluationResult:
        return evaluate_policy(agent_policy, self.config)


def evaluate_policy(agent_policy: AgentPolicy, config: EvaluationConfig) -> EvaluationResult:
    sampler = BatchRolloutSampler(
        env_config=config.env_config,
        mechanisms=config.mechanisms,
        map_pool=config.map_pool,
        spawn=config.spawn,
        reward_fn=config.reward_fn,
    )
    trajectories: list[Trajectory] = []
    for case_index, case in enumerate(config.cases):
        batch = sampler.collect(
            agent_policy,
            pair_count=1,
            seed=case.seed,
            map_ids=None if case.map_id is None else (case.map_id,),
            opponent_specs=None if case.opponent_spec is None else (case.opponent_spec,),
        )
        case_id = _case_id(case_index, case)
        for trajectory in batch.trajectories:
            trajectories.append(
                replace(
                    trajectory,
                    episode_id=f"{case_id}-{trajectory.pair_role}",
                    pair_id=case_id,
                )
            )
    combined = EpisodeBatch(trajectories=tuple(trajectories))
    return EvaluationResult(config=config, batch=combined, metrics=_evaluation_metrics(config, combined))


def _evaluation_metrics(config: EvaluationConfig, batch: EpisodeBatch) -> dict[str, Any]:
    metrics = dict(batch.metrics())
    margins = [_net_gold_margin(trajectory) for trajectory in batch.trajectories]
    agent_vision_spent = [_score(trajectory, "vision_spent", trajectory.agent_player_id) for trajectory in batch.trajectories]
    opponent_vision_spent = [_score(trajectory, "vision_spent", _opponent_player_id(trajectory)) for trajectory in batch.trajectories]
    agent_pickups = [_event_count(trajectory, "pickups", trajectory.agent_player_id) for trajectory in batch.trajectories]
    opponent_pickups = [_event_count(trajectory, "pickups", _opponent_player_id(trajectory)) for trajectory in batch.trajectories]
    agent_bombs = [_event_count(trajectory, "bomb_triggers", trajectory.agent_player_id) for trajectory in batch.trajectories]
    opponent_bombs = [_event_count(trajectory, "bomb_triggers", _opponent_player_id(trajectory)) for trajectory in batch.trajectories]

    metrics.update(
        {
            "case_count": len(config.cases),
            "pair_count": len(config.cases),
            "map_ids": tuple(sorted({trajectory.map_id for trajectory in batch.trajectories}, key=lambda item: -1 if item is None else item)),
            "opponent_count": len({_opponent_key(trajectory.opponent_spec) for trajectory in batch.trajectories}),
            "net_gold_margin_mean": mean(margins),
            "net_gold_margin_median": median(margins),
            "net_gold_margin_min": min(margins),
            "net_gold_margin_max": max(margins),
            "mean_agent_vision_spent": mean(agent_vision_spent),
            "mean_opponent_vision_spent": mean(opponent_vision_spent),
            "mean_agent_pickups": mean(agent_pickups),
            "mean_opponent_pickups": mean(opponent_pickups),
            "mean_agent_bomb_triggers": mean(agent_bombs),
            "mean_opponent_bomb_triggers": mean(opponent_bombs),
            "by_map": _group_metrics(batch.trajectories, lambda trajectory: str(trajectory.map_id)),
            "by_opponent": _group_metrics(batch.trajectories, lambda trajectory: _opponent_key(trajectory.opponent_spec)),
        }
    )
    return metrics


def _group_metrics(trajectories: tuple[Trajectory, ...], key_fn: Callable[[Trajectory], str]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Trajectory]] = {}
    for trajectory in trajectories:
        grouped.setdefault(key_fn(trajectory), []).append(trajectory)
    return {key: EpisodeBatch(trajectories=tuple(items)).metrics() for key, items in sorted(grouped.items())}


def _case_id(case_index: int, case: EvaluationCase) -> str:
    map_part = "map-auto" if case.map_id is None else f"map-{case.map_id}"
    tag_part = f"{case.tag}-" if case.tag else ""
    return f"{tag_part}case-{case_index:06d}-{map_part}-seed-{case.seed}"


def _net_gold_margin(trajectory: Trajectory) -> int:
    return _score(trajectory, "net_gold", trajectory.agent_player_id) - _score(trajectory, "net_gold", _opponent_player_id(trajectory))


def _score(trajectory: Trajectory, score_name: str, player_id: int) -> int:
    return trajectory.terminal_info["scores"][score_name][player_id]


def _event_count(trajectory: Trajectory, event_name: str, player_id: int) -> int:
    return trajectory.terminal_info["events"][event_name][player_id]


def _opponent_player_id(trajectory: Trajectory) -> int:
    return 2 if trajectory.agent_player_id == 1 else 1


def _opponent_key(spec: OpponentSpec) -> str:
    params = json.dumps(dict(spec.params), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"{spec.kind}:{spec.name}:{params}"


def _require_non_empty(name: str, values: Sequence[Any]) -> None:
    if not values:
        raise SimulatorRuleError(f"{name} cannot be empty")
