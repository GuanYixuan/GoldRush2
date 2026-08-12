from __future__ import annotations

import copy
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from ..config import EpisodeConfig
from ..errors import SimulatorRuleError
from ..mechanisms.bombs import BernoulliBombRefresher, BombRefreshEvent
from ..mechanisms.gold import CenterGoldGenerator, OuterGoldGenerator, OuterGoldState
from ..mechanisms.maps import MapPool, MapTemplate, SpawnConfig, build_initial_state, built_in_training_map_pool
from ..mechanisms.npc import M4aNpcPolicy
from ..observation.sdk import GameInput, make_game_input
from ..replay import SimulatorReplayRecorder
from ..randomness import make_simulator_rng_streams
from ..rules.scoring import GameResult, determine_winner
from ..rules.snapshot import SnapshotAccumulator
from ..rules.transition import TransitionResult, apply_gold_generation, transition_started_round
from ..state import GameState
from ..types import Action, GameOutput, GoldGenerationEvent, Snapshot


class PlayerPolicy(Protocol):
    def __call__(self, game_input: GameInput) -> GameOutput | Sequence[int]:
        ...


class DuelOrderMode(str, Enum):
    AGENT_AFTER_OPPONENT = "agent_after_opponent"
    AGENT_BEFORE_OPPONENT = "agent_before_opponent"
    RANDOM_ORDER = "random_order"
    COST_BASED = "cost_based"


@dataclass(frozen=True)
class DuelConfig:
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    order_mode: DuelOrderMode | str = DuelOrderMode.RANDOM_ORDER
    agent_player_id: int = 1

    def __post_init__(self) -> None:
        if self.agent_player_id not in (1, 2):
            raise SimulatorRuleError(f"agent_player_id must be 1 or 2, got {self.agent_player_id}")
        DuelOrderMode(self.order_mode)


@dataclass
class DuelMechanisms:
    center_gold: CenterGoldGenerator = field(default_factory=CenterGoldGenerator)
    outer_gold: OuterGoldGenerator = field(default_factory=OuterGoldGenerator)
    bomb_refresher: BernoulliBombRefresher = field(default_factory=BernoulliBombRefresher)
    npc_policy: M4aNpcPolicy = field(default_factory=M4aNpcPolicy)


@dataclass(frozen=True)
class RoundTrace:
    round_index: int
    first_player_id: int
    player_outputs: dict[int, GameOutput]
    policy_latency_ns: dict[int, int]
    npc_order: tuple[int, ...]
    npc_actions: dict[int, tuple[Action, ...]]
    gold_generated: tuple[GoldGenerationEvent, ...]
    bomb_refresh_event: BombRefreshEvent
    transition_result: TransitionResult


@dataclass(frozen=True)
class DuelRunResult:
    state: GameState
    game_result: GameResult
    map_template: MapTemplate
    rounds: tuple[RoundTrace, ...]
    replay: dict[str, Any] | None
    p90_latency_ns: dict[int, int]


def run_duel(
    player1: PlayerPolicy | Any,
    player2: PlayerPolicy | Any,
    *,
    config: DuelConfig | None = None,
    mechanisms: DuelMechanisms | None = None,
    map_pool: MapPool | None = None,
    spawn: SpawnConfig | None = None,
    record_replay: bool = False,
    player_names: dict[int, str] | None = None,
    p90_latency_ns: dict[int, int] | None = None,
) -> DuelRunResult:
    config = DuelConfig() if config is None else config
    mechanisms = DuelMechanisms() if mechanisms is None else mechanisms
    rng_streams = make_simulator_rng_streams(config.episode.seed)
    rng = rng_streams.environment
    template = _select_map(map_pool or built_in_training_map_pool(), config.episode.map_id, rng)
    state = build_initial_state(template, spawn)
    snapshot_accumulator = SnapshotAccumulator(config.episode.rules)
    visible_snapshot: Snapshot | None = None
    outer_state = mechanisms.outer_gold.initial_state(rng_streams.outer_gold, state.round_index)
    npc_profile = mechanisms.npc_policy.sample_profile(state, rng_streams.npc)
    recorder = (
        SimulatorReplayRecorder(
            map_template=template,
            seed=config.episode.seed,
            players=player_names,
            mechanisms=_mechanism_metadata(config, mechanisms),
        )
        if record_replay
        else None
    )

    policies = {1: player1, 2: player2}
    latencies: dict[int, list[int]] = {1: [], 2: []}
    traces: list[RoundTrace] = []

    for _ in range(config.episode.rules.round_count):
        bomb_event = mechanisms.bomb_refresher.refresh(state, rng_streams.bomb)
        gold_generated = _generate_gold(
            state,
            template,
            outer_state,
            mechanisms,
            rng_streams.center_gold,
            rng_streams.outer_gold,
        )
        apply_gold_generation(state, gold_generated)

        round_index = state.round_index
        round_start_state = copy.deepcopy(state) if recorder is not None else None
        player_outputs, policy_latency = _collect_player_outputs(policies, state, visible_snapshot)
        for player_id, latency_ns in policy_latency.items():
            latencies[player_id].append(latency_ns)
        first_player_id = _first_player_id(config, policy_latency, rng)
        npc_order = mechanisms.npc_policy.sample_order(state, rng_streams.npc)
        decision_state = state if isinstance(mechanisms.npc_policy, M4aNpcPolicy) else copy.deepcopy(state)
        npc_actions = mechanisms.npc_policy.decide_all(decision_state, template, npc_order, rng_streams.npc, npc_profile)

        transition_result = transition_started_round(
            state,
            player_outputs=player_outputs,
            gold_generated=gold_generated,
            first_player_id=first_player_id,
            npc_order=npc_order,
            npc_actions=npc_actions,
            rules=config.episode.rules,
            snapshot_accumulator=snapshot_accumulator,
        )
        visible_snapshot = transition_result.snapshot

        trace = RoundTrace(
            round_index=round_index,
            first_player_id=first_player_id,
            player_outputs=player_outputs,
            policy_latency_ns=policy_latency,
            npc_order=npc_order,
            npc_actions=npc_actions,
            gold_generated=gold_generated,
            bomb_refresh_event=bomb_event,
            transition_result=transition_result,
        )
        traces.append(trace)

        if recorder is not None:
            assert round_start_state is not None
            recorder.record_round(
                start_state=round_start_state,
                end_state=state,
                player_outputs=player_outputs,
                transition_result=transition_result,
                npc_actions=npc_actions,
                npc_order=npc_order,
                bomb_refresh_event=bomb_event,
            )

    final_p90 = dict(p90_latency_ns) if p90_latency_ns is not None else _p90_latency_ns(latencies)
    game_result = determine_winner(state, final_p90)
    return DuelRunResult(
        state=state,
        game_result=game_result,
        map_template=template,
        rounds=tuple(traces),
        replay=recorder.to_json() if recorder is not None else None,
        p90_latency_ns=final_p90,
    )


def _select_map(map_pool: MapPool, map_id: int | None, rng: random.Random) -> MapTemplate:
    if map_id is None:
        return map_pool.sample(rng)
    return map_pool.get(map_id)


def _generate_gold(
    state: GameState,
    template: MapTemplate,
    outer_state: OuterGoldState,
    mechanisms: DuelMechanisms,
    center_gold_rng: random.Random,
    outer_gold_rng: random.Random,
) -> tuple[GoldGenerationEvent, ...]:
    center_events = mechanisms.center_gold.generate(state, template, center_gold_rng)
    _validate_gold_generation_events(state, center_events)
    outer_events = mechanisms.outer_gold.generate(
        state,
        template,
        outer_state,
        outer_gold_rng,
    )
    return center_events + outer_events


def _validate_gold_generation_events(state: GameState, events: tuple[GoldGenerationEvent, ...]) -> None:
    for event in events:
        if event.amount <= 0:
            raise SimulatorRuleError(f"gold generation amount must be positive: {event}")
        if not event.position.in_bounds():
            raise SimulatorRuleError(f"gold generation position out of bounds: {event.position}")
        if event.position in state.obstacles:
            raise SimulatorRuleError(f"gold generation overlaps obstacle at {event.position}")
        if event.position in state.bombs:
            raise SimulatorRuleError(f"gold generation overlaps bomb at {event.position}")


def _collect_player_outputs(
    policies: dict[int, PlayerPolicy | Any],
    state: GameState,
    snapshot: Snapshot | None,
) -> tuple[dict[int, GameOutput], dict[int, int]]:
    outputs: dict[int, GameOutput] = {}
    latencies: dict[int, int] = {}
    for player_id in (1, 2):
        game_input = make_game_input(state, player_id, snapshot)
        started = time.perf_counter_ns()
        raw_output = _call_policy(policies[player_id], game_input)
        latencies[player_id] = time.perf_counter_ns() - started
        outputs[player_id] = _coerce_game_output(raw_output)
    return outputs, latencies


def _call_policy(policy: PlayerPolicy | Any, game_input: GameInput) -> GameOutput | Sequence[int]:
    if callable(policy):
        return policy(game_input)
    move_decision = getattr(policy, "move_decision", None)
    if move_decision is not None:
        return move_decision(game_input)
    raise TypeError(f"policy must be callable or expose move_decision(), got {type(policy).__name__}")


def _coerce_game_output(raw_output: GameOutput | Sequence[int]) -> GameOutput:
    if isinstance(raw_output, GameOutput):
        return raw_output
    values = tuple(int(value) for value in raw_output)
    if len(values) != 9:
        raise ValueError(f"policy sequence output must have 9 integers, got {len(values)}")
    return GameOutput(actions=values[:6], k=values[6], order=values[7], vp=values[8])


def _first_player_id(config: DuelConfig, policy_latency_ns: dict[int, int], rng: random.Random) -> int:
    mode = DuelOrderMode(config.order_mode)
    if mode == DuelOrderMode.AGENT_BEFORE_OPPONENT:
        return config.agent_player_id
    if mode == DuelOrderMode.AGENT_AFTER_OPPONENT:
        return 1 if config.agent_player_id == 2 else 2
    if mode == DuelOrderMode.RANDOM_ORDER:
        return 1 if rng.randrange(2) == 0 else 2
    if mode == DuelOrderMode.COST_BASED:
        if policy_latency_ns[1] <= policy_latency_ns[2]:
            return 1
        return 2
    raise SimulatorRuleError(f"unsupported duel order mode: {mode}")


def _p90_latency_ns(latencies: dict[int, list[int]]) -> dict[int, int]:
    return {player_id: _percentile_ceil(samples, 0.9) for player_id, samples in latencies.items()}


def _percentile_ceil(samples: list[int], q: float) -> int:
    if not samples:
        raise SimulatorRuleError("cannot compute P90 latency without samples")
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * q + 0.999999999) - 1)))
    return ordered[index]


def _mechanism_metadata(config: DuelConfig, mechanisms: DuelMechanisms) -> dict[str, Any]:
    return {
        "profile": config.episode.mechanism_profile,
        "center_gold": type(mechanisms.center_gold).__name__,
        "outer_gold": type(mechanisms.outer_gold).__name__,
        "bombs": type(mechanisms.bomb_refresher).__name__,
        "npc": type(mechanisms.npc_policy).__name__,
    }


__all__ = [
    "DuelConfig",
    "DuelMechanisms",
    "DuelOrderMode",
    "DuelRunResult",
    "PlayerPolicy",
    "RoundTrace",
    "run_duel",
]
