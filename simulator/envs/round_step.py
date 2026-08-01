from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..config import EpisodeConfig
from ..errors import SimulatorRuleError
from ..mechanisms.bombs import BernoulliBombRefresher, BombRefreshEvent
from ..mechanisms.gold import CenterGoldGenerator, OuterGoldGenerator, OuterGoldState
from ..mechanisms.maps import MapPool, MapTemplate, SpawnConfig, build_initial_state, built_in_public_map_pool
from ..mechanisms.npc import M4aNpcPolicy, NpcEpisodeProfile
from ..observation.sdk import GameInput, make_game_input
from ..replay import SimulatorReplayRecorder
from ..rules.scoring import GameResult, determine_winner
from ..rules.snapshot import SnapshotAccumulator
from ..rules.transition import TransitionResult, apply_gold_generation, transition_started_round
from ..state import GameState
from ..types import Action, GameOutput, GoldGenerationEvent, Snapshot


@dataclass(frozen=True)
class RoundStepConfig:
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)


@dataclass
class RoundStepMechanisms:
    center_gold: CenterGoldGenerator = field(default_factory=CenterGoldGenerator)
    outer_gold: OuterGoldGenerator = field(default_factory=OuterGoldGenerator)
    bomb_refresher: BernoulliBombRefresher = field(default_factory=BernoulliBombRefresher)
    npc_policy: M4aNpcPolicy = field(default_factory=M4aNpcPolicy)


@dataclass(frozen=True)
class RoundStepTrace:
    round_index: int
    first_player_id: int
    player_outputs: dict[int, GameOutput]
    npc_order: tuple[int, ...]
    npc_actions: dict[int, tuple[Action, ...]]
    gold_generated: tuple[GoldGenerationEvent, ...]
    bomb_refresh_event: BombRefreshEvent
    transition_result: TransitionResult


@dataclass(frozen=True)
class RoundStepResult:
    observations: dict[int, GameInput]
    terminated: bool
    trace: RoundStepTrace
    state: GameState
    game_result: GameResult | None = None
    replay: dict[str, Any] | None = None


class RoundStepEnv:
    """Round-level simulator adapter.

    The caller owns policy execution, first-player selection, reward, and any
    training semantics. This class only prepares action-time observations and
    advances the simulator by complete rounds.
    """

    def __init__(
        self,
        *,
        config: RoundStepConfig | None = None,
        mechanisms: RoundStepMechanisms | None = None,
        map_pool: MapPool | None = None,
        spawn: SpawnConfig | None = None,
        record_replay: bool = False,
        player_names: dict[int, str] | None = None,
        p90_latency_ns: dict[int, int] | None = None,
    ) -> None:
        self.config = RoundStepConfig() if config is None else config
        self.mechanisms = RoundStepMechanisms() if mechanisms is None else mechanisms
        self.map_pool = map_pool or built_in_public_map_pool()
        self.spawn = spawn
        self.record_replay = record_replay
        self.player_names = player_names
        self.p90_latency_ns = p90_latency_ns

        self.rng = random.Random()
        self.template: MapTemplate | None = None
        self.state: GameState | None = None
        self.snapshot_accumulator: SnapshotAccumulator | None = None
        self.visible_snapshot: Snapshot | None = None
        self.outer_state: OuterGoldState | None = None
        self.npc_profile: NpcEpisodeProfile | None = None
        self.recorder: SimulatorReplayRecorder | None = None
        self.pending_gold_generated: tuple[GoldGenerationEvent, ...] = ()
        self.pending_bomb_refresh_event: BombRefreshEvent | None = None
        self.terminated = False

    def reset(self, *, seed: int | None = None, map_id: int | None = None) -> dict[int, GameInput]:
        episode = self.config.episode
        if seed is not None:
            episode = replace(episode, seed=seed)
        if map_id is not None:
            episode = replace(episode, map_id=map_id)
        self.config = replace(self.config, episode=episode)

        self.rng = random.Random(episode.seed)
        self.template = _select_map(self.map_pool, episode.map_id, self.rng)
        self.state = build_initial_state(self.template, self.spawn)
        self.snapshot_accumulator = SnapshotAccumulator(episode.rules)
        self.visible_snapshot = None
        self.outer_state = self.mechanisms.outer_gold.initial_state(self.rng, self.state.round_index)
        self.npc_profile = self.mechanisms.npc_policy.sample_profile(self.state, self.rng)
        self.recorder = (
            SimulatorReplayRecorder(
                map_template=self.template,
                seed=episode.seed,
                players=self.player_names,
                mechanisms=_mechanism_metadata(episode, self.mechanisms),
            )
            if self.record_replay
            else None
        )
        self.terminated = False
        self._prepare_round()
        return self._observations()

    def step(
        self,
        player_outputs: dict[int, GameOutput | Sequence[int]],
        *,
        first_player_id: int,
    ) -> RoundStepResult:
        self._require_ready()
        if self.terminated:
            raise SimulatorRuleError("cannot step a terminated round-step environment; call reset() first")
        if first_player_id not in (1, 2):
            raise SimulatorRuleError(f"first_player_id must be 1 or 2, got {first_player_id}")
        outputs = _coerce_player_outputs(player_outputs)

        assert self.state is not None
        assert self.template is not None
        assert self.snapshot_accumulator is not None
        assert self.npc_profile is not None
        assert self.pending_bomb_refresh_event is not None
        round_start_state = copy.deepcopy(self.state)
        npc_order = self.mechanisms.npc_policy.sample_order(self.state, self.rng)
        npc_actions = self.mechanisms.npc_policy.decide_all(copy.deepcopy(self.state), self.template, npc_order, self.rng, self.npc_profile)

        transition_result = transition_started_round(
            self.state,
            player_outputs=outputs,
            gold_generated=self.pending_gold_generated,
            first_player_id=first_player_id,
            npc_order=npc_order,
            npc_actions=npc_actions,
            rules=self.config.episode.rules,
            snapshot_accumulator=self.snapshot_accumulator,
        )
        self.visible_snapshot = transition_result.snapshot
        trace = RoundStepTrace(
            round_index=round_start_state.round_index,
            first_player_id=first_player_id,
            player_outputs=outputs,
            npc_order=npc_order,
            npc_actions=npc_actions,
            gold_generated=self.pending_gold_generated,
            bomb_refresh_event=self.pending_bomb_refresh_event,
            transition_result=transition_result,
        )

        if self.recorder is not None:
            self.recorder.record_round(
                start_state=round_start_state,
                end_state=self.state,
                player_outputs=outputs,
                transition_result=transition_result,
                npc_actions=npc_actions,
                npc_order=npc_order,
                bomb_refresh_event=self.pending_bomb_refresh_event,
            )

        self.terminated = self.state.round_index >= self.config.episode.rules.round_count
        game_result = _determine_result_if_possible(self.state, self.p90_latency_ns) if self.terminated else None
        observations: dict[int, GameInput] = {}
        if not self.terminated:
            self._prepare_round()
            observations = self._observations()

        return RoundStepResult(
            observations=observations,
            terminated=self.terminated,
            trace=trace,
            state=self.state,
            game_result=game_result,
            replay=copy.deepcopy(self.recorder.to_json()) if self.recorder is not None else None,
        )

    def _prepare_round(self) -> None:
        assert self.state is not None
        assert self.template is not None
        assert self.outer_state is not None
        self.pending_bomb_refresh_event = self.mechanisms.bomb_refresher.refresh(self.state, self.rng)
        self.pending_gold_generated = _generate_gold(self.state, self.template, self.outer_state, self.mechanisms, self.rng)
        apply_gold_generation(self.state, self.pending_gold_generated)

    def _observations(self) -> dict[int, GameInput]:
        assert self.state is not None
        return {
            1: make_game_input(self.state, 1, self.visible_snapshot),
            2: make_game_input(self.state, 2, self.visible_snapshot),
        }

    def _require_ready(self) -> None:
        if self.state is None:
            raise SimulatorRuleError("round-step environment is not reset")


def _select_map(map_pool: MapPool, map_id: int | None, rng: random.Random) -> MapTemplate:
    if map_id is None:
        return map_pool.sample(rng)
    return map_pool.get(map_id)


def _generate_gold(
    state: GameState,
    template: MapTemplate,
    outer_state: OuterGoldState,
    mechanisms: RoundStepMechanisms,
    rng: random.Random,
) -> tuple[GoldGenerationEvent, ...]:
    center_events = mechanisms.center_gold.generate(state, template, rng)
    temp_state = copy.deepcopy(state)
    apply_gold_generation(temp_state, center_events)
    outer_events = mechanisms.outer_gold.generate(temp_state, template, outer_state, rng)
    return center_events + outer_events


def _coerce_player_outputs(player_outputs: dict[int, GameOutput | Sequence[int]]) -> dict[int, GameOutput]:
    if set(player_outputs) != {1, 2}:
        raise SimulatorRuleError(f"player_outputs must contain player ids 1 and 2, got {sorted(player_outputs)}")
    return {player_id: _coerce_game_output(output) for player_id, output in player_outputs.items()}


def _coerce_game_output(raw_output: GameOutput | Sequence[int]) -> GameOutput:
    if isinstance(raw_output, GameOutput):
        return raw_output
    values = tuple(int(value) for value in raw_output)
    if len(values) != 9:
        raise ValueError(f"policy sequence output must have 9 integers, got {len(values)}")
    return GameOutput(actions=values[:6], k=values[6], order=values[7], vp=values[8])


def _determine_result_if_possible(state: GameState, p90_latency_ns: dict[int, int] | None) -> GameResult | None:
    if state.players[1].net_gold == state.players[2].net_gold and p90_latency_ns is None:
        return None
    return determine_winner(state, p90_latency_ns)


def _mechanism_metadata(episode: EpisodeConfig, mechanisms: RoundStepMechanisms) -> dict[str, Any]:
    return {
        "profile": episode.mechanism_profile,
        "center_gold": type(mechanisms.center_gold).__name__,
        "outer_gold": type(mechanisms.outer_gold).__name__,
        "bombs": type(mechanisms.bomb_refresher).__name__,
        "npc": type(mechanisms.npc_policy).__name__,
    }


__all__ = [
    "RoundStepConfig",
    "RoundStepEnv",
    "RoundStepMechanisms",
    "RoundStepResult",
    "RoundStepTrace",
]
