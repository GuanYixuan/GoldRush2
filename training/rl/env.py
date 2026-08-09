from __future__ import annotations

import copy
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from simulator.config import EpisodeConfig
from simulator.envs.round_step import RoundStepConfig, RoundStepEnv, RoundStepMechanisms, RoundStepResult, RoundStepTrace
from simulator.errors import SimulatorRuleError
from simulator.mechanisms.maps import MapPool, SpawnConfig
from simulator.observation.sdk import GameInput
from simulator.types import GameOutput
from training.opponents import EpisodeContext, OpponentLeague, OpponentRunner, OpponentSpec, build_runner
from training.opponents.league import LeagueSplit

from .rewards import RewardFn, WinLossReward
from .types import ResetResult, StepResult


@dataclass(frozen=True)
class SingleAgentEnvConfig:
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    agent_player_id: int = 1
    opponent_spec: OpponentSpec | None = field(default_factory=lambda: OpponentSpec(kind="python", name="fast_probe_v3_like"))
    opponent_league: OpponentLeague | None = None
    league_split: LeagueSplit = "train"
    record_replay: bool = False

    def __post_init__(self) -> None:
        if self.agent_player_id not in (1, 2):
            raise SimulatorRuleError(f"agent_player_id must be 1 or 2, got {self.agent_player_id}")


class SingleAgentGoldRushEnv:
    """Single-agent training adapter around RoundStepEnv.

    The agent always acts after the opponent in this first version. Both sides
    still receive observations from the same action-time state, matching the
    platform semantics.
    """

    def __init__(
        self,
        *,
        config: SingleAgentEnvConfig | None = None,
        mechanisms: RoundStepMechanisms | None = None,
        map_pool: MapPool | None = None,
        spawn: SpawnConfig | None = None,
        reward_fn: RewardFn | None = None,
    ) -> None:
        self.config = SingleAgentEnvConfig() if config is None else config
        self.mechanisms = RoundStepMechanisms() if mechanisms is None else mechanisms
        self.map_pool = map_pool
        self.spawn = spawn
        self.reward_fn = WinLossReward() if reward_fn is None else reward_fn

        self.agent_player_id = self.config.agent_player_id
        self.opponent_player_id = _other_player(self.agent_player_id)
        self.opponent_spec: OpponentSpec | None = None
        self.opponent_runner: OpponentRunner | None = None
        self.round_env: RoundStepEnv | None = None
        self.observations: dict[int, GameInput] = {}
        self.terminated = True

    def reset(
        self,
        *,
        seed: int | None = None,
        map_id: int | None = None,
        agent_player_id: int | None = None,
        opponent_spec: OpponentSpec | None = None,
    ) -> ResetResult:
        self.agent_player_id = self.config.agent_player_id if agent_player_id is None else int(agent_player_id)
        if self.agent_player_id not in (1, 2):
            raise SimulatorRuleError(f"agent_player_id must be 1 or 2, got {self.agent_player_id}")
        self.opponent_player_id = _other_player(self.agent_player_id)

        episode_seed = self.config.episode.seed if seed is None else seed
        episode_map_id = self.config.episode.map_id if map_id is None else map_id
        sampled_spec = opponent_spec or self._sample_opponent_spec(episode_seed)
        self.opponent_spec = sampled_spec
        self.opponent_runner = build_runner(sampled_spec)
        self.opponent_runner.reset(
            _opponent_seed(episode_seed),
            EpisodeContext(
                player_id=self.opponent_player_id,
                opponent_id=self.agent_player_id,
                map_id=episode_map_id,
                seed=episode_seed,
                tags=("rl_env",),
            ),
        )

        self.round_env = RoundStepEnv(
            config=RoundStepConfig(self.config.episode),
            mechanisms=copy.deepcopy(self.mechanisms),
            map_pool=self.map_pool,
            spawn=self.spawn,
            record_replay=self.config.record_replay,
            player_names={self.agent_player_id: "agent", self.opponent_player_id: "opponent"},
            p90_latency_ns=_agent_slow_p90(self.agent_player_id),
        )
        self.observations = self.round_env.reset(seed=episode_seed, map_id=episode_map_id)
        reset_reward = getattr(self.reward_fn, "reset", None)
        if callable(reset_reward):
            assert self.round_env.state is not None
            reset_reward(self.round_env.state, agent_player_id=self.agent_player_id)
        self.terminated = False
        return ResetResult(observation=self.observations[self.agent_player_id], info=self._reset_info(episode_seed))

    def step(self, agent_output: GameOutput | Sequence[int]) -> StepResult:
        if self.round_env is None or self.opponent_runner is None or self.opponent_spec is None:
            raise SimulatorRuleError("single-agent RL environment is not reset")
        if self.terminated:
            raise SimulatorRuleError("cannot step a terminated single-agent RL environment; call reset() first")

        opponent_output = self.opponent_runner.act(self.observations[self.opponent_player_id])
        result = self.round_env.step(
            {
                self.agent_player_id: agent_output,
                self.opponent_player_id: opponent_output,
            },
            first_player_id=self.opponent_player_id,
        )
        reward = self.reward_fn(result, agent_player_id=self.agent_player_id)
        reward_components = getattr(self.reward_fn, "last_components", None)
        self.terminated = result.terminated
        self.observations = result.observations
        next_observation = None if result.terminated else result.observations[self.agent_player_id]
        return StepResult(
            observation=next_observation,
            reward=reward,
            terminated=result.terminated,
            truncated=False,
            info=_step_info(
                result,
                agent_player_id=self.agent_player_id,
                opponent_player_id=self.opponent_player_id,
                agent_output=agent_output,
                opponent_output=opponent_output,
                opponent_spec=self.opponent_spec,
                reward_components=reward_components,
            ),
        )

    def _sample_opponent_spec(self, seed: int | None) -> OpponentSpec:
        if self.config.opponent_league is not None:
            return self.config.opponent_league.sample_spec(random.Random(seed), self.config.league_split)
        if self.config.opponent_spec is None:
            raise SimulatorRuleError("reset requires opponent_spec when env config has no opponent_spec or opponent_league")
        return self.config.opponent_spec

    def _reset_info(self, seed: int | None) -> dict[str, Any]:
        assert self.round_env is not None
        assert self.round_env.template is not None
        assert self.opponent_spec is not None
        return {
            "seed": seed,
            "map_id": self.round_env.template.map_id,
            "map_name": self.round_env.template.name,
            "agent_player_id": self.agent_player_id,
            "opponent_player_id": self.opponent_player_id,
            "opponent_spec": self.opponent_spec,
            "order_mode": "agent_after_opponent",
            "p90_latency_ns": _agent_slow_p90(self.agent_player_id),
        }


def _step_info(
    result: RoundStepResult,
    *,
    agent_player_id: int,
    opponent_player_id: int,
    agent_output: GameOutput | Sequence[int],
    opponent_output: GameOutput,
    opponent_spec: OpponentSpec,
    reward_components: dict[str, float] | None,
) -> dict[str, Any]:
    return {
        "round_index": result.trace.round_index,
        "agent_player_id": agent_player_id,
        "opponent_player_id": opponent_player_id,
        "first_player_id": result.trace.first_player_id,
        "agent_output": agent_output,
        "opponent_output": opponent_output,
        "opponent_spec": opponent_spec,
        "scores": _scores(result),
        "events": _event_counts(result.trace),
        "reward_components": reward_components,
        "trace": result.trace,
        "game_result": result.game_result,
        "replay": result.replay,
    }


def _scores(result: RoundStepResult) -> dict[str, dict[int, int]]:
    return {
        "gross_gold": {player_id: result.state.players[player_id].gross_gold for player_id in (1, 2)},
        "net_gold": {player_id: result.state.players[player_id].net_gold for player_id in (1, 2)},
        "vision_spent": {player_id: result.state.players[player_id].vision_spent for player_id in (1, 2)},
    }


def _event_counts(trace: RoundStepTrace) -> dict[str, dict[int, int]]:
    counts = {
        "pickups": {1: 0, 2: 0},
        "pickup_gold": {1: 0, 2: 0},
        "bomb_triggers": {1: 0, 2: 0},
        "bomb_lost_gold": {1: 0, 2: 0},
        "tramples": {1: 0, 2: 0},
        "trample_penalty": {1: 0, 2: 0},
    }
    for interaction_events in trace.transition_result.interaction_events:
        for pickup in interaction_events.pickups:
            if pickup.actor.player_id in (1, 2):
                counts["pickups"][pickup.actor.player_id] += 1
                counts["pickup_gold"][pickup.actor.player_id] += int(pickup.picked_gold)
        for trigger in interaction_events.bomb_triggers:
            if trigger.actor.player_id in (1, 2):
                counts["bomb_triggers"][trigger.actor.player_id] += 1
                counts["bomb_lost_gold"][trigger.actor.player_id] += int(trigger.lost_gold)
        for trample in interaction_events.tramples:
            if trample.actor.player_id in (1, 2):
                counts["tramples"][trample.actor.player_id] += 1
                counts["trample_penalty"][trample.actor.player_id] += int(trample.penalty)
    return counts


def _other_player(player_id: int) -> int:
    if player_id == 1:
        return 2
    if player_id == 2:
        return 1
    raise SimulatorRuleError(f"player_id must be 1 or 2, got {player_id}")


def _agent_slow_p90(agent_player_id: int) -> dict[int, int]:
    opponent_player_id = _other_player(agent_player_id)
    return {agent_player_id: 2, opponent_player_id: 1}


def _opponent_seed(seed: int | None) -> int:
    base = 0 if seed is None else int(seed)
    return base * 2 + 1
