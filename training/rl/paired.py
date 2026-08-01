from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from simulator.mechanisms.maps import MapPool, SpawnConfig
from simulator.observation.sdk import GameInput
from simulator.types import GameOutput
from simulator.envs.round_step import RoundStepMechanisms
from training.opponents import OpponentSpec

from .env import SingleAgentEnvConfig, SingleAgentGoldRushEnv
from .rewards import RewardFn
from .types import ResetResult, StepResult


AgentPolicy = Callable[[GameInput], GameOutput]


@dataclass(frozen=True)
class EpisodeRollout:
    agent_player_id: int
    reset: ResetResult
    steps: tuple[StepResult, ...]

    @property
    def total_reward(self) -> float:
        return sum(step.reward for step in self.steps)

    @property
    def terminal_step(self) -> StepResult:
        if not self.steps or not self.steps[-1].terminated:
            raise ValueError("episode rollout is not terminal")
        return self.steps[-1]


@dataclass(frozen=True)
class PairedEpisodeResult:
    first: EpisodeRollout
    second: EpisodeRollout

    @property
    def pair_score(self) -> float:
        return (self.first.total_reward + self.second.total_reward) / 2.0


class PairedEpisodeSampler:
    """Run the same setting twice with swapped P1/P2 agent identity."""

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
        self.mechanisms = mechanisms
        self.map_pool = map_pool
        self.spawn = spawn
        self.reward_fn = reward_fn

    def rollout_pair(
        self,
        agent_policy: AgentPolicy,
        *,
        seed: int | None = None,
        map_id: int | None = None,
        opponent_spec: OpponentSpec | None = None,
    ) -> PairedEpisodeResult:
        first = self._rollout_one(agent_policy, agent_player_id=1, seed=seed, map_id=map_id, opponent_spec=opponent_spec)
        shared_spec = first.reset.info["opponent_spec"] if opponent_spec is None else opponent_spec
        second = self._rollout_one(agent_policy, agent_player_id=2, seed=seed, map_id=map_id, opponent_spec=shared_spec)
        return PairedEpisodeResult(first=first, second=second)

    def _rollout_one(
        self,
        agent_policy: AgentPolicy,
        *,
        agent_player_id: int,
        seed: int | None,
        map_id: int | None,
        opponent_spec: OpponentSpec | None,
    ) -> EpisodeRollout:
        env = SingleAgentGoldRushEnv(
            config=self.config,
            mechanisms=self.mechanisms,
            map_pool=self.map_pool,
            spawn=self.spawn,
            reward_fn=self.reward_fn,
        )
        reset = env.reset(seed=seed, map_id=map_id, agent_player_id=agent_player_id, opponent_spec=opponent_spec)
        steps: list[StepResult] = []
        observation: GameInput | None = reset.observation
        while observation is not None:
            step = env.step(agent_policy(observation))
            steps.append(step)
            observation = step.observation
        return EpisodeRollout(agent_player_id=agent_player_id, reset=reset, steps=tuple(steps))
