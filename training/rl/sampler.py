from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from simulator.envs.round_step import RoundStepMechanisms
from simulator.errors import SimulatorRuleError
from simulator.mechanisms.maps import MapPool, SpawnConfig
from simulator.observation.sdk import GameInput
from simulator.types import GameOutput
from training.opponents import OpponentSpec

from .env import SingleAgentEnvConfig, SingleAgentGoldRushEnv
from .rewards import RewardFn
from .rollout import EpisodeBatch, PairRole, Trajectory, Transition


AgentPolicy = Callable[[GameInput], GameOutput | Sequence[int]]


@dataclass
class BatchRolloutSampler:
    env_config: SingleAgentEnvConfig
    mechanisms: RoundStepMechanisms | None = None
    map_pool: MapPool | None = None
    spawn: SpawnConfig | None = None
    reward_fn: RewardFn | None = None

    def collect(
        self,
        agent_policy: AgentPolicy,
        *,
        pair_count: int,
        seed: int,
        map_ids: Sequence[int] | None = None,
        opponent_specs: Sequence[OpponentSpec] | None = None,
    ) -> EpisodeBatch:
        if pair_count <= 0:
            raise SimulatorRuleError(f"pair_count must be positive, got {pair_count}")
        if map_ids is not None and not map_ids:
            raise SimulatorRuleError("map_ids cannot be empty when provided")
        if opponent_specs is not None and not opponent_specs:
            raise SimulatorRuleError("opponent_specs cannot be empty when provided")

        trajectories: list[Trajectory] = []
        for pair_index in range(pair_count):
            episode_seed = seed + pair_index
            requested_map_id = map_ids[pair_index % len(map_ids)] if map_ids is not None else None
            requested_opponent_spec = opponent_specs[pair_index % len(opponent_specs)] if opponent_specs is not None else None
            pair_id = f"pair-{pair_index:06d}-seed-{episode_seed}"

            first = self._rollout_one(
                agent_policy,
                pair_id=pair_id,
                pair_role="first",
                seed=episode_seed,
                map_id=requested_map_id,
                agent_player_id=1,
                opponent_spec=requested_opponent_spec,
            )
            shared_opponent_spec = first.opponent_spec if requested_opponent_spec is None else requested_opponent_spec
            second = self._rollout_one(
                agent_policy,
                pair_id=pair_id,
                pair_role="second",
                seed=episode_seed,
                map_id=first.map_id,
                agent_player_id=2,
                opponent_spec=shared_opponent_spec,
            )
            trajectories.extend((first, second))
        return EpisodeBatch(trajectories=tuple(trajectories))

    def _rollout_one(
        self,
        agent_policy: AgentPolicy,
        *,
        pair_id: str,
        pair_role: PairRole,
        seed: int,
        map_id: int | None,
        agent_player_id: int,
        opponent_spec: OpponentSpec | None,
    ) -> Trajectory:
        env = SingleAgentGoldRushEnv(
            config=self.env_config,
            mechanisms=self.mechanisms,
            map_pool=self.map_pool,
            spawn=self.spawn,
            reward_fn=self.reward_fn,
        )
        reset = env.reset(seed=seed, map_id=map_id, agent_player_id=agent_player_id, opponent_spec=opponent_spec)
        transitions: list[Transition] = []
        observation: GameInput | None = reset.observation
        while observation is not None:
            action = _coerce_game_output(agent_policy(observation))
            step = env.step(action)
            transitions.append(
                Transition(
                    observation=observation,
                    action=action,
                    reward=step.reward,
                    next_observation=step.observation,
                    terminated=step.terminated,
                    truncated=step.truncated,
                    info=step.info,
                )
            )
            observation = step.observation

        return Trajectory(
            episode_id=f"{pair_id}-{pair_role}",
            pair_id=pair_id,
            pair_role=pair_role,
            seed=seed,
            map_id=reset.info["map_id"],
            agent_player_id=agent_player_id,
            opponent_spec=reset.info["opponent_spec"],
            transitions=tuple(transitions),
        )


def _coerce_game_output(raw_output: GameOutput | Sequence[int]) -> GameOutput:
    if isinstance(raw_output, GameOutput):
        return raw_output
    values = tuple(int(value) for value in raw_output)
    if len(values) != 9:
        raise ValueError(f"policy sequence output must have 9 integers, got {len(values)}")
    return GameOutput(actions=values[:6], k=values[6], order=values[7], vp=values[8])
