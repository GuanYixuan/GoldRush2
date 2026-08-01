from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Literal

from simulator.errors import SimulatorRuleError
from simulator.observation.sdk import GameInput
from simulator.types import GameOutput
from training.opponents import OpponentSpec


PairRole = Literal["first", "second"]


@dataclass(frozen=True)
class Transition:
    observation: GameInput
    action: GameOutput
    reward: float
    next_observation: GameInput | None
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class Trajectory:
    episode_id: str
    pair_id: str
    pair_role: PairRole
    seed: int | None
    map_id: int | None
    agent_player_id: int
    opponent_spec: OpponentSpec
    transitions: tuple[Transition, ...]

    @property
    def total_reward(self) -> float:
        return sum(transition.reward for transition in self.transitions)

    @property
    def terminal_info(self) -> dict[str, Any]:
        if not self.transitions or not self.transitions[-1].terminated:
            raise SimulatorRuleError(f"trajectory {self.episode_id!r} is not terminal")
        return self.transitions[-1].info


@dataclass(frozen=True)
class EpisodeBatch:
    trajectories: tuple[Trajectory, ...]

    @property
    def episode_count(self) -> int:
        return len(self.trajectories)

    @property
    def transition_count(self) -> int:
        return sum(len(trajectory.transitions) for trajectory in self.trajectories)

    def metrics(self) -> dict[str, Any]:
        if not self.trajectories:
            return {
                "episode_count": 0,
                "transition_count": 0,
                "mean_total_reward": 0.0,
                "win_rate": 0.0,
                "loss_rate": 0.0,
                "mean_episode_steps": 0.0,
                "mean_pair_score": 0.0,
                "mean_agent_net_gold": 0.0,
                "mean_opponent_net_gold": 0.0,
            }

        total_rewards = [trajectory.total_reward for trajectory in self.trajectories]
        wins = [_agent_won(trajectory) for trajectory in self.trajectories]
        agent_net_gold = [_terminal_net_gold(trajectory, trajectory.agent_player_id) for trajectory in self.trajectories]
        opponent_net_gold = [_terminal_net_gold(trajectory, _opponent_player_id(trajectory)) for trajectory in self.trajectories]
        return {
            "episode_count": self.episode_count,
            "transition_count": self.transition_count,
            "mean_total_reward": mean(total_rewards),
            "win_rate": mean(1.0 if won else 0.0 for won in wins),
            "loss_rate": mean(0.0 if won else 1.0 for won in wins),
            "mean_episode_steps": mean(len(trajectory.transitions) for trajectory in self.trajectories),
            "mean_pair_score": mean(_pair_scores(self.trajectories)),
            "mean_agent_net_gold": mean(agent_net_gold),
            "mean_opponent_net_gold": mean(opponent_net_gold),
        }


def _agent_won(trajectory: Trajectory) -> bool:
    game_result = trajectory.terminal_info["game_result"]
    return game_result.winner_id == trajectory.agent_player_id


def _terminal_net_gold(trajectory: Trajectory, player_id: int) -> int:
    return trajectory.terminal_info["scores"]["net_gold"][player_id]


def _opponent_player_id(trajectory: Trajectory) -> int:
    if trajectory.agent_player_id == 1:
        return 2
    if trajectory.agent_player_id == 2:
        return 1
    raise SimulatorRuleError(f"agent_player_id must be 1 or 2, got {trajectory.agent_player_id}")


def _pair_scores(trajectories: tuple[Trajectory, ...]) -> tuple[float, ...]:
    rewards_by_pair: dict[str, list[float]] = {}
    for trajectory in trajectories:
        rewards_by_pair.setdefault(trajectory.pair_id, []).append(trajectory.total_reward)
    return tuple(mean(rewards) for rewards in rewards_by_pair.values())
