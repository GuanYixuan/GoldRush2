from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from simulator.envs.round_step import RoundStepResult
from simulator.errors import SimulatorRuleError


class RewardFn(Protocol):
    def __call__(self, result: RoundStepResult, *, agent_player_id: int) -> float:
        ...


@dataclass(frozen=True)
class WinLossReward:
    win_reward: float = 1.0
    loss_reward: float = -1.0
    non_terminal_reward: float = 0.0

    def __call__(self, result: RoundStepResult, *, agent_player_id: int) -> float:
        if not result.terminated:
            return float(self.non_terminal_reward)
        if result.game_result is None:
            raise SimulatorRuleError("terminal win/loss reward requires game_result")
        if result.game_result.winner_id == agent_player_id:
            return float(self.win_reward)
        return float(self.loss_reward)
