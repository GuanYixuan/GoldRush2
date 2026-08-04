from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from simulator.envs.round_step import RoundStepResult
from simulator.errors import SimulatorRuleError
from simulator.state import GameState


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


@dataclass
class TerminalWinPlusMarginPotentialReward:
    gamma: float = 0.9999
    beta: float = 0.2
    margin_scale: float = 500.0
    win_reward: float = 1.0
    loss_reward: float = -1.0
    reward_schema: str = "terminal_win_plus_margin_potential_v1"
    _previous_phi: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if self.margin_scale <= 0.0:
            raise ValueError(f"margin_scale must be positive, got {self.margin_scale}")

    def reset(self, state: GameState, *, agent_player_id: int) -> None:
        self._previous_phi = self._phi(state, agent_player_id=agent_player_id)

    def __call__(self, result: RoundStepResult, *, agent_player_id: int) -> float:
        if self._previous_phi is None:
            raise SimulatorRuleError("terminal margin potential reward requires reset() before first step")

        phi_next = 0.0 if result.terminated else self._phi(result.state, agent_player_id=agent_player_id)
        shaping = self.gamma * phi_next - self._previous_phi
        self._previous_phi = phi_next

        return self._terminal_reward(result, agent_player_id=agent_player_id) + self.beta * shaping

    def _phi(self, state: GameState, *, agent_player_id: int) -> float:
        opponent_player_id = _opponent_player_id(agent_player_id)
        margin = state.players[agent_player_id].net_gold - state.players[opponent_player_id].net_gold
        return math.tanh(float(margin) / self.margin_scale)

    def _terminal_reward(self, result: RoundStepResult, *, agent_player_id: int) -> float:
        if not result.terminated:
            return 0.0
        if result.game_result is None:
            raise SimulatorRuleError("terminal margin potential reward requires game_result")
        if result.game_result.winner_id == agent_player_id:
            return float(self.win_reward)
        return float(self.loss_reward)


def _opponent_player_id(agent_player_id: int) -> int:
    if agent_player_id == 1:
        return 2
    if agent_player_id == 2:
        return 1
    raise SimulatorRuleError(f"agent_player_id must be 1 or 2, got {agent_player_id}")
