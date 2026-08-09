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
class TerminalWinMarginGoldGainReward:
    gamma: float = 0.9999
    beta_win: float = 1.0
    beta_margin: float = 0.2
    beta_gold_gain: float = 0.0
    beta_net_gold_gain: float = 0.1
    margin_scale: float = 500.0
    gold_gain_scale: float = 100.0
    net_gold_gain_scale: float = 50.0
    reward_schema: str = "terminal_win_margin_gold_gain_v1"
    _previous_phi: float | None = field(default=None, init=False, repr=False)
    _previous_gross_gold: int | None = field(default=None, init=False, repr=False)
    _previous_net_gold: int | None = field(default=None, init=False, repr=False)
    last_components: dict[str, float] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if self.margin_scale <= 0.0:
            raise ValueError(f"margin_scale must be positive, got {self.margin_scale}")
        if self.gold_gain_scale <= 0.0:
            raise ValueError(f"gold_gain_scale must be positive, got {self.gold_gain_scale}")
        if self.net_gold_gain_scale <= 0.0:
            raise ValueError(f"net_gold_gain_scale must be positive, got {self.net_gold_gain_scale}")

    def reset(self, state: GameState, *, agent_player_id: int) -> None:
        self._previous_phi = self._phi(state, agent_player_id=agent_player_id)
        self._previous_gross_gold = int(state.players[agent_player_id].gross_gold)
        self._previous_net_gold = int(state.players[agent_player_id].net_gold)
        self.last_components = None

    def __call__(self, result: RoundStepResult, *, agent_player_id: int) -> float:
        if self._previous_phi is None or self._previous_gross_gold is None or self._previous_net_gold is None:
            raise SimulatorRuleError("terminal win/margin/gold-gain/net-gold-gain reward requires reset() before first step")

        phi_next = 0.0 if result.terminated else self._phi(result.state, agent_player_id=agent_player_id)
        shaping = self.gamma * phi_next - self._previous_phi
        self._previous_phi = phi_next

        current_gross_gold = int(result.state.players[agent_player_id].gross_gold)
        gold_gain = max(0, current_gross_gold - self._previous_gross_gold)
        self._previous_gross_gold = current_gross_gold
        clipped_gold_gain = min(1.0, max(0.0, float(gold_gain) / self.gold_gain_scale))

        current_net_gold = int(result.state.players[agent_player_id].net_gold)
        net_gold_gain = current_net_gold - self._previous_net_gold
        self._previous_net_gold = current_net_gold
        clipped_net_gold_gain = min(1.0, max(-1.0, float(net_gold_gain) / self.net_gold_gain_scale))

        terminal = self._terminal_win_loss(result, agent_player_id=agent_player_id)
        terminal_reward = self.beta_win * terminal
        margin_reward = self.beta_margin * shaping
        gold_gain_reward = self.beta_gold_gain * clipped_gold_gain
        net_gold_gain_reward = self.beta_net_gold_gain * clipped_net_gold_gain
        total = terminal_reward + margin_reward + gold_gain_reward + net_gold_gain_reward
        self.last_components = {
            "terminal": terminal,
            "terminal_reward": terminal_reward,
            "margin_shaping": shaping,
            "margin_reward": margin_reward,
            "gold_gain": float(gold_gain),
            "clipped_gold_gain": clipped_gold_gain,
            "gold_gain_reward": gold_gain_reward,
            "net_gold_gain": float(net_gold_gain),
            "clipped_net_gold_gain": clipped_net_gold_gain,
            "net_gold_gain_reward": net_gold_gain_reward,
            "total": total,
        }
        return total

    def _phi(self, state: GameState, *, agent_player_id: int) -> float:
        opponent_player_id = _opponent_player_id(agent_player_id)
        margin = state.players[agent_player_id].net_gold - state.players[opponent_player_id].net_gold
        return math.tanh(float(margin) / self.margin_scale)

    def _terminal_win_loss(self, result: RoundStepResult, *, agent_player_id: int) -> float:
        if not result.terminated:
            return 0.0
        if result.game_result is None:
            raise SimulatorRuleError("terminal win/margin/gold-gain/net-gold-gain reward requires game_result")
        if result.game_result.winner_id == agent_player_id:
            return 1.0
        return -1.0


def _opponent_player_id(agent_player_id: int) -> int:
    if agent_player_id == 1:
        return 2
    if agent_player_id == 2:
        return 1
    raise SimulatorRuleError(f"agent_player_id must be 1 or 2, got {agent_player_id}")
