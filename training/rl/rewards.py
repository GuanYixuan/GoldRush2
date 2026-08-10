from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from simulator.constants import DEFAULT_VISION_RADIUS, GRID_SIZE
from simulator.envs.round_step import RoundStepResult
from simulator.errors import SimulatorRuleError
from simulator.state import GameState
from simulator.types import Position


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
    gamma: float = 0.97
    beta_win: float = 1.0
    beta_margin: float = 0.2
    beta_gold_gain: float = 0.0
    beta_net_gold_gain: float = 0.1
    margin_scale: float = 500.0
    gold_gain_scale: float = 100.0
    net_gold_gain_scale: float = 50.0
    beta_vision_info: float = 0.0
    vision_info_scale: float = 100.0
    vision_info_reward_cap: float = 0.03
    vision_info_recent_window: int = 5
    reward_schema: str = "terminal_win_margin_gold_gain_v1"
    _previous_phi: float | None = field(default=None, init=False, repr=False)
    _previous_gross_gold: int | None = field(default=None, init=False, repr=False)
    _previous_net_gold: int | None = field(default=None, init=False, repr=False)
    _last_seen_round: list[list[int]] = field(default_factory=list, init=False, repr=False)
    _last_gold_increase_round: list[list[int]] = field(default_factory=list, init=False, repr=False)
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
        if self.beta_vision_info < 0.0:
            raise ValueError(f"beta_vision_info must be non-negative, got {self.beta_vision_info}")
        if self.vision_info_scale <= 0.0:
            raise ValueError(f"vision_info_scale must be positive, got {self.vision_info_scale}")
        if self.vision_info_reward_cap < 0.0:
            raise ValueError(f"vision_info_reward_cap must be non-negative, got {self.vision_info_reward_cap}")
        if self.vision_info_recent_window < 0:
            raise ValueError(f"vision_info_recent_window must be non-negative, got {self.vision_info_recent_window}")

    def reset(self, state: GameState, *, agent_player_id: int) -> None:
        self._previous_phi = self._phi(state, agent_player_id=agent_player_id)
        self._previous_gross_gold = int(state.players[agent_player_id].gross_gold)
        self._previous_net_gold = int(state.players[agent_player_id].net_gold)
        self._last_seen_round = _new_round_grid()
        self._last_gold_increase_round = _new_round_grid()
        self._mark_seen(state, agent_player_id=agent_player_id)
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
        vision_info = self._vision_info(result, agent_player_id=agent_player_id)
        vision_info_reward = min(
            self.vision_info_reward_cap,
            self.beta_vision_info * min(1.0, vision_info["gold"] / self.vision_info_scale),
        )
        total = terminal_reward + margin_reward + gold_gain_reward + net_gold_gain_reward + vision_info_reward
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
            "vision_info_gold": vision_info["gold"],
            "vision_info_cells": vision_info["cells"],
            "vision_info_reward": vision_info_reward,
            "total": total,
        }
        return total

    def _vision_info(self, result: RoundStepResult, *, agent_player_id: int) -> dict[str, float]:
        self._mark_gold_increases(result)
        if result.terminated:
            return {"gold": 0.0, "cells": 0.0}
        if self.beta_vision_info == 0.0:
            self._mark_seen(result.state, agent_player_id=agent_player_id)
            return {"gold": 0.0, "cells": 0.0}
        if agent_player_id not in result.observations:
            raise SimulatorRuleError(f"missing next observation for agent player {agent_player_id}")
        if result.trace is None:
            raise SimulatorRuleError("vision info reward requires round-step trace")
        try:
            agent_vp = int(result.trace.player_outputs[agent_player_id].vp)
        except KeyError as exc:
            raise SimulatorRuleError(f"missing agent output for player {agent_player_id}") from exc

        observation = result.observations[agent_player_id]
        fresh_gold = 0
        fresh_cells = 0
        if agent_vp > 0:
            actual_cells = _visible_cells(result.state, player_id=agent_player_id, radius=None)
            base_cells = _visible_cells(result.state, player_id=agent_player_id, radius=DEFAULT_VISION_RADIUS)
            round_index = int(result.state.round_index)
            for pos in actual_cells - base_cells:
                amount = int(observation.grid[pos.row][pos.col])
                if amount <= 0:
                    continue
                last_seen = self._last_seen_round[pos.row][pos.col]
                last_gold_increase = self._last_gold_increase_round[pos.row][pos.col]
                if round_index - last_seen > self.vision_info_recent_window or last_seen < last_gold_increase:
                    fresh_gold += amount
                    fresh_cells += 1

        self._mark_seen(result.state, agent_player_id=agent_player_id)
        return {"gold": float(fresh_gold), "cells": float(fresh_cells)}

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

    def _mark_gold_increases(self, result: RoundStepResult) -> None:
        if not self._last_gold_increase_round:
            raise SimulatorRuleError("vision info reward requires reset() before first step")
        round_index = int(result.state.round_index)
        for event in result.next_round_gold_generated:
            self._last_gold_increase_round[event.position.row][event.position.col] = round_index

    def _mark_seen(self, state: GameState, *, agent_player_id: int) -> None:
        if not self._last_seen_round:
            raise SimulatorRuleError("vision info reward requires initialized seen grid")
        round_index = int(state.round_index)
        for pos in _visible_cells(state, player_id=agent_player_id, radius=None):
            self._last_seen_round[pos.row][pos.col] = round_index


def _opponent_player_id(agent_player_id: int) -> int:
    if agent_player_id == 1:
        return 2
    if agent_player_id == 2:
        return 1
    raise SimulatorRuleError(f"agent_player_id must be 1 or 2, got {agent_player_id}")


def _new_round_grid() -> list[list[int]]:
    return [[-1_000_000 for _col in range(GRID_SIZE)] for _row in range(GRID_SIZE)]


def _visible_cells(state: GameState, *, player_id: int, radius: int | None) -> set[Position]:
    player = state.players[player_id]
    vision_radius = int(player.active_vision_radius if radius is None else radius)
    cells: set[Position] = set()
    for unit in player.units:
        for row in range(unit.position.row - vision_radius, unit.position.row + vision_radius + 1):
            for col in range(unit.position.col - vision_radius, unit.position.col + vision_radius + 1):
                if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                    cells.add(Position(row, col))
    return cells
