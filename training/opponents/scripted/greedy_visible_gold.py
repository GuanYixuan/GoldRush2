from __future__ import annotations

import random
from dataclasses import dataclass, field

from simulator.constants import GRID_BOMB, GRID_FOG, GRID_OBSTACLE, MOVE_BUDGET
from simulator.observation.sdk import GameInput
from simulator.types import Action, GameOutput, Position

from ..base import EpisodeContext
from .common import ACTION_ORDER, validate_vp_policy, visible_gold, vp_from_policy


@dataclass
class GreedyVisibleGoldOpponent:
    target_score: str = "value_per_step"
    avoid_bombs: bool = True
    risk_weight: float = 1.0
    vp_policy: str = "never"
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.target_score not in ("nearest", "highest", "value_per_step"):
            raise ValueError(f"unsupported target_score: {self.target_score!r}")
        if self.risk_weight < 0:
            raise ValueError(f"risk_weight must be >= 0, got {self.risk_weight}")
        validate_vp_policy(self.vp_policy)

    def reset(self, seed: int, episode_context: EpisodeContext) -> None:
        self._rng.seed(seed)

    def act(self, game_input: GameInput) -> GameOutput:
        candidates = []
        for unit_id, unit_pos_raw in enumerate(game_input.my_units):
            start = Position(*unit_pos_raw)
            for target, amount in visible_gold(game_input):
                if target == start:
                    continue
                path = _shortest_visible_path(game_input, start, target, self.avoid_bombs)
                if path is None or len(path) > MOVE_BUDGET:
                    continue
                score = self._score_target(amount, len(path), start, path, game_input)
                candidates.append((score, -len(path), amount, unit_id, path))

        if not candidates:
            return GameOutput(actions=(int(Action.STAY),) * MOVE_BUDGET, k=3, order=0, vp=vp_from_policy(self.vp_policy, game_input.round))

        candidates.sort(reverse=True)
        _score, _neg_dist, _amount, unit_id, path = candidates[0]
        actions = tuple(int(action) for action in path[:MOVE_BUDGET])
        actions = actions + (int(Action.STAY),) * (MOVE_BUDGET - len(actions))
        if unit_id == 0:
            return GameOutput(actions=actions, k=len(path), order=0, vp=vp_from_policy(self.vp_policy, game_input.round))
        return GameOutput(actions=actions, k=0, order=1, vp=vp_from_policy(self.vp_policy, game_input.round))

    def __call__(self, game_input: GameInput) -> GameOutput:
        return self.act(game_input)

    def _score_target(self, amount: int, distance: int, start: Position, path: tuple[Action, ...], game_input: GameInput) -> float:
        distance = max(1, distance)
        if self.target_score == "nearest":
            base = -distance + amount * 0.01
        elif self.target_score == "highest":
            base = amount - distance * 0.1
        else:
            base = amount / distance
        bomb_steps = sum(1 for pos in _trace_positions(start, path) if game_input.grid[pos.row][pos.col] == GRID_BOMB)
        return base - self.risk_weight * bomb_steps


def _shortest_visible_path(game_input: GameInput, start: Position, target: Position, avoid_bombs: bool) -> tuple[Action, ...] | None:
    queue: list[tuple[Position, tuple[Action, ...]]] = [(start, ())]
    seen = {start}
    for pos, path in queue:
        if pos == target:
            return path
        if len(path) >= MOVE_BUDGET:
            continue
        for action in ACTION_ORDER:
            next_pos = pos.moved(action)
            if next_pos in seen or not next_pos.in_bounds():
                continue
            cell = game_input.grid[next_pos.row][next_pos.col]
            if cell in (GRID_FOG, GRID_OBSTACLE):
                continue
            if avoid_bombs and cell == GRID_BOMB:
                continue
            seen.add(next_pos)
            queue.append((next_pos, path + (action,)))
    return None


def _trace_positions(start: Position, path: tuple[Action, ...]) -> tuple[Position, ...]:
    pos = start
    positions = []
    for action in path:
        pos = pos.moved(action)
        positions.append(pos)
    return tuple(positions)
