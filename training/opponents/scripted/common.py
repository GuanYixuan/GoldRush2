from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from simulator.constants import GRID_SIZE, MOVE_BUDGET
from simulator.types import Action, GameOutput, Position


CENTER = Position(8, 8)
ACTION_ORDER = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)
DR = (-1, 1, 0, 0)
DC = (0, 0, -1, 1)
OPPOSITE = (Action.DOWN, Action.UP, Action.RIGHT, Action.LEFT)


def stay_output() -> GameOutput:
    return GameOutput(actions=(int(Action.STAY),) * MOVE_BUDGET, k=3, order=0, vp=0)


def vp_from_policy(vp_policy: str, round_index: int) -> int:
    if vp_policy == "never":
        return 0
    if vp_policy == "always_7x7":
        return 1
    if vp_policy == "always_9x9":
        return 2
    if vp_policy == "periodic_9x9":
        return 2 if round_index % 5 == 0 else 0
    raise ValueError(f"unsupported vp_policy: {vp_policy!r}")


def validate_vp_policy(vp_policy: str) -> None:
    vp_from_policy(vp_policy, 1)


def reject_unknown_params(name: str, params: Mapping[str, Any], allowed: tuple[str, ...]) -> None:
    unknown = set(params) - set(allowed)
    if unknown:
        raise ValueError(f"unknown params for {name}: {sorted(unknown)}")


def visible_gold(game_input) -> tuple[tuple[Position, int], ...]:
    targets = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            amount = game_input.grid[row][col]
            if amount > 0:
                targets.append((Position(row, col), amount))
    return tuple(targets)


def manhattan(a: Position, b: Position) -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)
