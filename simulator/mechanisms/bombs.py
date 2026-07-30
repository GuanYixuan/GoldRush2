from __future__ import annotations

import random
from dataclasses import dataclass

from ..constants import BOMB_REFRESH_PERIOD, GRID_SIZE
from ..errors import SimulatorRuleError
from ..state import GameState
from ..types import Position

DEFAULT_BOMB_SPAWN_PROBABILITY = 0.0795


@dataclass(frozen=True)
class BombConfig:
    spawn_probability: float = DEFAULT_BOMB_SPAWN_PROBABILITY

    def __post_init__(self) -> None:
        if not 0.0 <= self.spawn_probability <= 1.0:
            raise SimulatorRuleError(f"bomb spawn_probability must be in [0, 1], got {self.spawn_probability}")


@dataclass(frozen=True)
class BombRefreshEvent:
    cleared: frozenset[Position]
    spawned: frozenset[Position]


@dataclass(frozen=True)
class BernoulliBombRefresher:
    config: BombConfig = BombConfig()

    def refresh(self, state: GameState, rng: random.Random) -> BombRefreshEvent:
        _validate_existing_bombs(state)
        if state.round_index % BOMB_REFRESH_PERIOD != 0:
            return BombRefreshEvent(cleared=frozenset(), spawned=frozenset())

        cleared = frozenset(state.bombs)
        state.bombs.clear()
        spawned = frozenset(pos for pos in _candidate_positions(state) if rng.random() < self.config.spawn_probability)
        state.bombs.update(spawned)
        return BombRefreshEvent(cleared=cleared, spawned=spawned)


def _candidate_positions(state: GameState) -> tuple[Position, ...]:
    occupied = _occupied_positions(state)
    return tuple(
        pos
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
        for pos in (Position(row, col),)
        if pos not in state.obstacles and pos not in state.gold and pos not in occupied
    )


def _occupied_positions(state: GameState) -> frozenset[Position]:
    player_positions = (unit.position for _, unit in state.iter_player_units())
    npc_positions = (npc.position for npc in state.npcs.values())
    return frozenset((*player_positions, *npc_positions))


def _validate_existing_bombs(state: GameState) -> None:
    occupied = _occupied_positions(state)
    for pos in state.bombs:
        if not pos.in_bounds():
            raise SimulatorRuleError(f"existing bomb out of bounds: {pos}")
        if pos in state.obstacles:
            raise SimulatorRuleError(f"existing bomb overlaps obstacle at {pos}")
        if pos in state.gold:
            raise SimulatorRuleError(f"existing bomb overlaps gold at {pos}")
        npc_count = sum(1 for npc in state.npcs.values() if npc.position == pos)
        if npc_count >= 3:
            raise SimulatorRuleError(f"existing bomb overlaps {npc_count} NPCs at {pos}")
        if pos in occupied:
            raise SimulatorRuleError(f"existing bomb overlaps actor at {pos}")


__all__ = ["BernoulliBombRefresher", "BombConfig", "BombRefreshEvent", "DEFAULT_BOMB_SPAWN_PROBABILITY"]
