from __future__ import annotations

from dataclasses import dataclass

from ..constants import (
    GRID_BOMB,
    GRID_EMPTY,
    GRID_FOG,
    GRID_OBSTACLE,
    GRID_SIZE,
    MAX_NPCS,
)
from ..errors import SimulatorRuleError
from ..state import GameState
from ..types import Position, RegionStat, Snapshot

_POSITION_GRID = tuple(tuple(Position(row, col) for col in range(GRID_SIZE)) for row in range(GRID_SIZE))


@dataclass
class NpcInfo:
    id: int = 0
    row: int = -1
    col: int = -1


@dataclass
class GameInput:
    round: int
    grid: list[list[int]]
    my_units: list[tuple[int, int]]
    my_units_gold: tuple[int, int]
    gold_opp: int
    visible_enemies: list[tuple[int, int]]
    visible_npcs: list[NpcInfo]
    snapshot_valid: bool
    snapshot: Snapshot | None


def make_game_input(state: GameState, player_id: int, snapshot: Snapshot | None = None) -> GameInput:
    """Build a Python SDK-shaped GameInput from full simulator state."""
    if player_id not in state.players:
        raise ValueError(f"unknown player id: {player_id}")

    _validate_observable_layers(state)
    return _make_game_input_unchecked(state, player_id, snapshot)


def make_game_inputs(state: GameState, snapshot: Snapshot | None = None) -> dict[int, GameInput]:
    """Build both player observations while sharing invariant validation."""
    _validate_observable_layers(state)
    return {
        1: _make_game_input_unchecked(state, 1, snapshot),
        2: _make_game_input_unchecked(state, 2, snapshot),
    }


def _make_game_input_unchecked(state: GameState, player_id: int, snapshot: Snapshot | None = None) -> GameInput:
    visible = _visible_cells(state, player_id)
    opponent_id = _opponent_id(player_id)
    player = state.players[player_id]
    opponent = state.players[opponent_id]

    visible_enemy_positions = sorted(
        (unit.position for unit in opponent.units if unit.position in visible),
        key=lambda pos: (pos.row, pos.col),
    )

    return GameInput(
        round=state.round_index,
        grid=_make_grid(state, visible),
        my_units=[_position_tuple(unit.position) for unit in player.units],
        my_units_gold=(player.units[0].gold, player.units[1].gold),
        gold_opp=opponent.gross_gold,
        visible_enemies=[_position_tuple(pos) for pos in visible_enemy_positions] + [(-1, -1)] * (2 - len(visible_enemy_positions)),
        visible_npcs=[
            NpcInfo(id=npc.id, row=npc.position.row, col=npc.position.col)
            for npc in sorted(state.npcs.values(), key=lambda item: item.id)
            if npc.position in visible
        ][:MAX_NPCS],
        snapshot_valid=snapshot is not None,
        snapshot=snapshot,
    )


def _make_grid(state: GameState, visible: set[Position]) -> list[list[int]]:
    grid = [[GRID_FOG for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            pos = _POSITION_GRID[row][col]
            if pos not in visible:
                continue
            if pos in state.obstacles:
                grid[row][col] = GRID_OBSTACLE
            elif pos in state.bombs:
                grid[row][col] = GRID_BOMB
            else:
                grid[row][col] = state.gold.get(pos, GRID_EMPTY)
    return grid


def _visible_cells(state: GameState, player_id: int) -> set[Position]:
    visible: set[Position] = set()
    radius = state.players[player_id].active_vision_radius
    for unit in state.players[player_id].units:
        for row in range(unit.position.row - radius, unit.position.row + radius + 1):
            for col in range(unit.position.col - radius, unit.position.col + radius + 1):
                if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
                    visible.add(_POSITION_GRID[row][col])
    return visible


def _validate_observable_layers(state: GameState) -> None:
    for pos, amount in state.gold.items():
        if amount <= 0:
            raise SimulatorRuleError(f"gold amount must be positive at {pos}: {amount}")
        if pos in state.obstacles:
            raise SimulatorRuleError(f"gold overlaps obstacle at {pos}")
        if pos in state.bombs:
            raise SimulatorRuleError(f"gold overlaps bomb at {pos}")

    for pos in state.bombs:
        if pos in state.obstacles:
            raise SimulatorRuleError(f"bomb overlaps obstacle at {pos}")


def _opponent_id(player_id: int) -> int:
    if player_id == 1:
        return 2
    if player_id == 2:
        return 1
    raise ValueError(f"unknown player id: {player_id}")


def _position_tuple(position: Position) -> tuple[int, int]:
    return (position.row, position.col)


__all__ = ["GameInput", "NpcInfo", "RegionStat", "Snapshot", "make_game_input", "make_game_inputs"]
