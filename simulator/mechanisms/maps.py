from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Sequence

from ..constants import GRID_SIZE, MAX_NPCS, STATIC_EMPTY, STATIC_OBSTACLE, STATIC_SPECIAL_NON_BLOCKING
from ..errors import SimulatorRuleError
from ..rules.snapshot import region_id
from ..state import GameState, NpcState, PlayerState, UnitState
from ..types import Position

StaticGrid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MapTemplate:
    map_id: int
    name: str
    static_grid: StaticGrid
    obstacles: frozenset[Position]
    special_cells: frozenset[Position]

    @staticmethod
    def from_static_grid(map_id: int, name: str, static_grid: Sequence[Sequence[int]]) -> MapTemplate:
        grid = _normalize_static_grid(static_grid)
        obstacles = frozenset(Position(row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE) if grid[row][col] == STATIC_OBSTACLE)
        special_cells = frozenset(
            Position(row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE) if grid[row][col] == STATIC_SPECIAL_NON_BLOCKING
        )
        return MapTemplate(map_id=map_id, name=name, static_grid=grid, obstacles=obstacles, special_cells=special_cells)

    def __post_init__(self) -> None:
        grid = _normalize_static_grid(self.static_grid)
        expected_obstacles = frozenset(
            Position(row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE) if grid[row][col] == STATIC_OBSTACLE
        )
        expected_special = frozenset(
            Position(row, col) for row in range(GRID_SIZE) for col in range(GRID_SIZE) if grid[row][col] == STATIC_SPECIAL_NON_BLOCKING
        )
        if self.obstacles != expected_obstacles:
            raise SimulatorRuleError(f"map {self.map_id} obstacle set does not match static_grid")
        if self.special_cells != expected_special:
            raise SimulatorRuleError(f"map {self.map_id} special cell set does not match static_grid")


@dataclass(frozen=True)
class SpawnConfig:
    p1_units: tuple[Position, Position] = (Position(0, 0), Position(16, 16))
    p2_units: tuple[Position, Position] = (Position(0, 16), Position(16, 0))
    npc_ids: tuple[int, ...] = tuple(range(-1, -MAX_NPCS - 1, -1))
    npc_position: Position = Position(8, 8)

    def __post_init__(self) -> None:
        player_positions = self.p1_units + self.p2_units
        if len(set(player_positions)) != len(player_positions):
            raise SimulatorRuleError("player spawn positions must be unique")
        for pos in player_positions + (self.npc_position,):
            if not pos.in_bounds():
                raise SimulatorRuleError(f"spawn position out of bounds: {pos}")
        if len(set(self.npc_ids)) != len(self.npc_ids):
            raise SimulatorRuleError(f"NPC ids must be unique: {self.npc_ids}")


@dataclass(frozen=True)
class MapPool:
    templates: tuple[MapTemplate, ...]

    def __post_init__(self) -> None:
        if not self.templates:
            raise SimulatorRuleError("map pool cannot be empty")
        ids = [template.map_id for template in self.templates]
        if len(set(ids)) != len(ids):
            raise SimulatorRuleError(f"map ids must be unique: {ids}")

    def sample(self, rng: random.Random) -> MapTemplate:
        return self.templates[rng.randrange(len(self.templates))]

    def get(self, map_id: int) -> MapTemplate:
        for template in self.templates:
            if template.map_id == map_id:
                return template
        raise KeyError(f"unknown map_id: {map_id}")


def build_initial_state(template: MapTemplate, spawn: SpawnConfig | None = None) -> GameState:
    spawn = SpawnConfig() if spawn is None else spawn
    _validate_spawn_against_map(template, spawn)
    return GameState(
        round_index=0,
        players={
            1: PlayerState(1, [UnitState(0, spawn.p1_units[0]), UnitState(1, spawn.p1_units[1])]),
            2: PlayerState(2, [UnitState(0, spawn.p2_units[0]), UnitState(1, spawn.p2_units[1])]),
        },
        obstacles=template.obstacles,
        npcs={npc_id: NpcState(npc_id, spawn.npc_position) for npc_id in spawn.npc_ids},
    )


def built_in_public_map_pool() -> MapPool:
    return MapPool(
        templates=(
            _template_from_cells(1, "official_map_1", _MAP1_OBSTACLES, _MAP1_SPECIAL),
            _template_from_cells(2, "official_map_2", _MAP2_OBSTACLES, _MAP2_SPECIAL),
            _template_from_cells(3, "official_map_3", _MAP3_OBSTACLES, _MAP3_SPECIAL),
            _template_from_static_rows(4, "official_map_4", _MAP4_STATIC_ROWS),
        )
    )


def built_in_training_map_pool() -> MapPool:
    public_templates = built_in_public_map_pool().templates
    pool = MapPool(
        templates=(
            *public_templates,
            _template_from_cells(101, "training_axis_cross_101", _MAP101_OBSTACLES, _MAP101_SPECIAL),
            _template_from_cells(111, "training_left_right_111", _MAP111_OBSTACLES, _MAP111_SPECIAL),
            _template_from_cells(121, "training_up_down_121", _MAP121_OBSTACLES, _MAP121_SPECIAL),
        )
    )
    for template in pool.templates:
        _validate_spawn_against_map(template, SpawnConfig())
        _validate_spawn_reaches_center(template, SpawnConfig())
        if template.map_id >= 100:
            validate_competition_training_map(template)
    return pool


def _validate_spawn_against_map(template: MapTemplate, spawn: SpawnConfig) -> None:
    for pos in spawn.p1_units + spawn.p2_units + (spawn.npc_position,):
        if pos in template.obstacles:
            raise SimulatorRuleError(f"spawn position overlaps obstacle on map {template.map_id}: {pos}")


def validate_competition_training_map(template: MapTemplate, spawn: SpawnConfig | None = None) -> None:
    spawn = SpawnConfig() if spawn is None else spawn
    _validate_spawn_against_map(template, spawn)
    _validate_obstacle_symmetry(template)
    _validate_outer_special_cell_counts(template)
    _validate_spawn_reaches_center(template, spawn)


def _validate_obstacle_symmetry(template: MapTemplate) -> None:
    obstacles = template.obstacles
    up_down = all(Position(GRID_SIZE - 1 - pos.row, pos.col) in obstacles for pos in obstacles)
    left_right = all(Position(pos.row, GRID_SIZE - 1 - pos.col) in obstacles for pos in obstacles)
    if not (up_down or left_right):
        raise SimulatorRuleError(
            f"map {template.map_id} obstacles must satisfy at least up-down or left-right symmetry"
        )


def _validate_outer_special_cell_counts(template: MapTemplate) -> None:
    counts = {region: 0 for region in (2, 3, 4, 5)}
    for pos in template.special_cells:
        region = region_id(pos)
        if region in counts:
            counts[region] += 1
    bad = {region: count for region, count in counts.items() if count != 5}
    if bad:
        raise SimulatorRuleError(f"map {template.map_id} must have exactly 5 special cells in each outer region: {bad}")


def _validate_spawn_reaches_center(template: MapTemplate, spawn: SpawnConfig) -> None:
    center_targets = {
        Position(row, col)
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
        if region_id(Position(row, col)) == 1 and Position(row, col) not in template.obstacles
    }
    if not center_targets:
        raise SimulatorRuleError(f"map {template.map_id} has no non-obstacle center cells")
    for start in spawn.p1_units + spawn.p2_units:
        if not _can_reach_any(template, start, center_targets):
            raise SimulatorRuleError(f"spawn {start} cannot reach center on map {template.map_id}")


def _can_reach_any(template: MapTemplate, start: Position, targets: set[Position]) -> bool:
    if start in template.obstacles:
        return False
    frontier: deque[Position] = deque([start])
    visited = {start}
    while frontier:
        pos = frontier.popleft()
        if pos in targets:
            return True
        for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            next_pos = Position(pos.row + delta_row, pos.col + delta_col)
            if not next_pos.in_bounds() or next_pos in visited or next_pos in template.obstacles:
                continue
            visited.add(next_pos)
            frontier.append(next_pos)
    return False


def _template_from_cells(
    map_id: int,
    name: str,
    obstacle_coords: tuple[tuple[int, int], ...],
    special_coords: tuple[tuple[int, int], ...],
) -> MapTemplate:
    overlap = set(obstacle_coords) & set(special_coords)
    if overlap:
        raise SimulatorRuleError(f"map {map_id} obstacle/special overlap: {sorted(overlap)}")

    grid = [[STATIC_EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for row, col in obstacle_coords:
        grid[row][col] = STATIC_OBSTACLE
    for row, col in special_coords:
        grid[row][col] = STATIC_SPECIAL_NON_BLOCKING
    return MapTemplate.from_static_grid(map_id, name, grid)


def _template_from_static_rows(map_id: int, name: str, rows: tuple[str, ...]) -> MapTemplate:
    return MapTemplate.from_static_grid(map_id, name, tuple(tuple(int(value) for value in row) for row in rows))


def _normalize_static_grid(static_grid: Sequence[Sequence[int]]) -> StaticGrid:
    if len(static_grid) != GRID_SIZE:
        raise SimulatorRuleError(f"static grid must have {GRID_SIZE} rows")
    rows = []
    for row_index, row in enumerate(static_grid):
        if len(row) != GRID_SIZE:
            raise SimulatorRuleError(f"static grid row {row_index} must have {GRID_SIZE} columns")
        values = tuple(int(value) for value in row)
        invalid = [value for value in values if value not in (STATIC_EMPTY, STATIC_OBSTACLE, STATIC_SPECIAL_NON_BLOCKING)]
        if invalid:
            raise SimulatorRuleError(f"static grid row {row_index} has invalid values: {invalid}")
        rows.append(values)
    return tuple(rows)


_MAP1_OBSTACLES = (
    (0, 3),
    (0, 13),
    (2, 1),
    (2, 2),
    (2, 14),
    (2, 15),
    (3, 0),
    (3, 3),
    (3, 13),
    (3, 16),
    (4, 4),
    (4, 12),
    (5, 5),
    (5, 11),
    (6, 3),
    (6, 13),
    (7, 7),
    (7, 9),
    (8, 4),
    (8, 6),
    (8, 10),
    (8, 12),
    (9, 7),
    (9, 9),
    (10, 3),
    (10, 13),
    (11, 5),
    (11, 11),
    (12, 4),
    (12, 12),
    (13, 0),
    (13, 3),
    (13, 13),
    (13, 16),
    (14, 1),
    (14, 2),
    (14, 14),
    (14, 15),
    (16, 3),
    (16, 13),
)

_MAP1_SPECIAL = (
    (0, 4),
    (0, 5),
    (0, 12),
    (1, 6),
    (1, 10),
    (5, 0),
    (5, 16),
    (7, 1),
    (7, 15),
    (9, 2),
    (9, 14),
    (11, 0),
    (11, 16),
    (12, 3),
    (12, 13),
    (15, 6),
    (15, 10),
    (16, 4),
    (16, 5),
    (16, 12),
)

_MAP2_OBSTACLES = (
    (2, 2),
    (2, 6),
    (2, 10),
    (2, 14),
    (4, 4),
    (4, 8),
    (4, 12),
    (6, 2),
    (6, 6),
    (6, 10),
    (6, 14),
    (8, 4),
    (8, 12),
    (10, 2),
    (10, 6),
    (10, 10),
    (10, 14),
    (12, 4),
    (12, 8),
    (12, 12),
    (14, 2),
    (14, 6),
    (14, 10),
    (14, 14),
)

_MAP2_SPECIAL = (
    (0, 4),
    (0, 8),
    (1, 6),
    (1, 12),
    (3, 10),
    (5, 0),
    (5, 16),
    (7, 2),
    (7, 14),
    (9, 1),
    (9, 15),
    (11, 3),
    (11, 13),
    (12, 0),
    (12, 16),
    (13, 10),
    (15, 6),
    (15, 12),
    (16, 4),
    (16, 8),
)

_MAP3_OBSTACLES = (
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 12),
    (2, 13),
    (2, 14),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 12),
    (3, 13),
    (3, 14),
    (4, 4),
    (4, 5),
    (4, 6),
    (4, 7),
    (4, 9),
    (4, 10),
    (4, 11),
    (4, 12),
    (5, 4),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 9),
    (5, 10),
    (5, 11),
    (5, 12),
    (6, 4),
    (6, 5),
    (6, 6),
    (6, 7),
    (6, 9),
    (6, 10),
    (6, 11),
    (6, 12),
    (8, 4),
    (8, 5),
    (8, 6),
    (8, 10),
    (8, 11),
    (8, 12),
    (10, 4),
    (10, 5),
    (10, 6),
    (10, 7),
    (10, 9),
    (10, 10),
    (10, 11),
    (10, 12),
    (11, 4),
    (11, 5),
    (11, 6),
    (11, 7),
    (11, 9),
    (11, 10),
    (11, 11),
    (11, 12),
    (12, 4),
    (12, 5),
    (12, 6),
    (12, 7),
    (12, 9),
    (12, 10),
    (12, 11),
    (12, 12),
    (13, 2),
    (13, 3),
    (13, 4),
    (13, 12),
    (13, 13),
    (13, 14),
    (14, 2),
    (14, 3),
    (14, 4),
    (14, 12),
    (14, 13),
    (14, 14),
)

_MAP3_SPECIAL = (
    (0, 6),
    (0, 7),
    (0, 8),
    (0, 9),
    (0, 10),
    (6, 0),
    (6, 16),
    (7, 0),
    (7, 16),
    (8, 0),
    (8, 16),
    (9, 0),
    (9, 16),
    (10, 0),
    (10, 16),
    (16, 6),
    (16, 7),
    (16, 8),
    (16, 9),
    (16, 10),
)


_MAP4_STATIC_ROWS = (
    "00100010201000100",
    "01101011011010110",
    "00021000000012000",
    "01111111011111110",
    "00000001010000000",
    "11111101010111111",
    "00000100000100000",
    "01110111011101110",
    "00010000000001000",
    "01010111011101010",
    "01010100000101010",
    "01010101010101010",
    "01000101010100010",
    "01110101110101110",
    "00000000000000000",
    "01110111011101110",
    "00120010201002100",
)


_MAP101_OBSTACLES = (
    (2, 2),
    (2, 6),
    (2, 10),
    (2, 14),
    (3, 3),
    (3, 6),
    (3, 10),
    (3, 13),
    (4, 6),
    (4, 10),
    (5, 5),
    (5, 6),
    (5, 7),
    (5, 9),
    (5, 10),
    (5, 11),
    (8, 0),
    (8, 1),
    (8, 2),
    (8, 14),
    (8, 15),
    (8, 16),
    (11, 5),
    (11, 6),
    (11, 7),
    (11, 9),
    (11, 10),
    (11, 11),
    (12, 6),
    (12, 10),
    (13, 3),
    (13, 6),
    (13, 10),
    (13, 13),
    (14, 2),
    (14, 6),
    (14, 10),
    (14, 14),
)

_MAP101_SPECIAL = (
    (1, 6),
    (1, 10),
    (2, 3),
    (2, 13),
    (3, 2),
    (3, 8),
    (3, 14),
    (7, 0),
    (7, 16),
    (8, 3),
    (8, 13),
    (9, 0),
    (9, 16),
    (13, 2),
    (13, 8),
    (13, 14),
    (14, 3),
    (14, 13),
    (15, 6),
    (15, 10),
)


_MAP111_OBSTACLES = (
    (0, 7),
    (0, 9),
    (2, 5),
    (2, 11),
    (3, 3),
    (3, 6),
    (3, 10),
    (3, 13),
    (4, 2),
    (4, 7),
    (4, 9),
    (4, 14),
    (6, 4),
    (6, 12),
    (7, 3),
    (7, 6),
    (7, 10),
    (7, 13),
    (8, 2),
    (8, 7),
    (8, 9),
    (8, 14),
    (10, 0),
    (10, 5),
    (10, 11),
    (10, 16),
    (11, 3),
    (11, 6),
    (11, 10),
    (11, 13),
    (12, 2),
    (12, 7),
    (12, 9),
    (12, 14),
    (15, 4),
    (15, 6),
    (15, 10),
    (15, 12),
    (16, 3),
    (16, 7),
    (16, 9),
    (16, 13),
)

_MAP111_SPECIAL = (
    (0, 8),
    (1, 2),
    (1, 14),
    (2, 6),
    (2, 10),
    (3, 8),
    (5, 3),
    (5, 13),
    (6, 1),
    (6, 15),
    (9, 3),
    (9, 13),
    (11, 0),
    (11, 16),
    (13, 3),
    (13, 8),
    (13, 13),
    (14, 5),
    (14, 11),
    (16, 8),
)


_MAP121_OBSTACLES = (
    (2, 4),
    (2, 9),
    (2, 12),
    (3, 4),
    (3, 9),
    (3, 12),
    (4, 2),
    (4, 3),
    (4, 4),
    (4, 7),
    (4, 8),
    (4, 9),
    (4, 12),
    (4, 13),
    (4, 14),
    (6, 0),
    (6, 6),
    (6, 10),
    (6, 16),
    (7, 2),
    (7, 3),
    (7, 4),
    (7, 12),
    (7, 13),
    (7, 14),
    (8, 4),
    (8, 12),
    (9, 2),
    (9, 3),
    (9, 4),
    (9, 12),
    (9, 13),
    (9, 14),
    (10, 0),
    (10, 6),
    (10, 10),
    (10, 16),
    (12, 2),
    (12, 3),
    (12, 4),
    (12, 7),
    (12, 8),
    (12, 9),
    (12, 12),
    (12, 13),
    (12, 14),
    (13, 4),
    (13, 9),
    (13, 12),
    (14, 4),
    (14, 9),
    (14, 12),
)

_MAP121_SPECIAL = (
    (2, 5),
    (2, 8),
    (2, 11),
    (3, 3),
    (3, 8),
    (3, 13),
    (5, 2),
    (5, 14),
    (8, 2),
    (8, 3),
    (8, 13),
    (8, 14),
    (11, 2),
    (11, 14),
    (13, 3),
    (13, 8),
    (13, 13),
    (14, 5),
    (14, 8),
    (14, 11),
)


__all__ = [
    "MapPool",
    "MapTemplate",
    "SpawnConfig",
    "build_initial_state",
    "built_in_public_map_pool",
    "built_in_training_map_pool",
    "validate_competition_training_map",
]
