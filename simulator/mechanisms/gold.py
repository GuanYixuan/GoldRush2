from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..constants import GRID_SIZE, REGION_COUNT, STATIC_EMPTY, STATIC_SPECIAL_NON_BLOCKING
from ..errors import SimulatorRuleError
from ..rules.snapshot import region_id
from ..state import GameState
from ..types import GoldGenerationEvent, Position
from .maps import MapTemplate

DEFAULT_CENTER_A = 0.0596
DEFAULT_CENTER_B = 0.06735
CENTER_MIN_ROW = 4
CENTER_MAX_ROW = 12
CENTER_MIN_COL = 4
CENTER_MAX_COL = 12
CENTER_POINT = Position(8, 8)
OUTER_REGIONS = (2, 3, 4, 5)

DEFAULT_STATIC2_GAP_WEIGHTS = (
    (8, 1781),
    (9, 1771),
    (10, 1759),
    (11, 1514),
    (12, 1980),
    (13, 1747),
    (14, 1970),
    (15, 1605),
    (16, 1723),
)
DEFAULT_STATIC2_TOTAL_WEIGHTS = (
    (50, 2),
    (51, 1),
    (52, 1),
    (53, 17),
    (54, 13),
    (55, 5),
    (56, 35),
    (57, 10),
    (58, 28),
    (59, 18),
    (60, 106),
    (61, 58),
    (62, 31),
    (63, 91),
    (64, 181),
    (65, 65),
    (66, 155),
    (67, 82),
    (68, 62),
    (69, 91),
    (70, 80),
    (71, 40),
    (72, 156),
    (73, 88),
    (74, 80),
    (75, 113),
    (76, 105),
    (77, 44),
    (78, 82),
    (79, 62),
    (80, 427),
    (81, 338),
    (82, 140),
    (83, 205),
    (84, 284),
    (85, 196),
    (86, 187),
    (87, 108),
    (88, 217),
    (89, 141),
    (90, 132),
    (91, 104),
    (92, 386),
    (93, 151),
    (94, 132),
    (95, 175),
    (96, 321),
    (97, 152),
    (98, 210),
    (99, 170),
    (100, 297),
    (101, 135),
    (102, 154),
    (103, 134),
    (104, 220),
    (105, 148),
    (106, 101),
    (107, 169),
    (108, 172),
    (109, 141),
    (110, 135),
    (111, 124),
    (112, 151),
)
DEFAULT_OUTER_STATIC0_COUNT_WEIGHTS = (
    (0, 27),
    (1, 335),
    (2, 1951),
    (3, 4805),
    (4, 4946),
    (5, 2737),
    (6, 1146),
    (7, 348),
    (8, 66),
)
DEFAULT_OUTER_STATIC0_AMOUNT_WEIGHTS = (
    (1, 3921),
    (2, 3605),
    (3, 3292),
    (4, 3009),
    (5, 3252),
    (6, 2704),
    (7, 2892),
    (8, 2425),
    (9, 2345),
    (10, 1922),
    (11, 2089),
    (12, 40),
    (13, 47),
    (14, 29),
    (15, 31),
    (16, 4),
    (17, 17),
    (18, 13),
)
DEFAULT_REGION_WEIGHTS = tuple((region, 1) for region in OUTER_REGIONS)
DEFAULT_FIRST_ROUND_OFFSET_WEIGHTS = tuple((offset, 1) for offset in range(8, 15))

WeightedIntDistribution = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CenterGoldConfig:
    center_a: float = DEFAULT_CENTER_A
    center_b: float = DEFAULT_CENTER_B
    amount_min: int = 1
    amount_max: int = 11

    def __post_init__(self) -> None:
        if not 0.0 <= self.center_a <= 1.0:
            raise SimulatorRuleError(f"center_a must be in [0, 1], got {self.center_a}")
        if self.center_b < 0.0:
            raise SimulatorRuleError(f"center_b must be non-negative, got {self.center_b}")
        if self.amount_min <= 0:
            raise SimulatorRuleError(f"amount_min must be positive, got {self.amount_min}")
        if self.amount_max < self.amount_min:
            raise SimulatorRuleError(f"amount_max must be >= amount_min, got {self.amount_max} < {self.amount_min}")


@dataclass(frozen=True)
class CenterGoldGenerator:
    config: CenterGoldConfig = CenterGoldConfig()

    def generate(self, state: GameState, template: MapTemplate, rng: random.Random) -> tuple[GoldGenerationEvent, ...]:
        events: list[GoldGenerationEvent] = []
        occupied = _occupied_positions(state)
        for pos in center_candidate_cells(template):
            if pos in state.bombs or pos in occupied:
                continue
            if rng.random() < center_rate(pos, self.config):
                events.append(GoldGenerationEvent(pos, rng.randint(self.config.amount_min, self.config.amount_max)))
        return tuple(events)


@dataclass
class OuterGoldState:
    next_static2_round: int
    next_static2_region: int

    def __post_init__(self) -> None:
        if self.next_static2_round < 0:
            raise SimulatorRuleError(f"next_static2_round must be non-negative, got {self.next_static2_round}")
        if self.next_static2_region not in OUTER_REGIONS:
            raise SimulatorRuleError(
                f"next_static2_region must be one of {OUTER_REGIONS}, got {self.next_static2_region}"
            )


@dataclass(frozen=True)
class OuterGoldConfig:
    first_round_offset_weights: WeightedIntDistribution = DEFAULT_FIRST_ROUND_OFFSET_WEIGHTS
    gap_weights: WeightedIntDistribution = DEFAULT_STATIC2_GAP_WEIGHTS
    region_weights: WeightedIntDistribution = DEFAULT_REGION_WEIGHTS
    static2_total_weights: WeightedIntDistribution = DEFAULT_STATIC2_TOTAL_WEIGHTS
    outer_static0_count_weights: WeightedIntDistribution = DEFAULT_OUTER_STATIC0_COUNT_WEIGHTS
    outer_static0_amount_weights: WeightedIntDistribution = DEFAULT_OUTER_STATIC0_AMOUNT_WEIGHTS

    def __post_init__(self) -> None:
        _validate_weights("first_round_offset_weights", self.first_round_offset_weights, min_value=0)
        _validate_weights("gap_weights", self.gap_weights, min_value=1)
        _validate_weights("region_weights", self.region_weights, allowed_values=OUTER_REGIONS)
        _validate_weights("static2_total_weights", self.static2_total_weights, min_value=1)
        _validate_weights("outer_static0_count_weights", self.outer_static0_count_weights, min_value=0)
        _validate_weights("outer_static0_amount_weights", self.outer_static0_amount_weights, min_value=1)


@dataclass(frozen=True)
class OuterGoldGenerator:
    config: OuterGoldConfig = field(default_factory=OuterGoldConfig)

    def initial_state(self, rng: random.Random, round_index: int = 0) -> OuterGoldState:
        if round_index < 0:
            raise SimulatorRuleError(f"round_index must be non-negative, got {round_index}")
        return OuterGoldState(
            next_static2_round=round_index + _sample_weighted(self.config.first_round_offset_weights, rng),
            next_static2_region=_sample_weighted(self.config.region_weights, rng),
        )

    def generate(
        self,
        state: GameState,
        template: MapTemplate,
        outer_state: OuterGoldState,
        rng: random.Random,
    ) -> tuple[GoldGenerationEvent, ...]:
        if state.round_index < outer_state.next_static2_round:
            return ()
        if state.round_index > outer_state.next_static2_round:
            raise SimulatorRuleError(
                f"outer gold state missed scheduled static2 round {outer_state.next_static2_round}; current round is {state.round_index}"
            )

        events = self._generate_static2_batch(state, template, outer_state.next_static2_region, rng)
        self._schedule_next(outer_state, state.round_index, rng)
        return events

    def _schedule_next(self, outer_state: OuterGoldState, round_index: int, rng: random.Random) -> None:
        next_round = round_index + _sample_weighted(self.config.gap_weights, rng)
        next_region = _sample_weighted(self.config.region_weights, rng)
        outer_state.next_static2_round = next_round
        outer_state.next_static2_region = next_region

    def _generate_static2_batch(
        self,
        state: GameState,
        template: MapTemplate,
        high_region: int,
        rng: random.Random,
    ) -> tuple[GoldGenerationEvent, ...]:
        static2_cells = _generation_available_cells(
            outer_static2_candidate_cells(template, high_region),
            state,
        )
        if not static2_cells:
            # TODO: 全部 static2 候选格均不可用时官方行为未观测；第一版跳过本次 static2 分配并推进 gap。
            return ()

        total = _sample_weighted(self.config.static2_total_weights, rng)
        events = _split_batch_total(static2_cells, total, rng)

        static0_cells = _generation_available_cells(
            outer_static0_candidate_cells(template, exclude_region=high_region),
            state,
        )
        static0_count = _sample_weighted(self.config.outer_static0_count_weights, rng)
        static0_count = min(static0_count, len(static0_cells))
        selected_static0 = tuple(rng.sample(static0_cells, static0_count))
        events.extend(
            GoldGenerationEvent(pos, _sample_weighted(self.config.outer_static0_amount_weights, rng)) for pos in selected_static0
        )
        return tuple(events)


def center_candidate_cells(template: MapTemplate) -> tuple[Position, ...]:
    return tuple(
        pos
        for row in range(CENTER_MIN_ROW, CENTER_MAX_ROW + 1)
        for col in range(CENTER_MIN_COL, CENTER_MAX_COL + 1)
        for pos in (Position(row, col),)
        if template.static_grid[row][col] == STATIC_EMPTY and pos not in template.obstacles
    )


def outer_static2_candidate_cells(template: MapTemplate, id_: int) -> tuple[Position, ...]:
    if id_ not in OUTER_REGIONS:
        raise SimulatorRuleError(f"outer static2 region must be one of {OUTER_REGIONS}, got {id_}")
    cells = tuple(
        pos
        for pos in sorted(template.special_cells)
        if region_id(pos) == id_ and template.static_grid[pos.row][pos.col] == STATIC_SPECIAL_NON_BLOCKING
    )
    if len(cells) != 5:
        raise SimulatorRuleError(f"map {template.map_id} region {id_} must have exactly 5 static2 cells, got {len(cells)}")
    return cells


def outer_static0_candidate_cells(template: MapTemplate, exclude_region: int | None = None) -> tuple[Position, ...]:
    if exclude_region is not None and not 1 <= exclude_region <= REGION_COUNT:
        raise SimulatorRuleError(f"invalid excluded region: {exclude_region}")
    cells: list[Position] = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            pos = Position(row, col)
            pos_region = region_id(pos)
            if (
                pos_region in OUTER_REGIONS
                and pos_region != exclude_region
                and template.static_grid[row][col] == STATIC_EMPTY
                and pos not in template.obstacles
            ):
                cells.append(pos)
    return tuple(cells)


def center_rate(position: Position, config: CenterGoldConfig = CenterGoldConfig()) -> float:
    if not (CENTER_MIN_ROW <= position.row <= CENTER_MAX_ROW and CENTER_MIN_COL <= position.col <= CENTER_MAX_COL):
        raise SimulatorRuleError(f"center rate requested for non-center position: {position}")
    sq_distance = (position.row - CENTER_POINT.row) ** 2 + (position.col - CENTER_POINT.col) ** 2
    return config.center_a * math.exp(-config.center_b * sq_distance)


def _generation_available_cells(cells: tuple[Position, ...], state: GameState) -> tuple[Position, ...]:
    occupied = _occupied_positions(state)
    return tuple(pos for pos in cells if pos not in state.bombs and pos not in occupied)


def _occupied_positions(state: GameState) -> frozenset[Position]:
    player_positions = (unit.position for _, unit in state.iter_player_units())
    npc_positions = (npc.position for npc in state.npcs.values())
    return frozenset((*player_positions, *npc_positions))


def _split_batch_total(cells: tuple[Position, ...], total: int, rng: random.Random) -> list[GoldGenerationEvent]:
    if not cells:
        raise SimulatorRuleError("cannot split static2 batch total across no cells")
    if total <= 0:
        raise SimulatorRuleError(f"static2 batch total must be positive, got {total}")
    base = total // len(cells)
    remainder = total % len(cells)
    amounts = {cell: base for cell in cells}
    for cell in rng.sample(cells, remainder):
        amounts[cell] += 1
    return [GoldGenerationEvent(cell, amounts[cell]) for cell in cells]


def _sample_weighted(weights: WeightedIntDistribution, rng: random.Random) -> int:
    total_weight = sum(weight for _, weight in weights)
    ticket = rng.randrange(total_weight)
    running = 0
    for value, weight in weights:
        running += weight
        if ticket < running:
            return value
    raise SimulatorRuleError("weighted sampler failed due to invalid weights")


def _validate_weights(
    name: str,
    weights: WeightedIntDistribution,
    *,
    min_value: int | None = None,
    allowed_values: tuple[int, ...] | None = None,
) -> None:
    if not weights:
        raise SimulatorRuleError(f"{name} cannot be empty")
    seen: set[int] = set()
    allowed_set = set(allowed_values) if allowed_values is not None else None
    for value, weight in weights:
        if value in seen:
            raise SimulatorRuleError(f"{name} has duplicate value: {value}")
        seen.add(value)
        if allowed_set is not None and value not in allowed_set:
            raise SimulatorRuleError(f"{name} value must be in {allowed_values}, got {value}")
        if min_value is not None and value < min_value:
            raise SimulatorRuleError(f"{name} value must be >= {min_value}, got {value}")
        if weight <= 0:
            raise SimulatorRuleError(f"{name} weight must be positive for value {value}, got {weight}")


__all__ = [
    "CENTER_POINT",
    "OUTER_REGIONS",
    "CenterGoldConfig",
    "CenterGoldGenerator",
    "OuterGoldConfig",
    "OuterGoldGenerator",
    "OuterGoldState",
    "center_candidate_cells",
    "center_rate",
    "outer_static0_candidate_cells",
    "outer_static2_candidate_cells",
]
