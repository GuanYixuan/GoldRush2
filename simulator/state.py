from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .constants import PLAYER_UNIT_COUNT
from .types import Position


@dataclass
class UnitState:
    id: int
    position: Position
    gold: int = 0


@dataclass
class PlayerState:
    id: int
    units: list[UnitState]
    vision_spent: int = 0
    active_vision_radius: int = 2
    next_vision_radius: int = 2

    def __post_init__(self) -> None:
        if len(self.units) != PLAYER_UNIT_COUNT:
            raise ValueError(f"player {self.id} must have {PLAYER_UNIT_COUNT} units")
        expected_ids = list(range(PLAYER_UNIT_COUNT))
        actual_ids = [unit.id for unit in self.units]
        if actual_ids != expected_ids:
            raise ValueError(f"player {self.id} unit ids must be {expected_ids}, got {actual_ids}")

    @property
    def gross_gold(self) -> int:
        return sum(unit.gold for unit in self.units)

    @property
    def net_gold(self) -> int:
        return self.gross_gold - self.vision_spent


@dataclass
class NpcState:
    id: int
    position: Position


@dataclass
class GameState:
    round_index: int
    players: dict[int, PlayerState]
    obstacles: frozenset[Position] = frozenset()
    gold: dict[Position, int] = field(default_factory=dict)
    bombs: set[Position] = field(default_factory=set)
    npcs: dict[int, NpcState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.players) != {1, 2}:
            raise ValueError(f"players must contain ids {{1, 2}}, got {sorted(self.players)}")

    def player_unit(self, player_id: int, unit_id: int) -> UnitState:
        try:
            return self.players[player_id].units[unit_id]
        except KeyError as exc:
            raise ValueError(f"unknown player id: {player_id}") from exc
        except IndexError as exc:
            raise ValueError(f"unknown unit id for player {player_id}: {unit_id}") from exc

    def iter_player_units(self) -> Iterable[tuple[int, UnitState]]:
        for player_id in sorted(self.players):
            for unit in self.players[player_id].units:
                yield player_id, unit

    def player_unit_at(self, position: Position, *, exclude: tuple[int, int] | None = None) -> tuple[int, UnitState] | None:
        for player_id, unit in self.iter_player_units():
            if exclude == (player_id, unit.id):
                continue
            if unit.position == position:
                return player_id, unit
        return None
