from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

from .constants import ACTION_DELTAS, GRID_SIZE, MOVE_BUDGET, VALID_ACTIONS


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


class MoveStatus(str, Enum):
    MOVED = "moved"
    STAYED = "stayed"
    BLOCKED_OUT_OF_BOUNDS = "blocked_out_of_bounds"
    BLOCKED_OBSTACLE = "blocked_obstacle"
    BLOCKED_PLAYER = "blocked_player"


class ActorKind(str, Enum):
    PLAYER_UNIT = "player_unit"
    NPC = "npc"


@dataclass(frozen=True, order=True)
class Position:
    row: int
    col: int

    def moved(self, action: int | Action) -> Position:
        action_value = int(Action(action))
        dr, dc = ACTION_DELTAS[action_value]
        return Position(self.row + dr, self.col + dc)

    def in_bounds(self) -> bool:
        return 0 <= self.row < GRID_SIZE and 0 <= self.col < GRID_SIZE


@dataclass(frozen=True)
class GameOutput:
    actions: tuple[int, ...]
    k: int
    order: int
    vp: int

    def __post_init__(self) -> None:
        actions = tuple(int(action) for action in self.actions)
        if len(actions) != MOVE_BUDGET:
            raise ValueError(f"actions length must be {MOVE_BUDGET}, got {len(actions)}")
        invalid_actions = [action for action in actions if action not in VALID_ACTIONS]
        if invalid_actions:
            raise ValueError(f"invalid actions: {invalid_actions}")
        if not 0 <= int(self.k) <= MOVE_BUDGET:
            raise ValueError(f"k must be in [0, {MOVE_BUDGET}], got {self.k}")
        if int(self.order) not in (0, 1):
            raise ValueError(f"order must be 0 or 1, got {self.order}")
        if int(self.vp) not in (0, 1, 2):
            raise ValueError(f"vp must be 0, 1, or 2, got {self.vp}")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "k", int(self.k))
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(self, "vp", int(self.vp))


@dataclass(frozen=True)
class PlayerUnitRef:
    player_id: int
    unit_id: int


@dataclass(frozen=True)
class ActorRef:
    kind: ActorKind
    player_id: int | None = None
    unit_id: int | None = None
    npc_id: int | None = None

    @staticmethod
    def player_unit(player_id: int, unit_id: int) -> ActorRef:
        return ActorRef(kind=ActorKind.PLAYER_UNIT, player_id=player_id, unit_id=unit_id)

    @staticmethod
    def npc(npc_id: int) -> ActorRef:
        return ActorRef(kind=ActorKind.NPC, npc_id=npc_id)


@dataclass(frozen=True)
class MovementEvent:
    player_id: int
    unit_id: int
    action: Action
    from_pos: Position
    to_pos: Position
    status: MoveStatus
    blocked_by: PlayerUnitRef | None = None


@dataclass(frozen=True)
class PickupEvent:
    actor: ActorRef
    position: Position
    available_gold: int
    picked_gold: int
    remaining_gold: int


@dataclass(frozen=True)
class BombTriggerEvent:
    actor: ActorRef
    position: Position
    lost_gold: int


@dataclass(frozen=True)
class TrampleEvent:
    actor: ActorRef
    position: Position
    npc_count: int
    penalty: int


@dataclass(frozen=True)
class InteractionEvents:
    pickups: tuple[PickupEvent, ...] = ()
    bomb_triggers: tuple[BombTriggerEvent, ...] = ()
    tramples: tuple[TrampleEvent, ...] = ()
