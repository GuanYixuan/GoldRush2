from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from visualizer.replay_io import Position


class EntityKind(str, Enum):
    """Selectable entity kinds displayed on the board."""

    UNIT = "unit"
    NPC = "npc"
    PLAYER = "player"


class MotionStyle(str, Enum):
    """How confidently a movement path can be drawn."""

    SOLID_POLYLINE = "solid_polyline"
    DASHED_ENDPOINT = "dashed_endpoint"
    ENDPOINT_ONLY = "endpoint_only"


class Confidence(str, Enum):
    """Confidence of a visual annotation."""

    CERTAIN = "certain"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EntityRef:
    """Stable visual reference for selection and annotation binding."""

    kind: EntityKind
    owner_id: int | None
    entity_id: int


@dataclass(frozen=True)
class MotionAnnotation:
    """One round-level movement arrow or path marker."""

    ref: EntityRef
    start: Position | None
    end: Position | None
    points: tuple[Position, ...]
    style: MotionStyle
    confidence: Confidence


@dataclass(frozen=True)
class FloatingLabelAnnotation:
    """Short label drawn above an entity, such as '+5' or '-2'."""

    ref: EntityRef
    text: str
    confidence: Confidence


@dataclass(frozen=True)
class CellAnnotation:
    """Short marker drawn on a map cell."""

    position: Position
    marker: str
    confidence: Confidence

