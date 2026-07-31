from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

Grid = tuple[tuple[int, ...], ...]
RawObject = Mapping[str, Any]


class ReplayKind(str, Enum):
    """Supported top-level replay formats."""

    OFFICIAL = "official"
    MERGED = "merged"
    SIMULATOR_FULL = "simulator_full"


@dataclass(frozen=True)
class Position:
    """Grid position in row/col coordinates."""

    row: int
    col: int


@dataclass(frozen=True)
class ReplayRoundRecord:
    """One normalized round slot before visualizer-specific derivation."""

    round_index: int
    merged: RawObject | None
    raw_round: RawObject | None
    viewer_side: int | None


@dataclass(frozen=True)
class ReplayDocument:
    """Format-neutral replay document consumed by the model layer."""

    kind: ReplayKind
    source_path: Path
    players: Mapping[str, str]
    static_map: Grid
    rounds: tuple[ReplayRoundRecord, ...]
    forfeit: RawObject | None


class ReplayLoadError(ValueError):
    """Raised when a replay file cannot be read or parsed."""
