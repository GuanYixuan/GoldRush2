from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from simulator.observation.sdk import GameInput


@dataclass(frozen=True)
class ResetResult:
    observation: GameInput
    info: dict[str, Any]


@dataclass(frozen=True)
class StepResult:
    observation: GameInput | None
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
