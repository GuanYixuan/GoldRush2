from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from simulator.observation.sdk import GameInput
from simulator.types import GameOutput


OpponentKind = Literal["python", "cpp_subprocess", "cpp_binding"]


@dataclass(frozen=True)
class EpisodeContext:
    """Per-episode metadata passed to opponent runners."""

    player_id: int = 2
    opponent_id: int = 1
    map_id: int | None = None
    seed: int | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class OpponentSpec:
    """Serializable description of an opponent instance."""

    kind: OpponentKind
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("python", "cpp_subprocess", "cpp_binding"):
            raise ValueError(f"unsupported opponent kind: {self.kind!r}")
        if not self.name:
            raise ValueError("opponent name must be non-empty")
        object.__setattr__(self, "params", dict(self.params))


class OpponentRunner(Protocol):
    def reset(self, seed: int, episode_context: EpisodeContext) -> None:
        ...

    def act(self, game_input: GameInput) -> GameOutput:
        ...

    def __call__(self, game_input: GameInput) -> GameOutput:
        ...
