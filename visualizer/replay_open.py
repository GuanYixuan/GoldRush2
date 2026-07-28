from __future__ import annotations

from pathlib import Path

from visualizer.model.derived_state import DerivedStateBuilder
from visualizer.replay_io import ReplayDocument, ReplayLoadError, ReplayLoader


class ReplayOpenError(RuntimeError):
    """Raised when a replay file cannot be opened for visualization."""


def open_replay(path: str | Path) -> tuple[ReplayDocument, tuple]:
    try:
        document = ReplayLoader.load(path)
        bundles = DerivedStateBuilder.build(document)
    except (ReplayLoadError, ValueError) as exc:
        raise ReplayOpenError(str(exc)) from exc
    if not bundles:
        raise ReplayOpenError("replay 不包含可显示回合")
    return document, bundles


__all__ = ["ReplayOpenError", "open_replay"]

