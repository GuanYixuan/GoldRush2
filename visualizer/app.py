from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from PySide6.QtWidgets import QApplication

from visualizer.replay_open import ReplayOpenError
from visualizer.ui.main_window import MainWindow


class VisualizerBootstrapError(RuntimeError):
    """Raised when the visualizer application cannot start."""


def build_window(
    replay_path: str | Path | None = None,
    *,
    argv: Sequence[str] | None = None,
) -> tuple[QApplication, MainWindow]:
    app = QApplication.instance() or QApplication(list(argv) if argv is not None else list(sys.argv[:1]))
    window = MainWindow()
    if replay_path is not None:
        try:
            window.load_replay_path(replay_path)
        except ReplayOpenError as exc:
            raise VisualizerBootstrapError(str(exc)) from exc
    return app, window


__all__ = ["VisualizerBootstrapError", "build_window"]

