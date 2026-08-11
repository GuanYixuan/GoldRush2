from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from visualizer.replay_open import ReplayOpenError

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

    from visualizer.ui.main_window import MainWindow


class VisualizerBootstrapError(RuntimeError):
    """Raised when the visualizer application cannot start."""


def _load_qt_widgets():
    try:
        from PySide6.QtWidgets import QApplication

        from visualizer.ui.main_window import MainWindow
    except ImportError as exc:
        raise VisualizerBootstrapError(
            "无法导入 PySide6/Qt。若错误中包含 libGL/libEGL/libxkbcommon/libxcb，"
            "请先安装 visualizer README 中列出的系统运行库。"
        ) from exc
    return QApplication, MainWindow


def _ensure_display_available() -> None:
    if sys.platform != "linux":
        return
    if os.environ.get("QT_QPA_PLATFORM"):
        return
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return
    raise VisualizerBootstrapError(
        "未检测到图形显示端：DISPLAY 和 WAYLAND_DISPLAY 均为空。"
        "请在桌面/X11 forwarding/VNC 中运行；只做无界面加载检查时可设置 QT_QPA_PLATFORM=offscreen。"
    )


def build_window(
    replay_path: str | Path | None = None,
    *,
    argv: Sequence[str] | None = None,
) -> tuple["QApplication", "MainWindow"]:
    QApplication, MainWindow = _load_qt_widgets()
    _ensure_display_available()
    app = QApplication.instance() or QApplication(list(argv) if argv is not None else list(sys.argv[:1]))
    window = MainWindow()
    if replay_path is not None:
        try:
            window.load_replay_path(replay_path)
        except ReplayOpenError as exc:
            raise VisualizerBootstrapError(str(exc)) from exc
    return app, window


__all__ = ["VisualizerBootstrapError", "build_window"]
