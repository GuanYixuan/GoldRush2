from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from visualizer.model import FrameBundle, FramePhase
from visualizer.replay_io import ReplayDocument
from visualizer.replay_open import ReplayOpenError, open_replay

from .board_view import BoardView
from .playback import PlaybackController


class MainWindow(QMainWindow):
    """Minimal replay viewer for official and merged GoldRush2 replays."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GoldRush2 Replay Visualizer")
        self.resize(1480, 960)

        self.document: ReplayDocument | None = None
        self.playback = PlaybackController()
        self.phase = FramePhase.END
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._shortcuts: list[QShortcut] = []

        self.board = BoardView()
        self.overview = QTextEdit()
        self.overview.setReadOnly(True)
        self.events = QTextEdit()
        self.events.setReadOnly(True)
        self.round_label = QLabel("0 / 0")
        self.progress_round_label = QLabel("当前回合: 0")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.phase_combo = QComboBox()
        self.phase_combo.addItem("回合结束", FramePhase.END.value)
        self.phase_combo.addItem("回合开始", FramePhase.START.value)
        self.phase_combo.currentIndexChanged.connect(self._on_phase_changed)
        self.speed_combo = QComboBox()
        for label, speed in (("1x", 1.0), ("2x", 2.0), ("5x", 5.0), ("10x", 10.0)):
            self.speed_combo.addItem(label, speed)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        self.play_button = QToolButton()
        self.play_button.setText("播放")
        self.play_button.clicked.connect(self._toggle_play)

        self._build_layout()
        self._build_menu()
        self._register_shortcuts()
        self.statusBar().showMessage("打开 official NDJSON 或 merged JSON replay")

    def load_replay_path(self, path: str | Path) -> None:
        document, bundles = open_replay(path)
        self.document = document
        self.playback.load_bundles(bundles)
        self.slider.blockSignals(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, len(bundles) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.setWindowTitle(f"GoldRush2 Replay Visualizer - {Path(path).name}")
        self.statusBar().showMessage(f"已加载: {Path(path).name}", 5000)
        self._refresh()

    def _build_layout(self) -> None:
        top_bar = QHBoxLayout()
        open_button = QPushButton("打开")
        open_button.clicked.connect(self._open_dialog)
        prev_button = QPushButton("上一回合")
        prev_button.clicked.connect(self._step_backward)
        next_button = QPushButton("下一回合")
        next_button.clicked.connect(self._step_forward)
        top_bar.addWidget(open_button)
        top_bar.addWidget(prev_button)
        top_bar.addWidget(self.play_button)
        top_bar.addWidget(next_button)
        top_bar.addWidget(QLabel("阶段"))
        top_bar.addWidget(self.phase_combo)
        top_bar.addWidget(QLabel("倍速"))
        top_bar.addWidget(self.speed_combo)
        top_bar.addStretch(1)
        top_bar.addWidget(self.round_label)

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.addLayout(top_bar)
        center_layout.addWidget(self.board, 1)
        timeline_bar = QHBoxLayout()
        timeline_bar.addWidget(QLabel("进度"))
        timeline_bar.addWidget(self.slider, 1)
        timeline_bar.addWidget(self.progress_round_label)
        center_layout.addLayout(timeline_bar)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("回合概览"))
        left_layout.addWidget(self.overview)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("事件 / 详情"))
        right_layout.addWidget(self.events)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(center_widget)
        splitter.addWidget(right)
        splitter.setSizes([300, 820, 360])
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        open_action = file_menu.addAction("打开 replay")
        open_action.triggered.connect(self._open_dialog)

    def _register_shortcuts(self) -> None:
        self._bind_shortcut(Qt.Key.Key_Left, self._step_backward)
        self._bind_shortcut(Qt.Key.Key_Right, self._step_forward)
        self._bind_shortcut(Qt.Key.Key_Space, self._toggle_play)

    def _bind_shortcut(self, key: int, handler) -> None:
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        shortcut.activated.connect(handler)
        self._shortcuts.append(shortcut)

    def _open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "打开 replay",
            str(Path.cwd()),
            "Replay files (*.txt *.log *.ndjson *.json);;All files (*)",
        )
        if not path:
            return
        try:
            self.load_replay_path(path)
        except ReplayOpenError as exc:
            QMessageBox.warning(self, "打开失败", str(exc))

    def _step_forward(self) -> None:
        self.playback.step_forward()
        self._refresh()

    def _step_backward(self) -> None:
        self.playback.step_backward()
        self._refresh()

    def _toggle_play(self) -> None:
        if self.playback.state().is_playing:
            self.playback.pause()
            self.timer.stop()
        else:
            self.playback.play()
            self.timer.start(max(80, int(700 / self.playback.state().speed)))
        self._refresh_controls()

    def _tick(self) -> None:
        before = self.playback.state().current_round
        self.playback.step_forward()
        if self.playback.state().current_round == before and not self.playback.state().loop_enabled:
            self.timer.stop()
        self._refresh()

    def _on_slider_changed(self, value: int) -> None:
        self.playback.set_round(value)
        self._refresh()

    def _on_phase_changed(self) -> None:
        self.phase = FramePhase(self.phase_combo.currentData())
        self._refresh()

    def _on_speed_changed(self) -> None:
        self.playback.set_speed(float(self.speed_combo.currentData()))
        if self.playback.state().is_playing:
            self.timer.start(max(80, int(700 / self.playback.state().speed)))
        self._refresh_controls()

    def _refresh(self) -> None:
        bundle = self.playback.current_bundle()
        self.board.render_bundle(bundle, None if self.document is None else self.document.static_map, self.phase)
        self.overview.setPlainText(_format_overview(self.document, bundle, self.phase))
        self.events.setPlainText(_format_events(bundle))
        self.slider.blockSignals(True)
        self.slider.setValue(self.playback.state().current_round)
        self.slider.blockSignals(False)
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        total = len(self.playback.bundles())
        current = self.playback.state().current_round
        self.round_label.setText(f"{current} / {max(0, total - 1)}")
        self.progress_round_label.setText(f"当前回合: {current}")
        self.play_button.setText("暂停" if self.playback.state().is_playing else "播放")


def _format_overview(document: ReplayDocument | None, bundle: FrameBundle | None, phase: FramePhase) -> str:
    if document is None or bundle is None:
        return "未加载 replay"
    frame = bundle.start if phase == FramePhase.START else bundle.end
    lines = [
        f"文件: {document.source_path.name}",
        f"类型: {document.kind.value}",
        f"阶段: {phase.value}",
        f"回合: {bundle.round_index}",
        "",
        "玩家",
    ]
    for score in frame.players:
        name = document.players.get(f"player{score.player_id}", f"player{score.player_id}")
        lines.append(
            f"P{score.player_id} {name}: 毛金币 {score.gross_gold}, 视野花费 {score.vision_spent}, 净金币 {score.net_gold}"
        )
    lines.extend(["", "实体"])
    for entity in frame.entities:
        pos = "不可见" if entity.position is None else f"({entity.position.row}, {entity.position.col})"
        complete = "完整" if entity.is_complete else "局部"
        gold = "-" if entity.gold is None else str(entity.gold)
        pickup = "-" if entity.pickup is None else str(entity.pickup)
        lines.append(f"{entity.label}: {pos}, gold={gold}, pickup={pickup}, {complete}, visible_by={entity.visible_by}")
    return "\n".join(lines)


def _format_events(bundle: FrameBundle | None) -> str:
    if bundle is None:
        return "未加载 replay"
    if not bundle.events:
        return "本回合无派生事件"
    return "\n".join(str(event.get("text", event)) for event in bundle.events)
