from __future__ import annotations

import math
from collections import defaultdict

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from visualizer.model import FrameBundle, FramePhase, MotionStyle
from visualizer.model.annotations import EntityKind, EntityRef
from visualizer.model.derived_state import EntityDisplayState, FrameState
from visualizer.replay_io import Grid, Position

CELL = 38
BOARD = 17
PLAYER_RADIUS = 11
NPC_RADIUS = 9
GOLD_RADIUS = 10
PLAYER_COLORS = {
    1: "#2563eb",
    2: "#dc2626",
}
NPC_COLORS = (
    "#f59e0b",
    "#84cc16",
    "#a855f7",
    "#92400e",
    "#525252",
    "#d946ef",
    "#0f766e",
)


class BoardView(QGraphicsView):
    """QGraphicsView rendering a single GoldRush2 frame."""

    def __init__(self) -> None:
        self.scene = QGraphicsScene()
        super().__init__(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setBackgroundBrush(QBrush(QColor("#f4f1e8")))
        self.setSceneRect(QRectF(0, 0, CELL * BOARD, CELL * BOARD))
        self.setMinimumSize(CELL * BOARD + 24, CELL * BOARD + 24)

    def render_bundle(self, bundle: FrameBundle | None, static_map: Grid | None, phase: FramePhase) -> None:
        self.scene.clear()
        if bundle is None or static_map is None:
            return
        frame = bundle.start if phase == FramePhase.START else bundle.end
        self._draw_grid(frame, static_map)
        if phase == FramePhase.END:
            self._draw_motions(bundle)
        self._draw_entities(frame)
        self._draw_cell_annotations(frame)
        self._draw_floating_labels(frame)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _draw_grid(self, frame: FrameState, static_map: Grid) -> None:
        font = QFont("Sans Serif", 8)
        for row in range(BOARD):
            for col in range(BOARD):
                value = frame.grid[row][col]
                static_value = static_map[row][col]
                rect = QRectF(col * CELL, row * CELL, CELL, CELL)
                if value == -5:
                    brush = QBrush(QColor("#d5d0c7"))
                elif value == -1 or static_value == 1:
                    brush = QBrush(QColor("#5a544b"))
                elif static_value == 2:
                    brush = QBrush(QColor("#d8efc7"))
                else:
                    brush = QBrush(QColor("#f8f3df"))
                self.scene.addRect(rect, QPen(QColor("#c7bfae")), brush)

                if value == -3:
                    self._draw_bomb(col, row)
                elif value > 0:
                    center = _center(Position(row, col))
                    self.scene.addEllipse(
                        QRectF(center.x() - GOLD_RADIUS, center.y() - GOLD_RADIUS, GOLD_RADIUS * 2, GOLD_RADIUS * 2),
                        QPen(QColor("#b98500")),
                        QBrush(QColor("#f4c542")),
                    )
                    item = self.scene.addText(str(value), font)
                    item.setDefaultTextColor(QColor("#31260c"))
                    _center_text_item(item, center)

    def _draw_bomb(self, col: int, row: int) -> None:
        points = QPolygonF(
            [
                QPointF(col * CELL + CELL / 2, row * CELL + 8),
                QPointF(col * CELL + CELL - 8, row * CELL + CELL - 8),
                QPointF(col * CELL + 8, row * CELL + CELL - 8),
            ]
        )
        self.scene.addPolygon(points, QPen(QColor("#991b1b"), 1.4), QBrush(QColor("#dc2626")))

    def _draw_motions(self, bundle: FrameBundle) -> None:
        for motion in bundle.motions:
            if len(motion.points) < 2:
                continue
            if motion.style == MotionStyle.DASHED_ENDPOINT:
                color = _motion_color(motion.ref)
                color.setAlpha(170)
                pen = QPen(color, 2.2)
                pen.setStyle(Qt.PenStyle.DashLine)
                path = QPainterPath(_center(motion.points[0]))
                for point in motion.points[1:]:
                    path.lineTo(_center(point))
                self.scene.addPath(path, pen)
            else:
                self._draw_solid_motion_segments(motion.points, _motion_color(motion.ref))
            self._draw_arrow_head(motion.points[-2], motion.points[-1], _motion_color(motion.ref))

    def _draw_solid_motion_segments(self, points: tuple[Position, ...], base_color: QColor) -> None:
        segment_count = len(points) - 1
        for index in range(segment_count):
            color = QColor(base_color)
            color.setAlpha(_segment_alpha(index, segment_count))
            pen = QPen(color, 2.4)
            path = QPainterPath(_center(points[index]))
            path.lineTo(_center(points[index + 1]))
            self.scene.addPath(path, pen)

    def _draw_arrow_head(self, start: Position, end: Position, color: QColor) -> None:
        start_pt = _center(start)
        end_pt = _center(end)
        angle = math.atan2(end_pt.y() - start_pt.y(), end_pt.x() - start_pt.x())
        size = 7.0
        points = [
            end_pt,
            QPointF(end_pt.x() - size * math.cos(angle - 0.55), end_pt.y() - size * math.sin(angle - 0.55)),
            QPointF(end_pt.x() - size * math.cos(angle + 0.55), end_pt.y() - size * math.sin(angle + 0.55)),
        ]
        self.scene.addPolygon(QPolygonF(points), QPen(color), QBrush(color))

    def _draw_entities(self, frame: FrameState) -> None:
        by_cell: dict[tuple[int, int], list[EntityDisplayState]] = defaultdict(list)
        for entity in frame.entities:
            if entity.position is not None:
                by_cell[(entity.position.row, entity.position.col)].append(entity)
        entity_font = QFont("Sans Serif", 7, QFont.Weight.Bold)
        for entities in by_cell.values():
            for index, entity in enumerate(entities):
                point = _entity_point(entity.position, index, len(entities))
                color = _entity_color(entity)
                radius = _entity_radius(entity)
                self.scene.addEllipse(
                    QRectF(point.x() - radius, point.y() - radius, radius * 2, radius * 2),
                    QPen(QColor("#1f2937"), 1.2),
                    QBrush(color),
                )
                text = self.scene.addText(_entity_text(entity), entity_font)
                text.setDefaultTextColor(_entity_text_color(entity))
                _center_text_item(text, point)

    def _draw_cell_annotations(self, frame: FrameState) -> None:
        font = QFont("Sans Serif", 18, QFont.Weight.Bold)
        for annotation in frame.cell_annotations:
            item = self.scene.addText(annotation.marker, font)
            item.setDefaultTextColor(QColor("#dc2626") if annotation.marker == "!" else QColor("#be123c"))
            item.setPos(annotation.position.col * CELL + 12, annotation.position.row * CELL - 2)

    def _draw_floating_labels(self, frame: FrameState) -> None:
        entities = {entity.ref: entity for entity in frame.entities}
        font = QFont("Sans Serif", 11, QFont.Weight.Bold)
        offsets: dict[object, int] = defaultdict(int)
        for label in frame.floating_labels:
            entity = entities.get(label.ref)
            if entity is None or entity.position is None:
                continue
            point = _center(entity.position)
            offset = offsets[label.ref]
            offsets[label.ref] += 1
            item = self.scene.addText(label.text, font)
            item.setDefaultTextColor(QColor("#15803d") if label.text.startswith("+") else QColor("#dc2626"))
            item.setPos(point.x() - 10, point.y() - 30 - offset * 16)


def _center(position: Position) -> QPointF:
    return QPointF(position.col * CELL + CELL / 2, position.row * CELL + CELL / 2)


def _center_text_item(item, center: QPointF) -> None:
    bounds = item.boundingRect()
    item.setPos(center.x() - bounds.width() / 2, center.y() - bounds.height() / 2)


def _entity_point(position: Position | None, index: int, total: int) -> QPointF:
    if position is None:
        return QPointF(0, 0)
    base = _center(position)
    if total <= 1:
        return base
    angle = 2 * math.pi * index / total
    return QPointF(base.x() + math.cos(angle) * 9, base.y() + math.sin(angle) * 9)


def _entity_color(entity: EntityDisplayState) -> QColor:
    return _motion_color(entity.ref)


def _entity_radius(entity: EntityDisplayState) -> int:
    if entity.ref.kind == EntityKind.NPC:
        return NPC_RADIUS
    return PLAYER_RADIUS


def _motion_color(ref: EntityRef) -> QColor:
    if ref.kind == EntityKind.NPC:
        index = (abs(ref.entity_id) - 1) % len(NPC_COLORS)
        return QColor(NPC_COLORS[index])
    if ref.owner_id in PLAYER_COLORS:
        return QColor(PLAYER_COLORS[ref.owner_id])
    return QColor("#f59e0b")


def _entity_text(entity: EntityDisplayState) -> str:
    if entity.ref.kind == EntityKind.NPC:
        return str(abs(entity.ref.entity_id))
    if entity.ref.kind == EntityKind.UNIT:
        return str(entity.ref.entity_id)
    return entity.label


def _entity_text_color(entity: EntityDisplayState) -> QColor:
    color = _entity_color(entity)
    luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
    return QColor("#ffffff") if luminance < 120 else QColor("#111827")


def _segment_alpha(index: int, segment_count: int) -> int:
    if segment_count <= 1:
        return 230
    return int(70 + 170 * ((index + 1) / segment_count))
