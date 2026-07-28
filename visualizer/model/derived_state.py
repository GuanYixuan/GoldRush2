from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from visualizer.replay_io import Grid, Position, ReplayDocument, ReplayKind

from .annotations import CellAnnotation, EntityRef, FloatingLabelAnnotation, MotionAnnotation
from .annotations import Confidence, EntityKind, MotionStyle


class FramePhase(str, Enum):
    """The replay phase currently represented by a frame state."""

    START = "start"
    END = "end"


@dataclass(frozen=True)
class PlayerScore:
    """Round-local player score summary."""

    player_id: int
    gross_gold: int
    vision_spent: int

    @property
    def net_gold(self) -> int:
        return self.gross_gold - self.vision_spent


@dataclass(frozen=True)
class EntityDisplayState:
    """Entity state after replay visibility and source rules are normalized."""

    ref: EntityRef
    label: str
    position: Position | None
    gold: int | None
    actions: tuple[int, ...]
    pickup: int | None
    visible_by: int | None
    is_complete: bool


@dataclass(frozen=True)
class FrameState:
    """One start/end frame consumed by scene rendering."""

    round_index: int
    phase: FramePhase
    grid: Grid
    visible_by: Grid | None
    players: tuple[PlayerScore, ...]
    entities: tuple[EntityDisplayState, ...]
    cell_annotations: tuple[CellAnnotation, ...]
    floating_labels: tuple[FloatingLabelAnnotation, ...]


@dataclass(frozen=True)
class FrameBundle:
    """A replay round plus visualizer annotations for static playback."""

    round_index: int
    start: FrameState
    end: FrameState
    motions: tuple[MotionAnnotation, ...]
    events: tuple[dict[str, object], ...]


class DerivedStateBuilder:
    """Convert a ReplayDocument into FrameBundle objects.

    The implementation should be the only place that interprets GoldRush2 replay
    semantics for display. UI and scene code should consume FrameBundle only.
    """

    @staticmethod
    def build(document: ReplayDocument) -> tuple[FrameBundle, ...]:
        bundles: list[FrameBundle] = []
        for record in document.rounds:
            start_frame = _source_frame(document, record, "start")
            end_frame = _source_frame(document, record, "end")
            start = _build_frame(record.round_index, FramePhase.START, start_frame)
            end = _build_frame(record.round_index, FramePhase.END, end_frame)
            motions = _build_motions(start.entities, end.entities)
            cell_annotations = tuple(_build_cell_annotations(start_frame, end_frame))
            floating_labels = tuple(_build_floating_labels(start.entities, end.entities))
            end = FrameState(
                round_index=end.round_index,
                phase=end.phase,
                grid=end.grid,
                visible_by=end.visible_by,
                players=end.players,
                entities=end.entities,
                cell_annotations=cell_annotations,
                floating_labels=floating_labels,
            )
            events = tuple(_build_events(record.round_index, start_frame, end_frame, floating_labels, cell_annotations))
            bundles.append(FrameBundle(round_index=record.round_index, start=start, end=end, motions=motions, events=events))
        return tuple(bundles)


def _source_frame(document: ReplayDocument, record, phase: str) -> dict:
    if document.kind == ReplayKind.MERGED:
        if record.merged is None:
            raise ValueError(f"round {record.round_index} missing merged payload")
        value = record.merged.get(phase)
    else:
        if record.raw_round is None:
            raise ValueError(f"round {record.round_index} missing official payload")
        value = record.raw_round.get(phase)
    if not isinstance(value, dict):
        raise ValueError(f"round {record.round_index}.{phase} is not an object")
    return value


def _build_frame(round_index: int, phase: FramePhase, frame: dict) -> FrameState:
    grid = _grid(frame.get("grid"))
    visible_by = _visible_by(frame.get("visible_by"), grid)
    return FrameState(
        round_index=round_index,
        phase=phase,
        grid=grid,
        visible_by=visible_by,
        players=tuple(_build_scores(frame.get("players"))),
        entities=tuple(_build_entities(frame, visible_by)),
        cell_annotations=(),
        floating_labels=(),
    )


def _build_scores(players: object) -> list[PlayerScore]:
    if not isinstance(players, list):
        return []
    scores: list[PlayerScore] = []
    for player in players:
        if not isinstance(player, dict):
            continue
        pid = _int(player.get("id"), 0)
        if pid <= 0:
            continue
        scores.append(
            PlayerScore(
                player_id=pid,
                gross_gold=_int(player.get("gold"), 0),
                vision_spent=_int(player.get("vision_spent"), 0),
            )
        )
    return scores


def _build_entities(frame: dict, visible_by: Grid | None) -> list[EntityDisplayState]:
    result: list[EntityDisplayState] = []
    players = frame.get("players")
    if isinstance(players, list):
        for player in players:
            if not isinstance(player, dict):
                continue
            pid = _int(player.get("id"), 0)
            units = player.get("units")
            if not isinstance(units, list):
                continue
            for index, unit in enumerate(units):
                if not isinstance(unit, dict):
                    continue
                position = _position(unit.get("position"))
                actions = _actions(unit.get("actions"))
                result.append(
                    EntityDisplayState(
                        ref=EntityRef(EntityKind.UNIT, pid, index),
                        label=f"P{pid}U{index}",
                        position=position,
                        gold=_optional_int(unit.get("gold")),
                        actions=actions,
                        pickup=_optional_int(unit.get("pickup")),
                        visible_by=_entity_visible_by(unit, position, visible_by),
                        is_complete=_is_complete(unit, position, actions),
                    )
                )

    npcs = frame.get("npcs")
    if isinstance(npcs, list):
        for npc in npcs:
            if not isinstance(npc, dict):
                continue
            npc_id = _int(npc.get("id"), 0)
            if npc_id == 0:
                continue
            position = _position(npc.get("position"))
            actions = _actions(npc.get("actions"))
            result.append(
                EntityDisplayState(
                    ref=EntityRef(EntityKind.NPC, None, npc_id),
                    label=f"NPC {npc_id}",
                    position=position,
                    gold=_optional_int(npc.get("gold")),
                    actions=actions,
                    pickup=_optional_int(npc.get("pickup")),
                    visible_by=_entity_visible_by(npc, position, visible_by),
                    is_complete=_is_complete(npc, position, actions),
                )
            )
    return result


def _build_motions(
    start_entities: tuple[EntityDisplayState, ...],
    end_entities: tuple[EntityDisplayState, ...],
) -> tuple[MotionAnnotation, ...]:
    start_by_ref = {entity.ref: entity for entity in start_entities}
    motions: list[MotionAnnotation] = []
    for end_entity in end_entities:
        start_entity = start_by_ref.get(end_entity.ref)
        start = None if start_entity is None else start_entity.position
        end = end_entity.position
        if start is None or end is None:
            continue
        if start == end and not end_entity.actions:
            continue
        points = _trace_actions(start, end_entity.actions)
        if points and points[-1] == end:
            style = MotionStyle.SOLID_POLYLINE
            confidence = Confidence.CERTAIN if end_entity.is_complete else Confidence.PARTIAL
            motion_points = points
        elif start != end:
            style = MotionStyle.DASHED_ENDPOINT
            confidence = Confidence.PARTIAL
            motion_points = (start, end)
        else:
            style = MotionStyle.ENDPOINT_ONLY
            confidence = Confidence.UNKNOWN
            motion_points = (end,)
        motions.append(
            MotionAnnotation(
                ref=end_entity.ref,
                start=start,
                end=end,
                points=motion_points,
                style=style,
                confidence=confidence,
            )
        )
    return tuple(motions)


def _build_floating_labels(
    start_entities: tuple[EntityDisplayState, ...],
    end_entities: tuple[EntityDisplayState, ...],
) -> list[FloatingLabelAnnotation]:
    start_by_ref = {entity.ref: entity for entity in start_entities}
    labels: list[FloatingLabelAnnotation] = []
    for entity in end_entities:
        pickup = entity.pickup or 0
        if pickup > 0:
            labels.append(FloatingLabelAnnotation(entity.ref, f"+{pickup}", Confidence.CERTAIN))
        previous = start_by_ref.get(entity.ref)
        if previous is None or previous.gold is None or entity.gold is None:
            continue
        loss = previous.gold + pickup - entity.gold
        if loss > 0:
            labels.append(FloatingLabelAnnotation(entity.ref, f"-{loss}", Confidence.PARTIAL))
    return labels


def _build_cell_annotations(start_frame: dict, end_frame: dict) -> list[CellAnnotation]:
    annotations: list[CellAnnotation] = []
    start_grid = _grid(start_frame.get("grid"))
    end_grid = _grid(end_frame.get("grid"))
    for row in range(len(start_grid)):
        for col in range(len(start_grid[row])):
            if start_grid[row][col] == -3 and end_grid[row][col] not in {-5, -3}:
                annotations.append(CellAnnotation(Position(row, col), "!", Confidence.CERTAIN))
    for event in end_frame.get("trample_events") or []:
        if not isinstance(event, dict):
            continue
        pos = _position(event.get("pos"))
        if pos is not None:
            annotations.append(CellAnnotation(pos, "T", Confidence.CERTAIN))
    return annotations


def _build_events(
    round_index: int,
    start_frame: dict,
    end_frame: dict,
    floating_labels: tuple[FloatingLabelAnnotation, ...],
    cell_annotations: tuple[CellAnnotation, ...],
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for label in floating_labels:
        events.append({"round": round_index, "kind": "entity_label", "entity": str(label.ref), "text": label.text})
    burned = _int(end_frame.get("burned"), 0)
    if burned:
        events.append({"round": round_index, "kind": "burned", "text": f"玩家损失金币合计 {burned}"})
    for event in end_frame.get("trample_events") or []:
        if isinstance(event, dict):
            events.append({"round": round_index, "kind": "trample", "text": f"踩踏: {event}"})
    for annotation in cell_annotations:
        if annotation.marker == "!":
            events.append({"round": round_index, "kind": "bomb_triggered", "text": f"炸弹触发于 ({annotation.position.row}, {annotation.position.col})"})
    dispatch_order = end_frame.get("dispatch_order")
    if isinstance(dispatch_order, list):
        first = next((item for item in dispatch_order if isinstance(item, int) and item > 0), None)
        if first is not None:
            events.append({"round": round_index, "kind": "first_player", "text": f"P{first} 先行动"})
    return events


def _trace_actions(start: Position, actions: tuple[int, ...]) -> tuple[Position, ...]:
    row = start.row
    col = start.col
    points = [start]
    for action in actions:
        if action == 0:
            row -= 1
        elif action == 1:
            row += 1
        elif action == 2:
            col -= 1
        elif action == 3:
            col += 1
        elif action == 4:
            pass
        else:
            return tuple(points)
        next_pos = Position(row, col)
        if next_pos != points[-1]:
            points.append(next_pos)
    return tuple(points)


def _grid(value: object) -> Grid:
    if not isinstance(value, list):
        return tuple(tuple(-5 for _ in range(17)) for _ in range(17))
    rows: list[tuple[int, ...]] = []
    for row in value[:17]:
        if isinstance(row, list):
            rows.append(tuple(_int(cell, -5) for cell in row[:17]))
    while len(rows) < 17:
        rows.append(tuple(-5 for _ in range(17)))
    return tuple(row if len(row) == 17 else row + tuple(-5 for _ in range(17 - len(row))) for row in rows)


def _visible_by(value: object, grid: Grid) -> Grid:
    if isinstance(value, list):
        return _grid(value)
    return tuple(tuple(0 if cell == -5 else 1 for cell in row) for row in grid)


def _position(value: object) -> Position | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    row = _int(value[0], -1)
    col = _int(value[1], -1)
    if row < 0 or col < 0:
        return None
    return Position(row, col)


def _actions(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(_int(item, 4) for item in value)


def _entity_visible_by(entity: dict, position: Position | None, visible_by: Grid | None) -> int | None:
    explicit = _optional_int(entity.get("visible_by"))
    if explicit is not None:
        return explicit
    if position is None or visible_by is None:
        return None
    if 0 <= position.row < len(visible_by) and 0 <= position.col < len(visible_by[position.row]):
        return visible_by[position.row][position.col]
    return None


def _is_complete(entity: dict, position: Position | None, actions: tuple[int, ...]) -> bool:
    explicit = entity.get("is_complete")
    if isinstance(explicit, bool):
        return explicit
    return position is not None and "actions" in entity and "pickup" in entity


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return _int(value, 0)


def _int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
