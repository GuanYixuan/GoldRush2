from __future__ import annotations

from . import __doc__ as _rules_doc
from ..errors import SimulatorRuleError
from ..types import Action, GameOutput, MoveStatus, MovementEvent, PlayerUnitRef
from ..state import GameState

_ = _rules_doc


def apply_player_turn(state: GameState, player_id: int, output: GameOutput) -> list[MovementEvent]:
    """Apply one player's full six-step turn in-place and return movement events."""
    if player_id not in state.players:
        raise ValueError(f"unknown player id: {player_id}")

    unit_actions = _split_unit_actions(output)
    unit_order = (0, 1) if output.order == 0 else (1, 0)

    events: list[MovementEvent] = []
    for unit_id in unit_order:
        for action in unit_actions[unit_id]:
            events.append(apply_player_step(state, player_id, unit_id, action))
    return events


def apply_player_step(state: GameState, player_id: int, unit_id: int, action: int | Action) -> MovementEvent:
    """Apply one player unit movement step in-place.

    NPCs intentionally do not block player movement.
    """
    parsed_action = Action(action)
    unit = state.player_unit(player_id, unit_id)
    from_pos = unit.position

    if parsed_action == Action.STAY:
        return MovementEvent(
            player_id=player_id,
            unit_id=unit_id,
            action=parsed_action,
            from_pos=from_pos,
            to_pos=from_pos,
            status=MoveStatus.STAYED,
        )

    to_pos = from_pos.moved(parsed_action)
    if not to_pos.in_bounds():
        return MovementEvent(
            player_id=player_id,
            unit_id=unit_id,
            action=parsed_action,
            from_pos=from_pos,
            to_pos=from_pos,
            status=MoveStatus.BLOCKED_OUT_OF_BOUNDS,
        )

    if to_pos in state.obstacles:
        return MovementEvent(
            player_id=player_id,
            unit_id=unit_id,
            action=parsed_action,
            from_pos=from_pos,
            to_pos=from_pos,
            status=MoveStatus.BLOCKED_OBSTACLE,
        )

    blocker = state.player_unit_at(to_pos, exclude=(player_id, unit_id))
    if blocker is not None:
        blocker_player_id, blocker_unit = blocker
        return MovementEvent(
            player_id=player_id,
            unit_id=unit_id,
            action=parsed_action,
            from_pos=from_pos,
            to_pos=from_pos,
            status=MoveStatus.BLOCKED_PLAYER,
            blocked_by=PlayerUnitRef(blocker_player_id, blocker_unit.id),
        )

    unit.position = to_pos
    return MovementEvent(
        player_id=player_id,
        unit_id=unit_id,
        action=parsed_action,
        from_pos=from_pos,
        to_pos=to_pos,
        status=MoveStatus.MOVED,
    )


def apply_npc_step(state: GameState, npc_id: int, action: int | Action) -> MovementEvent:
    """Apply one NPC movement step in-place.

    NPC obstacle or out-of-bounds attempts are treated as fail-fast because
    current replay evidence indicates NPCs never generate such attempts.
    NPCs may overlap players and other NPCs, so player occupancy is ignored.
    """
    if npc_id not in state.npcs:
        raise ValueError(f"unknown npc id: {npc_id}")

    parsed_action = Action(action)
    npc = state.npcs[npc_id]
    from_pos = npc.position

    if parsed_action == Action.STAY:
        return MovementEvent(
            player_id=0,
            unit_id=npc_id,
            action=parsed_action,
            from_pos=from_pos,
            to_pos=from_pos,
            status=MoveStatus.STAYED,
        )

    to_pos = from_pos.moved(parsed_action)
    if not to_pos.in_bounds():
        raise SimulatorRuleError(f"NPC {npc_id} attempted to move out of bounds: {from_pos} -> {to_pos}")
    if to_pos in state.obstacles:
        raise SimulatorRuleError(f"NPC {npc_id} attempted to move into obstacle: {from_pos} -> {to_pos}")

    npc.position = to_pos
    return MovementEvent(
        player_id=0,
        unit_id=npc_id,
        action=parsed_action,
        from_pos=from_pos,
        to_pos=to_pos,
        status=MoveStatus.MOVED,
    )


def _split_unit_actions(output: GameOutput) -> dict[int, tuple[Action, ...]]:
    actions = tuple(Action(action) for action in output.actions)
    return {
        0: actions[: output.k],
        1: actions[output.k :],
    }
