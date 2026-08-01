from __future__ import annotations

import copy
from dataclasses import dataclass

from ..config import RulesConfig
from ..errors import SimulatorRuleError
from ..mechanisms.scripted import ScriptedMechanisms
from ..state import GameState
from ..types import Action, GameOutput, GoldGenerationEvent, InteractionEvents, MoveStatus, MovementEvent, Snapshot
from .interaction import apply_step_interactions
from .movement import apply_npc_step, apply_player_step, iter_player_steps
from .scoring import activate_pending_vision, apply_vision_purchase
from .snapshot import SnapshotAccumulator


@dataclass(frozen=True)
class TransitionResult:
    state: GameState
    gold_generated: tuple[GoldGenerationEvent, ...]
    movement_events: tuple[MovementEvent, ...]
    interaction_events: tuple[InteractionEvents, ...]
    dispatch_order: tuple[int, ...]
    snapshot: Snapshot | None = None


def transition_one_round(
    state: GameState,
    player_outputs: dict[int, GameOutput],
    mechanisms: ScriptedMechanisms,
    first_player_id: int,
    rules: RulesConfig | None = None,
    snapshot_accumulator: SnapshotAccumulator | None = None,
) -> TransitionResult:
    """Advance one round in-place and return structured transition events."""
    rules = RulesConfig() if rules is None else rules
    if set(player_outputs) != {1, 2}:
        raise ValueError(f"player_outputs must contain players 1 and 2, got {sorted(player_outputs)}")
    if first_player_id not in (1, 2):
        raise ValueError(f"first_player_id must be 1 or 2, got {first_player_id}")

    round_index = state.round_index
    start_state = copy.deepcopy(state)
    second_player_id = 1 if first_player_id == 2 else 2

    gold_generated = mechanisms.gold_generated(round_index)
    apply_gold_generation(state, gold_generated)

    movement_events: list[MovementEvent] = []
    interaction_events: list[InteractionEvents] = []
    dispatch_order = [first_player_id]

    _apply_player_action(state, first_player_id, player_outputs[first_player_id], rules, movement_events, interaction_events)

    for npc_id in sorted(state.npcs):
        actions = mechanisms.npc_actions(round_index, npc_id)
        if actions:
            dispatch_order.append(npc_id)
        for action in actions:
            movement = apply_npc_step(state, npc_id, action)
            movement_events.append(movement)
            interaction_events.append(apply_step_interactions(state, movement, rules))

    dispatch_order.append(second_player_id)
    _apply_player_action(state, second_player_id, player_outputs[second_player_id], rules, movement_events, interaction_events)

    for player_id, output in player_outputs.items():
        apply_vision_purchase(state, player_id, output.vp, rules)

    snapshot = None
    if snapshot_accumulator is not None:
        snapshot = snapshot_accumulator.record_round(
            round_index,
            start_state,
            state,
            gold_generated=gold_generated,
            interactions=tuple(interaction_events),
        )

    state.round_index += 1
    activate_pending_vision(state, rules)

    return TransitionResult(
        state=state,
        gold_generated=gold_generated,
        movement_events=tuple(movement_events),
        interaction_events=tuple(interaction_events),
        dispatch_order=tuple(dispatch_order),
        snapshot=snapshot,
    )


def transition_started_round(
    state: GameState,
    player_outputs: dict[int, GameOutput],
    gold_generated: tuple[GoldGenerationEvent, ...],
    first_player_id: int,
    npc_order: tuple[int, ...],
    npc_actions: dict[int, tuple[int | Action, ...]],
    rules: RulesConfig | None = None,
    snapshot_accumulator: SnapshotAccumulator | None = None,
) -> TransitionResult:
    """Advance one round after pre-round resource generation has been applied.

    The caller supplies player outputs, first-player order, and already planned
    NPC actions. This keeps rules.transition responsible for execution and
    settlement only; unknown NPC policy remains in mechanisms/envs.
    """
    rules = RulesConfig() if rules is None else rules
    if set(player_outputs) != {1, 2}:
        raise ValueError(f"player_outputs must contain players 1 and 2, got {sorted(player_outputs)}")
    if first_player_id not in (1, 2):
        raise ValueError(f"first_player_id must be 1 or 2, got {first_player_id}")
    if set(npc_order) != set(state.npcs):
        raise SimulatorRuleError(f"npc_order must contain exactly current NPC ids, got {npc_order}")
    if set(npc_actions) != set(npc_order):
        raise SimulatorRuleError(f"npc_actions must contain actions for NPC ids {npc_order}, got {sorted(npc_actions)}")

    round_index = state.round_index
    start_state = copy.deepcopy(state)
    second_player_id = 1 if first_player_id == 2 else 2

    movement_events: list[MovementEvent] = []
    interaction_events: list[InteractionEvents] = []
    dispatch_order = [first_player_id]

    _apply_player_action(state, first_player_id, player_outputs[first_player_id], rules, movement_events, interaction_events)

    for npc_id in npc_order:
        actions = tuple(Action(action) for action in npc_actions[npc_id])
        if actions:
            dispatch_order.append(npc_id)
        for action in actions:
            movement = apply_npc_step(state, npc_id, action)
            movement_events.append(movement)
            interaction_events.append(apply_step_interactions(state, movement, rules))

    dispatch_order.append(second_player_id)
    _apply_player_action(state, second_player_id, player_outputs[second_player_id], rules, movement_events, interaction_events)

    for player_id, output in player_outputs.items():
        apply_vision_purchase(state, player_id, output.vp, rules)

    snapshot = None
    if snapshot_accumulator is not None:
        snapshot = snapshot_accumulator.record_round(
            round_index,
            start_state,
            state,
            gold_generated=gold_generated,
            interactions=tuple(interaction_events),
        )

    state.round_index += 1
    activate_pending_vision(state, rules)

    return TransitionResult(
        state=state,
        gold_generated=gold_generated,
        movement_events=tuple(movement_events),
        interaction_events=tuple(interaction_events),
        dispatch_order=tuple(dispatch_order),
        snapshot=snapshot,
    )


def _apply_player_action(
    state: GameState,
    player_id: int,
    output: GameOutput,
    rules: RulesConfig,
    movement_events: list[MovementEvent],
    interaction_events: list[InteractionEvents],
) -> None:
    for unit_id, action in iter_player_steps(output):
        movement = apply_player_step(state, player_id, unit_id, action)
        movement_events.append(movement)
        if movement.status == MoveStatus.MOVED:
            interaction_events.append(apply_step_interactions(state, movement, rules))
        else:
            interaction_events.append(InteractionEvents())


def apply_gold_generation(state: GameState, gold_generated: tuple[GoldGenerationEvent, ...]) -> None:
    for event in gold_generated:
        if event.amount <= 0:
            raise SimulatorRuleError(f"gold generation amount must be positive: {event}")
        if not event.position.in_bounds():
            raise SimulatorRuleError(f"gold generation position out of bounds: {event.position}")
        if event.position in state.obstacles:
            raise SimulatorRuleError(f"gold generation overlaps obstacle at {event.position}")
        if event.position in state.bombs:
            raise SimulatorRuleError(f"gold generation overlaps bomb at {event.position}")
        state.gold[event.position] = state.gold.get(event.position, 0) + event.amount


__all__ = ["TransitionResult", "apply_gold_generation", "transition_one_round", "transition_started_round"]
