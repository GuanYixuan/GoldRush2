from __future__ import annotations

from ..config import RulesConfig
from ..errors import SimulatorRuleError
from ..types import (
    ActorKind,
    ActorRef,
    BombTriggerEvent,
    InteractionEvents,
    MoveStatus,
    MovementEvent,
    PickupEvent,
    Position,
    TrampleEvent,
)
from ..state import GameState, NpcState, UnitState


def apply_step_interactions(
    state: GameState,
    movement: MovementEvent,
    rules: RulesConfig | None = None,
) -> InteractionEvents:
    """Apply post-move cell interactions in-place.

    Only successful movement from one cell into another can trigger pickup,
    bomb, or trample. Staying in place and blocked movement do not interact
    with gold or bombs generated underfoot.

    Order implemented here for valid states:
    1. Gold pickup.
    2. Bomb trigger.
    3. Player trample penalty.

    A bomb cell with at least the trample threshold of NPCs is a forbidden state
    invariant and fails fast before any state mutation.
    """
    if movement.status != MoveStatus.MOVED:
        return InteractionEvents()

    rules = RulesConfig() if rules is None else rules
    actor, player_unit = _resolve_actor(state, movement)
    position = movement.to_pos
    npc_count = _npc_count_at(state, position) if position in state.bombs or actor.kind == ActorKind.PLAYER_UNIT else 0
    _validate_landing_cell_invariants(state, position, rules, npc_count)

    pickups: list[PickupEvent] = []
    bomb_triggers: list[BombTriggerEvent] = []
    tramples: list[TrampleEvent] = []

    pickup = _apply_gold_pickup(state, player_unit, actor, position, rules)
    if pickup is not None:
        pickups.append(pickup)

    bomb_trigger = _apply_bomb_trigger(state, player_unit, actor, position, rules)
    if bomb_trigger is not None:
        bomb_triggers.append(bomb_trigger)

    trample = _apply_player_trample(state, player_unit, actor, position, rules, npc_count)
    if trample is not None:
        tramples.append(trample)

    return InteractionEvents(
        pickups=tuple(pickups),
        bomb_triggers=tuple(bomb_triggers),
        tramples=tuple(tramples),
    )


def _apply_gold_pickup(
    state: GameState,
    player_unit: UnitState | None,
    actor: ActorRef,
    position: Position,
    rules: RulesConfig,
) -> PickupEvent | None:
    available = state.gold.get(position)
    if available is None:
        return None
    if available <= 0:
        raise ValueError(f"gold amount must be positive at {position}: {available}")

    picked = _ceil_fraction(available, rules.pickup_numerator, rules.pickup_denominator)
    remaining = available - picked
    if player_unit is not None:
        player_unit.gold += picked
    if remaining > 0:
        state.gold[position] = remaining
    else:
        del state.gold[position]
    return PickupEvent(
        actor=actor,
        position=position,
        available_gold=available,
        picked_gold=picked,
        remaining_gold=max(remaining, 0),
    )


def _validate_landing_cell_invariants(state: GameState, position: Position, rules: RulesConfig, npc_count: int) -> None:
    if position in state.bombs and npc_count >= rules.trample_npc_threshold:
        raise SimulatorRuleError(
            f"forbidden bomb/NPC overlap at {position}: bomb with {npc_count} NPCs "
            f"(threshold={rules.trample_npc_threshold})"
        )


def _apply_bomb_trigger(
    state: GameState,
    player_unit: UnitState | None,
    actor: ActorRef,
    position: Position,
    rules: RulesConfig,
) -> BombTriggerEvent | None:
    if position not in state.bombs:
        return None

    state.bombs.remove(position)
    if actor.kind == ActorKind.NPC:
        return BombTriggerEvent(actor=actor, position=position, lost_gold=0)

    if player_unit is None:
        raise ValueError(f"player bomb trigger has no player unit: {actor}")
    lost = _ceil_fraction(player_unit.gold, rules.bomb_loss_numerator, rules.bomb_loss_denominator)
    player_unit.gold -= lost
    return BombTriggerEvent(actor=actor, position=position, lost_gold=lost)


def _apply_player_trample(
    state: GameState,
    player_unit: UnitState | None,
    actor: ActorRef,
    position: Position,
    rules: RulesConfig,
    npc_count: int,
) -> TrampleEvent | None:
    if actor.kind != ActorKind.PLAYER_UNIT:
        return None
    if player_unit is None:
        raise ValueError(f"player trample has no player unit: {actor}")

    if npc_count < rules.trample_npc_threshold:
        return None

    penalty = _ceil_fraction(player_unit.gold, rules.trample_loss_numerator, rules.trample_loss_denominator)
    player_unit.gold -= penalty
    return TrampleEvent(actor=actor, position=position, npc_count=npc_count, penalty=penalty)


def _resolve_actor(state: GameState, movement: MovementEvent) -> tuple[ActorRef, UnitState | None]:
    if movement.player_id == 0:
        npc = state.npcs.get(movement.unit_id)
        if npc is None:
            raise ValueError(f"movement references unknown NPC: {movement.unit_id}")
        if npc.position != movement.to_pos:
            raise ValueError(f"NPC {movement.unit_id} position {npc.position} does not match movement target {movement.to_pos}")
        return ActorRef.npc(movement.unit_id), None

    unit = state.player_unit(movement.player_id, movement.unit_id)
    if unit.position != movement.to_pos:
        raise ValueError(
            f"player {movement.player_id} unit {movement.unit_id} position {unit.position} "
            f"does not match movement target {movement.to_pos}"
        )
    return ActorRef.player_unit(movement.player_id, movement.unit_id), unit


def _npc_count_at(state: GameState, position: Position) -> int:
    return sum(1 for npc in state.npcs.values() if npc.position == position)


def _ceil_fraction(value: int, numerator: int, denominator: int) -> int:
    if value <= 0:
        return 0
    return (value * numerator + denominator - 1) // denominator
