from __future__ import annotations

import unittest

from simulator.errors import SimulatorRuleError
from simulator.rules.interaction import apply_step_interactions
from simulator.state import GameState, NpcState, PlayerState, UnitState
from simulator.types import Action, ActorRef, MoveStatus, MovementEvent, Position


class InteractionTests(unittest.TestCase):
    def test_player_pickup_takes_ceil_65_percent_and_leaves_remainder(self) -> None:
        state = _state(p1_unit0=Position(1, 2), gold={Position(1, 2): 10})
        movement = _player_moved(Position(1, 1), Position(1, 2))

        events = apply_step_interactions(state, movement)

        self.assertEqual(state.player_unit(1, 0).gold, 7)
        self.assertEqual(state.gold[Position(1, 2)], 3)
        self.assertEqual(len(events.pickups), 1)
        self.assertEqual(events.pickups[0].picked_gold, 7)
        self.assertEqual(events.pickups[0].remaining_gold, 3)

    def test_blocked_or_stay_movement_does_not_pickup_underfoot_gold(self) -> None:
        state = _state(gold={Position(1, 1): 10})
        movement = MovementEvent(
            player_id=1,
            unit_id=0,
            action=Action.STAY,
            from_pos=Position(1, 1),
            to_pos=Position(1, 1),
            status=MoveStatus.STAYED,
        )

        events = apply_step_interactions(state, movement)

        self.assertEqual(state.player_unit(1, 0).gold, 0)
        self.assertEqual(state.gold[Position(1, 1)], 10)
        self.assertFalse(events.pickups)

    def test_npc_pickup_records_actor_and_removes_ground_gold_without_internal_ledger(self) -> None:
        state = _state(gold={Position(1, 2): 2}, npcs={-1: Position(1, 2)})
        movement = _npc_moved(-1, Position(1, 1), Position(1, 2))

        events = apply_step_interactions(state, movement)

        self.assertNotIn(Position(1, 2), state.gold)
        self.assertEqual(events.pickups[0].actor, ActorRef.npc(-1))
        self.assertEqual(events.pickups[0].picked_gold, 2)

    def test_player_bomb_trigger_removes_bomb_and_loses_ceil_10_percent(self) -> None:
        state = _state(p1_unit0=Position(1, 2), bombs={Position(1, 2)})
        state.player_unit(1, 0).gold = 21
        movement = _player_moved(Position(1, 1), Position(1, 2))

        events = apply_step_interactions(state, movement)

        self.assertEqual(state.player_unit(1, 0).gold, 18)
        self.assertNotIn(Position(1, 2), state.bombs)
        self.assertEqual(events.bomb_triggers[0].lost_gold, 3)

    def test_npc_bomb_trigger_only_removes_bomb(self) -> None:
        state = _state(bombs={Position(1, 2)}, npcs={-1: Position(1, 2)})
        movement = _npc_moved(-1, Position(1, 1), Position(1, 2))

        events = apply_step_interactions(state, movement)

        self.assertNotIn(Position(1, 2), state.bombs)
        self.assertEqual(events.bomb_triggers[0].lost_gold, 0)

    def test_player_trample_loses_ceil_5_percent_when_landing_on_three_npcs(self) -> None:
        pos = Position(1, 2)
        state = _state(p1_unit0=pos, npcs={-1: pos, -2: pos, -3: pos})
        state.player_unit(1, 0).gold = 21
        movement = _player_moved(Position(1, 1), pos)

        events = apply_step_interactions(state, movement)

        self.assertEqual(state.player_unit(1, 0).gold, 19)
        self.assertEqual(events.tramples[0].npc_count, 3)
        self.assertEqual(events.tramples[0].penalty, 2)

    def test_gold_pickup_happens_before_trample(self) -> None:
        pos = Position(1, 2)
        state = _state(p1_unit0=pos, gold={pos: 10}, npcs={-1: pos, -2: pos, -3: pos})
        movement = _player_moved(Position(1, 1), pos)

        events = apply_step_interactions(state, movement)

        self.assertEqual(events.pickups[0].picked_gold, 7)
        self.assertEqual(events.tramples[0].penalty, 1)
        self.assertEqual(state.player_unit(1, 0).gold, 6)

    def test_bomb_with_three_npcs_is_forbidden_state(self) -> None:
        pos = Position(1, 2)
        state = _state(p1_unit0=pos, bombs={pos}, npcs={-1: pos, -2: pos, -3: pos})
        state.player_unit(1, 0).gold = 21
        movement = _player_moved(Position(1, 1), pos)

        with self.assertRaises(SimulatorRuleError):
            apply_step_interactions(state, movement)

        self.assertEqual(state.player_unit(1, 0).gold, 21)
        self.assertIn(pos, state.bombs)

    def test_npc_landing_on_three_npcs_does_not_trigger_trample(self) -> None:
        pos = Position(1, 2)
        state = _state(npcs={-1: pos, -2: pos, -3: pos})
        movement = _npc_moved(-1, Position(1, 1), pos)

        events = apply_step_interactions(state, movement)

        self.assertFalse(events.tramples)

    def test_interaction_requires_state_position_to_match_movement_target(self) -> None:
        state = _state()
        movement = _player_moved(Position(1, 1), Position(1, 2))

        with self.assertRaises(ValueError):
            apply_step_interactions(state, movement)


def _player_moved(from_pos: Position, to_pos: Position) -> MovementEvent:
    return MovementEvent(
        player_id=1,
        unit_id=0,
        action=Action.RIGHT,
        from_pos=from_pos,
        to_pos=to_pos,
        status=MoveStatus.MOVED,
    )


def _npc_moved(npc_id: int, from_pos: Position, to_pos: Position) -> MovementEvent:
    return MovementEvent(
        player_id=0,
        unit_id=npc_id,
        action=Action.RIGHT,
        from_pos=from_pos,
        to_pos=to_pos,
        status=MoveStatus.MOVED,
    )


def _state(
    *,
    p1_unit0: Position = Position(1, 1),
    gold: dict[Position, int] | None = None,
    bombs: set[Position] | None = None,
    npcs: dict[int, Position] | None = None,
) -> GameState:
    return GameState(
        round_index=0,
        players={
            1: PlayerState(1, [UnitState(0, p1_unit0), UnitState(1, Position(5, 5))]),
            2: PlayerState(2, [UnitState(0, Position(0, 16)), UnitState(1, Position(16, 0))]),
        },
        gold=dict(gold or {}),
        bombs=set(bombs or set()),
        npcs={npc_id: NpcState(npc_id, position) for npc_id, position in (npcs or {}).items()},
    )


if __name__ == "__main__":
    unittest.main()
