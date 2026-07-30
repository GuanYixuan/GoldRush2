from __future__ import annotations

import unittest

from simulator.config import RulesConfig
from simulator.mechanisms.scripted import ScriptedMechanisms, ScriptedRound
from simulator.rules.snapshot import SnapshotAccumulator
from simulator.rules.transition import transition_one_round
from simulator.state import GameState, NpcState, PlayerState, UnitState
from simulator.types import Action, ActorRef, GameOutput, GoldGenerationEvent, MoveStatus, Position


class TransitionTests(unittest.TestCase):
    def test_scripted_transition_is_deterministic_and_orders_interactions(self) -> None:
        state = _state()
        mechanisms = ScriptedMechanisms(
            {
                0: ScriptedRound(
                    gold_generated=(GoldGenerationEvent(Position(5, 6), 10),),
                    npc_actions={-1: (Action.LEFT,)},
                )
            }
        )
        p1 = GameOutput(actions=(Action.RIGHT, Action.STAY, Action.STAY, Action.STAY, Action.STAY, Action.STAY), k=1, order=0, vp=1)
        p2 = GameOutput(actions=(Action.STAY,) * 6, k=3, order=0, vp=0)
        snapshots = SnapshotAccumulator(RulesConfig(snapshot_period=1))

        result = transition_one_round(
            state,
            player_outputs={1: p1, 2: p2},
            mechanisms=mechanisms,
            first_player_id=1,
            snapshot_accumulator=snapshots,
        )

        self.assertIs(result.state, state)
        self.assertEqual(state.round_index, 1)
        self.assertEqual(state.player_unit(1, 0).position, Position(5, 6))
        self.assertEqual(state.player_unit(1, 0).gold, 7)
        self.assertEqual(state.npcs[-1].position, Position(5, 6))
        self.assertEqual(state.gold[Position(5, 6)], 1)
        self.assertEqual(state.players[1].vision_spent, 2)
        self.assertEqual(state.players[1].active_vision_radius, 3)
        self.assertEqual(state.players[1].next_vision_radius, 2)
        self.assertEqual(result.dispatch_order, (1, -1, 2))
        self.assertEqual(result.movement_events[0].status, MoveStatus.MOVED)
        self.assertEqual(result.interaction_events[0].pickups[0].actor, ActorRef.player_unit(1, 0))
        self.assertEqual(result.interaction_events[0].pickups[0].picked_gold, 7)
        npc_pickups = [event.pickups[0] for event in result.interaction_events if event.pickups and event.pickups[0].actor == ActorRef.npc(-1)]
        self.assertEqual(npc_pickups[0].picked_gold, 2)
        self.assertIsNotNone(result.snapshot)
        assert result.snapshot is not None
        region1 = result.snapshot.regions[0]
        self.assertEqual(region1.gold_generated, 10)
        self.assertEqual(region1.gold_collected, 9)
        self.assertEqual(region1.gold_remaining, 1)
        self.assertEqual(region1.occupants, 3)

    def test_second_player_can_be_first_actor(self) -> None:
        state = _state()
        p1 = GameOutput(actions=(Action.STAY,) * 6, k=3, order=0, vp=0)
        p2 = GameOutput(actions=(Action.LEFT, Action.STAY, Action.STAY, Action.STAY, Action.STAY, Action.STAY), k=1, order=0, vp=0)

        result = transition_one_round(
            state,
            player_outputs={1: p1, 2: p2},
            mechanisms=ScriptedMechanisms(),
            first_player_id=2,
        )

        self.assertEqual(result.dispatch_order, (2, 1))
        self.assertEqual(state.player_unit(2, 0).position, Position(0, 15))


def _state() -> GameState:
    return GameState(
        round_index=0,
        players={
            1: PlayerState(1, [UnitState(0, Position(5, 5)), UnitState(1, Position(10, 10))]),
            2: PlayerState(2, [UnitState(0, Position(0, 16)), UnitState(1, Position(16, 0))]),
        },
        npcs={-1: NpcState(-1, Position(5, 7))},
    )


if __name__ == "__main__":
    unittest.main()
