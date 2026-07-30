from __future__ import annotations

import unittest

from simulator.config import RulesConfig
from simulator.errors import SimulatorRuleError
from simulator.rules.snapshot import SnapshotAccumulator, region_id
from simulator.state import GameState, NpcState, PlayerState, UnitState
from simulator.types import ActorRef, GoldGenerationEvent, InteractionEvents, PickupEvent, Position


class SnapshotTests(unittest.TestCase):
    def test_region_id_matches_documented_areas(self) -> None:
        self.assertEqual(region_id(Position(8, 8)), 1)
        self.assertEqual(region_id(Position(0, 12)), 2)
        self.assertEqual(region_id(Position(16, 0)), 3)
        self.assertEqual(region_id(Position(13, 16)), 4)
        self.assertEqual(region_id(Position(0, 16)), 5)

    def test_snapshot_accumulates_window_stats_and_current_state_fields(self) -> None:
        accumulator = SnapshotAccumulator(RulesConfig(snapshot_period=2))
        state0 = _state(p1_u0=Position(3, 3), npc=-1, npc_pos=Position(4, 4))
        state1 = _state(p1_u0=Position(3, 3), npc=-1, npc_pos=Position(4, 4))

        first = accumulator.record_round(
            0,
            state0,
            state1,
            gold_generated=(GoldGenerationEvent(Position(8, 8), 10),),
        )

        self.assertIsNone(first)

        state2 = _state(
            p1_u0=Position(4, 3),
            npc=-1,
            npc_pos=Position(4, 13),
            gold={Position(8, 8): 5, Position(0, 13): 2},
        )
        interactions = (
            InteractionEvents(
                pickups=(
                    PickupEvent(
                        actor=ActorRef.player_unit(1, 0),
                        position=Position(13, 4),
                        available_gold=10,
                        picked_gold=7,
                        remaining_gold=3,
                    ),
                )
            ),
        )

        snapshot = accumulator.record_round(1, state1, state2, interactions=interactions)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.window_begin, 0)
        self.assertEqual(snapshot.window_end, 1)
        regions = {region.id: region for region in snapshot.regions}
        self.assertEqual(regions[1].gold_generated, 10)
        self.assertEqual(regions[1].gold_remaining, 5)
        self.assertEqual(regions[2].leave, 1)
        self.assertEqual(regions[3].enter, 1)
        self.assertEqual(regions[4].gold_collected, 7)
        self.assertEqual(regions[5].enter, 1)
        self.assertEqual(regions[5].gold_remaining, 2)
        self.assertEqual(regions[5].occupants, 2)

    def test_next_window_starts_after_snapshot_round(self) -> None:
        accumulator = SnapshotAccumulator(RulesConfig(snapshot_period=1))
        state = _state()

        snapshot = accumulator.record_round(3, state, state)

        self.assertIsNotNone(snapshot)
        self.assertEqual(accumulator.window_begin, 4)

    def test_out_of_order_round_fails_fast(self) -> None:
        accumulator = SnapshotAccumulator(RulesConfig(snapshot_period=2))
        state = _state()
        accumulator.record_round(0, state, state)

        with self.assertRaises(SimulatorRuleError):
            accumulator.record_round(2, state, state)

    def test_changed_occupant_set_fails_fast(self) -> None:
        accumulator = SnapshotAccumulator(RulesConfig(snapshot_period=2))
        start = _state(npc=-1, npc_pos=Position(8, 8))
        end = _state(npc=-2, npc_pos=Position(8, 8))

        with self.assertRaises(SimulatorRuleError):
            accumulator.record_round(0, start, end)


def _state(
    *,
    p1_u0: Position = Position(0, 0),
    npc: int = -1,
    npc_pos: Position = Position(8, 8),
    gold: dict[Position, int] | None = None,
) -> GameState:
    return GameState(
        round_index=0,
        players={
            1: PlayerState(1, [UnitState(0, p1_u0), UnitState(1, Position(16, 16))]),
            2: PlayerState(2, [UnitState(0, Position(0, 16)), UnitState(1, Position(16, 0))]),
        },
        gold=dict(gold or {}),
        npcs={npc: NpcState(npc, npc_pos)},
    )


if __name__ == "__main__":
    unittest.main()
