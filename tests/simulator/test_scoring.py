from __future__ import annotations

import unittest

from simulator.config import RulesConfig
from simulator.errors import SimulatorRuleError
from simulator.rules.scoring import (
    activate_pending_vision,
    apply_vision_purchase,
    determine_winner,
    gross_gold,
    net_gold,
)
from simulator.state import GameState, NpcState, PlayerState, UnitState
from simulator.types import Position


class ScoringTests(unittest.TestCase):
    def test_gross_and_net_gold_use_player_units_and_vision_spent(self) -> None:
        state = _state()
        state.player_unit(1, 0).gold = 7
        state.player_unit(1, 1).gold = 5
        state.players[1].vision_spent = 3

        self.assertEqual(gross_gold(state, 1), 12)
        self.assertEqual(net_gold(state, 1), 9)

    def test_vision_purchase_accumulates_cost_without_gold_requirement(self) -> None:
        state = _state()

        apply_vision_purchase(state, 1, vp=2)

        self.assertEqual(state.players[1].vision_spent, 3)
        self.assertEqual(state.players[1].next_vision_radius, 4)

    def test_activate_pending_vision_applies_one_round_effect_and_resets_pending(self) -> None:
        state = _state()
        state.players[1].next_vision_radius = 4

        activate_pending_vision(state)

        self.assertEqual(state.players[1].active_vision_radius, 4)
        self.assertEqual(state.players[1].next_vision_radius, 2)

    def test_determine_winner_by_net_gold(self) -> None:
        state = _state()
        state.player_unit(1, 0).gold = 10
        state.player_unit(2, 0).gold = 9

        result = determine_winner(state)

        self.assertEqual(result.winner_id, 1)
        self.assertEqual(result.loser_id, 2)
        self.assertEqual(result.reason, "net_gold")

    def test_determine_winner_uses_lower_p90_latency_on_net_tie(self) -> None:
        state = _state()
        state.player_unit(1, 0).gold = 10
        state.player_unit(2, 0).gold = 10

        result = determine_winner(state, p90_latency_ns={1: 500, 2: 700})

        self.assertEqual(result.winner_id, 1)
        self.assertEqual(result.reason, "p90_latency")

    def test_net_tie_without_p90_fails_fast(self) -> None:
        state = _state()

        with self.assertRaises(SimulatorRuleError):
            determine_winner(state)

    def test_custom_vision_config_is_respected(self) -> None:
        state = _state()
        rules = RulesConfig(vision_cost_by_vp={0: 0, 1: 4, 2: 9}, vision_radius_by_vp={0: 2, 1: 5, 2: 8})

        apply_vision_purchase(state, 1, vp=1, rules=rules)

        self.assertEqual(state.players[1].vision_spent, 4)
        self.assertEqual(state.players[1].next_vision_radius, 5)


def _state() -> GameState:
    return GameState(
        round_index=0,
        players={
            1: PlayerState(1, [UnitState(0, Position(0, 0)), UnitState(1, Position(16, 16))]),
            2: PlayerState(2, [UnitState(0, Position(0, 16)), UnitState(1, Position(16, 0))]),
        },
        npcs={-1: NpcState(-1, Position(8, 8))},
    )


if __name__ == "__main__":
    unittest.main()
