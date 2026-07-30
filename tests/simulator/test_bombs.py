from __future__ import annotations

import random
import unittest

from simulator.constants import BOMB_REFRESH_PERIOD
from simulator.errors import SimulatorRuleError
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig, DEFAULT_BOMB_SPAWN_PROBABILITY
from simulator.mechanisms.maps import build_initial_state, built_in_public_map_pool
from simulator.state import GameState, NpcState, PlayerState, UnitState
from simulator.types import Position


class BombsTests(unittest.TestCase):
    def test_default_probability_is_documented_estimate(self) -> None:
        self.assertEqual(DEFAULT_BOMB_SPAWN_PROBABILITY, 0.0795)

    def test_non_refresh_round_does_not_change_bombs(self) -> None:
        state = _state(round_index=1)
        state.bombs = {Position(4, 5)}

        event = BernoulliBombRefresher(BombConfig(1.0)).refresh(state, random.Random(1))

        self.assertEqual(event.cleared, frozenset())
        self.assertEqual(event.spawned, frozenset())
        self.assertEqual(state.bombs, {Position(4, 5)})

    def test_refresh_clears_old_bombs_before_resampling(self) -> None:
        state = _state(round_index=0)
        old_bomb = Position(4, 5)
        state.bombs = {old_bomb}

        event = BernoulliBombRefresher(BombConfig(1.0)).refresh(state, random.Random(1))

        self.assertEqual(event.cleared, frozenset({old_bomb}))
        self.assertIn(old_bomb, event.spawned)
        self.assertEqual(state.bombs, set(event.spawned))

    def test_refresh_respects_excluded_cells(self) -> None:
        state = _state(round_index=BOMB_REFRESH_PERIOD)
        state.gold[Position(4, 5)] = 9
        player_pos = state.player_unit(1, 0).position
        npc_pos = next(iter(state.npcs.values())).position

        event = BernoulliBombRefresher(BombConfig(1.0)).refresh(state, random.Random(1))

        self.assertNotIn(Position(0, 3), event.spawned)
        self.assertNotIn(Position(4, 5), event.spawned)
        self.assertNotIn(player_pos, event.spawned)
        self.assertNotIn(npc_pos, event.spawned)
        self.assertIn(Position(0, 4), event.spawned)  # static_map=2 is allowed.

    def test_refresh_is_reproducible_with_seed(self) -> None:
        state1 = _state(round_index=0)
        state2 = _state(round_index=0)
        refresher = BernoulliBombRefresher()

        event1 = refresher.refresh(state1, random.Random(20260730))
        event2 = refresher.refresh(state2, random.Random(20260730))

        self.assertEqual(event1, event2)
        self.assertEqual(state1.bombs, state2.bombs)

    def test_invalid_probability_fails_fast(self) -> None:
        with self.assertRaises(SimulatorRuleError):
            BombConfig(-0.1)
        with self.assertRaises(SimulatorRuleError):
            BombConfig(1.1)

    def test_existing_bomb_overlap_fails_fast(self) -> None:
        state = _state(round_index=0)
        state.bombs = {state.player_unit(1, 0).position}

        with self.assertRaises(SimulatorRuleError):
            BernoulliBombRefresher().refresh(state, random.Random(1))


def _state(*, round_index: int) -> GameState:
    state = build_initial_state(built_in_public_map_pool().get(1))
    state.round_index = round_index
    return state


if __name__ == "__main__":
    unittest.main()
