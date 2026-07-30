from __future__ import annotations

import unittest

from simulator.errors import SimulatorRuleError
from simulator.observation.sdk import NpcInfo, RegionStat, Snapshot, make_game_input
from simulator.state import GameState, NpcState, PlayerState, UnitState
from simulator.types import Position


class SdkObservationTests(unittest.TestCase):
    def test_grid_contains_visible_layers_and_fog_elsewhere(self) -> None:
        state = _state(
            p1=(Position(5, 5), Position(10, 10)),
            obstacles={Position(5, 6)},
            gold={Position(4, 5): 7, Position(0, 0): 9},
            bombs={Position(6, 5)},
        )

        game_input = make_game_input(state, 1)

        self.assertEqual(game_input.grid[5][6], -1)
        self.assertEqual(game_input.grid[4][5], 7)
        self.assertEqual(game_input.grid[6][5], -3)
        self.assertEqual(game_input.grid[5][5], 0)
        self.assertEqual(game_input.grid[0][0], -5)

    def test_player_fields_match_official_python_shape(self) -> None:
        state = _state(
            p1=(Position(5, 5), Position(10, 10)),
            p2=(Position(5, 7), Position(16, 16)),
        )
        state.player_unit(1, 0).gold = 3
        state.player_unit(1, 1).gold = 5
        state.player_unit(2, 0).gold = 11
        state.player_unit(2, 1).gold = 13

        game_input = make_game_input(state, 1)

        self.assertEqual(game_input.round, 12)
        self.assertEqual(game_input.my_units, [(5, 5), (10, 10)])
        self.assertEqual(game_input.my_units_gold, (3, 5))
        self.assertEqual(game_input.gold_opp, 24)
        self.assertEqual(game_input.visible_enemies, [(5, 7), (-1, -1)])
        self.assertFalse(game_input.snapshot_valid)
        self.assertIsNone(game_input.snapshot)

    def test_visible_enemies_are_sorted_by_position_not_unit_id(self) -> None:
        state = _state(
            p1=(Position(5, 5), Position(10, 10)),
            p2=(Position(6, 6), Position(4, 4)),
        )

        game_input = make_game_input(state, 1)

        self.assertEqual(game_input.visible_enemies, [(4, 4), (6, 6)])

    def test_visible_npcs_include_id_and_position(self) -> None:
        state = _state(
            p1=(Position(5, 5), Position(10, 10)),
            npcs={-3: Position(5, 7), -1: Position(0, 0), -2: Position(4, 4)},
        )

        game_input = make_game_input(state, 1)

        self.assertEqual(game_input.visible_npcs, [NpcInfo(-3, 5, 7), NpcInfo(-2, 4, 4)])

    def test_active_vision_radius_controls_visible_area(self) -> None:
        state = _state(
            p1=(Position(5, 5), Position(10, 10)),
            p2=(Position(8, 5), Position(16, 16)),
        )

        default_input = make_game_input(state, 1)
        state.players[1].active_vision_radius = 3
        expanded_input = make_game_input(state, 1)

        self.assertEqual(default_input.visible_enemies, [(-1, -1), (-1, -1)])
        self.assertEqual(expanded_input.visible_enemies, [(8, 5), (-1, -1)])

    def test_snapshot_is_passed_through(self) -> None:
        state = _state()
        snapshot = Snapshot(window_begin=0, window_end=4, regions=[RegionStat(id=index + 1) for index in range(5)])

        game_input = make_game_input(state, 1, snapshot=snapshot)

        self.assertTrue(game_input.snapshot_valid)
        self.assertIs(game_input.snapshot, snapshot)

    def test_invalid_gold_and_bomb_overlap_fails_fast(self) -> None:
        pos = Position(5, 6)
        state = _state(gold={pos: 3}, bombs={pos})

        with self.assertRaises(SimulatorRuleError):
            make_game_input(state, 1)


def _state(
    *,
    p1: tuple[Position, Position] = (Position(5, 5), Position(10, 10)),
    p2: tuple[Position, Position] = (Position(0, 16), Position(16, 0)),
    obstacles: set[Position] | None = None,
    gold: dict[Position, int] | None = None,
    bombs: set[Position] | None = None,
    npcs: dict[int, Position] | None = None,
) -> GameState:
    return GameState(
        round_index=12,
        players={
            1: PlayerState(1, [UnitState(0, p1[0]), UnitState(1, p1[1])]),
            2: PlayerState(2, [UnitState(0, p2[0]), UnitState(1, p2[1])]),
        },
        obstacles=frozenset(obstacles or set()),
        gold=dict(gold or {}),
        bombs=set(bombs or set()),
        npcs={npc_id: NpcState(npc_id, position) for npc_id, position in (npcs or {}).items()},
    )


if __name__ == "__main__":
    unittest.main()
