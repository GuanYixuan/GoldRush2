from __future__ import annotations

import random
import unittest

from simulator.constants import MAX_NPCS, STATIC_OBSTACLE, STATIC_SPECIAL_NON_BLOCKING
from simulator.errors import SimulatorRuleError
from simulator.mechanisms.maps import MapPool, MapTemplate, SpawnConfig, build_initial_state, built_in_public_map_pool
from simulator.observation.sdk import make_game_input
from simulator.types import Position


class MapsTests(unittest.TestCase):
    def test_built_in_public_pool_contains_three_uniform_templates(self) -> None:
        pool = built_in_public_map_pool()

        self.assertEqual([template.map_id for template in pool.templates], [1, 2, 3])
        self.assertEqual([len(template.obstacles) for template in pool.templates], [40, 24, 78])
        self.assertEqual([len(template.special_cells) for template in pool.templates], [20, 20, 20])

    def test_static_two_cells_are_non_blocking_special_cells(self) -> None:
        template = built_in_public_map_pool().get(2)
        special = next(iter(template.special_cells))

        self.assertNotIn(special, template.obstacles)
        self.assertEqual(template.static_grid[special.row][special.col], STATIC_SPECIAL_NON_BLOCKING)

    def test_build_initial_state_uses_cross_map_stable_spawns(self) -> None:
        template = built_in_public_map_pool().get(3)

        state = build_initial_state(template)

        self.assertEqual([unit.position for unit in state.players[1].units], [Position(0, 0), Position(16, 16)])
        self.assertEqual([unit.position for unit in state.players[2].units], [Position(0, 16), Position(16, 0)])
        self.assertEqual(len(state.npcs), MAX_NPCS)
        self.assertEqual({npc.position for npc in state.npcs.values()}, {Position(8, 8)})
        self.assertEqual(state.obstacles, template.obstacles)
        self.assertEqual(state.round_index, 0)

    def test_sampling_is_reproducible_with_seed(self) -> None:
        pool = built_in_public_map_pool()
        rng1 = random.Random(20260730)
        rng2 = random.Random(20260730)

        ids1 = [pool.sample(rng1).map_id for _ in range(20)]
        ids2 = [pool.sample(rng2).map_id for _ in range(20)]

        self.assertEqual(ids1, ids2)
        self.assertEqual(set(ids1), {1, 2, 3})

    def test_spawn_over_obstacle_fails_fast(self) -> None:
        template = built_in_public_map_pool().get(1)
        spawn = SpawnConfig(p1_units=(Position(0, 3), Position(16, 16)))

        with self.assertRaises(SimulatorRuleError):
            build_initial_state(template, spawn)

    def test_map_template_validates_static_grid_and_cell_sets(self) -> None:
        grid = [[0 for _ in range(17)] for _ in range(17)]
        grid[0][0] = STATIC_OBSTACLE
        template = MapTemplate.from_static_grid(99, "test", grid)

        self.assertEqual(template.obstacles, frozenset({Position(0, 0)}))

        with self.assertRaises(SimulatorRuleError):
            MapTemplate(99, "bad", template.static_grid, frozenset(), template.special_cells)

    def test_initial_state_can_build_sdk_observation(self) -> None:
        state = build_initial_state(built_in_public_map_pool().get(1))

        game_input = make_game_input(state, 1)

        self.assertEqual(game_input.my_units, [(0, 0), (16, 16)])
        self.assertEqual(game_input.grid[2][1], -1)

    def test_empty_pool_fails_fast(self) -> None:
        with self.assertRaises(SimulatorRuleError):
            MapPool(())


if __name__ == "__main__":
    unittest.main()
