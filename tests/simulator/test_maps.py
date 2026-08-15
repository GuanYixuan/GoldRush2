from __future__ import annotations

import random
import unittest

from simulator.constants import GRID_SIZE, MAX_NPCS, STATIC_OBSTACLE, STATIC_SPECIAL_NON_BLOCKING
from simulator.errors import SimulatorRuleError
from simulator.mechanisms.maps import (
    MapPool,
    MapTemplate,
    SpawnConfig,
    build_initial_state,
    built_in_public_map_pool,
    built_in_training_map_pool,
    validate_competition_training_map,
)
from simulator.observation.sdk import make_game_input
from simulator.rules.snapshot import region_id
from simulator.types import Position


class MapsTests(unittest.TestCase):
    def test_built_in_public_pool_contains_observed_templates(self) -> None:
        pool = built_in_public_map_pool()

        self.assertEqual([template.map_id for template in pool.templates], [1, 2, 3, 4])
        self.assertEqual([len(template.obstacles) for template in pool.templates], [40, 24, 78, 119])
        self.assertEqual([len(template.special_cells) for template in pool.templates], [20, 20, 20, 6])

    def test_built_in_training_pool_extends_public_pool(self) -> None:
        public_pool = built_in_public_map_pool()
        training_pool = built_in_training_map_pool()

        self.assertEqual([template.map_id for template in training_pool.templates], [1, 2, 3, 4, 101, 111, 121])
        self.assertEqual(
            [template.static_grid for template in training_pool.templates[:4]],
            [template.static_grid for template in public_pool.templates],
        )
        self.assertEqual(len(training_pool.get(101).obstacles), 38)
        self.assertEqual(len(training_pool.get(101).special_cells), 20)
        self.assertEqual(len(training_pool.get(111).obstacles), 42)
        self.assertEqual(len(training_pool.get(111).special_cells), 20)
        self.assertEqual(len(training_pool.get(121).obstacles), 52)
        self.assertEqual(len(training_pool.get(121).special_cells), 20)

    def test_synthetic_training_maps_satisfy_competition_constraints(self) -> None:
        for template in built_in_training_map_pool().templates:
            if template.map_id >= 100:
                validate_competition_training_map(template)

    def test_public_map_four_has_sparse_static_two_cells(self) -> None:
        template = built_in_public_map_pool().get(4)

        counts = {region: 0 for region in (2, 3, 4, 5)}
        for pos in template.special_cells:
            counts[region_id(pos)] += 1

        self.assertEqual(counts, {2: 2, 3: 1, 4: 2, 5: 1})

    def test_training_map_validation_rejects_non_axis_symmetric_obstacles(self) -> None:
        grid = _valid_training_grid()
        grid[1][2] = STATIC_OBSTACLE

        template = MapTemplate.from_static_grid(99, "bad_symmetry", grid)

        with self.assertRaisesRegex(SimulatorRuleError, "symmetry"):
            validate_competition_training_map(template)

    def test_training_map_validation_rejects_bad_outer_special_counts(self) -> None:
        grid = _valid_training_grid()
        grid[0][4] = 0

        template = MapTemplate.from_static_grid(99, "bad_static2_count", grid)

        with self.assertRaisesRegex(SimulatorRuleError, "exactly 5 special"):
            validate_competition_training_map(template)

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
        self.assertEqual(set(ids1), {1, 2, 3, 4})

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


def _valid_training_grid() -> list[list[int]]:
    grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for row, col in (
        (0, 4),
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (4, 0),
        (5, 0),
        (6, 0),
        (7, 0),
        (8, 0),
        (16, 4),
        (16, 5),
        (16, 6),
        (16, 7),
        (16, 8),
        (4, 16),
        (5, 16),
        (6, 16),
        (7, 16),
        (8, 16),
    ):
        grid[row][col] = STATIC_SPECIAL_NON_BLOCKING
    return grid


if __name__ == "__main__":
    unittest.main()
