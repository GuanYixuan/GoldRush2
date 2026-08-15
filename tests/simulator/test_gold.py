from __future__ import annotations

import random
import unittest

from simulator.errors import SimulatorRuleError
from simulator.mechanisms.gold import (
    CenterGoldConfig,
    CenterGoldGenerator,
    OuterGoldConfig,
    OuterGoldGenerator,
    OuterGoldState,
    center_candidate_cells,
    center_rate,
    outer_static0_candidate_cells,
    outer_static2_candidate_cells,
)
from simulator.mechanisms.maps import build_initial_state, built_in_public_map_pool
from simulator.randomness import make_simulator_rng_streams
from simulator.rules.snapshot import region_id
from simulator.rules.transition import apply_gold_generation
from simulator.types import GoldGenerationEvent, Position


class CenterGoldTests(unittest.TestCase):
    def test_center_candidate_cells_are_center_static_zero_only(self) -> None:
        template = built_in_public_map_pool().get(1)
        candidates = set(center_candidate_cells(template))

        self.assertNotIn(Position(4, 4), candidates)  # obstacle
        self.assertNotIn(Position(12, 13), candidates)  # static_map=2
        self.assertIn(Position(8, 8), candidates)
        self.assertTrue(all(4 <= pos.row <= 12 and 4 <= pos.col <= 12 for pos in candidates))

    def test_center_rate_uses_squared_distance_decay(self) -> None:
        config = CenterGoldConfig(center_a=0.1, center_b=0.5)

        self.assertAlmostEqual(center_rate(Position(8, 8), config), 0.1)
        self.assertAlmostEqual(center_rate(Position(8, 9), config), 0.1 * 2.718281828459045 ** -0.5)
        self.assertLess(center_rate(Position(4, 4), config), center_rate(Position(8, 9), config))

    def test_generate_is_reproducible_and_does_not_mutate_state(self) -> None:
        template = built_in_public_map_pool().get(1)
        state1 = build_initial_state(template)
        state2 = build_initial_state(template)
        generator = CenterGoldGenerator()

        events1 = generator.generate(state1, template, random.Random(20260730))
        events2 = generator.generate(state2, template, random.Random(20260730))

        self.assertEqual(events1, events2)
        self.assertEqual(state1.gold, {})
        self.assertTrue(all(1 <= event.amount <= 11 for event in events1))

    def test_generate_allows_existing_gold_but_filters_bombs_players_and_npcs(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)
        state.players[1].units[0].position = Position(8, 8)
        state.npcs[-1].position = Position(8, 7)
        state.gold[Position(8, 9)] = 3
        state.bombs.add(Position(9, 8))
        generator = CenterGoldGenerator(CenterGoldConfig(center_a=1.0, center_b=0.0))

        events = generator.generate(state, template, random.Random(1))
        positions = {event.position for event in events}

        self.assertNotIn(Position(8, 8), positions)
        self.assertNotIn(Position(8, 7), positions)
        self.assertIn(Position(8, 9), positions)
        self.assertNotIn(Position(9, 8), positions)
        self.assertIn(Position(7, 8), positions)

    def test_config_validation_fails_fast(self) -> None:
        with self.assertRaises(SimulatorRuleError):
            CenterGoldConfig(center_a=-0.1)
        with self.assertRaises(SimulatorRuleError):
            CenterGoldConfig(center_a=1.1)
        with self.assertRaises(SimulatorRuleError):
            CenterGoldConfig(center_b=-0.1)
        with self.assertRaises(SimulatorRuleError):
            CenterGoldConfig(amount_min=0)
        with self.assertRaises(SimulatorRuleError):
            CenterGoldConfig(amount_min=2, amount_max=1)

    def test_center_rate_for_non_center_position_fails_fast(self) -> None:
        with self.assertRaises(SimulatorRuleError):
            center_rate(Position(3, 8))


class OuterGoldTests(unittest.TestCase):
    def test_outer_candidate_cells_match_static_layers_and_regions(self) -> None:
        template = built_in_public_map_pool().get(1)

        static2 = outer_static2_candidate_cells(template, 2)
        static0 = outer_static0_candidate_cells(template, exclude_region=2)

        self.assertEqual(len(static2), 5)
        self.assertTrue(all(region_id(pos) == 2 for pos in static2))
        self.assertTrue(all(template.static_grid[pos.row][pos.col] == 2 for pos in static2))
        self.assertTrue(all(region_id(pos) in (3, 4, 5) for pos in static0))
        self.assertTrue(all(template.static_grid[pos.row][pos.col] == 0 for pos in static0))

    def test_outer_static2_candidates_follow_map_specific_static_layer(self) -> None:
        template = built_in_public_map_pool().get(4)

        counts = {region: len(outer_static2_candidate_cells(template, region)) for region in (2, 3, 4, 5)}

        self.assertEqual(counts, {2: 2, 3: 1, 4: 2, 5: 1})

    def test_initial_state_is_reproducible(self) -> None:
        generator = OuterGoldGenerator()

        self.assertEqual(generator.initial_state(random.Random(7)), generator.initial_state(random.Random(7)))

    def test_default_initial_state_uses_provisional_eight_to_fourteen_offset(self) -> None:
        generator = OuterGoldGenerator()

        initial_states = {seed: generator.initial_state(random.Random(seed)) for seed in range(200)}
        offsets = {state.next_static2_round for state in initial_states.values()}

        self.assertTrue(offsets)
        self.assertTrue(all(8 <= offset <= 14 for offset in offsets))
        self.assertIn(8, offsets)
        self.assertIn(14, offsets)
        self.assertTrue(all(state.next_static2_region in (2, 3, 4, 5) for state in initial_states.values()))

    def test_non_scheduled_round_does_not_generate(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)
        outer_state = OuterGoldState(next_static2_round=3, next_static2_region=2)

        events = OuterGoldGenerator().generate(state, template, outer_state, random.Random(1))

        self.assertEqual(events, ())
        self.assertEqual(outer_state.next_static2_round, 3)
        self.assertEqual(outer_state.next_static2_region, 2)

    def test_missed_scheduled_round_fails_fast(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)
        state.round_index = 4

        with self.assertRaises(SimulatorRuleError):
            OuterGoldGenerator().generate(
                state,
                template,
                OuterGoldState(next_static2_round=3, next_static2_region=2),
                random.Random(1),
            )

    def test_static2_batch_is_split_evenly_and_outer_static0_excludes_high_region(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)
        generator = OuterGoldGenerator(
            OuterGoldConfig(
                gap_weights=((8, 1),),
                region_weights=((2, 1),),
                static2_total_weights=((88, 1),),
                outer_static0_count_weights=((3, 1),),
                outer_static0_amount_weights=((6, 1),),
            )
        )
        outer_state = OuterGoldState(next_static2_round=0, next_static2_region=2)

        events = generator.generate(state, template, outer_state, random.Random(11))

        self.assertEqual(outer_state.next_static2_round, 8)
        self.assertEqual(outer_state.next_static2_region, 2)
        static2_positions = set(outer_static2_candidate_cells(template, 2))
        static2_events = [event for event in events if event.position in static2_positions]
        static0_events = [event for event in events if event.position not in static2_positions]
        self.assertEqual(len(static2_events), 5)
        self.assertEqual(sum(event.amount for event in static2_events), 88)
        self.assertEqual({event.amount for event in static2_events}, {17, 18})
        self.assertEqual(len(static0_events), 3)
        self.assertTrue(all(region_id(event.position) in (3, 4, 5) for event in static0_events))
        self.assertTrue(all(event.amount == 6 for event in static0_events))
        self.assertEqual(state.gold, {})

    def test_dynamic_occupied_cells_are_filtered(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)
        blocked_static2 = outer_static2_candidate_cells(template, 2)[0]
        state.players[1].units[0].position = blocked_static2
        generator = OuterGoldGenerator(
            OuterGoldConfig(
                gap_weights=((8, 1),),
                region_weights=((2, 1),),
                static2_total_weights=((88, 1),),
                outer_static0_count_weights=((0, 1),),
                outer_static0_amount_weights=((6, 1),),
            )
        )

        events = generator.generate(
            state,
            template,
            OuterGoldState(next_static2_round=0, next_static2_region=2),
            random.Random(5),
        )
        positions = {event.position for event in events}
        static2_positions = set(outer_static2_candidate_cells(template, 2))
        static2_events = [event for event in events if event.position in static2_positions]

        self.assertNotIn(blocked_static2, positions)
        self.assertEqual(len(static2_events), 4)
        self.assertEqual(sum(event.amount for event in static2_events), 88)
        self.assertEqual(positions, static2_positions - {blocked_static2})

    def test_zero_available_static2_falls_back_to_other_outer_static0_cells(self) -> None:
        template = built_in_public_map_pool().get(4)
        state = build_initial_state(template)
        blocked_static2 = outer_static2_candidate_cells(template, 5)[0]
        state.players[2].units[0].position = blocked_static2
        generator = OuterGoldGenerator(
            OuterGoldConfig(
                gap_weights=((8, 1),),
                region_weights=((5, 1),),
                static2_total_weights=((90, 1),),
                static2_zero_fallback_count_weights=((3, 1),),
                outer_static0_count_weights=((0, 1),),
            )
        )

        events = generator.generate(
            state,
            template,
            OuterGoldState(next_static2_round=0, next_static2_region=5),
            random.Random(5),
        )

        self.assertEqual(len(events), 3)
        self.assertEqual(sum(event.amount for event in events), 90)
        self.assertEqual({event.amount for event in events}, {30})
        self.assertTrue(all(region_id(event.position) in (2, 3, 4) for event in events))
        self.assertTrue(all(template.static_grid[event.position.row][event.position.col] == 0 for event in events))
        self.assertNotIn(blocked_static2, {event.position for event in events})

    def test_static2_batch_allows_existing_gold_cells(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)
        existing_gold_static2 = outer_static2_candidate_cells(template, 2)[0]
        state.gold[existing_gold_static2] = 4
        generator = OuterGoldGenerator(
            OuterGoldConfig(
                gap_weights=((8, 1),),
                region_weights=((2, 1),),
                static2_total_weights=((88, 1),),
                outer_static0_count_weights=((0, 1),),
            )
        )

        events = generator.generate(
            state,
            template,
            OuterGoldState(next_static2_round=0, next_static2_region=2),
            random.Random(5),
        )
        static2_positions = set(outer_static2_candidate_cells(template, 2))
        static2_events = [event for event in events if event.position in static2_positions]

        self.assertIn(existing_gold_static2, {event.position for event in static2_events})
        self.assertEqual(len(static2_events), 5)
        self.assertEqual(sum(event.amount for event in static2_events), 88)

    def test_outer_static0_generation_allows_existing_gold_cells(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)
        existing_gold_static0 = outer_static0_candidate_cells(template, exclude_region=2)[0]
        state.gold[existing_gold_static0] = 4
        generator = OuterGoldGenerator(
            OuterGoldConfig(
                gap_weights=((8, 1),),
                region_weights=((2, 1),),
                static2_total_weights=((88, 1),),
                outer_static0_count_weights=((999, 1),),
                outer_static0_amount_weights=((6, 1),),
            )
        )

        events = generator.generate(
            state,
            template,
            OuterGoldState(next_static2_round=0, next_static2_region=2),
            random.Random(5),
        )
        static2_positions = set(outer_static2_candidate_cells(template, 2))
        static0_events = [event for event in events if event.position not in static2_positions]

        self.assertIn(existing_gold_static0, {event.position for event in static0_events})

    def test_scheduled_region_controls_current_batch_and_next_schedule_is_complete(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)
        generator = OuterGoldGenerator(
            OuterGoldConfig(
                gap_weights=((8, 1),),
                region_weights=((3, 1),),
                static2_total_weights=((88, 1),),
                outer_static0_count_weights=((0, 1),),
            )
        )
        outer_state = OuterGoldState(next_static2_round=0, next_static2_region=2)

        events = generator.generate(state, template, outer_state, random.Random(5))

        self.assertEqual({event.position for event in events}, set(outer_static2_candidate_cells(template, 2)))
        self.assertEqual(outer_state.next_static2_round, 8)
        self.assertEqual(outer_state.next_static2_region, 3)

    def test_invalid_outer_state_fails_fast(self) -> None:
        with self.assertRaises(SimulatorRuleError):
            OuterGoldState(next_static2_round=-1, next_static2_region=2)
        with self.assertRaises(SimulatorRuleError):
            OuterGoldState(next_static2_round=0, next_static2_region=1)

    def test_apply_gold_generation_stacks_on_existing_gold(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)
        position = Position(8, 9)
        state.gold[position] = 3

        apply_gold_generation(state, (GoldGenerationEvent(position, 5),))

        self.assertEqual(state.gold[position], 8)

    def test_invalid_outer_config_fails_fast(self) -> None:
        with self.assertRaises(SimulatorRuleError):
            OuterGoldState(next_static2_round=-1, next_static2_region=2)
        with self.assertRaises(SimulatorRuleError):
            OuterGoldConfig(gap_weights=())
        with self.assertRaises(SimulatorRuleError):
            OuterGoldConfig(region_weights=((1, 1),))
        with self.assertRaises(SimulatorRuleError):
            OuterGoldConfig(static2_total_weights=((88, 0),))
        with self.assertRaises(SimulatorRuleError):
            outer_static2_candidate_cells(built_in_public_map_pool().get(1), 1)


class SimulatorRngStreamTests(unittest.TestCase):
    def test_named_streams_are_reproducible_and_isolated(self) -> None:
        first = make_simulator_rng_streams(20260807)
        second = make_simulator_rng_streams(20260807)

        for _ in range(100):
            first.outer_gold.random()

        self.assertEqual(
            [first.center_gold.random() for _ in range(10)],
            [second.center_gold.random() for _ in range(10)],
        )
        self.assertEqual(
            [first.npc.random() for _ in range(10)],
            [second.npc.random() for _ in range(10)],
        )


if __name__ == "__main__":
    unittest.main()
