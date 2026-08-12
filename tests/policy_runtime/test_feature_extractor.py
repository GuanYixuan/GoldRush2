from __future__ import annotations

import unittest

from policy_runtime import FeatureExtractor, channel_names, feature_schema, scalar_names
from simulator.observation.sdk import GameInput, NpcInfo, RegionStat, Snapshot
from simulator.types import Action, GameOutput


class FeatureExtractorTests(unittest.TestCase):
    def test_schema_names_and_shapes(self) -> None:
        extractor = FeatureExtractor(player_id=1)

        features = extractor.observe(_basic_input(round_index=0))

        self.assertEqual(feature_schema(), "goldrush2_feature_v2")
        self.assertEqual(features["feature_schema"], "goldrush2_feature_v2")
        self.assertEqual(features["planes"].shape, (43, 17, 17))
        self.assertEqual(features["scalars"].shape, (10,))
        self.assertEqual(tuple(features["channel_names"]), tuple(channel_names()))
        self.assertEqual(tuple(features["scalar_names"]), tuple(scalar_names()))

    def test_temporal_planes_shift_and_scale(self) -> None:
        extractor = FeatureExtractor(player_id=1)
        first = _fog_input(round_index=0)
        first.grid[1][2] = 40
        first.grid[3][4] = -3
        first.visible_enemies = [(5, 6), (-1, -1)]

        features0 = extractor.observe(first)
        features1 = extractor.observe(_fog_input(round_index=1))

        ch = _channel_index()
        self.assertEqual(features0["planes"][ch["visible_mask_t0"], 1, 2], 1.0)
        self.assertEqual(features0["planes"][ch["gold_count_t0"], 1, 2], 2.0)
        self.assertEqual(features0["planes"][ch["bomb_mask_t0"], 3, 4], 1.0)
        self.assertEqual(features0["planes"][ch["visible_enemy_count_t0"], 5, 6], 1.0)
        self.assertEqual(features1["planes"][ch["visible_mask_t0"], 1, 2], 0.0)
        self.assertEqual(features1["planes"][ch["visible_mask_t1"], 1, 2], 1.0)
        self.assertEqual(features1["planes"][ch["gold_count_t1"], 1, 2], 2.0)
        self.assertEqual(features1["planes"][ch["bomb_mask_t1"], 3, 4], 1.0)
        self.assertEqual(features1["planes"][ch["visible_enemy_count_t1"], 5, 6], 1.0)
        self.assertEqual(features1["planes"][ch["visible_mask_t4"], 1, 2], 0.0)

    def test_obstacle_memory_axis_symmetry_and_direct_override(self) -> None:
        extractor = FeatureExtractor(player_id=1)
        first = _fog_input(round_index=0)
        first.grid[1][2] = -1

        features0 = extractor.observe(first)
        second = _fog_input(round_index=1)
        second.grid[15][2] = 0
        features1 = extractor.observe(second)

        ch = _channel_index()
        self.assertEqual(features0["planes"][ch["obstacle_known_mask"], 1, 2], 1.0)
        self.assertEqual(features0["planes"][ch["obstacle_mask"], 1, 2], 1.0)
        self.assertEqual(features0["planes"][ch["obstacle_known_mask"], 1, 14], 0.0)
        self.assertEqual(features0["planes"][ch["obstacle_known_mask"], 15, 2], 0.0)
        self.assertEqual(features1["planes"][ch["obstacle_known_mask"], 15, 2], 1.0)
        self.assertEqual(features1["planes"][ch["obstacle_mask"], 15, 2], 0.0)
        self.assertEqual(features1["planes"][ch["obstacle_known_mask"], 1, 14], 1.0)
        self.assertEqual(features1["planes"][ch["obstacle_mask"], 1, 14], 1.0)

    def test_obstacle_memory_keeps_unknown_when_both_axes_disagree_or_lack_evidence(self) -> None:
        extractor = FeatureExtractor(player_id=1)
        game_input = _fog_input(round_index=0)
        game_input.grid[1][2] = -1

        features = extractor.observe(game_input)

        ch = _channel_index()
        self.assertEqual(features["planes"][ch["obstacle_known_mask"], 1, 14], 0.0)
        self.assertEqual(features["planes"][ch["obstacle_mask"], 1, 14], 0.0)

    def test_visible_npc_count_and_crowded_mask(self) -> None:
        game_input = _fog_input(round_index=0)
        game_input.visible_npcs = [
            NpcInfo(id=1, row=2, col=2),
            NpcInfo(id=2, row=2, col=2),
            NpcInfo(id=3, row=2, col=2),
            NpcInfo(id=4, row=3, col=3),
        ]

        features = FeatureExtractor(player_id=1).observe(game_input)

        ch = _channel_index()
        self.assertEqual(features["planes"][ch["visible_npc_count"], 2, 2], 1.0)
        self.assertEqual(features["planes"][ch["npc_crowded_mask"], 2, 2], 1.0)
        self.assertAlmostEqual(float(features["planes"][ch["visible_npc_count"], 3, 3]), 1.0 / 3.0)
        self.assertEqual(features["planes"][ch["npc_crowded_mask"], 3, 3], 0.0)

    def test_own_units_and_distance_maps(self) -> None:
        extractor = FeatureExtractor(player_id=1)
        game_input = _fog_input(round_index=0)
        game_input.grid[0][1] = -1

        features = extractor.observe(game_input)

        ch = _channel_index()
        self.assertEqual(features["planes"][ch["own_unit0_mask"], 0, 0], 1.0)
        self.assertEqual(features["planes"][ch["own_unit1_mask"], 16, 16], 1.0)
        self.assertEqual(features["planes"][ch["unit0_manhattan_distance"], 16, 16], 1.0)
        self.assertEqual(features["planes"][ch["unit1_manhattan_distance"], 0, 0], 1.0)
        self.assertEqual(features["planes"][ch["unit0_known_obstacle_distance"], 0, 1], 2.0)
        self.assertAlmostEqual(float(features["planes"][ch["unit0_known_obstacle_distance"], 0, 2]), 4.0 / 32.0)

    def test_regions_and_snapshot_memory(self) -> None:
        extractor = FeatureExtractor(player_id=1)
        first = _basic_input(round_index=0)
        first.snapshot_valid = True
        first.snapshot = _snapshot(
            gold=(100, 200, 300, 400, 500),
            occupants=(1, 2, 3, 4, 8),
            generated=(5, 10, 15, 20, 25),
        )

        features0 = extractor.observe(first)
        features1 = extractor.observe(_basic_input(round_index=1))
        second = _basic_input(round_index=5)
        second.snapshot_valid = True
        second.snapshot = _snapshot(
            gold=(50, 60, 70, 80, 90),
            occupants=(0, 1, 2, 3, 4),
            generated=(30, 40, 50, 60, 70),
        )
        features2 = extractor.observe(second)

        ch = _channel_index()
        self.assertEqual(features0["planes"][ch["region_1_center_mask"], 8, 8], 1.0)
        self.assertEqual(features0["planes"][ch["region_2_up_mask"], 0, 0], 1.0)
        self.assertEqual(features0["planes"][ch["region_3_left_mask"], 4, 0], 1.0)
        self.assertEqual(features0["planes"][ch["region_4_down_mask"], 16, 16], 1.0)
        self.assertEqual(features0["planes"][ch["region_5_right_mask"], 0, 16], 1.0)
        self.assertEqual(features0["planes"][ch["last_snapshot_gold_remaining_map"], 8, 8], 1.0)
        self.assertEqual(features1["planes"][ch["last_snapshot_gold_remaining_map"], 0, 16], 2.0)
        self.assertEqual(features1["planes"][ch["last_snapshot_occupants_map"], 0, 16], 2.0)
        self.assertEqual(features1["planes"][ch["last_snapshot_gold_generated_map"], 0, 16], 0.25)
        self.assertEqual(features2["planes"][ch["last_snapshot_gold_remaining_map"], 8, 8], 0.5)
        self.assertEqual(features2["planes"][ch["prev_snapshot_gold_remaining_map"], 8, 8], 1.0)
        self.assertEqual(features2["planes"][ch["last_snapshot_gold_generated_map"], 8, 8], 0.3)

    def test_scalar_features(self) -> None:
        game_input = _basic_input(round_index=0)
        game_input.my_units_gold = (1000, 500)
        game_input.gold_opp = 500
        game_input.snapshot_valid = True
        game_input.snapshot = _snapshot(gold=(0, 0, 0, 0, 0), occupants=(0, 0, 0, 0, 0))

        features = FeatureExtractor(player_id=1).observe(game_input)

        sc = _scalar_index()
        self.assertAlmostEqual(float(features["scalars"][sc["game_phase_sin"]]), -1.0, places=6)
        self.assertAlmostEqual(float(features["scalars"][sc["game_phase_cos"]]), 0.0, places=6)
        self.assertEqual(features["scalars"][sc["own_gold_scaled"]], 0.75)
        self.assertEqual(features["scalars"][sc["opp_gold_scaled"]], 0.25)
        self.assertEqual(features["scalars"][sc["gold_margin_scaled"]], 2.0)
        self.assertAlmostEqual(float(features["scalars"][sc["outer_gold_phase_sin"]]), 0.0, places=6)
        self.assertAlmostEqual(float(features["scalars"][sc["outer_gold_phase_cos"]]), 1.0, places=6)
        self.assertAlmostEqual(float(features["scalars"][sc["snapshot_sin"]]), 0.0, places=6)
        self.assertAlmostEqual(float(features["scalars"][sc["snapshot_cos"]]), 1.0, places=6)
        self.assertEqual(features["scalars"][sc["last_snapshot_valid"]], 1.0)

    def test_commit_action_does_not_change_feature_schema(self) -> None:
        extractor = FeatureExtractor(player_id=1)

        before = extractor.observe(_basic_input(round_index=0))
        extractor.commit_action(_right_output())
        after = extractor.observe(_basic_input(round_index=1))

        self.assertEqual(before["feature_schema"], "goldrush2_feature_v2")
        self.assertEqual(after["feature_schema"], "goldrush2_feature_v2")
        self.assertEqual(after["planes"].shape, (43, 17, 17))
        self.assertEqual(after["scalars"].shape, (10,))

    def test_reset_clears_memory(self) -> None:
        extractor = FeatureExtractor(player_id=1)
        first = _fog_input(round_index=0)
        first.grid[1][2] = -1
        first.grid[0][5] = 16
        extractor.observe(first)

        extractor.reset(1)
        features = extractor.observe(_fog_input(round_index=0))

        ch = _channel_index()
        self.assertEqual(features["planes"][ch["obstacle_known_mask"], 1, 2], 0.0)
        self.assertEqual(features["planes"][ch["visible_mask_t1"], 1, 2], 0.0)
        self.assertEqual(features["planes"][ch["bomb_belief_mask"], 1, 2], 0.0795)
        self.assertEqual(features["planes"][ch["static_2_mask"], 0, 5], 0.0)

    def test_round_regression_fails_fast(self) -> None:
        extractor = FeatureExtractor(player_id=1)

        extractor.observe(_basic_input(round_index=2))

        with self.assertRaisesRegex(Exception, "round regression"):
            extractor.observe(_basic_input(round_index=1))

    def test_center_gold_spawn_prior_masks_known_obstacles(self) -> None:
        game_input = _fog_input(round_index=0)
        game_input.grid[4][4] = -1

        features = FeatureExtractor(player_id=1).observe(game_input)

        ch = _channel_index()
        self.assertEqual(features["planes"][ch["center_gold_spawn_prob"], 8, 8], 1.0)
        self.assertEqual(features["planes"][ch["center_gold_spawn_prob"], 4, 4], 0.0)
        self.assertGreater(features["planes"][ch["center_gold_spawn_prob"], 4, 5], 0.0)
        self.assertEqual(features["planes"][ch["center_gold_spawn_prob"], 3, 8], 0.0)

    def test_bomb_belief_refresh_visibility_and_invisible_persistence(self) -> None:
        extractor = FeatureExtractor(player_id=1)
        first = _fog_input(round_index=0)
        first.grid[3][4] = -3

        features0 = extractor.observe(first)
        features1 = extractor.observe(_fog_input(round_index=1))
        features20 = extractor.observe(_fog_input(round_index=20))

        ch = _channel_index()
        self.assertEqual(features0["planes"][ch["bomb_belief_mask"], 3, 4], 1.0)
        self.assertEqual(features0["planes"][ch["bomb_belief_mask"], 0, 0], 0.0)
        self.assertEqual(features0["planes"][ch["bomb_belief_mask"], 2, 2], 0.0795)
        self.assertEqual(features1["planes"][ch["bomb_belief_mask"], 3, 4], 1.0)
        self.assertEqual(features20["planes"][ch["bomb_belief_mask"], 3, 4], 0.0795)

    def test_static2_online_discovery_and_distance(self) -> None:
        extractor = FeatureExtractor(player_id=1)

        initial = extractor.observe(_fog_input(round_index=0))
        online = _basic_input(round_index=1)
        online.grid[0][4] = 16
        online.grid[8][8] = 16
        discovered = extractor.observe(online)

        ch = _channel_index()
        self.assertEqual(initial["planes"][ch["static_2_mask"], 0, 5], 0.0)
        self.assertEqual(initial["planes"][ch["to_static2_distance"], 0, 5], 2.0)
        self.assertEqual(discovered["planes"][ch["static_2_mask"], 0, 4], 1.0)
        self.assertEqual(discovered["planes"][ch["static_2_mask"], 8, 8], 0.0)
        self.assertEqual(discovered["planes"][ch["to_static2_distance"], 0, 4], 0.0)


def _channel_index() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(channel_names())}


def _scalar_index() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(scalar_names())}


def _basic_input(round_index: int) -> GameInput:
    grid = [[0 for _ in range(17)] for _ in range(17)]
    return GameInput(
        round=round_index,
        grid=grid,
        my_units=[(0, 0), (16, 16)],
        my_units_gold=(3, 4),
        gold_opp=5,
        visible_enemies=[(-1, -1), (-1, -1)],
        visible_npcs=[],
        snapshot_valid=False,
        snapshot=None,
    )


def _fog_input(round_index: int) -> GameInput:
    game_input = _basic_input(round_index)
    game_input.grid = [[-5 for _ in range(17)] for _ in range(17)]
    return game_input


def _snapshot(
    *,
    gold: tuple[int, int, int, int, int],
    occupants: tuple[int, int, int, int, int],
    generated: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0),
) -> Snapshot:
    return Snapshot(
        window_begin=0,
        window_end=4,
        regions=[
            RegionStat(
                id=idx + 1,
                gold_remaining=gold[idx],
                occupants=occupants[idx],
                gold_generated=generated[idx],
            )
            for idx in range(5)
        ],
    )


def _right_output() -> GameOutput:
    return GameOutput(
        actions=(
            int(Action.RIGHT),
            int(Action.STAY),
            int(Action.STAY),
            int(Action.STAY),
            int(Action.STAY),
            int(Action.STAY),
        ),
        k=1,
        order=0,
        vp=0,
    )


if __name__ == "__main__":
    unittest.main()
