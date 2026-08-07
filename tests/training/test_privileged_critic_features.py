from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from simulator.mechanisms.gold import outer_static2_candidate_cells
from simulator.mechanisms.maps import build_initial_state, built_in_public_map_pool
from simulator.state import NpcState
from simulator.types import Position
from training.rl.privileged_critic_features import (
    channel_names,
    extract_privileged_critic_features,
    feature_schema,
    scalar_names,
)


@dataclass(frozen=True)
class _OuterState:
    next_static2_round: int
    next_static2_region: int


class PrivilegedCriticFeatureTests(unittest.TestCase):
    def test_schema_names_and_shapes(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)

        features = extract_privileged_critic_features(
            state=state,
            template=template,
            outer_state=_OuterState(next_static2_round=8, next_static2_region=2),
            agent_player_id=1,
        )

        self.assertEqual(feature_schema(), "goldrush2_privileged_critic_feature_v1")
        self.assertEqual(features["feature_schema"], "goldrush2_privileged_critic_feature_v1")
        self.assertEqual(features["planes"].shape, (26, 17, 17))
        self.assertEqual(features["scalars"].shape, (17,))
        self.assertEqual(tuple(features["channel_names"]), channel_names())
        self.assertEqual(tuple(features["scalar_names"]), scalar_names())

    def test_static2_region_and_available_masks(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)
        region2_cells = outer_static2_candidate_cells(template, 2)
        blocked_by_bomb = region2_cells[0]
        blocked_by_npc = region2_cells[1]
        blocked_by_player = region2_cells[2]
        available = region2_cells[3]
        state.bombs.add(blocked_by_bomb)
        state.npcs[-1] = NpcState(-1, blocked_by_npc)
        state.players[1].units[0].position = blocked_by_player

        features = extract_privileged_critic_features(
            state=state,
            template=template,
            outer_state=_OuterState(next_static2_round=8, next_static2_region=2),
            agent_player_id=1,
        )

        ch = _channel_index()
        self.assertEqual(features["planes"][ch["static_2_mask"], available.row, available.col], 1.0)
        self.assertEqual(features["planes"][ch["static_2_available_mask"], available.row, available.col], 1.0)
        self.assertEqual(features["planes"][ch["static_2_available_mask"], blocked_by_bomb.row, blocked_by_bomb.col], 0.0)
        self.assertEqual(features["planes"][ch["static_2_available_mask"], blocked_by_npc.row, blocked_by_npc.col], 0.0)
        self.assertEqual(features["planes"][ch["static_2_available_mask"], blocked_by_player.row, blocked_by_player.col], 0.0)
        self.assertEqual(features["planes"][ch["next_static2_region_mask"], available.row, available.col], 1.0)
        self.assertEqual(features["planes"][ch["next_static2_region_mask"], 8, 8], 0.0)

    def test_role_view_and_unit_numbering(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)
        state.players[1].units[0].position = Position(0, 0)
        state.players[1].units[1].position = Position(16, 16)
        state.players[2].units[0].position = Position(0, 16)
        state.players[2].units[1].position = Position(16, 0)
        state.players[1].units[0].gold = 100
        state.players[1].units[1].gold = 200
        state.players[2].units[0].gold = 300
        state.players[2].units[1].gold = 400
        state.players[1].vision_spent = 10
        state.players[2].vision_spent = 20

        features = extract_privileged_critic_features(
            state=state,
            template=template,
            outer_state=_OuterState(next_static2_round=8, next_static2_region=3),
            agent_player_id=2,
        )

        ch = _channel_index()
        sc = _scalar_index()
        self.assertEqual(features["planes"][ch["own_unit0_mask"], 0, 16], 1.0)
        self.assertEqual(features["planes"][ch["own_unit1_mask"], 16, 0], 1.0)
        self.assertEqual(features["planes"][ch["enemy_unit0_mask"], 0, 0], 1.0)
        self.assertEqual(features["planes"][ch["enemy_unit1_mask"], 16, 16], 1.0)
        self.assertEqual(features["scalars"][sc["own_unit0_gold_scaled"]], 0.3)
        self.assertEqual(features["scalars"][sc["own_unit1_gold_scaled"]], 0.4)
        self.assertEqual(features["scalars"][sc["enemy_unit0_gold_scaled"]], 0.1)
        self.assertEqual(features["scalars"][sc["enemy_unit1_gold_scaled"]], 0.2)
        self.assertAlmostEqual(float(features["scalars"][sc["own_net_gold_scaled"]]), (700 - 20) / 2000.0)
        self.assertAlmostEqual(float(features["scalars"][sc["opp_net_gold_scaled"]]), (300 - 10) / 2000.0)

    def test_gold_npc_distances_regions_and_scalars(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)
        state.round_index = 7
        state.gold = {Position(8, 8): 100, Position(0, 0): 40}
        state.bombs.add(Position(1, 1))
        state.npcs = {
            -1: NpcState(-1, Position(2, 2)),
            -2: NpcState(-2, Position(2, 2)),
            -3: NpcState(-3, Position(2, 2)),
        }

        features = extract_privileged_critic_features(
            state=state,
            template=template,
            outer_state=_OuterState(next_static2_round=8, next_static2_region=4),
            agent_player_id=1,
        )

        ch = _channel_index()
        sc = _scalar_index()
        self.assertEqual(features["planes"][ch["gold_count"], 8, 8], 3.0)
        self.assertEqual(features["planes"][ch["gold_count"], 0, 0], 2.0)
        self.assertEqual(features["planes"][ch["bomb_mask"], 1, 1], 1.0)
        self.assertEqual(features["planes"][ch["npc_count"], 2, 2], 1.0)
        self.assertEqual(features["planes"][ch["npc_crowded_mask"], 2, 2], 1.0)
        self.assertEqual(features["planes"][ch["region_1_center_mask"], 8, 8], 1.0)
        self.assertEqual(features["planes"][ch["region_4_down_mask"], 16, 16], 1.0)
        self.assertEqual(features["planes"][ch["center_gold_spawn_prob"], 8, 8], 1.0)
        self.assertEqual(features["planes"][ch["obstacle_mask"], 0, 3], 1.0)
        self.assertEqual(features["planes"][ch["unit0_obstacle_distance"], 0, 3], 2.0)
        self.assertEqual(features["scalars"][sc["static2_high_imminence"]], 1.0)
        self.assertAlmostEqual(float(features["scalars"][sc["total_gold_on_map_scaled"]]), 140 / 2000.0)
        self.assertAlmostEqual(float(features["scalars"][sc["center_gold_on_map_scaled"]]), 100 / 1000.0)
        self.assertAlmostEqual(float(features["scalars"][sc["bomb_refresh_phase_sin"]]), math.sin(2.0 * math.pi * 7 / 20.0))

    def test_next_static2_round_must_be_future(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)
        state.round_index = 8

        with self.assertRaisesRegex(Exception, "next_static2_round"):
            extract_privileged_critic_features(
                state=state,
                template=template,
                outer_state=_OuterState(next_static2_round=8, next_static2_region=2),
                agent_player_id=1,
            )

    def test_single_round_count_is_supported_for_smoke_tests(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)

        features = extract_privileged_critic_features(
            state=state,
            template=template,
            outer_state=_OuterState(next_static2_round=8, next_static2_region=2),
            agent_player_id=1,
            round_count=1,
        )

        sc = _scalar_index()
        self.assertAlmostEqual(float(features["scalars"][sc["game_phase_sin"]]), -1.0, places=6)
        self.assertEqual(features["scalars"][sc["remaining_rounds_scaled"]], 1.0)

    def test_missing_next_static2_region_fails_fast(self) -> None:
        template = built_in_public_map_pool().get(1)
        state = build_initial_state(template)

        with self.assertRaisesRegex(Exception, "next_static2_region"):
            extract_privileged_critic_features(
                state=state,
                template=template,
                outer_state=type("_MissingRegion", (), {"next_static2_round": 8})(),
                agent_player_id=1,
            )


def _channel_index() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(channel_names())}


def _scalar_index() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(scalar_names())}


if __name__ == "__main__":
    unittest.main()
