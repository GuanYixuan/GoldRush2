from __future__ import annotations

import unittest

import numpy as np

from policy_runtime import (
    FastRuntimeState,
    FeatureExtractor,
    infer_fast_role,
    simulate_known_gold_pickups,
    try_fast_gold_grab,
)
from simulator.observation.sdk import GameInput, NpcInfo
from simulator.types import Action, GameOutput


class FastOptionTests(unittest.TestCase):
    def test_try_fast_gold_grab_uses_first_hit_and_fused_output(self) -> None:
        game_input = _basic_input(round_index=3)
        game_input.my_units = [(8, 8), (12, 12)]
        game_input.grid[7][8] = 15
        game_input.grid[8][7] = 30

        result = try_fast_gold_grab(game_input, 12)

        self.assertTrue(result["success"])
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["role"], 0)
        self.assertEqual(result["target"], (7, 8))
        self.assertEqual(result["action_count"], 3)
        self.assertEqual(tuple(result["output"]["actions"]), (int(Action.UP), int(Action.DOWN), int(Action.UP), 4, 4, 4))
        self.assertEqual(result["output"]["k"], 3)
        self.assertEqual(result["output"]["order"], 0)

    def test_try_fast_gold_grab_reports_miss_and_path_fail(self) -> None:
        miss = _basic_input(round_index=1)
        self.assertEqual(try_fast_gold_grab(miss, 12)["status"], "miss_no_target")

        blocked = _basic_input(round_index=1)
        blocked.my_units = [(8, 8), (16, 16)]
        blocked.grid[7][8] = -1
        blocked.grid[8][7] = -1
        blocked.grid[8][9] = -1
        blocked.grid[9][8] = -1
        blocked.grid[6][8] = 20

        result = try_fast_gold_grab(blocked, 12)

        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "path_fail")
        self.assertEqual(result["target"], (6, 8))

    def test_try_fast_gold_grab_ignores_current_cell_gold(self) -> None:
        center_only = _basic_input(round_index=1)
        center_only.my_units = [(8, 8), (16, 16)]
        center_only.grid[8][8] = 30

        self.assertEqual(try_fast_gold_grab(center_only, 12)["status"], "miss_no_target")

        with_neighbor = _basic_input(round_index=1)
        with_neighbor.my_units = [(8, 8), (16, 16)]
        with_neighbor.grid[8][8] = 30
        with_neighbor.grid[8][9] = 20

        result = try_fast_gold_grab(with_neighbor, 12)

        self.assertTrue(result["success"])
        self.assertEqual(result["target"], (8, 9))
        self.assertGreater(result["action_count"], 0)

    def test_simulate_known_gold_pickups_counts_path_and_bounce(self) -> None:
        game_input = _basic_input(round_index=0)
        game_input.my_units = [(8, 8), (16, 16)]
        game_input.grid[7][8] = 10
        game_input.grid[6][8] = 20
        game_input.grid[5][8] = 30
        output = GameOutput(
            actions=(int(Action.UP), int(Action.UP), int(Action.UP), int(Action.DOWN), int(Action.STAY), int(Action.STAY)),
            k=4,
            order=0,
            vp=0,
        )

        self.assertEqual(simulate_known_gold_pickups(game_input, output, 0), 45)
        self.assertEqual(infer_fast_role(game_input, output), 0)

    def test_runtime_belief_updates_on_pending_backfill_before_neural_features(self) -> None:
        runtime = FastRuntimeState(player_id=1)
        runtime.set_next_threshold(12)
        fast_input = _basic_input(round_index=1)
        fast_input.my_units = [(8, 8), (16, 16)]
        fast_input.my_units_gold = (100, 200)
        fast_input.grid[7][8] = 20

        fast = runtime.try_fast(fast_input)
        self.assertTrue(fast["success"])
        self.assertTrue(runtime.pending_valid())

        now = _basic_input(round_index=2)
        now.my_units = [(7, 8), (16, 16)]
        now.my_units_gold = (113, 200)
        prepared = runtime.prepare_neural(now)

        self.assertFalse(runtime.pending_valid())
        self.assertEqual(tuple(round(value, 6) for value in prepared["fast_scalars"]), (0.787037, 0.3))
        diagnostics = runtime.diagnostics()
        self.assertEqual(diagnostics["fast_success"], 1)
        self.assertEqual(diagnostics["fast_effective_updates"], 1)
        self.assertEqual(diagnostics["fast_expected_gain_sum"], 18.0)
        self.assertEqual(diagnostics["fast_actual_delta_sum"], 13.0)
        self.assertEqual(diagnostics["one_step_fast_delta_sum"], -5.0)

    def test_pending_backfill_matches_full_observe_commit_for_feature_state(self) -> None:
        runtime = FastRuntimeState(player_id=1)
        original = FeatureExtractor(player_id=1)

        first = _feature_rich_input(round_index=0)
        fast_round = _feature_rich_input(round_index=1)
        fast_round.grid[7][8] = 20
        now = _feature_rich_input(round_index=2)

        first_runtime = runtime.prepare_neural(first)["actor_features"]
        first_original = original.observe(first)
        self.assertTrue(_same_features(first_runtime, first_original))
        stay = GameOutput(actions=(4, 4, 4, 4, 4, 4), k=0, order=0, vp=0)
        runtime.commit_neural(stay)
        original.commit_action(stay)

        runtime.set_next_threshold(12)
        fast = runtime.try_fast(fast_round)
        self.assertTrue(fast["success"])
        fast_output = _output_from_dict(fast["output"])
        original.observe(fast_round)
        original.commit_action(fast_output)

        restored = runtime.prepare_neural(now)["actor_features"]
        expected = original.observe(now)

        self.assertTrue(_same_features(restored, expected))


def _basic_input(round_index: int) -> GameInput:
    return GameInput(
        round=round_index,
        grid=[[0 for _ in range(17)] for _ in range(17)],
        my_units=[(0, 0), (16, 16)],
        my_units_gold=(0, 0),
        gold_opp=0,
        visible_enemies=[(-1, -1), (-1, -1)],
        visible_npcs=[],
        snapshot_valid=False,
        snapshot=None,
    )


def _feature_rich_input(round_index: int) -> GameInput:
    game_input = _basic_input(round_index)
    game_input.my_units = [(8, 8), (12, 12)]
    game_input.my_units_gold = (50 + round_index, 80 + round_index)
    game_input.gold_opp = 90 + round_index
    game_input.grid = [[-5 for _ in range(17)] for _ in range(17)]
    for row in range(6, 11):
        for col in range(6, 11):
            game_input.grid[row][col] = 0
    game_input.grid[6][6] = -1
    game_input.grid[6][7] = -3
    game_input.grid[0][4] = 16
    game_input.visible_enemies = [(9, 9), (-1, -1)]
    game_input.visible_npcs = [NpcInfo(id=1, row=8, col=9), NpcInfo(id=2, row=8, col=9), NpcInfo(id=3, row=8, col=9)]
    return game_input


def _output_from_dict(payload: dict) -> GameOutput:
    return GameOutput(
        actions=tuple(int(value) for value in payload["actions"]),
        k=int(payload["k"]),
        order=int(payload["order"]),
        vp=int(payload["vp"]),
    )


def _same_features(lhs: dict, rhs: dict) -> bool:
    return bool(np.allclose(lhs["planes"], rhs["planes"], atol=1.0e-6) and np.allclose(lhs["scalars"], rhs["scalars"], atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
