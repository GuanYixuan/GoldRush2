from __future__ import annotations

import unittest

from policy_runtime import FeatureExtractor, channel_names, scalar_names
from simulator.observation.sdk import GameInput, NpcInfo
from simulator.types import Action, GameOutput


class FeatureExtractorTests(unittest.TestCase):
    def test_observe_outputs_expected_shapes(self) -> None:
        extractor = FeatureExtractor(player_id=1)

        features = extractor.observe(_basic_input(round_index=0))

        self.assertEqual(features["planes"].shape, (11, 17, 17))
        self.assertEqual(features["scalars"].shape, (14,))
        self.assertEqual(tuple(features["channel_names"]), tuple(channel_names()))
        self.assertEqual(tuple(features["scalar_names"]), tuple(scalar_names()))

    def test_visible_cells_update_explored_and_last_seen(self) -> None:
        extractor = FeatureExtractor(player_id=1)
        first = _basic_input(round_index=0)
        first.grid[0][1] = 7

        features0 = extractor.observe(first)
        second = _fog_input(round_index=1)
        features1 = extractor.observe(second)

        self.assertEqual(features0["planes"][0, 0, 1], 1.0)
        self.assertEqual(features0["planes"][1, 0, 1], 1.0)
        self.assertEqual(features0["planes"][5, 0, 1], 0.07)
        self.assertEqual(features1["planes"][0, 0, 1], 0.0)
        self.assertEqual(features1["planes"][1, 0, 1], 1.0)
        self.assertAlmostEqual(float(features1["planes"][10, 0, 1]), 1.0 / 500.0)

    def test_commit_action_updates_last_action_only(self) -> None:
        extractor = FeatureExtractor(player_id=1)

        before = extractor.observe(_basic_input(round_index=0))
        extractor.commit_action(_right_output())
        after = extractor.observe(_basic_input(round_index=1))

        self.assertEqual(before["scalars"][4], 0.0)
        self.assertEqual(after["scalars"][4], 1.0)
        self.assertAlmostEqual(float(after["scalars"][5]), 1.0 / 6.0)
        self.assertEqual(after["scalars"][6], 0.0)
        self.assertEqual(after["scalars"][8], 0.75)

    def test_reset_clears_memory(self) -> None:
        extractor = FeatureExtractor(player_id=1)
        first = _basic_input(round_index=0)
        first.grid[0][1] = 7

        extractor.observe(first)
        extractor.reset(1)
        features = extractor.observe(_fog_input(round_index=0))

        self.assertEqual(features["planes"][1, 0, 1], 0.0)

    def test_round_regression_fails_fast(self) -> None:
        extractor = FeatureExtractor(player_id=1)

        extractor.observe(_basic_input(round_index=2))

        with self.assertRaisesRegex(Exception, "round regression"):
            extractor.observe(_basic_input(round_index=1))

    def test_commit_action_does_not_move_units_in_memory(self) -> None:
        extractor = FeatureExtractor(player_id=1)

        extractor.observe(_basic_input(round_index=0))
        extractor.commit_action(_right_output())
        features = extractor.observe(_basic_input(round_index=1))

        self.assertEqual(features["planes"][6, 0, 0], 1.0)
        self.assertEqual(features["planes"][6, 0, 1], 0.0)

    def test_visible_entities_are_encoded(self) -> None:
        game_input = _basic_input(round_index=0)
        game_input.visible_enemies = [(1, 1), (-1, -1)]
        game_input.visible_npcs = [NpcInfo(id=-1, row=2, col=2)]

        features = FeatureExtractor(player_id=1).observe(game_input)

        self.assertEqual(features["planes"][8, 1, 1], 1.0)
        self.assertEqual(features["planes"][9, 2, 2], 1.0)


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


def _right_output() -> GameOutput:
    return GameOutput(actions=(int(Action.RIGHT), int(Action.STAY), int(Action.STAY), int(Action.STAY), int(Action.STAY), int(Action.STAY)), k=1, order=0, vp=0)


if __name__ == "__main__":
    unittest.main()
