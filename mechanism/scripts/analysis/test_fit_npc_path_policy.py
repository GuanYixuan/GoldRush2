from __future__ import annotations

import unittest

import numpy as np

from mechanism.scripts.analysis import fit_npc_path_policy as fit


class FitNpcPathPolicyTest(unittest.TestCase):
    def test_path_features_apply_dynamic_pickup_and_bomb_removal(self) -> None:
        candidate = fit.PathCandidate(
            actions=(3, 2, 3),
            positions=((8, 9), (8, 8), (8, 9)),
        )

        features = fit.path_features(
            start_pos=(8, 8),
            candidate=candidate,
            gold_remaining={(8, 9): 10},
            bombs_remaining={(8, 9)},
            bomb_trapped_start=False,
        )

        self.assertEqual((0.9, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0), features)

    def test_path_features_can_use_has_pickup_mode(self) -> None:
        candidate = fit.PathCandidate(
            actions=(3, 2, 3),
            positions=((8, 9), (8, 8), (8, 9)),
        )

        features = fit.path_features(
            start_pos=(8, 8),
            candidate=candidate,
            gold_remaining={(8, 9): 10},
            bombs_remaining=set(),
            bomb_trapped_start=False,
            pickup_mode="has",
        )

        self.assertEqual(0.9, features[0])
        self.assertEqual(1.0, features[1])

    def test_path_features_can_use_static_pickup_count_mode(self) -> None:
        candidate = fit.PathCandidate(
            actions=(3, 2, 3),
            positions=((8, 9), (8, 8), (8, 9)),
        )

        features = fit.path_features(
            start_pos=(8, 8),
            candidate=candidate,
            gold_remaining={(8, 9): 1},
            bombs_remaining=set(),
            bomb_trapped_start=False,
            pickup_mode="static-count",
        )

        self.assertEqual(0.1, features[0])
        self.assertEqual(2.0, features[1])

    def test_bomb_trapped_stay_feature(self) -> None:
        maps = tuple(tuple(0 for _ in range(17)) for _ in range(17))
        start_pos = (8, 8)
        bombs = {(7, 8), (9, 8), (8, 7), (8, 9)}
        candidate = fit.PathCandidate(
            actions=(4, 4, 4),
            positions=(start_pos, start_pos, start_pos),
        )

        self.assertTrue(fit.all_legal_neighbors_are_bombs(start_pos, maps, bombs))
        features = fit.path_features(
            start_pos=start_pos,
            candidate=candidate,
            gold_remaining={},
            bombs_remaining=bombs,
            bomb_trapped_start=True,
        )

        self.assertEqual(1.0, features[7])

    def test_pair_same_and_sandwich_features(self) -> None:
        candidate = fit.PathCandidate(
            actions=(0, 1, 0),
            positions=((7, 8), (8, 8), (7, 8)),
        )

        features = fit.path_features(
            start_pos=(8, 8),
            candidate=candidate,
            gold_remaining={},
            bombs_remaining=set(),
            bomb_trapped_start=False,
        )

        self.assertEqual(0.0, features[8])
        self.assertEqual(0.0, features[9])
        self.assertEqual(1.0, features[10])

    def test_fit_model_learns_positive_reward_weight(self) -> None:
        samples = [
            fit.Sample(
                features=np.asarray(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                ),
                actual_index=0,
                split="train",
            )
            for _ in range(20)
        ]

        result = fit.fit_model(samples, [0], l2=1e-3, maxiter=50)

        self.assertTrue(result["success"])
        self.assertGreater(result["weights"][0], 0.0)

    def test_static_pickup_mode_uses_m4e_model_name(self) -> None:
        model_features = fit.model_features_for_mode("static-count")

        self.assertIn("M4e_m4a_static_pickup_center", model_features)
        self.assertIn("M5g_m5c_static_pickup_center", model_features)
        self.assertNotIn("M4d_m4a_pickup_center", model_features)


if __name__ == "__main__":
    unittest.main()
