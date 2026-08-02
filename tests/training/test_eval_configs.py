from __future__ import annotations

import unittest
from collections import Counter

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepMechanisms
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from simulator.types import Action, GameOutput
from training.configs.eval import ANCHOR_EVAL_MAP_IDS, ANCHOR_EVAL_OPPONENT_SPECS, ANCHOR_EVAL_SEEDS, anchor_eval_config
from training.rl import evaluate_policy


class EvalConfigTests(unittest.TestCase):
    def test_anchor_eval_v1_has_expected_case_matrix(self) -> None:
        config = anchor_eval_config()

        self.assertEqual(len(config.cases), 60)
        self.assertEqual(ANCHOR_EVAL_SEEDS, tuple(range(2026080200, 2026080210)))
        self.assertEqual(ANCHOR_EVAL_MAP_IDS, (1, 2, 3))
        self.assertEqual([spec.name for spec in ANCHOR_EVAL_OPPONENT_SPECS], ["greedy_visible_gold", "fast_probe_v3_like"])
        self.assertEqual(Counter(case.map_id for case in config.cases), {1: 20, 2: 20, 3: 20})
        self.assertEqual(Counter(case.opponent_spec.name for case in config.cases), {"greedy_visible_gold": 30, "fast_probe_v3_like": 30})
        self.assertTrue(all(case.tag == "anchor_eval_v1" for case in config.cases))

    def test_anchor_eval_v1_runs_expected_episode_count_with_short_episode(self) -> None:
        config = anchor_eval_config(
            episode=EpisodeConfig(rules=RulesConfig(round_count=1, snapshot_period=1), seed=1, map_id=1),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )

        result = evaluate_policy(_stay_policy, config)

        self.assertEqual(result.metrics["case_count"], 60)
        self.assertEqual(result.batch.episode_count, 120)
        self.assertEqual(result.batch.transition_count, 120)
        self.assertEqual(result.metrics["map_ids"], (1, 2, 3))
        self.assertEqual(result.metrics["opponent_count"], 2)


def _quiet_mechanisms() -> RoundStepMechanisms:
    return RoundStepMechanisms(
        center_gold=CenterGoldGenerator(CenterGoldConfig(center_a=0.0)),
        outer_gold=OuterGoldGenerator(OuterGoldConfig(first_round_offset_weights=((999, 1),))),
        bomb_refresher=BernoulliBombRefresher(BombConfig(0.0)),
    )


def _stay_policy(_observation) -> GameOutput:
    return GameOutput(actions=(int(Action.STAY),) * 6, k=3, order=0, vp=0)


if __name__ == "__main__":
    unittest.main()
