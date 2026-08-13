from __future__ import annotations

import unittest

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepMechanisms
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from training.models import GoldRushPolicyNetwork, PolicyNetworkConfig
from training.opponents import OpponentSpec
from training.rl import EvalTask, ParallelEvalConfig, ParallelEvalPool, SingleAgentEnvConfig, evaluate_parallel
from training.rl.eval_mp.scheduler import _request_uniform_row
from training.rl.eval_mp.types import EvalFeatureRequest


class ParallelEvalTests(unittest.TestCase):
    def test_evaluate_parallel_returns_episode_summaries_and_metrics(self) -> None:
        model = _small_model()
        tasks = (
            EvalTask(
                task_id="eval-a",
                seed=5,
                map_id=1,
                opponent_spec=_stay_opponent_spec(),
                agent_player_id=1,
                round_count=1,
                setting="smoke",
                policy_sample_key="eval-a",
                policy_sample_seed=123,
            ),
            EvalTask(
                task_id="eval-b",
                seed=6,
                map_id=1,
                opponent_spec=_stay_opponent_spec(),
                agent_player_id=2,
                round_count=1,
                setting="smoke",
                policy_sample_key="eval-b",
                policy_sample_seed=123,
            ),
        )

        summaries, stats = evaluate_parallel(
            model,
            tasks,
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            device="cpu",
            config=ParallelEvalConfig(num_workers=2, max_inference_batch_size=4, inference_timeout_ms=1.0),
        )

        self.assertEqual([summary.task_id for summary in summaries], ["eval-a", "eval-b"])
        self.assertEqual([summary.episode_length for summary in summaries], [1, 1])
        self.assertEqual([summary.round_count for summary in summaries], [1, 1])
        self.assertEqual({summary.agent_player_id for summary in summaries}, {1, 2})
        self.assertEqual({summary.opponent for summary in summaries}, {"python:stay:"})
        self.assertEqual(stats["eval_mode"], "parallel")
        self.assertFalse(stats["eval_deterministic"])
        self.assertEqual(stats["eval_task_count"], 2)
        self.assertEqual(stats["eval_episode_started"], 2)
        self.assertEqual(stats["eval_episode_count"], 2)
        self.assertGreaterEqual(stats["eval_feature_batches"], 1)
        self.assertGreaterEqual(stats["eval_worker_ready_count"], 2)
        self.assertFalse(stats["eval_worker_pool_reused"])
        self.assertEqual(stats["eval_worker_profile_episode_count"], 2)
        self.assertEqual(stats["eval_worker_profile_transition_count"], 2)
        self.assertEqual(stats["eval_worker_sum_steps"], 2)
        self.assertEqual(stats["eval_episode_count"], 2)
        self.assertIn("eval_mean_agent_net_gold", stats)
        self.assertIn("eval_win_rate", stats)
        self.assertIn("eval_by_setting", stats)

    def test_parallel_eval_pool_reuses_workers(self) -> None:
        model = _small_model()
        env_config = SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec())

        with ParallelEvalPool(
            env_config=env_config,
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            config=ParallelEvalConfig(num_workers=2, max_inference_batch_size=4, inference_timeout_ms=1.0),
        ) as pool:
            first, first_stats = pool.evaluate(model, _tasks(seed=5), device="cpu")
            second, second_stats = pool.evaluate(model, _tasks(seed=6), device="cpu")

        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertFalse(first_stats["eval_worker_pool_reused"])
        self.assertTrue(second_stats["eval_worker_pool_reused"])
        self.assertGreater(first_stats["eval_worker_all_ready_ms"], 0.0)
        self.assertEqual(second_stats["eval_worker_startup_ms"], 0.0)
        self.assertEqual(second_stats["eval_worker_all_ready_ms"], 0.0)
        self.assertEqual(second_stats["eval_worker_ready_count"], 2)

    def test_crn_uniform_row_covers_threshold_sample(self) -> None:
        request = EvalFeatureRequest(
            worker_id=0,
            eval_id="eval",
            task_id="task",
            request_id="request",
            round_index=3,
            feature_slot=0,
            policy_sample_key="policy",
            policy_sample_seed=123,
        )

        first = _request_uniform_row(request)
        second = _request_uniform_row(request)

        self.assertEqual(len(first), 9)
        self.assertEqual(first, second)
        self.assertTrue(all(0.0 < value < 1.0 for value in first))


def _tasks(seed: int) -> tuple[EvalTask, EvalTask]:
    return (
        EvalTask(
            task_id=f"eval-{seed}-p1",
            seed=seed,
            map_id=1,
            opponent_spec=_stay_opponent_spec(),
            agent_player_id=1,
            round_count=1,
        ),
        EvalTask(
            task_id=f"eval-{seed}-p2",
            seed=seed,
            map_id=1,
            opponent_spec=_stay_opponent_spec(),
            agent_player_id=2,
            round_count=1,
        ),
    )


def _small_model() -> GoldRushPolicyNetwork:
    return GoldRushPolicyNetwork(
        PolicyNetworkConfig(
            width=16,
            residual_blocks=1,
            se_reduction=4,
            scalar_hidden=(16, 16),
            actor_hidden=32,
            critic_hidden=(32, 16),
        )
    )


def _one_round_episode() -> EpisodeConfig:
    return EpisodeConfig(rules=RulesConfig(round_count=1, snapshot_period=1), seed=7, map_id=1)


def _quiet_mechanisms() -> RoundStepMechanisms:
    return RoundStepMechanisms(
        center_gold=CenterGoldGenerator(CenterGoldConfig(center_a=0.0)),
        outer_gold=OuterGoldGenerator(OuterGoldConfig(first_round_offset_weights=((999, 1),))),
        bomb_refresher=BernoulliBombRefresher(BombConfig(0.0)),
    )


def _stay_opponent_spec() -> OpponentSpec:
    return OpponentSpec(kind="python", name="stay")


if __name__ == "__main__":
    unittest.main()
