from __future__ import annotations

import unittest

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepMechanisms
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from training.models import GoldRushPolicyNetwork, PolicyNetworkConfig
from training.opponents import OpponentSpec
from training.rl import EvalTask, FastOrderConfig, ParallelEvalConfig, ParallelEvalPool, SingleAgentEnvConfig, evaluate_parallel
from training.rl.eval_mp import evaluate_fast_runtime_crn_pair
from training.rl.eval_mp.scheduler import _request_uniform_row
from training.rl.eval_mp.types import EvalFeatureRequest
from training.rl.eval_mp.worker import _env_config_for_task


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
        self.assertIn("eval_threshold_raw_mean", stats)
        self.assertIn("eval_threshold_int_p50", stats)
        self.assertIn("eval_threshold_base_raw_mean", stats)
        self.assertIn("eval_inference_threshold_residual_raw_mean", stats)
        self.assertIn("eval_fast_success_per_episode", stats)
        self.assertEqual(stats["eval_fast_success_per_episode"], 0.0)
        self.assertIn("threshold_raw_mean", summaries[0].extra)
        self.assertIn("threshold_mu_raw_mean", summaries[0].extra)
        self.assertIn("fast_success_per_episode", summaries[0].extra)

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

    def test_parallel_eval_fast_runtime_feature_mode_runs(self) -> None:
        model = _small_model()
        summaries, stats = evaluate_parallel(
            model,
            _tasks(seed=7),
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            device="cpu",
            config=ParallelEvalConfig(
                num_workers=2,
                max_inference_batch_size=4,
                inference_timeout_ms=1.0,
                enable_fast_runtime_features=True,
            ),
        )

        self.assertEqual(len(summaries), 2)
        self.assertEqual(stats["eval_episode_count"], 2)
        self.assertIn("eval_fast_miss_no_target_per_episode", stats)
        self.assertIn("eval_p_fast_effective_mean", stats)
        self.assertIn("eval_inference_threshold_entropy", stats)
        for summary in summaries:
            self.assertIn("fast_miss_no_target_per_episode", summary.extra)
            self.assertIn("neural_fallback_per_episode", summary.extra)
            self.assertIn("p_fast_effective_mean", summary.extra)
            self.assertGreaterEqual(summary.extra["threshold_int_mean"], 4.0)
            self.assertLessEqual(summary.extra["threshold_int_mean"], 30.0)

    def test_parallel_eval_fixed_threshold_overrides_runtime_threshold(self) -> None:
        model = _small_model()
        summaries, stats = evaluate_parallel(
            model,
            _tasks(seed=8),
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            device="cpu",
            config=ParallelEvalConfig(
                num_workers=2,
                max_inference_batch_size=4,
                inference_timeout_ms=1.0,
                enable_fast_runtime_features=True,
                fixed_threshold_int=8,
            ),
        )

        self.assertEqual(stats["eval_fixed_threshold_int"], 8)
        self.assertEqual(stats["eval_threshold_int_mean"], 8.0)
        self.assertEqual(stats["eval_threshold_int_p50"], 8.0)
        self.assertEqual(stats["eval_inference_threshold_int_mean"], 8.0)
        for summary in summaries:
            self.assertEqual(summary.extra["threshold_int_mean"], 8.0)

    def test_eval_task_env_config_preserves_fast_order(self) -> None:
        fast_order = FastOrderConfig(latent_first_rate_mixture=((1.0, 0.5, 0.5),))
        base = SingleAgentEnvConfig(
            episode=_one_round_episode(),
            opponent_spec=_stay_opponent_spec(),
            fast_order=fast_order,
        )
        same_round = _env_config_for_task(
            {"env_config": base},
            EvalTask(
                task_id="same",
                seed=1,
                map_id=1,
                opponent_spec=_stay_opponent_spec(),
                agent_player_id=1,
                round_count=1,
            ),
        )
        different_round = _env_config_for_task(
            {"env_config": base},
            EvalTask(
                task_id="different",
                seed=1,
                map_id=1,
                opponent_spec=_stay_opponent_spec(),
                agent_player_id=1,
                round_count=2,
            ),
        )

        self.assertEqual(same_round.fast_order, fast_order)
        self.assertEqual(different_round.fast_order, fast_order)

    def test_fast_runtime_crn_pair_reports_paired_deltas(self) -> None:
        model = _small_model()
        result = evaluate_fast_runtime_crn_pair(
            model,
            _tasks(seed=9),
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            device="cpu",
            config=ParallelEvalConfig(num_workers=2, max_inference_batch_size=4, inference_timeout_ms=1.0),
            policy_sample_seed=77,
        )

        self.assertEqual([summary.task_id for summary in result.fast_off_summaries], ["eval-9-p1", "eval-9-p2"])
        self.assertEqual([summary.task_id for summary in result.fast_on_summaries], ["eval-9-p1", "eval-9-p2"])
        self.assertEqual(result.fast_off_stats["eval_episode_count"], 2)
        self.assertEqual(result.fast_on_stats["eval_episode_count"], 2)
        self.assertEqual(result.paired_stats["paired_episode_count"], 2)
        self.assertIn("paired_mean_agent_net_gold_delta", result.paired_stats)
        self.assertFalse(result.fast_off_stats["eval_worker_pool_reused"])
        self.assertFalse(result.fast_on_stats["eval_worker_pool_reused"])

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
