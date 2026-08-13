from __future__ import annotations

import unittest

import torch

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepMechanisms
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from training.models import INITIAL_FAST_SCALARS, GoldRushPolicyNetwork, PolicyNetworkConfig
from training.opponents import OpponentSpec
from training.rl import (
    BatchRolloutSampler,
    MultiprocessRolloutConfig,
    MultiprocessRolloutPool,
    PpoConfig,
    SingleAgentEnvConfig,
    collect_multiprocess_ppo_rollouts,
    ppo_update,
)
from training.rl.rollout_mp.worker import _add_event_counts, _empty_event_counts, transition_info
from training.rl.rollout_mp.shared_memory import (
    attach_feature_shared_memory,
    attach_transition_shared_memory,
    create_feature_shared_memory,
    create_transition_shared_memory,
)


class MultiprocessRolloutTests(unittest.TestCase):
    def test_collect_multiprocess_rollouts_can_feed_gae_and_update(self) -> None:
        model = _small_model()
        sampler = BatchRolloutSampler(
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )

        batch, stats = collect_multiprocess_ppo_rollouts(
            model,
            sampler,
            pair_count=1,
            seed=5,
            map_ids=(1,),
            opponent_specs=(_stay_opponent_spec(),),
            device="cpu",
            config=MultiprocessRolloutConfig(num_workers=2, max_inference_batch_size=4, inference_timeout_ms=1.0),
        )
        batch = batch.compute_gae(gamma=1.0, gae_lambda=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        update_stats = ppo_update(
            model,
            optimizer,
            batch,
            PpoConfig(update_epochs=1, minibatch_size=batch.transition_count, target_joint_kl=None),
        )

        self.assertEqual(batch.transition_count, 2)
        self.assertEqual(tuple(batch.spatial_planes.shape), (2, 43, 17, 17))
        self.assertEqual(tuple(batch.scalars.shape), (2, 10))
        self.assertEqual(tuple(batch.fast_scalars.shape), (2, 2))
        self.assertEqual(tuple(batch.critic_planes.shape), (2, 26, 17, 17))
        self.assertEqual(tuple(batch.critic_scalars.shape), (2, 17))
        self.assertTrue(torch.allclose(batch.fast_scalars, torch.tensor(INITIAL_FAST_SCALARS).expand(2, -1)))
        self.assertEqual(tuple(batch.threshold_raw.shape), (2,))
        self.assertEqual(tuple(batch.threshold_int.shape), (2,))
        self.assertTrue(torch.isfinite(batch.threshold_raw).all().item())
        self.assertTrue(((batch.threshold_int >= 4) & (batch.threshold_int <= 30)).all().item())
        self.assertEqual(int(batch.dones.sum().item()), 2)
        self.assertEqual(batch.episode_ids[0], "pair-000000-seed-5-first")
        self.assertEqual(batch.episode_ids[1], "pair-000000-seed-5-second")
        self.assertEqual(set(batch.infos[0]), {"scores", "events", "game_result"})
        self.assertEqual(set(batch.infos[1]), {"scores", "events", "game_result"})
        self.assertEqual(stats["rollout_mode"], "multiprocess")
        self.assertEqual(stats["first_started"], 1)
        self.assertEqual(stats["second_started"], 1)
        self.assertEqual(stats["first_episodes"], 1)
        self.assertEqual(stats["second_episodes"], 1)
        self.assertGreaterEqual(stats["feature_batches"], 1)
        self.assertGreaterEqual(stats["worker_startup_ms"], 0.0)
        self.assertGreaterEqual(stats["scheduler_ms"], 0.0)
        self.assertGreaterEqual(stats["batch_assembly_ms"], 0.0)
        self.assertEqual(stats["worker_ready_count"], 2)
        self.assertIsNotNone(stats["worker_all_ready_ms"])
        self.assertGreaterEqual(stats["scheduler_queue_get_ms"], 0.0)
        self.assertGreaterEqual(stats["scheduler_message_handle_ms"], 0.0)
        self.assertGreaterEqual(stats["scheduler_task_dispatch_ms"], 0.0)
        self.assertGreaterEqual(stats["scheduler_payload_sort_ms"], 0.0)
        self.assertGreaterEqual(stats["inference_stack_ms"], 0.0)
        self.assertGreaterEqual(stats["inference_model_sample_ms"], 0.0)
        self.assertGreaterEqual(stats["inference_action_send_ms"], 0.0)
        self.assertGreaterEqual(stats["inference_total_with_action_send_ms"], stats["inference_ms"])
        self.assertFalse(stats["worker_pool_reused"])
        self.assertGreaterEqual(stats["worker_configure_ms"], 0.0)
        self.assertGreaterEqual(stats["worker_release_ms"], 0.0)
        self.assertEqual(stats["worker_profile_episode_count"], 2)
        self.assertEqual(stats["worker_profile_transition_count"], 2)
        self.assertEqual(stats["worker_sum_steps"], 2)
        self.assertGreaterEqual(stats["worker_sum_episode_wall_ms"], 0.0)
        self.assertGreaterEqual(stats["worker_per_transition_action_wait_ms"], 0.0)
        self.assertGreaterEqual(stats["scheduler_accounted_ms"], 0.0)
        self.assertEqual(update_stats.update_count, 1)

    def test_multiprocess_rollout_pool_reuses_workers(self) -> None:
        model = _small_model()
        sampler = BatchRolloutSampler(
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )

        with MultiprocessRolloutPool(
            sampler,
            MultiprocessRolloutConfig(num_workers=2, max_inference_batch_size=4, inference_timeout_ms=1.0),
        ) as pool:
            first, first_stats = pool.collect(
                model,
                pair_count=1,
                seed=5,
                map_ids=(1,),
                opponent_specs=(_stay_opponent_spec(),),
                device="cpu",
            )
            second, second_stats = pool.collect(
                model,
                pair_count=1,
                seed=6,
                map_ids=(1,),
                opponent_specs=(_stay_opponent_spec(),),
                device="cpu",
            )

        self.assertEqual(first.transition_count, 2)
        self.assertEqual(second.transition_count, 2)
        self.assertFalse(first_stats["worker_pool_reused"])
        self.assertTrue(second_stats["worker_pool_reused"])
        self.assertGreater(first_stats["worker_all_ready_ms"], 0.0)
        self.assertEqual(second_stats["worker_startup_ms"], 0.0)
        self.assertEqual(second_stats["worker_all_ready_ms"], 0.0)
        self.assertEqual(second_stats["worker_ready_count"], 2)

    def test_training_info_mode_keeps_non_terminal_infos_empty(self) -> None:
        model = _small_model()
        sampler = BatchRolloutSampler(
            env_config=SingleAgentEnvConfig(episode=_two_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )

        batch, _stats = collect_multiprocess_ppo_rollouts(
            model,
            sampler,
            pair_count=1,
            seed=5,
            map_ids=(1,),
            opponent_specs=(_stay_opponent_spec(),),
            device="cpu",
            config=MultiprocessRolloutConfig(num_workers=2, max_inference_batch_size=4, inference_timeout_ms=1.0),
        )

        self.assertEqual(batch.transition_count, 4)
        self.assertEqual(batch.infos[0], {})
        self.assertEqual(set(batch.infos[1]), {"scores", "events", "game_result"})
        self.assertEqual(batch.infos[2], {})
        self.assertEqual(set(batch.infos[3]), {"scores", "events", "game_result"})

    def test_training_info_mode_reports_episode_event_totals_on_terminal_step(self) -> None:
        episode_events = _empty_event_counts()
        _add_event_counts(
            episode_events,
            {
                "pickups": {1: 1, 2: 0},
                "pickup_gold": {1: 4, 2: 0},
                "bomb_triggers": {1: 0, 2: 1},
                "bomb_lost_gold": {1: 0, 2: 3},
                "tramples": {1: 0, 2: 0},
                "trample_penalty": {1: 0, 2: 0},
            },
        )

        non_terminal = transition_info(
            {"events": {"pickups": {1: 999}}, "reward_components": {"total": 0.25}},
            done=False,
            mode="training",
            episode_events=episode_events,
        )

        _add_event_counts(
            episode_events,
            {
                "pickups": {1: 2, 2: 1},
                "pickup_gold": {1: 7, 2: 5},
                "bomb_triggers": {1: 1, 2: 0},
                "bomb_lost_gold": {1: 6, 2: 0},
                "tramples": {1: 1, 2: 0},
                "trample_penalty": {1: 2, 2: 0},
            },
        )
        terminal = transition_info(
            {
                "scores": {"net_gold": {1: 10, 2: 5}},
                "events": {"pickups": {1: 2, 2: 1}},
                "game_result": object(),
            },
            done=True,
            mode="training",
            episode_events=episode_events,
        )

        self.assertEqual(non_terminal, {"reward_components": {"total": 0.25}})
        self.assertEqual(terminal["events"]["pickups"], {1: 3, 2: 1})
        self.assertEqual(terminal["events"]["pickup_gold"], {1: 11, 2: 5})
        self.assertEqual(terminal["events"]["bomb_triggers"], {1: 1, 2: 1})
        self.assertEqual(terminal["events"]["bomb_lost_gold"], {1: 6, 2: 3})
        self.assertEqual(terminal["events"]["tramples"], {1: 1, 2: 0})
        self.assertEqual(terminal["events"]["trample_penalty"], {1: 2, 2: 0})

    def test_debug_info_mode_keeps_full_step_info(self) -> None:
        model = _small_model()
        sampler = BatchRolloutSampler(
            env_config=SingleAgentEnvConfig(episode=_two_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )

        batch, _stats = collect_multiprocess_ppo_rollouts(
            model,
            sampler,
            pair_count=1,
            seed=5,
            map_ids=(1,),
            opponent_specs=(_stay_opponent_spec(),),
            device="cpu",
            config=MultiprocessRolloutConfig(
                num_workers=2,
                max_inference_batch_size=4,
                inference_timeout_ms=1.0,
                transition_info_mode="debug",
            ),
        )

        self.assertEqual(batch.transition_count, 4)
        self.assertIn("trace", batch.infos[0])
        self.assertIn("agent_output", batch.infos[0])
        self.assertIn("opponent_output", batch.infos[0])

    def test_rollout_shared_memory_roundtrip_includes_fast_threshold_fields(self) -> None:
        feature_shared = create_feature_shared_memory(num_workers=1)
        transition_shared = create_transition_shared_memory(episode_count=1, round_count=2)
        try:
            feature_shared.fast_scalars[0, :] = (0.75, 0.5)
            transition_shared.fast_scalars[0, 1, :] = (0.6, 0.4)
            transition_shared.threshold_raw[0, 1] = -0.25
            transition_shared.threshold_int[0, 1] = 12

            attached_feature = attach_feature_shared_memory(feature_shared.config())
            attached_transition = attach_transition_shared_memory(transition_shared.config())
            try:
                self.assertEqual(tuple(attached_feature.fast_scalars.shape), (1, 2))
                self.assertEqual(tuple(attached_transition.fast_scalars.shape), (1, 2, 2))
                self.assertEqual(tuple(attached_transition.threshold_raw.shape), (1, 2))
                self.assertEqual(tuple(attached_transition.threshold_int.shape), (1, 2))
                self.assertTrue(torch.allclose(torch.as_tensor(attached_feature.fast_scalars[0]), torch.tensor([0.75, 0.5])))
                self.assertTrue(torch.allclose(torch.as_tensor(attached_transition.fast_scalars[0, 1]), torch.tensor([0.6, 0.4])))
                self.assertAlmostEqual(float(attached_transition.threshold_raw[0, 1]), -0.25)
                self.assertEqual(int(attached_transition.threshold_int[0, 1]), 12)
            finally:
                attached_feature.close()
                attached_transition.close()
        finally:
            feature_shared.close()
            feature_shared.unlink()
            transition_shared.close()
            transition_shared.unlink()


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


def _two_round_episode() -> EpisodeConfig:
    return EpisodeConfig(rules=RulesConfig(round_count=2, snapshot_period=1), seed=7, map_id=1)


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
