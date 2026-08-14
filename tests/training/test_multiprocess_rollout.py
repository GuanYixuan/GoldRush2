from __future__ import annotations

import queue
import unittest
from unittest.mock import patch

import torch

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepMechanisms
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from simulator.observation.sdk import GameInput
from simulator.types import GameOutput
from training.models import INITIAL_FAST_SCALARS, GoldRushPolicyNetwork, PolicyNetworkConfig
from training.opponents import OpponentSpec
from training.rl import (
    BatchRolloutSampler,
    MultiprocessRolloutConfig,
    MultiprocessRolloutPool,
    PpoConfig,
    SingleAgentEnvConfig,
    ResetResult,
    StepResult,
    collect_multiprocess_ppo_rollouts,
    ppo_update,
)
from training.rl.ppo_buffer import FAST_STATUS_SUCCESS
from training.rl.rollout_mp.types import EpisodeTask
from training.rl.rollout_mp.worker import _add_event_counts, _empty_event_counts, run_worker_episode, transition_info
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
        self.assertIn("scores", batch.infos[0])
        self.assertIn("events", batch.infos[0])
        self.assertIn("game_result", batch.infos[0])
        self.assertIn("latent_first_rate", batch.infos[0])
        self.assertIn("scores", batch.infos[1])
        self.assertIn("events", batch.infos[1])
        self.assertIn("game_result", batch.infos[1])
        self.assertIn("latent_first_rate", batch.infos[1])
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

    def test_fast_runtime_feature_mode_runs_without_enabling_fast_action_skip(self) -> None:
        model = _small_model()
        sampler = BatchRolloutSampler(
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
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
                enable_fast_runtime_features=True,
            ),
        )

        self.assertEqual(batch.transition_count, 2)
        self.assertTrue(torch.allclose(batch.fast_scalars, torch.tensor(INITIAL_FAST_SCALARS).expand(2, -1)))
        self.assertTrue(((batch.threshold_int >= 4) & (batch.threshold_int <= 30)).all().item())

    def test_training_info_mode_keeps_only_light_non_terminal_infos(self) -> None:
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
        self.assertEqual(
            set(batch.infos[0]),
            {"first_player_id", "agent_decision_mode", "latent_first_rate", "fast_order_sampled", "agent_first"},
        )
        self.assertEqual(batch.infos[0]["agent_decision_mode"], "neural")
        self.assertFalse(batch.infos[0]["fast_order_sampled"])
        self.assertIn("scores", batch.infos[1])
        self.assertIn("events", batch.infos[1])
        self.assertIn("game_result", batch.infos[1])
        self.assertIn("latent_first_rate", batch.infos[1])
        self.assertEqual(
            set(batch.infos[2]),
            {"first_player_id", "agent_decision_mode", "latent_first_rate", "fast_order_sampled", "agent_first"},
        )
        self.assertIn("scores", batch.infos[3])
        self.assertIn("events", batch.infos[3])
        self.assertIn("game_result", batch.infos[3])

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

        step_order_info = {
            "first_player_id": 2,
            "agent_decision_mode": "neural",
            "latent_first_rate": 0.8,
            "fast_order_sampled": False,
            "agent_first": False,
        }
        non_terminal = transition_info(
            {"events": {"pickups": {1: 999}}, "reward_components": {"total": 0.25}, **step_order_info},
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
                **step_order_info,
            },
            done=True,
            mode="training",
            episode_events=episode_events,
        )

        self.assertEqual(non_terminal, {**step_order_info, "reward_components": {"total": 0.25}})
        self.assertEqual(terminal["first_player_id"], 2)
        self.assertEqual(terminal["agent_decision_mode"], "neural")
        self.assertEqual(terminal["latent_first_rate"], 0.8)
        self.assertFalse(terminal["fast_order_sampled"])
        self.assertFalse(terminal["agent_first"])
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

    def test_worker_folds_successful_fast_step_into_previous_policy_transition(self) -> None:
        feature_shared = create_feature_shared_memory(num_workers=1)
        transition_shared = create_transition_shared_memory(episode_count=1, round_count=2)
        command_queue = queue.Queue()
        result_queue = queue.Queue()
        command_queue.put(
            {
                "type": "action_result",
                "rollout_id": "rollout",
                "request_id": "task-round-0000",
                "action": {"actions": (4, 4, 4, 4, 4, 4), "k": 0, "order": 0, "vp": 0},
                "threshold_raw": 0.0,
                "threshold_int": 12,
                "threshold_mu_raw": 0.0,
                "threshold_base_raw": 0.0,
                "threshold_residual_raw": 0.0,
                "old_logprob": -0.5,
                "value": 0.25,
            }
        )
        try:
            with (
                patch("training.rl.rollout_mp.worker.SingleAgentGoldRushEnv", _FastFoldEnv),
                patch("training.rl.rollout_mp.worker.FastRuntimeState", _FastFoldRuntime),
                patch("training.rl.rollout_mp.worker._extract_critic_features", _fake_critic_features),
            ):
                run_worker_episode(
                    0,
                    "rollout",
                    EpisodeTask(
                        task_id="task",
                        pair_id="pair",
                        pair_role="first",
                        seed=1,
                        map_id=1,
                        agent_player_id=1,
                        opponent_spec=_stay_opponent_spec(),
                        transition_slot=0,
                    ),
                    command_queue,
                    result_queue,
                    {
                        "env_config": SingleAgentEnvConfig(episode=_two_round_episode(), opponent_spec=_stay_opponent_spec()),
                        "mechanisms": _quiet_mechanisms(),
                        "map_pool": None,
                        "spawn": SpawnConfig(npc_ids=()),
                        "reward_fn": None,
                        "transition_info_mode": "training",
                        "enable_fast_runtime_features": True,
                        "reward_fold_gamma": 0.5,
                        "round_count": 2,
                    },
                    feature_shared,
                    transition_shared,
                )

            messages = [result_queue.get_nowait(), result_queue.get_nowait(), result_queue.get_nowait()]
            self.assertEqual([message["type"] for message in messages], ["episode_started", "feature_request", "episode_done"])
            done_msg = messages[2]
            self.assertEqual(done_msg["episode_length"], 1)
            self.assertEqual(done_msg["policy_step_count"], 1)
            self.assertEqual(done_msg["env_step_count"], 2)
            self.assertEqual(transition_shared.reward[0, 0], 1.5)
            self.assertEqual(transition_shared.reward_sum[0, 0], 2.75)
            self.assertEqual(int(transition_shared.tau[0, 0]), 2)
            self.assertTrue(bool(transition_shared.fast_success[0, 0]))
            self.assertEqual(int(transition_shared.fast_status[0, 0]), FAST_STATUS_SUCCESS)
            self.assertTrue(bool(transition_shared.done[0, 0]))
            self.assertEqual(transition_shared.round_index[0, 0], 0)
            self.assertEqual(done_msg["infos"][0]["agent_decision_mode"], "fast")
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


class _FastFoldEnv:
    def __init__(self, **_: object) -> None:
        self.steps = 0
        self.round_env = object()

    def reset(
        self,
        *,
        seed: int | None = None,
        map_id: int | None = None,
        agent_player_id: int | None = None,
        opponent_spec: OpponentSpec | None = None,
    ) -> ResetResult:
        return ResetResult(
            observation=_fake_observation(0),
            info={"map_id": 1, "opponent_spec": opponent_spec or _stay_opponent_spec()},
        )

    def step(self, agent_output: GameOutput, *, agent_decision_mode: str = "neural") -> StepResult:
        self.steps += 1
        terminated = self.steps >= 2
        reward = 1.5 if agent_decision_mode == "neural" else 2.5
        return StepResult(
            observation=None if terminated else _fake_observation(self.steps),
            reward=reward,
            terminated=terminated,
            truncated=False,
            info={
                "first_player_id": 1,
                "agent_decision_mode": agent_decision_mode,
                "latent_first_rate": 1.0,
                "fast_order_sampled": agent_decision_mode == "fast",
                "agent_first": agent_decision_mode == "fast",
                "scores": {"net_gold": {1: 4, 2: 0}},
                "events": _empty_event_counts(),
                "game_result": object(),
                "reward_components": {"total": reward},
            },
        )


class _FastFoldRuntime:
    def __init__(self, player_id: int) -> None:
        self.player_id = player_id
        self.threshold = 12

    def prepare_neural(self, observation: GameInput) -> dict:
        return {
            "actor_features": {
                "feature_schema": "goldrush2_feature_v2",
                "planes": torch.zeros(43, 17, 17).numpy(),
                "scalars": torch.zeros(10).numpy(),
            },
            "fast_scalars": tuple(INITIAL_FAST_SCALARS),
        }

    def commit_neural(self, output: GameOutput) -> None:
        return None

    def set_next_threshold(self, threshold_int: int) -> None:
        self.threshold = int(threshold_int)

    def try_fast(self, observation: GameInput) -> dict:
        return {
            "status": "success",
            "output": {"actions": (4, 4, 4, 4, 4, 4), "k": 0, "order": 0, "vp": 0},
        }


def _fake_observation(round_index: int) -> GameInput:
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


def _fake_critic_features(*_: object, **__: object) -> dict:
    return {
        "planes": torch.zeros(26, 17, 17).numpy(),
        "scalars": torch.zeros(17).numpy(),
    }


if __name__ == "__main__":
    unittest.main()
