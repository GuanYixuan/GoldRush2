from __future__ import annotations

import unittest

import torch

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepMechanisms
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from training.models import GoldRushPolicyNetwork, PolicyNetworkConfig
from training.opponents import OpponentSpec
from training.rl import (
    BatchRolloutSampler,
    MultiprocessRolloutConfig,
    PpoConfig,
    SingleAgentEnvConfig,
    collect_multiprocess_ppo_rollouts,
    ppo_update,
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
        self.assertEqual(int(batch.dones.sum().item()), 2)
        self.assertEqual(batch.episode_ids[0], "pair-000000-seed-5-first")
        self.assertEqual(batch.episode_ids[1], "pair-000000-seed-5-second")
        self.assertEqual(stats["rollout_mode"], "multiprocess")
        self.assertEqual(stats["first_started"], 1)
        self.assertEqual(stats["second_started"], 1)
        self.assertEqual(stats["first_episodes"], 1)
        self.assertEqual(stats["second_episodes"], 1)
        self.assertGreaterEqual(stats["feature_batches"], 1)
        self.assertEqual(update_stats.update_count, 1)


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
