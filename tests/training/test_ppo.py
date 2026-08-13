from __future__ import annotations

import unittest
from dataclasses import replace

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
    PpoBatch,
    PpoConfig,
    PpoTransition,
    SingleAgentEnvConfig,
    collect_ppo_rollouts,
    evaluate_actions,
    ppo_update,
)


class PpoTests(unittest.TestCase):
    def test_evaluate_actions_returns_expected_shapes(self) -> None:
        model = _small_model()
        batch = _batch_from_model(model).compute_gae(gamma=1.0, gae_lambda=1.0)

        evaluation = evaluate_actions(model, batch)

        self.assertEqual(tuple(evaluation.logprob.shape), (batch.transition_count,))
        self.assertEqual(tuple(evaluation.value.shape), (batch.transition_count,))
        self.assertEqual(tuple(evaluation.normalized_entropy.shape), (batch.transition_count,))
        self.assertTrue(torch.isfinite(evaluation.logprob).all().item())

    def test_ppo_update_runs_optimizer_step(self) -> None:
        model = _small_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = _batch_from_model(model).compute_gae(gamma=1.0, gae_lambda=1.0)
        before = _parameter_vector(model)

        stats = ppo_update(
            model,
            optimizer,
            batch,
            PpoConfig(update_epochs=1, minibatch_size=batch.transition_count, target_joint_kl=None),
        )
        after = _parameter_vector(model)

        self.assertEqual(stats.update_count, 1)
        self.assertFalse(stats.early_stopped)
        self.assertTrue(torch.isfinite(torch.tensor(stats.loss)).item())
        self.assertFalse(torch.allclose(before, after))

    def test_ppo_update_can_early_stop_on_joint_kl(self) -> None:
        model = _small_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = _batch_from_model(model).compute_gae(gamma=1.0, gae_lambda=1.0)
        batch = replace(batch, old_logprob=batch.old_logprob + 10.0)

        stats = ppo_update(
            model,
            optimizer,
            batch,
            PpoConfig(update_epochs=3, minibatch_size=batch.transition_count, target_joint_kl=0.0),
        )

        self.assertTrue(stats.early_stopped)
        self.assertEqual(stats.update_count, 1)

    def test_collect_ppo_rollouts_can_feed_gae_and_update(self) -> None:
        model = _small_model()
        sampler = BatchRolloutSampler(
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )

        batch = collect_ppo_rollouts(model, sampler, pair_count=1, seed=5, map_ids=(1,))
        batch = batch.compute_gae(gamma=1.0, gae_lambda=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        stats = ppo_update(
            model,
            optimizer,
            batch,
            PpoConfig(update_epochs=1, minibatch_size=batch.transition_count, target_joint_kl=None),
        )

        self.assertEqual(batch.transition_count, 2)
        self.assertEqual(batch.episode_ids[0], "pair-000000-seed-5-first")
        self.assertEqual(batch.episode_ids[1], "pair-000000-seed-5-second")
        self.assertEqual(tuple(batch.spatial_planes.shape), (2, 43, 17, 17))
        self.assertEqual(tuple(batch.critic_planes.shape), (2, 26, 17, 17))
        self.assertEqual(tuple(batch.critic_scalars.shape), (2, 17))
        self.assertEqual(stats.update_count, 1)


def _small_model() -> GoldRushPolicyNetwork:
    return GoldRushPolicyNetwork(
        PolicyNetworkConfig(
            width=16,
            residual_blocks=1,
            se_reduction=4,
            scalar_hidden=(16, 16),
            actor_hidden=32,
            critic_hidden=(32, 16),
            decoder_hidden=16,
            decoder_embedding=4,
        )
    )


def _batch_from_model(model: GoldRushPolicyNetwork) -> PpoBatch:
    transitions: list[PpoTransition] = []
    spatial, scalars = _feature_tensors(batch_size=4)
    fast_scalars = torch.tensor([[0.8, 0.25]], dtype=torch.float32).expand(4, -1).clone()
    critic_spatial, critic_scalars = _critic_feature_tensors(batch_size=4)
    with torch.no_grad():
        action = model.act(spatial, scalars, fast_scalars, critic_spatial, critic_scalars)
    rewards = (1.0, -1.0, 0.5, -0.5)
    for idx, reward in enumerate(rewards):
        transitions.append(
            PpoTransition(
                spatial_planes=spatial[idx],
                scalars=scalars[idx],
                fast_scalars=fast_scalars[idx],
                critic_planes=critic_spatial[idx],
                critic_scalars=critic_scalars[idx],
                actions=action.actions[idx],
                k=action.k[idx],
                order=action.order[idx],
                vp=action.vp[idx],
                threshold_raw=action.threshold_raw[idx],
                threshold_int=action.threshold_int[idx],
                old_logprob=action.logprob[idx],
                value=action.value[idx],
                reward=reward,
                done=True,
                episode_id=f"episode-{idx}",
                round_index=0,
                map_id=1,
                agent_player_id=1,
                info={},
            )
        )
    return PpoBatch.from_transitions(transitions)


def _feature_tensors(*, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    spatial = torch.zeros(batch_size, 43, 17, 17)
    scalars = torch.zeros(batch_size, 10)
    spatial[:, 24, 0, 0] = 1.0
    spatial[:, 25, 16, 16] = 1.0
    scalars[:, 0] = torch.linspace(-1.0, 1.0, batch_size)
    return spatial, scalars


def _critic_feature_tensors(*, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    spatial = torch.zeros(batch_size, 26, 17, 17)
    scalars = torch.zeros(batch_size, 17)
    spatial[:, 9, 0, 0] = 1.0
    spatial[:, 10, 16, 16] = 1.0
    spatial[:, 11, 0, 16] = 1.0
    spatial[:, 12, 16, 0] = 1.0
    return spatial, scalars


def _parameter_vector(model: GoldRushPolicyNetwork) -> torch.Tensor:
    return torch.cat([parameter.detach().flatten().cpu() for parameter in model.parameters()])


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
