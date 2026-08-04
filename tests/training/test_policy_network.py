from __future__ import annotations

import unittest

import torch

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepMechanisms
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from simulator.types import GameOutput
from training.models import (
    GoldRushPolicyNetwork,
    PolicyNetworkOutput,
    TorchFeaturePolicy,
    argmax_action,
    policy_action_to_game_output,
    safe_game_output,
    sample_action,
)
from training.opponents import OpponentSpec
from training.rl import BatchRolloutSampler, RuntimePolicyWrapper, SingleAgentEnvConfig


class PolicyNetworkTests(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        model = GoldRushPolicyNetwork()

        output = model(*_feature_tensors(batch_size=2))

        self.assertEqual(tuple(output.action_logits.shape), (2, 6, 5))
        self.assertEqual(tuple(output.k_logits.shape), (2, 7))
        self.assertEqual(tuple(output.order_logits.shape), (2, 2))
        self.assertEqual(tuple(output.vp_logits.shape), (2, 3))
        self.assertEqual(tuple(output.value.shape), (2,))

    def test_input_shape_validation_fails_fast(self) -> None:
        model = GoldRushPolicyNetwork()
        spatial, scalars = _feature_tensors(batch_size=1)

        with self.assertRaisesRegex(ValueError, "spatial_planes must have shape"):
            model(spatial[:, :37], scalars)
        with self.assertRaisesRegex(ValueError, "scalars must have 10 features"):
            model(spatial, scalars[:, :9])

    def test_initialization_matches_v1_priors(self) -> None:
        model = GoldRushPolicyNetwork()

        film_last = model.scalar_tower[-1]
        self.assertTrue(torch.count_nonzero(film_last.weight).item() == 0)
        self.assertTrue(torch.count_nonzero(film_last.bias).item() == 0)
        for block in model.blocks:
            self.assertTrue(torch.count_nonzero(block.se_fc2.weight).item() == 0)
            self.assertTrue(torch.count_nonzero(block.se_fc2.bias).item() == 0)

        vp_prior = torch.softmax(model.vp_head.bias.detach(), dim=0)
        self.assertTrue(torch.allclose(vp_prior, torch.tensor([0.90, 0.07, 0.03]), atol=1e-6))

    def test_sample_action_returns_valid_game_output_and_stats(self) -> None:
        model = GoldRushPolicyNetwork()
        output = model(*_feature_tensors(batch_size=4))

        action = sample_action(output)
        game_output = policy_action_to_game_output(action, batch_index=0)

        self.assertIsInstance(game_output, GameOutput)
        self.assertEqual(tuple(action.actions.shape), (4, 6))
        self.assertEqual(tuple(action.logprob.shape), (4,))
        self.assertEqual(tuple(action.value.shape), (4,))
        self.assertEqual(tuple(action.normalized_entropy.shape), (4,))
        self.assertTrue(torch.isfinite(action.logprob).all().item())
        self.assertTrue(torch.isfinite(action.normalized_entropy).all().item())

    def test_argmax_action_is_deterministic(self) -> None:
        model = GoldRushPolicyNetwork()
        output = model(*_feature_tensors(batch_size=2))

        first = argmax_action(output)
        second = argmax_action(output)

        self.assertTrue(torch.equal(first.actions, second.actions))
        self.assertTrue(torch.equal(first.k, second.k))
        self.assertTrue(torch.equal(first.order, second.order))
        self.assertTrue(torch.equal(first.vp, second.vp))

    def test_safe_game_output_is_valid(self) -> None:
        self.assertEqual(safe_game_output(), GameOutput(actions=(4, 4, 4, 4, 4, 4), k=3, order=0, vp=0))

    def test_torch_feature_policy_falls_back_on_nan_output(self) -> None:
        model = GoldRushPolicyNetwork()

        def bad_forward(_spatial, _scalars):
            batch = 1
            return PolicyNetworkOutput(
                action_logits=torch.full((batch, 6, 5), float("nan")),
                k_logits=torch.zeros(batch, 7),
                order_logits=torch.zeros(batch, 2),
                vp_logits=torch.zeros(batch, 3),
                value=torch.zeros(batch),
            )

        model.forward = bad_forward  # type: ignore[method-assign]
        policy = TorchFeaturePolicy(model)

        output = policy(_feature_dict())

        self.assertEqual(output, safe_game_output())
        self.assertIsNotNone(policy.last_fallback_error)

    def test_runtime_wrapper_can_rollout_with_torch_policy(self) -> None:
        model = GoldRushPolicyNetwork()
        feature_policy = TorchFeaturePolicy(model, deterministic=True)
        runtime_policy = RuntimePolicyWrapper(feature_policy)
        sampler = BatchRolloutSampler(
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )

        batch = sampler.collect(runtime_policy, pair_count=1, seed=42, map_ids=(1,))

        self.assertEqual(batch.episode_count, 2)
        self.assertEqual(batch.transition_count, 2)
        self.assertIsNone(feature_policy.last_fallback_error)


def _feature_tensors(*, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    spatial = torch.zeros(batch_size, 38, 17, 17)
    scalars = torch.zeros(batch_size, 10)
    spatial[:, 24, 0, 0] = 1.0
    spatial[:, 25, 16, 16] = 1.0
    return spatial, scalars


def _feature_dict() -> dict[str, object]:
    spatial, scalars = _feature_tensors(batch_size=1)
    return {
        "feature_schema": "goldrush2_feature_v1",
        "planes": spatial[0].numpy(),
        "scalars": scalars[0].numpy(),
    }


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
