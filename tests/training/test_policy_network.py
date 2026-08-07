from __future__ import annotations

import unittest

import torch

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepMechanisms
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from simulator.types import Action, GameOutput
from training.models import (
    GoldRushPolicyNetwork,
    PolicyNetworkConfig,
    TorchFeaturePolicy,
    policy_action_to_game_output,
    safe_game_output,
)
from training.opponents import OpponentSpec
from training.rl import BatchRolloutSampler, RuntimePolicyWrapper, SingleAgentEnvConfig


class PolicyNetworkTests(unittest.TestCase):
    def test_forward_returns_deterministic_policy_action_shapes(self) -> None:
        model = _small_model()

        output = model(*_feature_tensors(batch_size=2))

        self.assertEqual(tuple(output.actions.shape), (2, 6))
        self.assertEqual(tuple(output.k.shape), (2,))
        self.assertEqual(tuple(output.order.shape), (2,))
        self.assertEqual(tuple(output.vp.shape), (2,))
        self.assertEqual(tuple(output.logprob.shape), (2,))
        self.assertEqual(tuple(output.value.shape), (2,))

    def test_input_shape_validation_fails_fast(self) -> None:
        model = _small_model()
        spatial, scalars = _feature_tensors(batch_size=1)

        with self.assertRaisesRegex(ValueError, "spatial_planes must have shape"):
            model.act(spatial[:, :37], scalars)
        with self.assertRaisesRegex(ValueError, "scalars must have 10 features"):
            model.act(spatial, scalars[:, :9])

    def test_initialization_preserves_feature_and_vp_priors(self) -> None:
        model = _small_model()

        for encoder in (model.actor_encoder, model.critic_encoder):
            film_last = encoder.scalar_tower[-1]
            self.assertEqual(torch.count_nonzero(film_last.weight).item(), 0)
            self.assertEqual(torch.count_nonzero(film_last.bias).item(), 0)
            for block in encoder.blocks:
                self.assertEqual(torch.count_nonzero(block.se_fc2.weight).item(), 0)
                self.assertEqual(torch.count_nonzero(block.se_fc2.bias).item(), 0)

        vp_prior = torch.softmax(model.vp_head.bias.detach(), dim=0)
        self.assertTrue(torch.allclose(vp_prior, torch.tensor([0.90, 0.07, 0.03]), atol=1e-6))

    def test_actor_and_critic_parameters_are_disjoint(self) -> None:
        model = _small_model()

        actor_ids = {id(parameter) for parameter in model.actor_parameters()}
        critic_ids = {id(parameter) for parameter in model.critic_parameters()}
        all_ids = {id(parameter) for parameter in model.parameters()}

        self.assertFalse(actor_ids & critic_ids)
        self.assertEqual(actor_ids | critic_ids, all_ids)

    def test_execution_tables_cover_all_ko_values(self) -> None:
        model = _small_model()

        for k in range(7):
            for order in range(2):
                ko = 2 * k + order
                if order == 0:
                    expected_roles = [0] * k + [1] * (6 - k)
                    expected_slots = list(range(6))
                else:
                    expected_roles = [1] * (6 - k) + [0] * k
                    expected_slots = list(range(k, 6)) + list(range(k))
                self.assertEqual(model.execution_role_table[ko].tolist(), expected_roles)
                self.assertEqual(model.official_slot_table[ko].tolist(), expected_slots)

    def test_sample_action_returns_valid_game_output_and_stats(self) -> None:
        model = _small_model()
        action = model.act(*_feature_tensors(batch_size=4))
        game_output = policy_action_to_game_output(action, batch_index=0)

        self.assertIsInstance(game_output, GameOutput)
        self.assertEqual(tuple(action.actions.shape), (4, 6))
        self.assertEqual(tuple(action.logprob.shape), (4,))
        self.assertEqual(tuple(action.normalized_entropy.shape), (4,))
        self.assertTrue(torch.isfinite(action.logprob).all().item())
        self.assertTrue(torch.isfinite(action.normalized_entropy).all().item())

    def test_deterministic_action_is_repeatable(self) -> None:
        model = _small_model()
        features = _feature_tensors(batch_size=2)

        first = model.act(*features, deterministic=True)
        second = model.act(*features, deterministic=True)

        self.assertTrue(torch.equal(first.actions, second.actions))
        self.assertTrue(torch.equal(first.k, second.k))
        self.assertTrue(torch.equal(first.order, second.order))
        self.assertTrue(torch.equal(first.vp, second.vp))

    def test_teacher_forcing_exactly_recomputes_sampled_logprob(self) -> None:
        model = _small_model()
        spatial, scalars = _feature_tensors(batch_size=8)
        with torch.no_grad():
            action = model.act(spatial, scalars)
            evaluation = model.evaluate_actions(
                spatial, scalars, action.actions, action.k, action.order, action.vp
            )

        self.assertTrue(torch.allclose(action.logprob, evaluation.logprob, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(action.value, evaluation.value, atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(action.normalized_entropy, evaluation.normalized_entropy, atol=1e-6))

    def test_movement_mask_blocks_boundaries_obstacles_and_other_unit(self) -> None:
        model = _small_model()
        spatial, scalars = _feature_tensors(batch_size=1, unit0=(0, 0), unit1=(0, 1))
        spatial[:, 20, 1, 0] = 1.0
        spatial[:, 21, 1, 0] = 1.0
        encoded = model._encode(spatial, scalars)

        valid, _candidates = model._movement_candidates(
            encoded.unit0_position, encoded.unit1_position, encoded.known_obstacles
        )

        self.assertEqual(valid[0].tolist(), [False, False, False, False, True])

    def test_unknown_obstacle_plane_does_not_mask_movement(self) -> None:
        model = _small_model()
        spatial, scalars = _feature_tensors(batch_size=1, unit0=(5, 5), unit1=(15, 15))
        spatial[:, 21, 5, 6] = 1.0
        encoded = model._encode(spatial, scalars)

        valid, _candidates = model._movement_candidates(
            encoded.unit0_position, encoded.unit1_position, encoded.known_obstacles
        )

        self.assertTrue(valid[0, Action.RIGHT].item())

    def test_decoder_simulates_positions_and_preserves_official_slots(self) -> None:
        model = _small_model()
        spatial, scalars = _feature_tensors(batch_size=1, unit0=(1, 1), unit1=(15, 15))
        official_actions = torch.tensor(
            [[Action.DOWN, Action.RIGHT, Action.DOWN, Action.UP, Action.LEFT, Action.UP]], dtype=torch.long
        )
        encoded = model._encode(spatial, scalars)

        decoded = model._decode(
            encoded,
            torch.tensor([6]),  # k=3, order=0
            deterministic=False,
            forced_actions=official_actions,
        )

        self.assertTrue(decoded.all_forced_actions_valid.all().item())
        self.assertTrue(torch.equal(decoded.actions, official_actions))
        self.assertEqual(decoded.final_unit0_position.item(), 3 * 17 + 2)
        self.assertEqual(decoded.final_unit1_position.item(), 13 * 17 + 14)

    def test_decoder_matches_reference_slot_assignment_for_all_ko_values(self) -> None:
        model = _small_model()
        spatial, scalars = _feature_tensors(batch_size=1, unit0=(5, 5), unit1=(11, 11))
        official_actions = torch.tensor(
            [[Action.DOWN, Action.RIGHT, Action.UP, Action.LEFT, Action.DOWN, Action.UP]], dtype=torch.long
        )
        encoded = model._encode(spatial, scalars)

        for k in range(7):
            for order in range(2):
                decoded = model._decode(
                    encoded,
                    torch.tensor([2 * k + order]),
                    deterministic=False,
                    forced_actions=official_actions,
                )
                expected_positions, invalid = _reference_move(
                    official_actions[0].tolist(), k=k, order=order, unit0=(5, 5), unit1=(11, 11)
                )
                self.assertEqual(invalid, [])
                self.assertTrue(decoded.all_forced_actions_valid.all().item())
                self.assertTrue(torch.equal(decoded.actions, official_actions))
                self.assertEqual(decoded.final_unit0_position.item(), _flat(expected_positions[0]))
                self.assertEqual(decoded.final_unit1_position.item(), _flat(expected_positions[1]))

    def test_order_allows_second_unit_to_vacate_before_first_unit_moves(self) -> None:
        model = _small_model()
        spatial, scalars = _feature_tensors(batch_size=1, unit0=(1, 1), unit1=(1, 2))
        official_actions = torch.tensor(
            [[Action.RIGHT, Action.RIGHT, Action.STAY, Action.STAY, Action.STAY, Action.STAY]], dtype=torch.long
        )
        encoded = model._encode(spatial, scalars)

        decoded = model._decode(
            encoded,
            torch.tensor([3]),  # k=1, order=1
            deterministic=False,
            forced_actions=official_actions,
        )

        self.assertTrue(decoded.all_forced_actions_valid.all().item())
        self.assertEqual(decoded.final_unit0_position.item(), 1 * 17 + 2)
        self.assertEqual(decoded.final_unit1_position.item(), 1 * 17 + 3)

    def test_teacher_forcing_rejects_masked_action(self) -> None:
        model = _small_model()
        spatial, scalars = _feature_tensors(batch_size=1, unit0=(0, 0), unit1=(16, 16))
        actions = torch.tensor([[Action.UP] + [Action.STAY] * 5], dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "teacher-forced action is invalid"):
            model.evaluate_actions(
                spatial,
                scalars,
                actions,
                torch.tensor([6]),
                torch.tensor([0]),
                torch.tensor([0]),
            )

    def test_blocked_forced_step_keeps_position_before_later_step(self) -> None:
        model = _small_model()
        spatial, scalars = _feature_tensors(batch_size=1, unit0=(0, 0), unit1=(16, 16))
        actions = torch.tensor(
            [[Action.UP, Action.RIGHT, Action.STAY, Action.STAY, Action.STAY, Action.STAY]], dtype=torch.long
        )
        encoded = model._encode(spatial, scalars)

        decoded = model._decode(
            encoded,
            torch.tensor([12]),  # k=6, order=0
            deterministic=False,
            forced_actions=actions,
        )

        self.assertFalse(decoded.all_forced_actions_valid.item())
        self.assertEqual(decoded.final_unit0_position.item(), 1)

    def test_sampled_actions_never_use_deterministically_blocked_moves(self) -> None:
        torch.manual_seed(123)
        model = _small_model()
        spatial, scalars = _feature_tensors(batch_size=64, unit0=(0, 0), unit1=(0, 1))
        spatial[:, 20, 1, 0] = 1.0
        spatial[:, 21, 1, 0] = 1.0

        action = model.act(spatial, scalars)

        for batch_index in range(64):
            _positions, invalid = _reference_move(
                action.actions[batch_index].tolist(),
                k=int(action.k[batch_index]),
                order=int(action.order[batch_index]),
                unit0=(0, 0),
                unit1=(0, 1),
                obstacles={(1, 0)},
            )
            self.assertEqual(invalid, [])

    def test_safe_game_output_is_valid(self) -> None:
        self.assertEqual(safe_game_output(), GameOutput(actions=(4, 4, 4, 4, 4, 4), k=3, order=0, vp=0))

    def test_torch_feature_policy_falls_back_on_model_error(self) -> None:
        model = _small_model()

        def bad_act(_spatial, _scalars, *, deterministic=False):
            raise ValueError("injected model failure")

        model.act = bad_act  # type: ignore[method-assign]
        policy = TorchFeaturePolicy(model)

        output = policy(_feature_dict())

        self.assertEqual(output, safe_game_output())
        self.assertIn("injected model failure", policy.last_fallback_error or "")

    def test_runtime_wrapper_can_rollout_with_torch_policy(self) -> None:
        model = _small_model()
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


def _feature_tensors(
    *, batch_size: int, unit0: tuple[int, int] = (1, 1), unit1: tuple[int, int] = (15, 15)
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial = torch.zeros(batch_size, 38, 17, 17)
    scalars = torch.zeros(batch_size, 10)
    spatial[:, 24, unit0[0], unit0[1]] = 1.0
    spatial[:, 25, unit1[0], unit1[1]] = 1.0
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


def _reference_move(
    actions: list[int],
    *,
    k: int,
    order: int,
    unit0: tuple[int, int],
    unit1: tuple[int, int],
    obstacles: set[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], list[int]]:
    positions = [unit0, unit1]
    obstacles = set() if obstacles is None else obstacles
    slots = list(range(6)) if order == 0 else list(range(k, 6)) + list(range(k))
    invalid: list[int] = []
    deltas = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
    for slot in slots:
        role = 0 if slot < k else 1
        row, col = positions[role]
        dr, dc = deltas[actions[slot]]
        candidate = (row + dr, col + dc)
        blocked = (
            not (0 <= candidate[0] < 17 and 0 <= candidate[1] < 17)
            or candidate in obstacles
            or candidate == positions[1 - role]
        )
        if blocked:
            invalid.append(slot)
        else:
            positions[role] = candidate
    return positions, invalid


def _flat(position: tuple[int, int]) -> int:
    return position[0] * 17 + position[1]


if __name__ == "__main__":
    unittest.main()
