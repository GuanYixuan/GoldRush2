from __future__ import annotations

import torch

from training.models import GoldRushPolicyNetwork, PolicyNetworkConfig
from training.scripts.inflate_candidate_cell_residual_head_checkpoint import (
    inflate_checkpoint_payload as inflate_action_head_checkpoint_payload,
)
from training.scripts.inflate_actor_feature_v2_checkpoint import inflate_checkpoint_payload
from training.scripts.inflate_fast_threshold_head_checkpoint import (
    inflate_checkpoint_payload as inflate_fast_threshold_checkpoint_payload,
)


def test_inflate_bc_checkpoint_to_actor_feature_v2() -> None:
    weight = torch.arange(2 * 38 * 3 * 3, dtype=torch.float32).reshape(2, 38, 3, 3)
    checkpoint = {
        "schema": "bc_train_v1",
        "feature_schema": "goldrush2_feature_v1",
        "model_config": {
            "actor_spatial_channels": 38,
            "actor_scalar_features": 10,
            "width": 2,
        },
        "model_state_dict": {
            "actor_encoder.stem.0.weight": weight,
            "actor_encoder.stem.0.bias": torch.ones(2),
        },
        "optimizer_state_dict": {"state": {"stale": True}},
    }

    inflated = inflate_checkpoint_payload(checkpoint)

    inflated_weight = inflated["model_state_dict"]["actor_encoder.stem.0.weight"]
    assert inflated["feature_schema"] == "goldrush2_feature_v2"
    assert inflated["model_config"]["actor_spatial_channels"] == 43
    assert "optimizer_state_dict" not in inflated
    assert inflated["optimizer_state_dict_dropped_for_actor_feature_inflation"] is True
    assert tuple(inflated_weight.shape) == (2, 43, 3, 3)
    assert torch.equal(inflated_weight[:, :38], weight)
    assert torch.count_nonzero(inflated_weight[:, 38:]).item() == 0
    assert torch.equal(inflated["model_state_dict"]["actor_encoder.stem.0.bias"], torch.ones(2))


def test_inflate_ppo_checkpoint_updates_train_config_model() -> None:
    weight = torch.ones(4, 38, 3, 3)
    checkpoint = {
        "schema": "ppo_train_v1",
        "train_config": {
            "model": {
                "actor_spatial_channels": 38,
                "actor_scalar_features": 10,
                "critic_spatial_channels": 26,
            }
        },
        "model_state_dict": {
            "actor_encoder.stem.0.weight": weight,
        },
    }

    inflated = inflate_checkpoint_payload(checkpoint)

    assert inflated["train_config"]["model"]["actor_spatial_channels"] == 43
    assert inflated["model_state_dict"]["actor_encoder.stem.0.weight"].shape[1] == 43
    assert inflated["actor_feature_inflation"]["inflated_weight_keys"] == ["actor_encoder.stem.0.weight"]


def test_inflate_rejects_non_v1_feature_schema() -> None:
    checkpoint = {
        "schema": "bc_train_v1",
        "feature_schema": "goldrush2_feature_v2",
        "model_config": {"actor_spatial_channels": 38},
        "model_state_dict": {"actor_encoder.stem.0.weight": torch.ones(2, 38, 3, 3)},
    }

    try:
        inflate_checkpoint_payload(checkpoint)
    except ValueError as exc:
        assert "feature_schema" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_inflate_candidate_residual_head_checkpoint_loads_and_is_noop() -> None:
    old_model = GoldRushPolicyNetwork(_small_config())
    old_state = old_model.state_dict()
    legacy_state = {
        key: value
        for key, value in old_state.items()
        if not (key.startswith("action_candidate_embedding.") or key.startswith("candidate_action_head."))
    }
    checkpoint = {
        "schema": "ppo_train_v1",
        "train_config": {"model": _legacy_model_config()},
        "model_state_dict": legacy_state,
        "optimizer_state_dict": {"state": {"stale": True}},
    }

    inflated = inflate_action_head_checkpoint_payload(checkpoint)
    model = GoldRushPolicyNetwork(_small_config())
    model.load_state_dict(inflated["model_state_dict"])

    hidden = torch.randn(3, model.config.decoder_hidden)
    spatial_features = torch.randn(3, model.config.width, 17, 17)
    candidate_positions = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [17, 18, 19, 20, 21],
            [34, 35, 36, 37, 38],
        ],
        dtype=torch.long,
    )

    assert inflated["train_config"]["model"]["action_head_schema"] == "candidate_cell_residual_v1"
    assert "optimizer_state_dict" not in inflated
    assert inflated["optimizer_state_dict_dropped_for_action_head_inflation"] is True
    assert "action_candidate_embedding.weight" in inflated["model_state_dict"]
    assert "candidate_action_head.2.weight" in inflated["model_state_dict"]
    assert torch.count_nonzero(inflated["model_state_dict"]["candidate_action_head.2.weight"]).item() == 0
    assert torch.count_nonzero(inflated["model_state_dict"]["candidate_action_head.2.bias"]).item() == 0
    assert torch.allclose(model._action_logits(hidden, spatial_features, candidate_positions), model.decoder_action_head(hidden))


def test_inflate_candidate_residual_head_rejects_non_v2_feature_shape() -> None:
    checkpoint = {
        "schema": "ppo_train_v1",
        "train_config": {"model": {**_legacy_model_config(), "actor_spatial_channels": 38}},
        "model_state_dict": {},
    }

    try:
        inflate_action_head_checkpoint_payload(checkpoint)
    except ValueError as exc:
        assert "actor feature shape" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_inflate_fast_threshold_checkpoint_loads_and_preserves_old_value() -> None:
    candidate_config = _small_config()
    candidate_config = PolicyNetworkConfig(
        width=candidate_config.width,
        residual_blocks=candidate_config.residual_blocks,
        se_reduction=candidate_config.se_reduction,
        scalar_hidden=candidate_config.scalar_hidden,
        actor_hidden=candidate_config.actor_hidden,
        critic_hidden=candidate_config.critic_hidden,
        decoder_hidden=candidate_config.decoder_hidden,
        decoder_embedding=candidate_config.decoder_embedding,
        action_head_schema="candidate_cell_residual_v1",
    )
    old_model = GoldRushPolicyNetwork(candidate_config)
    old_state = {
        key: value
        for key, value in old_model.state_dict().items()
        if not (key.startswith("threshold_mlp.") or key == "threshold_log_std")
    }
    old_state["critic_mlp.0.weight"] = old_state["critic_mlp.0.weight"][:, :-2].clone()
    checkpoint = {
        "schema": "ppo_train_v1",
        "train_config": {"model": _candidate_model_config()},
        "model_state_dict": old_state,
        "optimizer_state_dict": {"state": {"stale": True}},
    }
    critic_planes = torch.zeros(2, 26, 17, 17)
    critic_scalars = torch.zeros(2, 17)
    critic_planes[:, 9, 1, 1] = 1.0
    critic_planes[:, 10, 15, 15] = 1.0
    critic_planes[:, 11, 2, 14] = 1.0
    critic_planes[:, 12, 14, 2] = 1.0
    old_context = old_model.critic_encoder(critic_planes, critic_scalars)
    old_input = torch.cat(
        (
            old_context.avg,
            old_context.max_pool,
            old_context.own_unit0,
            old_context.own_unit1,
            old_context.enemy_unit0,
            old_context.enemy_unit1,
        ),
        dim=1,
    )
    old_hidden0 = torch.nn.functional.linear(old_input, old_state["critic_mlp.0.weight"], old_state["critic_mlp.0.bias"])
    old_hidden0 = torch.nn.functional.silu(old_hidden0)
    old_hidden1 = torch.nn.functional.linear(old_hidden0, old_state["critic_mlp.2.weight"], old_state["critic_mlp.2.bias"])
    old_hidden1 = torch.nn.functional.silu(old_hidden1)
    old_value = torch.nn.functional.linear(old_hidden1, old_state["critic_mlp.4.weight"], old_state["critic_mlp.4.bias"])

    inflated = inflate_fast_threshold_checkpoint_payload(checkpoint)
    model = GoldRushPolicyNetwork(_small_config())
    model.load_state_dict(inflated["model_state_dict"])
    new_value = model._critic_value(critic_planes, critic_scalars, torch.zeros(2, 2)).unsqueeze(1)

    assert (
        inflated["train_config"]["model"]["action_head_schema"]
        == "candidate_cell_residual_v1_fast_threshold_calibrated_base_v1"
    )
    assert "optimizer_state_dict" not in inflated
    assert inflated["optimizer_state_dict_dropped_for_fast_threshold_inflation"] is True
    assert "threshold_mlp.4.weight" in inflated["model_state_dict"]
    assert "threshold_log_std" in inflated["model_state_dict"]
    assert inflated["model_state_dict"]["critic_mlp.0.weight"].shape[1] == old_state["critic_mlp.0.weight"].shape[1] + 2
    assert torch.allclose(old_value, new_value, atol=1e-7, rtol=1e-7)


def test_inflate_fast_threshold_rejects_non_candidate_schema() -> None:
    checkpoint = {
        "schema": "ppo_train_v1",
        "train_config": {"model": {**_candidate_model_config(), "action_head_schema": "autoregressive_head_v1"}},
        "model_state_dict": {},
    }

    try:
        inflate_fast_threshold_checkpoint_payload(checkpoint)
    except ValueError as exc:
        assert "action_head_schema" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def _small_config() -> PolicyNetworkConfig:
    return PolicyNetworkConfig(
        width=16,
        residual_blocks=1,
        se_reduction=4,
        scalar_hidden=(16, 16),
        actor_hidden=32,
        critic_hidden=(32, 16),
        decoder_hidden=16,
        decoder_embedding=4,
    )


def _legacy_model_config() -> dict[str, object]:
    config = _small_config()
    return {
        "actor_spatial_channels": config.actor_spatial_channels,
        "actor_scalar_features": config.actor_scalar_features,
        "critic_spatial_channels": config.critic_spatial_channels,
        "critic_scalar_features": config.critic_scalar_features,
        "width": config.width,
        "residual_blocks": config.residual_blocks,
        "se_reduction": config.se_reduction,
        "scalar_hidden": config.scalar_hidden,
        "actor_hidden": config.actor_hidden,
        "critic_hidden": config.critic_hidden,
        "decoder_hidden": config.decoder_hidden,
        "decoder_embedding": config.decoder_embedding,
        "action_head_schema": "autoregressive_head_v1",
        "activation": config.activation,
    }


def _candidate_model_config() -> dict[str, object]:
    config = _small_config()
    return {
        "actor_spatial_channels": config.actor_spatial_channels,
        "actor_scalar_features": config.actor_scalar_features,
        "critic_spatial_channels": config.critic_spatial_channels,
        "critic_scalar_features": config.critic_scalar_features,
        "width": config.width,
        "residual_blocks": config.residual_blocks,
        "se_reduction": config.se_reduction,
        "scalar_hidden": config.scalar_hidden,
        "actor_hidden": config.actor_hidden,
        "critic_hidden": config.critic_hidden,
        "decoder_hidden": config.decoder_hidden,
        "decoder_embedding": config.decoder_embedding,
        "action_head_schema": "candidate_cell_residual_v1",
        "activation": config.activation,
    }
