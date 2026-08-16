from __future__ import annotations

from dataclasses import asdict

import torch

from training.models import GoldRushPolicyNetwork, PolicyNetworkConfig
from training.scripts.inflate_candidate_cell_residual_head_checkpoint import (
    inflate_checkpoint_payload as inflate_action_head_checkpoint_payload,
)
from training.scripts.inflate_actor_feature_v2_checkpoint import inflate_checkpoint_payload
from training.scripts.inflate_fast_threshold_head_checkpoint import (
    inflate_checkpoint_payload as inflate_fast_threshold_checkpoint_payload,
)
from training.scripts.inflate_vp_final_position_head_checkpoint import (
    inflate_checkpoint_payload as inflate_vp_head_checkpoint_payload,
)
from training.scripts.inflate_privileged_critic_feature_v2_checkpoint import (
    inflate_checkpoint_payload as inflate_critic_feature_checkpoint_payload,
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
    critic_planes = torch.zeros(2, 39, 17, 17)
    critic_scalars = torch.zeros(2, 20)
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


def test_inflate_privileged_critic_feature_v2_preserves_old_value() -> None:
    old_config = _small_v1_critic_config()
    reference_model = GoldRushPolicyNetwork(_small_config())
    reference_state = {
        key: value.clone()
        for key, value in reference_model.state_dict().items()
    }
    reference_state["critic_encoder.stem.0.weight"][:, 26:] = 0.0
    reference_state["critic_encoder.scalar_tower.0.weight"][:, 17:] = 0.0
    reference_model.load_state_dict(reference_state)
    checkpoint_state = {
        key: value.clone()
        for key, value in reference_state.items()
    }
    checkpoint_state["critic_encoder.stem.0.weight"] = checkpoint_state["critic_encoder.stem.0.weight"][:, :26].clone()
    checkpoint_state["critic_encoder.scalar_tower.0.weight"] = checkpoint_state["critic_encoder.scalar_tower.0.weight"][:, :17].clone()
    checkpoint = {
        "schema": "ppo_train_v1",
        "feature_schema": "goldrush2_feature_v2",
        "critic_feature_schema": "goldrush2_privileged_critic_feature_v1",
        "train_config": {"model": asdict(old_config)},
        "model_state_dict": checkpoint_state,
        "optimizer_state_dict": {"state": {"stale": True}},
    }
    old_critic_planes = torch.zeros(2, 26, 17, 17)
    old_critic_scalars = torch.zeros(2, 17)
    old_critic_planes[:, 9, 1, 1] = 1.0
    old_critic_planes[:, 10, 15, 15] = 1.0
    old_critic_planes[:, 11, 2, 14] = 1.0
    old_critic_planes[:, 12, 14, 2] = 1.0
    fast_scalars = torch.zeros(2, 2)

    inflated = inflate_critic_feature_checkpoint_payload(checkpoint)
    model = GoldRushPolicyNetwork(_small_config())
    model.load_state_dict(inflated["model_state_dict"])
    new_critic_planes = torch.randn(2, 39, 17, 17)
    new_critic_scalars = torch.randn(2, 20)
    new_critic_planes[:, :26] = old_critic_planes
    new_critic_scalars[:, :17] = old_critic_scalars
    with torch.no_grad():
        old_value = reference_model._critic_value(new_critic_planes, new_critic_scalars, fast_scalars)
        new_value = model._critic_value(new_critic_planes, new_critic_scalars, fast_scalars)

    stem_weight = inflated["model_state_dict"]["critic_encoder.stem.0.weight"]
    scalar_weight = inflated["model_state_dict"]["critic_encoder.scalar_tower.0.weight"]
    assert inflated["train_config"]["model"]["critic_spatial_channels"] == 39
    assert inflated["train_config"]["model"]["critic_scalar_features"] == 20
    assert inflated["critic_feature_schema"] == "goldrush2_privileged_critic_feature_v2"
    assert inflated["inflated_from_critic_feature_schema"] == "goldrush2_privileged_critic_feature_v1"
    assert inflated["critic_inflation"] == "v1_to_v2_zero_actor_info"
    assert "optimizer_state_dict" not in inflated
    assert inflated["optimizer_state_dict_dropped_for_critic_feature_inflation"] is True
    assert torch.equal(stem_weight[:, :26], checkpoint["model_state_dict"]["critic_encoder.stem.0.weight"])
    assert torch.count_nonzero(stem_weight[:, 26:]).item() == 0
    assert torch.equal(scalar_weight[:, :17], checkpoint["model_state_dict"]["critic_encoder.scalar_tower.0.weight"])
    assert torch.count_nonzero(scalar_weight[:, 17:]).item() == 0
    assert torch.allclose(old_value, new_value, atol=1e-6, rtol=1e-6)


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


def test_inflate_vp_final_position_head_checkpoint_loads_and_is_noop() -> None:
    model = GoldRushPolicyNetwork(_small_config())
    state = model.state_dict()
    old_weight = torch.randn(3, model.config.actor_hidden)
    old_bias = torch.randn(3)
    checkpoint_state = dict(state)
    checkpoint_state["vp_head.weight"] = old_weight.clone()
    checkpoint_state["vp_head.bias"] = old_bias.clone()
    checkpoint = {
        "schema": "ppo_train_v1",
        "feature_schema": "goldrush2_feature_v2",
        "train_config": {"model": _fast_threshold_model_config()},
        "model_state_dict": checkpoint_state,
        "optimizer_state_dict": {"state": {"stale": True}},
    }

    inflated = inflate_vp_head_checkpoint_payload(checkpoint)
    inflated_weight = inflated["model_state_dict"]["vp_head.weight"]
    inflated_bias = inflated["model_state_dict"]["vp_head.bias"]

    assert (
        inflated["train_config"]["model"]["action_head_schema"]
        == "candidate_cell_residual_v1_fast_threshold_calibrated_base_v1_vp_final_position_v1"
    )
    assert "optimizer_state_dict" not in inflated
    assert inflated["optimizer_state_dict_dropped_for_vp_head_inflation"] is True
    assert tuple(inflated_weight.shape) == (3, model.config.actor_hidden + model.config.width * 2)
    assert torch.equal(inflated_weight[:, : model.config.actor_hidden], old_weight)
    assert torch.count_nonzero(inflated_weight[:, model.config.actor_hidden :]).item() == 0
    assert torch.equal(inflated_bias, old_bias)
    context = torch.randn(5, model.config.actor_hidden)
    position_features = torch.randn(5, model.config.width * 2)
    old_logits = torch.nn.functional.linear(context, old_weight, old_bias)
    new_logits = torch.nn.functional.linear(torch.cat((context, position_features), dim=1), inflated_weight, inflated_bias)
    assert torch.allclose(old_logits, new_logits, atol=0.0, rtol=0.0)

    loaded = GoldRushPolicyNetwork(_small_config())
    loaded.load_state_dict(inflated["model_state_dict"])


def test_inflate_vp_final_position_can_chain_before_critic_feature_v2() -> None:
    model = GoldRushPolicyNetwork(_small_config())
    checkpoint_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
    }
    old_weight = torch.randn(3, model.config.actor_hidden)
    old_bias = torch.randn(3)
    checkpoint_state["vp_head.weight"] = old_weight.clone()
    checkpoint_state["vp_head.bias"] = old_bias.clone()
    checkpoint_state["critic_encoder.stem.0.weight"] = checkpoint_state["critic_encoder.stem.0.weight"][:, :26].clone()
    checkpoint_state["critic_encoder.scalar_tower.0.weight"] = (
        checkpoint_state["critic_encoder.scalar_tower.0.weight"][:, :17].clone()
    )
    raw_config = asdict(_small_v1_critic_config())
    raw_config["action_head_schema"] = "candidate_cell_residual_v1_fast_threshold_calibrated_base_v1"
    checkpoint = {
        "schema": "ppo_train_v1",
        "feature_schema": "goldrush2_feature_v2",
        "critic_feature_schema": "goldrush2_privileged_critic_feature_v1",
        "train_config": {"model": raw_config},
        "model_state_dict": checkpoint_state,
        "optimizer_state_dict": {"state": {"stale": True}},
    }

    vp_inflated = inflate_vp_head_checkpoint_payload(checkpoint)
    critic_inflated = inflate_critic_feature_checkpoint_payload(vp_inflated)

    assert critic_inflated["train_config"]["model"]["action_head_schema"] == _small_config().action_head_schema
    assert critic_inflated["train_config"]["model"]["critic_spatial_channels"] == 39
    assert critic_inflated["train_config"]["model"]["critic_scalar_features"] == 20
    assert "optimizer_state_dict" not in critic_inflated
    assert critic_inflated["optimizer_state_dict_dropped_for_vp_head_inflation"] is True
    assert "optimizer_state_dict_dropped_for_critic_feature_inflation" not in critic_inflated
    loaded = GoldRushPolicyNetwork(_small_config())
    loaded.load_state_dict(critic_inflated["model_state_dict"])


def test_inflate_vp_final_position_head_can_reset_vp_prior() -> None:
    model = GoldRushPolicyNetwork(_small_config())
    state = model.state_dict()
    old_weight = torch.randn(3, model.config.actor_hidden)
    old_bias = torch.randn(3)
    checkpoint_state = dict(state)
    checkpoint_state["vp_head.weight"] = old_weight.clone()
    checkpoint_state["vp_head.bias"] = old_bias.clone()
    checkpoint = {
        "schema": "ppo_train_v1",
        "feature_schema": "goldrush2_feature_v2",
        "train_config": {"model": _fast_threshold_model_config()},
        "model_state_dict": checkpoint_state,
    }
    prior = (0.92, 0.06, 0.02)

    inflated = inflate_vp_head_checkpoint_payload(checkpoint, reset_vp_head_prior=prior)
    inflated_weight = inflated["model_state_dict"]["vp_head.weight"]
    inflated_bias = inflated["model_state_dict"]["vp_head.bias"]

    assert torch.count_nonzero(inflated_weight).item() == 0
    assert torch.allclose(inflated_bias, torch.log(torch.tensor(prior, dtype=inflated_bias.dtype)))
    assert inflated["vp_head_inflation"]["weight_init"] == "zero"
    assert inflated["vp_head_inflation"]["bias_init"] == "log_prior"
    assert inflated["vp_head_inflation"]["vp_prior"] == list(prior)

    loaded = GoldRushPolicyNetwork(_small_config())
    loaded.load_state_dict(inflated["model_state_dict"])


def test_inflate_vp_head_rejects_non_fast_threshold_schema() -> None:
    checkpoint = {
        "schema": "ppo_train_v1",
        "train_config": {"model": {**_fast_threshold_model_config(), "action_head_schema": "candidate_cell_residual_v1"}},
        "model_state_dict": {},
    }

    try:
        inflate_vp_head_checkpoint_payload(checkpoint)
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


def _small_v1_critic_config() -> PolicyNetworkConfig:
    base = _small_config()
    return PolicyNetworkConfig(
        actor_spatial_channels=base.actor_spatial_channels,
        actor_scalar_features=base.actor_scalar_features,
        critic_spatial_channels=26,
        critic_scalar_features=17,
        width=base.width,
        residual_blocks=base.residual_blocks,
        se_reduction=base.se_reduction,
        scalar_hidden=base.scalar_hidden,
        actor_hidden=base.actor_hidden,
        critic_hidden=base.critic_hidden,
        decoder_hidden=base.decoder_hidden,
        decoder_embedding=base.decoder_embedding,
        action_head_schema=base.action_head_schema,
        fast_scalar_features=base.fast_scalar_features,
        threshold_initial=base.threshold_initial,
        threshold_hidden=base.threshold_hidden,
        threshold_log_std_initial=base.threshold_log_std_initial,
        threshold_log_std_min=base.threshold_log_std_min,
        threshold_log_std_max=base.threshold_log_std_max,
        threshold_entropy_coef=base.threshold_entropy_coef,
        activation=base.activation,
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


def _fast_threshold_model_config() -> dict[str, object]:
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
        "action_head_schema": "candidate_cell_residual_v1_fast_threshold_calibrated_base_v1",
        "fast_scalar_features": config.fast_scalar_features,
        "threshold_initial": config.threshold_initial,
        "threshold_hidden": config.threshold_hidden,
        "threshold_log_std_initial": config.threshold_log_std_initial,
        "threshold_log_std_min": config.threshold_log_std_min,
        "threshold_log_std_max": config.threshold_log_std_max,
        "threshold_entropy_coef": config.threshold_entropy_coef,
        "activation": config.activation,
    }
