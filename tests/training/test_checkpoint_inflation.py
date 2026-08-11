from __future__ import annotations

import torch

from training.scripts.inflate_actor_feature_v2_checkpoint import inflate_checkpoint_payload


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
