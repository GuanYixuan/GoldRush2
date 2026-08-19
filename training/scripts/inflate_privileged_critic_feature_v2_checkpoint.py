from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from training.models.policy_network import (
    CRITIC_FEATURE_SCHEMA,
    ACTION_HEAD_SCHEMA,
    FEATURE_SCHEMA,
    GoldRushPolicyNetwork,
    PolicyNetworkConfig,
)
from training.rl.privileged_critic_features import (
    REQUIRED_ACTOR_FEATURE_SCHEMA,
    V1_FEATURE_SCHEMA,
    V1_SCALAR_FEATURES,
    V1_SPATIAL_CHANNELS,
)


OLD_CRITIC_SPATIAL_CHANNELS = V1_SPATIAL_CHANNELS
OLD_CRITIC_SCALAR_FEATURES = V1_SCALAR_FEATURES
NEW_CRITIC_SPATIAL_CHANNELS = 39
NEW_CRITIC_SCALAR_FEATURES = 20
INFLATION_TAG = "v1_to_v2_zero_actor_info"


def inflate_checkpoint_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    inflated = copy.deepcopy(checkpoint)
    raw_config = _raw_model_config(inflated)
    _require_v1_critic_checkpoint(inflated, raw_config)
    config = _current_model_config(raw_config)
    initialized = GoldRushPolicyNetwork(config).state_dict()
    state = inflated.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint missing model_state_dict")

    _inflate_critic_stem(state, initialized)
    _inflate_critic_scalar_tower(state, initialized)
    _update_model_configs(inflated, asdict(config))
    if "optimizer_state_dict" in inflated:
        del inflated["optimizer_state_dict"]
        inflated["optimizer_state_dict_dropped_for_critic_feature_inflation"] = True
    inflated["critic_feature_schema"] = CRITIC_FEATURE_SCHEMA
    inflated["required_actor_feature_schema"] = REQUIRED_ACTOR_FEATURE_SCHEMA
    inflated["inflated_from_critic_feature_schema"] = V1_FEATURE_SCHEMA
    inflated["critic_inflation"] = INFLATION_TAG
    inflated["critic_feature_inflation"] = {
        "from_critic_feature_schema": V1_FEATURE_SCHEMA,
        "to_critic_feature_schema": CRITIC_FEATURE_SCHEMA,
        "old_critic_spatial_channels": OLD_CRITIC_SPATIAL_CHANNELS,
        "new_critic_spatial_channels": NEW_CRITIC_SPATIAL_CHANNELS,
        "old_critic_scalar_features": OLD_CRITIC_SCALAR_FEATURES,
        "new_critic_scalar_features": NEW_CRITIC_SCALAR_FEATURES,
        "new_actor_info_channel_init": "zero",
        "new_actor_info_scalar_init": "zero",
    }
    return inflated


def _raw_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if isinstance(checkpoint.get("model_config"), dict):
        return dict(checkpoint["model_config"])
    train_config = checkpoint.get("train_config")
    if isinstance(train_config, dict) and isinstance(train_config.get("model"), dict):
        return dict(train_config["model"])
    raise ValueError("checkpoint missing model_config or train_config.model")


def _require_v1_critic_checkpoint(checkpoint: dict[str, Any], raw_config: dict[str, Any]) -> None:
    if int(raw_config.get("critic_spatial_channels", -1)) != OLD_CRITIC_SPATIAL_CHANNELS:
        raise ValueError(
            "checkpoint critic_spatial_channels must be "
            f"{OLD_CRITIC_SPATIAL_CHANNELS}, got {raw_config.get('critic_spatial_channels')!r}"
        )
    if int(raw_config.get("critic_scalar_features", -1)) != OLD_CRITIC_SCALAR_FEATURES:
        raise ValueError(
            "checkpoint critic_scalar_features must be "
            f"{OLD_CRITIC_SCALAR_FEATURES}, got {raw_config.get('critic_scalar_features')!r}"
        )
    action_schema = str(raw_config.get("action_head_schema", ""))
    if action_schema != ACTION_HEAD_SCHEMA:
        raise ValueError(f"checkpoint action_head_schema must be {ACTION_HEAD_SCHEMA!r}, got {action_schema!r}")
    actor_schema = checkpoint.get("feature_schema")
    if actor_schema is not None and actor_schema != FEATURE_SCHEMA:
        raise ValueError(f"checkpoint feature_schema must be {FEATURE_SCHEMA!r}, got {actor_schema!r}")
    critic_schema = checkpoint.get("critic_feature_schema")
    if critic_schema is not None and critic_schema != V1_FEATURE_SCHEMA:
        raise ValueError(f"checkpoint critic_feature_schema must be {V1_FEATURE_SCHEMA!r}, got {critic_schema!r}")


def _current_model_config(raw: dict[str, Any]) -> PolicyNetworkConfig:
    return PolicyNetworkConfig(
        actor_spatial_channels=int(raw["actor_spatial_channels"]),
        actor_scalar_features=int(raw["actor_scalar_features"]),
        critic_spatial_channels=NEW_CRITIC_SPATIAL_CHANNELS,
        critic_scalar_features=NEW_CRITIC_SCALAR_FEATURES,
        width=int(raw["width"]),
        residual_blocks=int(raw["residual_blocks"]),
        se_reduction=int(raw["se_reduction"]),
        scalar_hidden=tuple(int(value) for value in raw["scalar_hidden"]),
        actor_hidden=int(raw["actor_hidden"]),
        critic_hidden=tuple(int(value) for value in raw["critic_hidden"]),
        decoder_hidden=int(raw["decoder_hidden"]),
        decoder_embedding=int(raw["decoder_embedding"]),
        action_head_schema=str(raw["action_head_schema"]),
        fast_scalar_features=int(raw.get("fast_scalar_features", 2)),
        threshold_initial=float(raw.get("threshold_initial", 12.0)),
        threshold_hidden=tuple(int(value) for value in raw.get("threshold_hidden", (128, 64))),
        threshold_log_std_initial=float(raw.get("threshold_log_std_initial", -1.3)),
        threshold_log_std_min=float(raw.get("threshold_log_std_min", -3.0)),
        threshold_log_std_max=float(raw.get("threshold_log_std_max", 0.0)),
        threshold_entropy_coef=float(raw.get("threshold_entropy_coef", 0.0)),
        activation=str(raw["activation"]),
    )


def _inflate_critic_stem(state: dict[str, Any], initialized: dict[str, torch.Tensor]) -> None:
    weight_key = "critic_encoder.stem.0.weight"
    bias_key = "critic_encoder.stem.0.bias"
    old_weight = state.get(weight_key)
    old_bias = state.get(bias_key)
    if not isinstance(old_weight, torch.Tensor):
        raise ValueError(f"checkpoint missing {weight_key}")
    if not isinstance(old_bias, torch.Tensor):
        raise ValueError(f"checkpoint missing {bias_key}")
    new_weight = initialized[weight_key]
    new_bias = initialized[bias_key]
    expected_old_shape = (new_weight.shape[0], OLD_CRITIC_SPATIAL_CHANNELS, *new_weight.shape[2:])
    if tuple(old_weight.shape) != tuple(expected_old_shape):
        raise ValueError(f"{weight_key} must have old shape {tuple(expected_old_shape)}, got {tuple(old_weight.shape)}")
    if tuple(new_weight.shape[1:]) != (NEW_CRITIC_SPATIAL_CHANNELS, 3, 3):
        raise ValueError(f"new {weight_key} must have input shape 39x3x3, got {tuple(new_weight.shape)}")
    if tuple(old_bias.shape) != tuple(new_bias.shape):
        raise ValueError(f"{bias_key} shape mismatch: old={tuple(old_bias.shape)} new={tuple(new_bias.shape)}")
    inflated_weight = old_weight.new_zeros(new_weight.shape)
    inflated_weight[:, :OLD_CRITIC_SPATIAL_CHANNELS] = old_weight
    state[weight_key] = inflated_weight
    state[bias_key] = old_bias


def _inflate_critic_scalar_tower(state: dict[str, Any], initialized: dict[str, torch.Tensor]) -> None:
    weight_key = "critic_encoder.scalar_tower.0.weight"
    bias_key = "critic_encoder.scalar_tower.0.bias"
    old_weight = state.get(weight_key)
    old_bias = state.get(bias_key)
    if not isinstance(old_weight, torch.Tensor):
        raise ValueError(f"checkpoint missing {weight_key}")
    if not isinstance(old_bias, torch.Tensor):
        raise ValueError(f"checkpoint missing {bias_key}")
    new_weight = initialized[weight_key]
    new_bias = initialized[bias_key]
    expected_old_shape = (new_weight.shape[0], OLD_CRITIC_SCALAR_FEATURES)
    if tuple(old_weight.shape) != tuple(expected_old_shape):
        raise ValueError(f"{weight_key} must have old shape {tuple(expected_old_shape)}, got {tuple(old_weight.shape)}")
    if tuple(new_weight.shape) != (new_weight.shape[0], NEW_CRITIC_SCALAR_FEATURES):
        raise ValueError(f"new {weight_key} must have {NEW_CRITIC_SCALAR_FEATURES} input columns, got {tuple(new_weight.shape)}")
    if tuple(old_bias.shape) != tuple(new_bias.shape):
        raise ValueError(f"{bias_key} shape mismatch: old={tuple(old_bias.shape)} new={tuple(new_bias.shape)}")
    inflated_weight = old_weight.new_zeros(new_weight.shape)
    inflated_weight[:, :OLD_CRITIC_SCALAR_FEATURES] = old_weight
    state[weight_key] = inflated_weight
    state[bias_key] = old_bias


def _update_model_configs(checkpoint: dict[str, Any], config: dict[str, Any]) -> None:
    if isinstance(checkpoint.get("model_config"), dict):
        checkpoint["model_config"] = dict(config)
    train_config = checkpoint.get("train_config")
    if isinstance(train_config, dict) and isinstance(train_config.get("model"), dict):
        train_config["model"] = dict(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inflate a privileged critic v1 checkpoint to critic feature v2.")
    parser.add_argument("--input", required=True, type=Path, help="Path to a PPO checkpoint with critic feature v1 shape.")
    parser.add_argument("--output", required=True, type=Path, help="Path for the inflated checkpoint.")
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must be a dict")
    inflated = inflate_checkpoint_payload(checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inflated, args.output)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "critic_feature_inflation": inflated["critic_feature_inflation"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
