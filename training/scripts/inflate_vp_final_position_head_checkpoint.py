from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from training.models.policy_network import (
    ACTION_HEAD_SCHEMA,
    FAST_THRESHOLD_ACTION_HEAD_SCHEMA,
    FEATURE_SCHEMA,
    GoldRushPolicyNetwork,
    PolicyNetworkConfig,
)


def inflate_checkpoint_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    inflated = copy.deepcopy(checkpoint)
    raw_config = _raw_model_config(inflated)
    _require_fast_threshold_checkpoint(inflated, raw_config)

    config = _current_model_config(raw_config)
    model = GoldRushPolicyNetwork(config)
    initialized = model.state_dict()
    state = inflated.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint missing model_state_dict")

    _inflate_vp_head(state, initialized, actor_hidden=config.actor_hidden, width=config.width)
    _update_model_configs(inflated, asdict(config))
    if "feature_schema" in inflated and inflated["feature_schema"] != FEATURE_SCHEMA:
        raise ValueError(f"checkpoint feature_schema must be {FEATURE_SCHEMA!r}, got {inflated['feature_schema']!r}")
    if "optimizer_state_dict" in inflated:
        del inflated["optimizer_state_dict"]
        inflated["optimizer_state_dict_dropped_for_vp_head_inflation"] = True
    inflated["vp_head_inflation"] = {
        "from_action_head_schema": FAST_THRESHOLD_ACTION_HEAD_SCHEMA,
        "to_action_head_schema": ACTION_HEAD_SCHEMA,
        "old_actor_context_columns": int(config.actor_hidden),
        "new_position_columns": int(config.width) * 2,
        "new_position_init": "zero",
    }
    return inflated


def _raw_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if isinstance(checkpoint.get("model_config"), dict):
        return dict(checkpoint["model_config"])
    train_config = checkpoint.get("train_config")
    if isinstance(train_config, dict) and isinstance(train_config.get("model"), dict):
        return dict(train_config["model"])
    raise ValueError("checkpoint missing model_config or train_config.model")


def _require_fast_threshold_checkpoint(checkpoint: dict[str, Any], raw_config: dict[str, Any]) -> None:
    schema = str(raw_config.get("action_head_schema", ""))
    if schema != FAST_THRESHOLD_ACTION_HEAD_SCHEMA:
        raise ValueError(f"checkpoint action_head_schema must be {FAST_THRESHOLD_ACTION_HEAD_SCHEMA!r}, got {schema!r}")
    feature_schema = checkpoint.get("feature_schema")
    if feature_schema is not None and feature_schema != FEATURE_SCHEMA:
        raise ValueError(f"checkpoint feature_schema must be {FEATURE_SCHEMA!r}, got {feature_schema!r}")


def _current_model_config(raw: dict[str, Any]) -> PolicyNetworkConfig:
    return PolicyNetworkConfig(
        actor_spatial_channels=int(raw["actor_spatial_channels"]),
        actor_scalar_features=int(raw["actor_scalar_features"]),
        critic_spatial_channels=int(raw["critic_spatial_channels"]),
        critic_scalar_features=int(raw["critic_scalar_features"]),
        width=int(raw["width"]),
        residual_blocks=int(raw["residual_blocks"]),
        se_reduction=int(raw["se_reduction"]),
        scalar_hidden=tuple(int(value) for value in raw["scalar_hidden"]),
        actor_hidden=int(raw["actor_hidden"]),
        critic_hidden=tuple(int(value) for value in raw["critic_hidden"]),
        decoder_hidden=int(raw["decoder_hidden"]),
        decoder_embedding=int(raw["decoder_embedding"]),
        action_head_schema=ACTION_HEAD_SCHEMA,
        fast_scalar_features=int(raw.get("fast_scalar_features", 2)),
        threshold_initial=float(raw.get("threshold_initial", 12.0)),
        threshold_hidden=tuple(int(value) for value in raw.get("threshold_hidden", (128, 64))),
        threshold_log_std_initial=float(raw.get("threshold_log_std_initial", -1.3)),
        threshold_log_std_min=float(raw.get("threshold_log_std_min", -3.0)),
        threshold_log_std_max=float(raw.get("threshold_log_std_max", 0.0)),
        threshold_entropy_coef=float(raw.get("threshold_entropy_coef", 0.0)),
        activation=str(raw["activation"]),
    )


def _inflate_vp_head(
    state: dict[str, Any],
    initialized: dict[str, torch.Tensor],
    *,
    actor_hidden: int,
    width: int,
) -> None:
    weight_key = "vp_head.weight"
    bias_key = "vp_head.bias"
    old_weight = state.get(weight_key)
    old_bias = state.get(bias_key)
    new_weight = initialized[weight_key]
    new_bias = initialized[bias_key]
    if not isinstance(old_weight, torch.Tensor):
        raise ValueError(f"checkpoint missing {weight_key}")
    if not isinstance(old_bias, torch.Tensor):
        raise ValueError(f"checkpoint missing {bias_key}")
    expected_old_shape = (3, int(actor_hidden))
    expected_new_shape = (3, int(actor_hidden) + int(width) * 2)
    if tuple(old_weight.shape) != expected_old_shape:
        raise ValueError(f"{weight_key} must have old shape {expected_old_shape}, got {tuple(old_weight.shape)}")
    if tuple(new_weight.shape) != expected_new_shape:
        raise ValueError(f"new {weight_key} must have shape {expected_new_shape}, got {tuple(new_weight.shape)}")
    if tuple(old_bias.shape) != tuple(new_bias.shape):
        raise ValueError(f"{bias_key} shape mismatch: old={tuple(old_bias.shape)} new={tuple(new_bias.shape)}")
    inflated_weight = old_weight.new_zeros(new_weight.shape)
    inflated_weight[:, : old_weight.shape[1]] = old_weight
    state[weight_key] = inflated_weight
    state[bias_key] = old_bias


def _update_model_configs(checkpoint: dict[str, Any], config: dict[str, Any]) -> None:
    if isinstance(checkpoint.get("model_config"), dict):
        checkpoint["model_config"] = dict(config)
    train_config = checkpoint.get("train_config")
    if isinstance(train_config, dict) and isinstance(train_config.get("model"), dict):
        train_config["model"] = dict(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inflate a fast-threshold checkpoint to final-position VP head schema.")
    parser.add_argument("--input", required=True, type=Path, help="Path to a fast-threshold PPO checkpoint.")
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
                "vp_head_inflation": inflated["vp_head_inflation"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
