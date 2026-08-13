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
    CANDIDATE_ACTION_HEAD_SCHEMA,
    FEATURE_SCHEMA,
    GoldRushPolicyNetwork,
    PolicyNetworkConfig,
)


def inflate_checkpoint_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    inflated = copy.deepcopy(checkpoint)
    raw_config = _raw_model_config(inflated)
    _require_candidate_checkpoint(inflated, raw_config)

    config = _current_model_config(raw_config)
    model = GoldRushPolicyNetwork(config)
    initialized = model.state_dict()
    state = inflated.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint missing model_state_dict")

    added_keys: list[str] = []
    for key in initialized:
        if key in state:
            continue
        if _is_fast_threshold_key(key):
            state[key] = initialized[key]
            added_keys.append(key)

    _inflate_critic_fast_scalar_columns(state, initialized, added_keys)
    expected_threshold_keys = sorted(key for key in initialized if _is_fast_threshold_key(key))
    added_threshold_keys = sorted(key for key in added_keys if _is_fast_threshold_key(key))
    if added_threshold_keys != expected_threshold_keys:
        missing = sorted(set(expected_threshold_keys) - set(added_threshold_keys))
        raise ValueError(f"failed to add fast threshold keys: {missing}")

    _update_model_configs(inflated, asdict(config))
    if "feature_schema" in inflated and inflated["feature_schema"] != FEATURE_SCHEMA:
        raise ValueError(f"checkpoint feature_schema must be {FEATURE_SCHEMA!r}, got {inflated['feature_schema']!r}")
    if "optimizer_state_dict" in inflated:
        del inflated["optimizer_state_dict"]
        inflated["optimizer_state_dict_dropped_for_fast_threshold_inflation"] = True
    inflated["fast_threshold_inflation"] = {
        "from_action_head_schema": CANDIDATE_ACTION_HEAD_SCHEMA,
        "to_action_head_schema": ACTION_HEAD_SCHEMA,
        "threshold_initial": config.threshold_initial,
        "threshold_log_std_initial": config.threshold_log_std_initial,
        "critic_fast_scalar_columns": "zero",
        "added_weight_keys": sorted(added_keys),
    }
    return inflated


def _raw_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if isinstance(checkpoint.get("model_config"), dict):
        return dict(checkpoint["model_config"])
    train_config = checkpoint.get("train_config")
    if isinstance(train_config, dict) and isinstance(train_config.get("model"), dict):
        return dict(train_config["model"])
    raise ValueError("checkpoint missing model_config or train_config.model")


def _require_candidate_checkpoint(checkpoint: dict[str, Any], raw_config: dict[str, Any]) -> None:
    schema = str(raw_config.get("action_head_schema", ""))
    if schema != CANDIDATE_ACTION_HEAD_SCHEMA:
        raise ValueError(f"checkpoint action_head_schema must be {CANDIDATE_ACTION_HEAD_SCHEMA!r}, got {schema!r}")
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


def _is_fast_threshold_key(key: str) -> bool:
    return key.startswith("threshold_mlp.") or key == "threshold_log_std"


def _inflate_critic_fast_scalar_columns(
    state: dict[str, Any],
    initialized: dict[str, torch.Tensor],
    added_keys: list[str],
) -> None:
    key = "critic_mlp.0.weight"
    old = state.get(key)
    new = initialized[key]
    if not isinstance(old, torch.Tensor):
        raise ValueError(f"checkpoint missing {key}")
    if old.shape == new.shape:
        return
    if old.ndim != 2 or new.ndim != 2 or old.shape[0] != new.shape[0] or old.shape[1] + 2 != new.shape[1]:
        raise ValueError(f"cannot inflate {key}: old={tuple(old.shape)} new={tuple(new.shape)}")
    inflated = old.new_zeros(new.shape)
    inflated[:, : old.shape[1]] = old
    state[key] = inflated
    added_keys.append(key)


def _update_model_configs(checkpoint: dict[str, Any], config: dict[str, Any]) -> None:
    if isinstance(checkpoint.get("model_config"), dict):
        checkpoint["model_config"] = dict(config)
    train_config = checkpoint.get("train_config")
    if isinstance(train_config, dict) and isinstance(train_config.get("model"), dict):
        train_config["model"] = dict(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inflate a candidate-cell v2 checkpoint to fast threshold head schema.")
    parser.add_argument("--input", required=True, type=Path, help="Path to a candidate-cell v2 PPO checkpoint.")
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
                "fast_threshold_inflation": inflated["fast_threshold_inflation"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
