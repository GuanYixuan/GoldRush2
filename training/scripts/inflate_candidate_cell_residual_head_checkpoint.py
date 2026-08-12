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
    FEATURE_SCHEMA,
    LEGACY_ACTION_HEAD_SCHEMA,
    SCALAR_FEATURES,
    SPATIAL_CHANNELS,
    GoldRushPolicyNetwork,
    PolicyNetworkConfig,
)


def inflate_checkpoint_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    inflated = copy.deepcopy(checkpoint)
    raw_config = _raw_model_config(inflated)
    _require_legacy_head(raw_config)
    _require_actor_feature_v2(inflated, raw_config)

    config = _current_model_config(raw_config)
    model = GoldRushPolicyNetwork(config)
    state = inflated.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint missing model_state_dict")

    initialized = model.state_dict()
    added_keys = []
    for key in initialized:
        if key in state:
            continue
        if _is_candidate_head_key(key):
            state[key] = initialized[key]
            added_keys.append(key)

    expected_added = sorted(key for key in initialized if _is_candidate_head_key(key))
    if sorted(added_keys) != expected_added:
        missing = sorted(set(expected_added) - set(added_keys))
        raise ValueError(f"failed to add candidate residual head keys: {missing}")

    _update_model_configs(inflated, asdict(config))
    if "feature_schema" in inflated and inflated["feature_schema"] != FEATURE_SCHEMA:
        raise ValueError(f"checkpoint feature_schema must be {FEATURE_SCHEMA!r}, got {inflated['feature_schema']!r}")
    if "optimizer_state_dict" in inflated:
        del inflated["optimizer_state_dict"]
        inflated["optimizer_state_dict_dropped_for_action_head_inflation"] = True
    inflated["actor_head_inflation"] = {
        "from_action_head_schema": LEGACY_ACTION_HEAD_SCHEMA,
        "to_action_head_schema": ACTION_HEAD_SCHEMA,
        "residual_init": "zero",
        "added_weight_keys": added_keys,
    }
    return inflated


def _raw_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if isinstance(checkpoint.get("model_config"), dict):
        return dict(checkpoint["model_config"])
    train_config = checkpoint.get("train_config")
    if isinstance(train_config, dict) and isinstance(train_config.get("model"), dict):
        return dict(train_config["model"])
    raise ValueError("checkpoint missing model_config or train_config.model")


def _require_legacy_head(raw_config: dict[str, Any]) -> None:
    schema = str(raw_config.get("action_head_schema", LEGACY_ACTION_HEAD_SCHEMA))
    if schema != LEGACY_ACTION_HEAD_SCHEMA:
        raise ValueError(f"checkpoint action_head_schema must be {LEGACY_ACTION_HEAD_SCHEMA!r}, got {schema!r}")


def _require_actor_feature_v2(checkpoint: dict[str, Any], raw_config: dict[str, Any]) -> None:
    feature_schema = checkpoint.get("feature_schema")
    if feature_schema is not None and feature_schema != FEATURE_SCHEMA:
        raise ValueError(f"checkpoint feature_schema must be {FEATURE_SCHEMA!r}, got {feature_schema!r}")
    actor_channels = int(raw_config.get("actor_spatial_channels", raw_config.get("spatial_channels", -1)))
    actor_scalars = int(raw_config.get("actor_scalar_features", raw_config.get("scalar_features", -1)))
    if actor_channels != SPATIAL_CHANNELS or actor_scalars != SCALAR_FEATURES:
        raise ValueError(
            "checkpoint actor feature shape must be "
            f"{SPATIAL_CHANNELS} spatial channels and {SCALAR_FEATURES} scalars, "
            f"got actor_spatial_channels={actor_channels}, actor_scalar_features={actor_scalars}"
        )


def _current_model_config(raw: dict[str, Any]) -> PolicyNetworkConfig:
    return PolicyNetworkConfig(
        actor_spatial_channels=SPATIAL_CHANNELS,
        actor_scalar_features=SCALAR_FEATURES,
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
        activation=str(raw["activation"]),
    )


def _is_candidate_head_key(key: str) -> bool:
    return key.startswith("action_candidate_embedding.") or key.startswith("candidate_action_head.")


def _update_model_configs(checkpoint: dict[str, Any], config: dict[str, Any]) -> None:
    if isinstance(checkpoint.get("model_config"), dict):
        checkpoint["model_config"] = dict(config)
    train_config = checkpoint.get("train_config")
    if isinstance(train_config, dict) and isinstance(train_config.get("model"), dict):
        train_config["model"] = dict(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inflate a v2 old-head checkpoint to candidate-cell residual head.")
    parser.add_argument("--input", required=True, type=Path, help="Path to a v2 old-head BC/PPO checkpoint.")
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
                "actor_head_inflation": inflated["actor_head_inflation"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
