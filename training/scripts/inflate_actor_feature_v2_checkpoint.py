from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch


OLD_FEATURE_SCHEMA = "goldrush2_feature_v1"
NEW_FEATURE_SCHEMA = "goldrush2_feature_v2"
OLD_ACTOR_SPATIAL_CHANNELS = 38
NEW_ACTOR_SPATIAL_CHANNELS = 43


def inflate_checkpoint_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    inflated = copy.deepcopy(checkpoint)
    state = inflated.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint missing model_state_dict")

    inflated_keys = []
    for key, value in list(state.items()):
        if not _is_actor_stem_weight(key, value):
            continue
        state[key] = _inflate_stem_weight(value)
        inflated_keys.append(key)

    if not inflated_keys:
        raise ValueError("checkpoint does not contain a 38-channel actor stem weight")

    _update_feature_schema(inflated)
    _update_model_configs(inflated)
    if "optimizer_state_dict" in inflated:
        del inflated["optimizer_state_dict"]
        inflated["optimizer_state_dict_dropped_for_actor_feature_inflation"] = True
    inflated["actor_feature_inflation"] = {
        "from_feature_schema": OLD_FEATURE_SCHEMA,
        "to_feature_schema": NEW_FEATURE_SCHEMA,
        "from_actor_spatial_channels": OLD_ACTOR_SPATIAL_CHANNELS,
        "to_actor_spatial_channels": NEW_ACTOR_SPATIAL_CHANNELS,
        "inflated_weight_keys": inflated_keys,
        "new_channel_init": "zero",
    }
    return inflated


def _is_actor_stem_weight(key: str, value: Any) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and key.endswith("stem.0.weight")
        and value.ndim == 4
        and int(value.shape[1]) == OLD_ACTOR_SPATIAL_CHANNELS
    )


def _inflate_stem_weight(weight: torch.Tensor) -> torch.Tensor:
    inflated = weight.new_zeros(
        (
            int(weight.shape[0]),
            NEW_ACTOR_SPATIAL_CHANNELS,
            int(weight.shape[2]),
            int(weight.shape[3]),
        )
    )
    inflated[:, :OLD_ACTOR_SPATIAL_CHANNELS, :, :] = weight
    return inflated


def _update_feature_schema(checkpoint: dict[str, Any]) -> None:
    if "feature_schema" not in checkpoint:
        return
    if checkpoint["feature_schema"] != OLD_FEATURE_SCHEMA:
        raise ValueError(f"checkpoint feature_schema must be {OLD_FEATURE_SCHEMA!r}, got {checkpoint['feature_schema']!r}")
    checkpoint["feature_schema"] = NEW_FEATURE_SCHEMA


def _update_model_configs(checkpoint: dict[str, Any]) -> None:
    changed = False
    if isinstance(checkpoint.get("model_config"), dict):
        _update_model_config(checkpoint["model_config"])
        changed = True
    train_config = checkpoint.get("train_config")
    if isinstance(train_config, dict) and isinstance(train_config.get("model"), dict):
        _update_model_config(train_config["model"])
        changed = True
    if not changed:
        raise ValueError("checkpoint missing model_config or train_config.model")


def _update_model_config(config: dict[str, Any]) -> None:
    if "actor_spatial_channels" in config:
        _replace_channel_value(config, "actor_spatial_channels")
    elif "spatial_channels" in config:
        _replace_channel_value(config, "spatial_channels")
    else:
        raise ValueError("model config missing actor_spatial_channels")


def _replace_channel_value(config: dict[str, Any], key: str) -> None:
    value = int(config[key])
    if value != OLD_ACTOR_SPATIAL_CHANNELS:
        raise ValueError(f"model config {key} must be {OLD_ACTOR_SPATIAL_CHANNELS}, got {value}")
    config[key] = NEW_ACTOR_SPATIAL_CHANNELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Inflate a v1 actor checkpoint to goldrush2_feature_v2.")
    parser.add_argument("--input", required=True, type=Path, help="Path to a v1 BC/PPO checkpoint.")
    parser.add_argument("--output", required=True, type=Path, help="Path for the inflated v2 checkpoint.")
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
                "actor_feature_inflation": inflated["actor_feature_inflation"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
