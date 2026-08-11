#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from training.models import GoldRushPolicyNetwork, PolicyNetworkConfig
from training.models.policy_network import (
    ACTION_COUNT,
    FEATURE_SCHEMA,
    GRID_SIZE,
    KO_COUNT,
    MOVE_BUDGET,
    SCALAR_FEATURES,
    SPATIAL_CHANNELS,
    _EncodedState,
    _gather_position,
)


class StochasticActorExport(nn.Module):
    def __init__(self, model: GoldRushPolicyNetwork) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        actor_planes: Tensor,
        actor_scalars: Tensor,
        rand_ko: Tensor,
        rand_vp: Tensor,
        rand_action: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        actor_planes = actor_planes.float()
        actor_scalars = actor_scalars.float()
        encoded = self.model._encode_actor(actor_planes, actor_scalars)

        ko_logits = self.model.ko_head(encoded.actor_context)
        vp_logits = self.model.vp_head(encoded.actor_context)
        ko = _gumbel_argmax(ko_logits, rand_ko)
        vp = _gumbel_argmax(vp_logits, rand_vp)
        actions = self._decode(encoded, ko, rand_action)
        k = torch.div(ko, 2, rounding_mode="floor")
        order = torch.remainder(ko, 2)
        return actions, k, order, vp

    def _decode(self, encoded: _EncodedState, ko: Tensor, rand_action: Tensor) -> Tensor:
        batch_size = encoded.actor_context.shape[0]
        roles = self.model.execution_role_table[ko]
        official_slots = self.model.official_slot_table[ko]
        ko_embedding = self.model.ko_embedding(ko)
        hidden = torch.tanh(self.model.decoder_initial(encoded.actor_context))
        previous_action = torch.full(
            (batch_size,),
            ACTION_COUNT,
            dtype=torch.long,
            device=encoded.actor_context.device,
        )
        unit0_position = encoded.unit0_position
        unit1_position = encoded.unit1_position
        execution_actions: list[Tensor] = []

        for step in range(MOVE_BUDGET):
            role = roles[:, step]
            current_position = torch.where(role == 0, unit0_position, unit1_position)
            other_position = torch.where(role == 0, unit1_position, unit0_position)
            current_local = _gather_position(encoded.spatial_features, current_position)
            other_local = _gather_position(encoded.spatial_features, other_position)
            step_index = torch.full((batch_size,), step, dtype=torch.long, device=role.device)
            decoder_input = torch.cat(
                (
                    encoded.actor_context,
                    ko_embedding,
                    self.model.role_embedding(role),
                    current_local,
                    other_local,
                    self.model.previous_action_embedding(previous_action),
                    self.model.step_embedding(step_index),
                ),
                dim=1,
            )
            hidden = self.model.decoder(decoder_input, hidden)
            logits = self.model.decoder_action_head(hidden)
            valid_actions, candidate_positions = self.model._movement_candidates(
                current_position,
                other_position,
                encoded.known_obstacles,
            )
            masked_logits = torch.where(valid_actions, logits, torch.full_like(logits, -1.0e9))
            selected = _gumbel_argmax(masked_logits, rand_action[:, step, :])
            selected_position = candidate_positions.gather(1, selected.unsqueeze(1)).squeeze(1)
            selected_valid = valid_actions.gather(1, selected.unsqueeze(1)).squeeze(1)
            next_position = torch.where(selected_valid, selected_position, current_position)
            unit0_position = torch.where(role == 0, next_position, unit0_position)
            unit1_position = torch.where(role == 1, next_position, unit1_position)
            execution_actions.append(selected)
            previous_action = selected

        actions_in_execution_order = torch.stack(execution_actions, dim=1)
        return torch.zeros_like(actions_in_execution_order).scatter(1, official_slots, actions_in_execution_order)


def _gumbel_argmax(logits: Tensor, rand: Tensor) -> Tensor:
    u = rand.float().clamp(1.0e-6, 1.0 - 1.0e-6)
    gumbel = -torch.log(-torch.log(u))
    return torch.argmax(logits + gumbel, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export actor-only stochastic GoldRush policy to FP32 ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--seed", type=int, default=810)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    model, checkpoint = _load_model(checkpoint_path)
    wrapper = StochasticActorExport(model).eval()

    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    sample = _sample_inputs(generator)
    with torch.no_grad():
        torch_outputs = wrapper(*sample)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        sample,
        output_path,
        input_names=["actor_planes", "actor_scalars", "rand_ko", "rand_vp", "rand_action"],
        output_names=["actions", "k", "order", "vp"],
        opset_version=int(args.opset),
        do_constant_folding=True,
        dynamic_axes=None,
    )

    ort_outputs = _run_onnx(output_path, sample)
    _require_equal_outputs(torch_outputs, ort_outputs)
    ops = _require_deployable_onnx(output_path)
    metadata = {
        "schema": "goldrush2_stochastic_actor_onnx_export_v1",
        "feature_schema": FEATURE_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpoint_schema": checkpoint.get("schema"),
        "checkpoint_update": checkpoint.get("update_index"),
        "onnx": str(output_path),
        "onnx_bytes": output_path.stat().st_size,
        "opset": int(args.opset),
        "operator_count": len(ops),
        "operator_types": sorted(set(ops)),
        "stochastic": True,
        "critic_exported": False,
        "model_config": asdict(model.config),
        "inputs": {
            "actor_planes": [1, SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE],
            "actor_scalars": [1, SCALAR_FEATURES],
            "rand_ko": [1, KO_COUNT],
            "rand_vp": [1, 3],
            "rand_action": [1, MOVE_BUDGET, ACTION_COUNT],
        },
        "outputs": {
            "actions": [1, MOVE_BUDGET],
            "k": [1],
            "order": [1],
            "vp": [1],
        },
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"onnx": str(output_path), "metadata": str(metadata_path), "onnx_bytes": output_path.stat().st_size}, sort_keys=True))


def _load_model(path: Path) -> tuple[GoldRushPolicyNetwork, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "ppo_train_v1":
        raise RuntimeError(f"checkpoint must be ppo_train_v1, got {checkpoint.get('schema')!r}")
    raw_config = checkpoint.get("train_config", {}).get("model")
    if not isinstance(raw_config, dict):
        raise RuntimeError("checkpoint missing train_config.model")
    config = PolicyNetworkConfig(
        actor_spatial_channels=int(raw_config["actor_spatial_channels"]),
        actor_scalar_features=int(raw_config["actor_scalar_features"]),
        critic_spatial_channels=int(raw_config["critic_spatial_channels"]),
        critic_scalar_features=int(raw_config["critic_scalar_features"]),
        width=int(raw_config["width"]),
        residual_blocks=int(raw_config["residual_blocks"]),
        se_reduction=int(raw_config["se_reduction"]),
        scalar_hidden=tuple(int(v) for v in raw_config["scalar_hidden"]),
        actor_hidden=int(raw_config["actor_hidden"]),
        critic_hidden=tuple(int(v) for v in raw_config["critic_hidden"]),
        decoder_hidden=int(raw_config["decoder_hidden"]),
        decoder_embedding=int(raw_config["decoder_embedding"]),
        activation=str(raw_config["activation"]),
    )
    if config.actor_spatial_channels != SPATIAL_CHANNELS or config.actor_scalar_features != SCALAR_FEATURES:
        raise RuntimeError(f"unexpected actor feature shape in checkpoint: {config}")
    model = GoldRushPolicyNetwork(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _sample_inputs(generator: torch.Generator) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    actor_planes = torch.randn((1, SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE), generator=generator)
    actor_scalars = torch.randn((1, SCALAR_FEATURES), generator=generator)
    rand_ko = torch.rand((1, KO_COUNT), generator=generator).clamp(1.0e-6, 1.0 - 1.0e-6)
    rand_vp = torch.rand((1, 3), generator=generator).clamp(1.0e-6, 1.0 - 1.0e-6)
    rand_action = torch.rand((1, MOVE_BUDGET, ACTION_COUNT), generator=generator).clamp(1.0e-6, 1.0 - 1.0e-6)
    return actor_planes, actor_scalars, rand_ko, rand_vp, rand_action


def _run_onnx(path: Path, inputs: tuple[Tensor, Tensor, Tensor, Tensor, Tensor]) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = ["actor_planes", "actor_scalars", "rand_ko", "rand_vp", "rand_action"]
    feed = {name: tensor.detach().cpu().numpy().astype(np.float32) for name, tensor in zip(names, inputs)}
    return session.run(["actions", "k", "order", "vp"], feed)


def _require_deployable_onnx(path: Path) -> list[str]:
    import onnx

    forbidden = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike", "Multinomial"}
    model = onnx.load(str(path))
    ops = [node.op_type for node in model.graph.node]
    found = sorted(forbidden.intersection(ops))
    if found:
        raise RuntimeError(f"ONNX graph contains non-deployable random ops: {found}")
    return ops


def _require_equal_outputs(torch_outputs: tuple[Tensor, Tensor, Tensor, Tensor], ort_outputs: list[np.ndarray]) -> None:
    for name, torch_value, ort_value in zip(("actions", "k", "order", "vp"), torch_outputs, ort_outputs):
        expected = torch_value.detach().cpu().numpy()
        if not np.array_equal(expected, ort_value):
            raise RuntimeError(f"ONNX output mismatch for {name}: torch={expected.tolist()} ort={ort_value.tolist()}")


if __name__ == "__main__":
    main()
