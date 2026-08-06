from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from simulator.errors import SimulatorRuleError

from .schema import BC_DATASET_SCHEMA, BC_SHARD_SCHEMA, FEATURE_SCHEMA, BcDatasetSplit


@dataclass(frozen=True)
class BcBatch:
    planes: Tensor
    scalars: Tensor
    actions: Tensor
    k: Tensor
    order: Tensor
    vp: Tensor

    def to(self, device: torch.device | str) -> BcBatch:
        return BcBatch(
            planes=self.planes.to(device),
            scalars=self.scalars.to(device),
            actions=self.actions.to(device),
            k=self.k.to(device),
            order=self.order.to(device),
            vp=self.vp.to(device),
        )


class BcTensorDataset(Dataset[BcBatch]):
    def __init__(self, *, planes: Tensor, scalars: Tensor, actions: Tensor, k: Tensor, order: Tensor, vp: Tensor) -> None:
        _validate_arrays(planes, scalars, actions, k, order, vp)
        self.planes = planes.float()
        self.scalars = scalars.float()
        self.actions = actions.long()
        self.k = k.long()
        self.order = order.long()
        self.vp = vp.long()

    def __len__(self) -> int:
        return int(self.k.shape[0])

    def __getitem__(self, index: int) -> BcBatch:
        return BcBatch(
            planes=self.planes[index],
            scalars=self.scalars[index],
            actions=self.actions[index],
            k=self.k[index],
            order=self.order[index],
            vp=self.vp[index],
        )


def load_bc_dataset(dataset_dir: Path, split: BcDatasetSplit) -> BcTensorDataset:
    dataset_dir = Path(dataset_dir)
    manifest = load_manifest(dataset_dir)
    shard_paths = [dataset_dir / shard["path"] for shard in manifest["shards"] if shard["split"] == split]
    if not shard_paths:
        raise SimulatorRuleError(f"BC dataset split {split!r} has no shards")
    shards = [load_shard(path, expected_config_hash=manifest["config_hash"]) for path in shard_paths]
    return BcTensorDataset(
        planes=torch.cat([shard["planes"] for shard in shards], dim=0),
        scalars=torch.cat([shard["scalars"] for shard in shards], dim=0),
        actions=torch.cat([shard["actions"] for shard in shards], dim=0),
        k=torch.cat([shard["k"] for shard in shards], dim=0),
        order=torch.cat([shard["order"] for shard in shards], dim=0),
        vp=torch.cat([shard["vp"] for shard in shards], dim=0),
    )


def load_manifest(dataset_dir: Path) -> dict[str, Any]:
    path = Path(dataset_dir) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != BC_DATASET_SCHEMA:
        raise SimulatorRuleError(f"unsupported BC dataset schema: {manifest.get('schema')!r}")
    if manifest.get("feature_schema") != FEATURE_SCHEMA:
        raise SimulatorRuleError(f"unsupported BC feature schema: {manifest.get('feature_schema')!r}")
    return manifest


def load_shard(path: Path, *, expected_config_hash: str | None = None) -> dict[str, Any]:
    shard = torch.load(Path(path), map_location="cpu", weights_only=False)
    if shard.get("schema") != BC_SHARD_SCHEMA:
        raise SimulatorRuleError(f"unsupported BC shard schema in {path}: {shard.get('schema')!r}")
    if shard.get("feature_schema") != FEATURE_SCHEMA:
        raise SimulatorRuleError(f"unsupported BC shard feature schema in {path}: {shard.get('feature_schema')!r}")
    if expected_config_hash is not None and shard.get("config_hash") != expected_config_hash:
        raise SimulatorRuleError(f"BC shard config hash mismatch in {path}")
    _validate_arrays(shard["planes"], shard["scalars"], shard["actions"], shard["k"], shard["order"], shard["vp"])
    return shard


def collate_bc_batches(items: list[BcBatch]) -> BcBatch:
    if not items:
        raise SimulatorRuleError("cannot collate empty BC batch")
    return BcBatch(
        planes=torch.stack([item.planes for item in items]),
        scalars=torch.stack([item.scalars for item in items]),
        actions=torch.stack([item.actions for item in items]),
        k=torch.stack([item.k.reshape(()) for item in items]),
        order=torch.stack([item.order.reshape(()) for item in items]),
        vp=torch.stack([item.vp.reshape(()) for item in items]),
    )


def _validate_arrays(planes: Tensor, scalars: Tensor, actions: Tensor, k: Tensor, order: Tensor, vp: Tensor) -> None:
    n = int(k.shape[0])
    if tuple(planes.shape) != (n, 38, 17, 17):
        raise ValueError(f"planes must have shape Nx38x17x17, got {tuple(planes.shape)}")
    if tuple(scalars.shape) != (n, 10):
        raise ValueError(f"scalars must have shape Nx10, got {tuple(scalars.shape)}")
    if tuple(actions.shape) != (n, 6):
        raise ValueError(f"actions must have shape Nx6, got {tuple(actions.shape)}")
    for name, tensor in (("order", order), ("vp", vp)):
        if tuple(tensor.shape) != (n,):
            raise ValueError(f"{name} must have shape N, got {tuple(tensor.shape)}")
    if tuple(k.shape) != (n,):
        raise ValueError(f"k must have shape N, got {tuple(k.shape)}")
    if bool(((actions < 0) | (actions > 4)).any().item()):
        raise ValueError("actions must be in [0, 4]")
    if bool(((k < 0) | (k > 6)).any().item()):
        raise ValueError("k must be in [0, 6]")
    if bool(((order < 0) | (order > 1)).any().item()):
        raise ValueError("order must be 0 or 1")
    if bool(((vp < 0) | (vp > 2)).any().item()):
        raise ValueError("vp must be in [0, 2]")
