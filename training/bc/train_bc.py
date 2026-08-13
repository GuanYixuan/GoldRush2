from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader

from simulator.errors import SimulatorRuleError
from training.models import GoldRushPolicyNetwork, PolicyNetworkConfig

from .dataset import collate_bc_batches, load_bc_dataset, load_manifest
from .losses import BcLossConfig, bc_loss
from .schema import BC_CHECKPOINT_SCHEMA, FEATURE_SCHEMA, json_hash, write_json


@dataclass(frozen=True)
class TrainBcConfig:
    dataset_dir: Path
    output_dir: Path
    seed: int = 20260806
    device: str = "cpu"
    epochs: int = 1
    batch_size: int = 1024
    learning_rate: float = 2.0e-4
    adam_eps: float = 1.0e-5
    num_workers: int = 0
    loss: BcLossConfig = field(default_factory=BcLossConfig)
    model: PolicyNetworkConfig = field(default_factory=PolicyNetworkConfig)


@dataclass(frozen=True)
class TrainBcResult:
    output_dir: Path
    latest_checkpoint: Path
    metrics: tuple[dict[str, Any], ...]


def run_bc_training(config: TrainBcConfig) -> TrainBcResult:
    _validate_config(config)
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(config.dataset_dir)
    dataset_hash = json_hash(manifest)
    write_json(output_dir / "config.json", _config_jsonable(config, dataset_hash))

    train_dataset = load_bc_dataset(config.dataset_dir, "train")
    val_dataset = None
    if any(shard["split"] == "val" for shard in manifest["shards"]):
        val_dataset = load_bc_dataset(config.dataset_dir, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_bc_batches,
    )
    val_loader = None if val_dataset is None else DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_bc_batches,
    )

    device = torch.device(config.device)
    model = GoldRushPolicyNetwork(config.model).to(device)
    optimizer = torch.optim.Adam(model.ordinary_actor_parameters(), lr=config.learning_rate, eps=config.adam_eps)
    metrics: list[dict[str, Any]] = []
    metrics_path = output_dir / "metrics.jsonl"

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_record = _run_epoch(model, optimizer, train_loader, config=config, device=device, epoch=epoch)
        _append_jsonl(metrics_path, train_record)
        metrics.append(train_record)
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_record = _run_validation(model, val_loader, config=config, device=device, epoch=epoch)
            _append_jsonl(metrics_path, val_record)
            metrics.append(val_record)
        latest = _save_checkpoint(
            checkpoint_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            dataset_manifest_hash=dataset_hash,
            epoch=epoch,
            metrics=metrics[-1],
        )
        _save_checkpoint(
            checkpoint_dir / f"epoch_{epoch:04d}.pt",
            model=model,
            optimizer=optimizer,
            config=config,
            dataset_manifest_hash=dataset_hash,
            epoch=epoch,
            metrics=metrics[-1],
        )

    return TrainBcResult(output_dir=output_dir, latest_checkpoint=latest, metrics=tuple(metrics))


def _run_epoch(
    model: GoldRushPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    *,
    config: TrainBcConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    losses: list[float] = []
    ko: list[float] = []
    action: list[float] = []
    vp: list[float] = []
    acc_ko: list[float] = []
    acc_vp: list[float] = []
    for batch in loader:
        batch = batch.to(device)
        result = bc_loss(model, batch, config.loss)
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        optimizer.step()
        _append_loss_stats(result, losses, ko, action, vp, acc_ko, acc_vp)
    return _metrics("train", epoch, losses, ko, action, vp, acc_ko, acc_vp)


def _run_validation(
    model: GoldRushPolicyNetwork,
    loader: DataLoader,
    *,
    config: TrainBcConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, Any]:
    losses: list[float] = []
    ko: list[float] = []
    action: list[float] = []
    vp: list[float] = []
    acc_ko: list[float] = []
    acc_vp: list[float] = []
    for batch in loader:
        result = bc_loss(model, batch.to(device), config.loss)
        _append_loss_stats(result, losses, ko, action, vp, acc_ko, acc_vp)
    return _metrics("val", epoch, losses, ko, action, vp, acc_ko, acc_vp)


def _append_loss_stats(result, losses, ko, action, vp, acc_ko, acc_vp) -> None:
    losses.append(float(result.loss.detach().cpu().item()))
    ko.append(float(result.nll_ko.detach().cpu().item()))
    action.append(float(result.nll_action.detach().cpu().item()))
    vp.append(float(result.nll_vp.detach().cpu().item()))
    acc_ko.append(float(result.accuracy_ko.detach().cpu().item()))
    acc_vp.append(float(result.accuracy_vp.detach().cpu().item()))


def _metrics(kind: str, epoch: int, losses, ko, action, vp, acc_ko, acc_vp) -> dict[str, Any]:
    return {
        "kind": kind,
        "epoch": epoch,
        "loss": mean(losses),
        "nll_ko": mean(ko),
        "nll_action": mean(action),
        "nll_vp": mean(vp),
        "accuracy_ko": mean(acc_ko),
        "accuracy_vp": mean(acc_vp),
    }


def _save_checkpoint(
    path: Path,
    *,
    model: GoldRushPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    config: TrainBcConfig,
    dataset_manifest_hash: str,
    epoch: int,
    metrics: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": BC_CHECKPOINT_SCHEMA,
            "feature_schema": FEATURE_SCHEMA,
            "dataset_manifest_hash": dataset_manifest_hash,
            "bc_config": _config_jsonable(config, dataset_manifest_hash),
            "model_config": asdict(config.model),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )
    return path


def _config_jsonable(config: TrainBcConfig, dataset_hash: str) -> dict[str, Any]:
    return {
        "dataset_dir": str(config.dataset_dir),
        "output_dir": str(config.output_dir),
        "seed": config.seed,
        "device": config.device,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "adam_eps": config.adam_eps,
        "num_workers": config.num_workers,
        "loss": asdict(config.loss),
        "model": asdict(config.model),
        "dataset_manifest_hash": dataset_hash,
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")


def _validate_config(config: TrainBcConfig) -> None:
    if config.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {config.epochs}")
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.batch_size}")
    if config.learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {config.learning_rate}")
    if config.num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {config.num_workers}")
    if not Path(config.dataset_dir).exists():
        raise FileNotFoundError(config.dataset_dir)
