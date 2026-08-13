from __future__ import annotations

from pathlib import Path

import torch

from simulator.config import EpisodeConfig, RulesConfig
from training.bc.audit import audit_dataset
from training.bc.collect import collect_dataset
from training.bc.dataset import BcBatch, load_bc_dataset
from training.bc.losses import BcLossConfig, bc_loss
from training.bc.schema import BcCollectionConfig, BcTeacherSpec
from training.bc.train_bc import TrainBcConfig, run_bc_training
from training.models import GoldRushPolicyNetwork, PolicyNetworkConfig
from training.rl import PpoConfig
from training.scripts.train_ppo import TrainPpoConfig, run_training


def test_collect_bc_dataset_writes_expected_transitions(tmp_path: Path) -> None:
    config = _collection_config(train_seeds=(100,), round_count=4, num_workers=1)
    result = collect_dataset(tmp_path / "dataset", config)

    assert result.manifest.config_hash == config.config_hash
    assert len(result.manifest.shards) == 1
    assert result.manifest.shards[0].transition_count == 8

    dataset = load_bc_dataset(tmp_path / "dataset", "train")
    assert len(dataset) == 8
    assert tuple(dataset.planes.shape) == (8, 43, 17, 17)
    assert tuple(dataset.scalars.shape) == (8, 10)
    assert tuple(dataset.actions.shape) == (8, 6)
    assert set(dataset.k.tolist()) <= set(range(7))


def test_parallel_collection_preserves_sample_metadata(tmp_path: Path) -> None:
    config_serial = _collection_config(train_seeds=(110, 111), round_count=3, num_workers=1)
    config_parallel = _collection_config(train_seeds=(110, 111), round_count=3, num_workers=2)

    collect_dataset(tmp_path / "serial", config_serial)
    collect_dataset(tmp_path / "parallel", config_parallel)

    serial = load_bc_dataset(tmp_path / "serial", "train")
    parallel = load_bc_dataset(tmp_path / "parallel", "train")
    assert torch.equal(serial.actions, parallel.actions)
    assert torch.equal(serial.k, parallel.k)
    assert torch.equal(serial.order, parallel.order)
    assert torch.equal(serial.vp, parallel.vp)


def test_collection_resume_reuses_existing_shard(tmp_path: Path) -> None:
    config = _collection_config(train_seeds=(115,), round_count=3, num_workers=1)
    first = collect_dataset(tmp_path / "dataset", config)
    shard_path = tmp_path / "dataset" / first.manifest.shards[0].path
    first_mtime = shard_path.stat().st_mtime_ns

    second = collect_dataset(tmp_path / "dataset", config, resume=True)

    assert second.manifest.shards[0].transition_count == first.manifest.shards[0].transition_count
    assert shard_path.stat().st_mtime_ns == first_mtime


def test_audit_and_bc_loss_run_on_collected_dataset(tmp_path: Path) -> None:
    config = _collection_config(train_seeds=(120,), round_count=3, num_workers=1)
    collect_dataset(tmp_path / "dataset", config)

    audit = audit_dataset(
        tmp_path / "dataset",
        model_config=_small_model_config(),
        sample_limit=6,
    )
    assert audit.summary["transition_count"] == 6
    assert "hard_mask_incompatibility_count" in audit.summary

    dataset = load_bc_dataset(tmp_path / "dataset", "train")
    batch = BcBatch(
        planes=dataset.planes[:4],
        scalars=dataset.scalars[:4],
        actions=dataset.actions[:4],
        k=dataset.k[:4],
        order=dataset.order[:4],
        vp=dataset.vp[:4],
    )
    model = GoldRushPolicyNetwork(_small_model_config())
    result = bc_loss(model, batch, BcLossConfig())
    assert torch.isfinite(result.loss)


def test_train_bc_and_init_ppo_from_checkpoint(tmp_path: Path) -> None:
    collect_dataset(
        tmp_path / "dataset",
        _collection_config(train_seeds=(130,), val_seeds=(230,), round_count=3, num_workers=1),
    )
    bc_result = run_bc_training(
        TrainBcConfig(
            dataset_dir=tmp_path / "dataset",
            output_dir=tmp_path / "bc_run",
            epochs=1,
            batch_size=4,
            model=_small_model_config(),
        )
    )
    assert bc_result.latest_checkpoint.exists()

    ppo_result = run_training(
        TrainPpoConfig(
            total_updates=1,
            pair_count=1,
            output_dir=tmp_path / "ppo_run",
            episode=EpisodeConfig(rules=RulesConfig(round_count=2)),
            model=_small_model_config(),
            ppo=PpoConfig(minibatch_size=4, update_epochs=1, target_joint_kl=None),
            init_model_checkpoint=bc_result.latest_checkpoint,
        )
    )
    assert ppo_result.latest_checkpoint.exists()


def test_bc_training_freezes_threshold_head(tmp_path: Path) -> None:
    collect_dataset(
        tmp_path / "dataset",
        _collection_config(train_seeds=(140,), round_count=3, num_workers=1),
    )
    config = TrainBcConfig(
        dataset_dir=tmp_path / "dataset",
        output_dir=tmp_path / "bc_run",
        epochs=1,
        batch_size=4,
        learning_rate=1.0e-3,
        model=_small_model_config(),
    )
    torch.manual_seed(config.seed)
    initial_model = GoldRushPolicyNetwork(config.model)

    result = run_bc_training(config)
    checkpoint = torch.load(result.latest_checkpoint, map_location="cpu", weights_only=False)
    trained_state = checkpoint["model_state_dict"]

    for key, value in initial_model.state_dict().items():
        if key.startswith("threshold_mlp.") or key == "threshold_log_std":
            assert torch.equal(trained_state[key], value)


def _collection_config(
    *,
    train_seeds: tuple[int, ...],
    val_seeds: tuple[int, ...] = (),
    round_count: int,
    num_workers: int,
) -> BcCollectionConfig:
    return BcCollectionConfig(
        teacher=BcTeacherSpec(kind="scripted", name="fast_probe_v3_like"),
        map_ids=(1,),
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        round_count=round_count,
        num_workers=num_workers,
        task_chunk_size=1,
        shard_transition_limit=1000,
    )


def _small_model_config() -> PolicyNetworkConfig:
    return PolicyNetworkConfig(
        width=16,
        residual_blocks=1,
        se_reduction=4,
        scalar_hidden=(16, 16),
        actor_hidden=32,
        critic_hidden=(32, 16),
        decoder_hidden=16,
        decoder_embedding=8,
    )
