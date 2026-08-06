from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


BC_DATASET_SCHEMA = "goldrush2_bc_dataset_v1"
BC_SHARD_SCHEMA = "goldrush2_bc_shard_v1"
BC_CHECKPOINT_SCHEMA = "bc_train_v1"
FEATURE_SCHEMA = "goldrush2_feature_v1"

BcDatasetSplit = Literal["train", "val", "holdout"]


@dataclass(frozen=True)
class BcTeacherSpec:
    kind: str = "scripted"
    name: str = "fast_probe_v3_like"
    params: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "params": dict(self.params)}


@dataclass(frozen=True)
class BcTask:
    task_id: int
    split: BcDatasetSplit
    paired_seed: int
    map_id: int

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BcCollectionConfig:
    teacher: BcTeacherSpec = field(default_factory=BcTeacherSpec)
    map_ids: tuple[int, ...] = (1,)
    train_seeds: tuple[int, ...] = ()
    val_seeds: tuple[int, ...] = ()
    holdout_seeds: tuple[int, ...] = ()
    round_count: int = 500
    num_workers: int = 1
    task_chunk_size: int = 1
    shard_transition_limit: int = 20_000
    base_seed: int = 20260806

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "teacher": self.teacher.to_jsonable(),
            "map_ids": list(self.map_ids),
            "train_seeds": list(self.train_seeds),
            "val_seeds": list(self.val_seeds),
            "holdout_seeds": list(self.holdout_seeds),
            "round_count": self.round_count,
            "num_workers": self.num_workers,
            "task_chunk_size": self.task_chunk_size,
            "shard_transition_limit": self.shard_transition_limit,
            "base_seed": self.base_seed,
        }

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_jsonable(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BcShardSummary:
    path: str
    split: BcDatasetSplit
    task_ids: tuple[int, ...]
    transition_count: int
    episode_count: int
    paired_seed_count: int

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "split": self.split,
            "task_ids": list(self.task_ids),
            "transition_count": self.transition_count,
            "episode_count": self.episode_count,
            "paired_seed_count": self.paired_seed_count,
        }


@dataclass(frozen=True)
class BcDatasetManifest:
    schema: str
    feature_schema: str
    config_hash: str
    config: dict[str, Any]
    shards: tuple[BcShardSummary, ...]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "feature_schema": self.feature_schema,
            "config_hash": self.config_hash,
            "config": self.config,
            "shards": [shard.to_jsonable() for shard in self.shards],
        }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
