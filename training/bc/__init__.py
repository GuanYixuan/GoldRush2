"""Behavior cloning data and training utilities for GoldRush2."""

from .schema import (
    BC_DATASET_SCHEMA,
    BC_SHARD_SCHEMA,
    BcCollectionConfig,
    BcDatasetManifest,
    BcDatasetSplit,
    BcShardSummary,
    BcTeacherSpec,
)
from .teachers import build_teacher

__all__ = [
    "BC_DATASET_SCHEMA",
    "BC_SHARD_SCHEMA",
    "BcCollectionConfig",
    "BcDatasetManifest",
    "BcDatasetSplit",
    "BcShardSummary",
    "BcTeacherSpec",
    "build_teacher",
]
