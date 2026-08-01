from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from .base import OpponentSpec
from .params import ParamSpace


LeagueSplit = Literal["train", "eval", "holdout"]


@dataclass(frozen=True)
class LeagueEntry:
    id: str
    weight: float
    spec_template: OpponentSpec
    param_space: ParamSpace = field(default_factory=ParamSpace)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("league entry id must be non-empty")
        if self.weight <= 0:
            raise ValueError(f"league entry weight must be positive, got {self.weight}")

    def sample_spec(self, rng: random.Random) -> OpponentSpec:
        params = dict(self.spec_template.params)
        params.update(self.param_space.sample(rng))
        return OpponentSpec(kind=self.spec_template.kind, name=self.spec_template.name, params=params)


@dataclass(frozen=True)
class OpponentLeague:
    train: tuple[LeagueEntry, ...] = ()
    eval: tuple[LeagueEntry, ...] = ()
    holdout: tuple[LeagueEntry, ...] = ()

    @staticmethod
    def from_splits(splits: Mapping[str, Sequence[LeagueEntry]]) -> OpponentLeague:
        unknown = set(splits) - {"train", "eval", "holdout"}
        if unknown:
            raise ValueError(f"unknown league splits: {sorted(unknown)}")
        return OpponentLeague(
            train=tuple(splits.get("train", ())),
            eval=tuple(splits.get("eval", ())),
            holdout=tuple(splits.get("holdout", ())),
        )

    def entries(self, split: LeagueSplit) -> tuple[LeagueEntry, ...]:
        if split == "train":
            return self.train
        if split == "eval":
            return self.eval
        if split == "holdout":
            return self.holdout
        raise ValueError(f"unknown league split: {split!r}")

    def sample_entry(self, rng: random.Random, split: LeagueSplit = "train") -> LeagueEntry:
        entries = self.entries(split)
        if not entries:
            raise ValueError(f"league split {split!r} is empty")
        total = sum(entry.weight for entry in entries)
        target = rng.random() * total
        cumulative = 0.0
        for entry in entries:
            cumulative += entry.weight
            if target < cumulative:
                return entry
        return entries[-1]

    def sample_spec(self, rng: random.Random, split: LeagueSplit = "train") -> OpponentSpec:
        return self.sample_entry(rng, split).sample_spec(rng)
