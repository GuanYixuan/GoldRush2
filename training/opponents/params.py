from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal


ParamKind = Literal["constant", "choice", "categorical", "uniform", "log_uniform", "int_uniform", "maybe"]


@dataclass(frozen=True)
class ParamSpec:
    kind: ParamKind
    value: Any = None
    choices: tuple[Any, ...] = ()
    weights: tuple[float, ...] = ()
    low: float | int | None = None
    high: float | int | None = None
    probability: float | None = None
    nested: Any = None

    def __post_init__(self) -> None:
        if self.kind == "choice":
            if not self.choices:
                raise ValueError("choice ParamSpec requires non-empty choices")
        elif self.kind == "categorical":
            if not self.choices:
                raise ValueError("categorical ParamSpec requires non-empty choices")
            if len(self.choices) != len(self.weights):
                raise ValueError("categorical choices and weights must have same length")
            if any(weight < 0 for weight in self.weights) or sum(self.weights) <= 0:
                raise ValueError("categorical weights must be non-negative with positive sum")
        elif self.kind in ("uniform", "log_uniform", "int_uniform"):
            if self.low is None or self.high is None:
                raise ValueError(f"{self.kind} ParamSpec requires low/high")
            if self.low > self.high:
                raise ValueError(f"{self.kind} ParamSpec requires low <= high")
            if self.kind == "log_uniform" and (self.low <= 0 or self.high <= 0):
                raise ValueError("log_uniform ParamSpec requires positive low/high")
        elif self.kind == "maybe":
            if self.probability is None or not 0.0 <= self.probability <= 1.0:
                raise ValueError("maybe ParamSpec requires probability in [0, 1]")
        elif self.kind != "constant":
            raise ValueError(f"unsupported ParamSpec kind: {self.kind!r}")

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "constant":
            return self.value
        if self.kind == "choice":
            return self.choices[rng.randrange(len(self.choices))]
        if self.kind == "categorical":
            return _weighted_choice(rng, self.choices, self.weights)
        if self.kind == "uniform":
            return rng.uniform(float(self.low), float(self.high))
        if self.kind == "log_uniform":
            return math.exp(rng.uniform(math.log(float(self.low)), math.log(float(self.high))))
        if self.kind == "int_uniform":
            return rng.randint(int(self.low), int(self.high))
        if self.kind == "maybe":
            if rng.random() >= float(self.probability):
                return None
            return _coerce_param_spec(self.nested).sample(rng)
        raise ValueError(f"unsupported ParamSpec kind: {self.kind!r}")


@dataclass(frozen=True)
class ParamSpace:
    specs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coerced = {name: _coerce_param_spec(spec) for name, spec in self.specs.items()}
        object.__setattr__(self, "specs", coerced)

    def sample(self, rng: random.Random) -> dict[str, Any]:
        return {name: spec.sample(rng) for name, spec in self.specs.items()}


def constant(value: Any) -> ParamSpec:
    return ParamSpec("constant", value=value)


def choice(*choices: Any) -> ParamSpec:
    return ParamSpec("choice", choices=tuple(choices))


def categorical(choices: Mapping[Any, float]) -> ParamSpec:
    return ParamSpec("categorical", choices=tuple(choices.keys()), weights=tuple(float(value) for value in choices.values()))


def uniform(low: float, high: float) -> ParamSpec:
    return ParamSpec("uniform", low=low, high=high)


def log_uniform(low: float, high: float) -> ParamSpec:
    return ParamSpec("log_uniform", low=low, high=high)


def int_uniform(low: int, high: int) -> ParamSpec:
    return ParamSpec("int_uniform", low=low, high=high)


def maybe(probability: float, nested: Any) -> ParamSpec:
    return ParamSpec("maybe", probability=probability, nested=nested)


def _coerce_param_spec(spec: Any) -> ParamSpec:
    if isinstance(spec, ParamSpec):
        return spec
    if isinstance(spec, Mapping):
        if "kind" not in spec:
            raise ValueError(f"param spec mapping requires kind: {spec}")
        known = {"kind", "value", "choices", "weights", "low", "high", "probability", "nested"}
        unknown = set(spec) - known
        if unknown:
            raise ValueError(f"unknown param spec fields: {sorted(unknown)}")
        return ParamSpec(
            kind=spec["kind"],
            value=spec.get("value"),
            choices=tuple(spec.get("choices", ())),
            weights=tuple(float(weight) for weight in spec.get("weights", ())),
            low=spec.get("low"),
            high=spec.get("high"),
            probability=spec.get("probability"),
            nested=spec.get("nested"),
        )
    return constant(spec)


def _weighted_choice(rng: random.Random, choices: tuple[Any, ...], weights: tuple[float, ...]) -> Any:
    total = sum(weights)
    target = rng.random() * total
    cumulative = 0.0
    for choice_value, weight in zip(choices, weights, strict=True):
        cumulative += weight
        if target < cumulative:
            return choice_value
    return choices[-1]
