from __future__ import annotations

try:
    from ._runtime import (
        FastRuntimeState,
        FeatureExtractor,
        channel_names,
        feature_schema,
        infer_fast_role,
        scalar_names,
        simulate_known_gold_pickups,
        try_fast_gold_grab,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "policy_runtime C++ extension is not built. Run: "
        "conda run --no-capture-output -n goldrush python setup.py build_ext --inplace"
    ) from exc


__all__ = [
    "FastRuntimeState",
    "FeatureExtractor",
    "channel_names",
    "feature_schema",
    "infer_fast_role",
    "scalar_names",
    "simulate_known_gold_pickups",
    "try_fast_gold_grab",
]
