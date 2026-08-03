from __future__ import annotations

try:
    from ._runtime import FeatureExtractor, channel_names, scalar_names
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "policy_runtime C++ extension is not built. Run: "
        "conda run --no-capture-output -n goldrush python setup.py build_ext --inplace"
    ) from exc


__all__ = ["FeatureExtractor", "channel_names", "scalar_names"]
