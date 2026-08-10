from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import reject_unknown_params
from .fast_probe_v3_bfs import FastProbeV3BfsOpponent
from .fast_probe_v3_like import FastProbeV3LikeOpponent
from .greedy_visible_gold import GreedyVisibleGoldOpponent
from .random import RandomOpponent
from .stay import StayOpponent

_FAST_PROBE_PARAMS = (
    "bad_rounds_before_vp",
    "enable_vision",
    "max_target_distance",
    "good_gold_threshold",
)


def build_scripted_opponent(
    name: str,
    params: Mapping[str, Any],
) -> (
    StayOpponent
    | RandomOpponent
    | GreedyVisibleGoldOpponent
    | FastProbeV3LikeOpponent
    | FastProbeV3BfsOpponent
):
    if name == "stay":
        reject_unknown_params(name, params, ())
        return StayOpponent()
    if name == "random":
        reject_unknown_params(name, params, ("stay_prob", "vp_policy"))
        return RandomOpponent(
            stay_prob=float(params.get("stay_prob", 0.2)),
            vp_policy=str(params.get("vp_policy", "never")),
        )
    if name == "greedy_visible_gold":
        reject_unknown_params(name, params, ("target_score", "avoid_bombs", "risk_weight", "vp_policy"))
        return GreedyVisibleGoldOpponent(
            target_score=str(params.get("target_score", "value_per_step")),
            avoid_bombs=bool(params.get("avoid_bombs", True)),
            risk_weight=float(params.get("risk_weight", 1.0)),
            vp_policy=str(params.get("vp_policy", "never")),
        )
    if name == "fast_probe_v3_like":
        reject_unknown_params(name, params, _FAST_PROBE_PARAMS)
        return FastProbeV3LikeOpponent(
            bad_rounds_before_vp=int(params.get("bad_rounds_before_vp", 2)),
            enable_vision=bool(params.get("enable_vision", True)),
            max_target_distance=int(params.get("max_target_distance", 6)),
            good_gold_threshold=int(params.get("good_gold_threshold", 2)),
        )
    if name == "fast_probe_v3_bfs":
        reject_unknown_params(name, params, _FAST_PROBE_PARAMS)
        return FastProbeV3BfsOpponent(
            bad_rounds_before_vp=int(params.get("bad_rounds_before_vp", 2)),
            enable_vision=bool(params.get("enable_vision", True)),
            max_target_distance=int(params.get("max_target_distance", 6)),
            good_gold_threshold=int(params.get("good_gold_threshold", 2)),
        )
    raise ValueError(f"unknown python opponent: {name!r}")


__all__ = [
    "FastProbeV3BfsOpponent",
    "FastProbeV3LikeOpponent",
    "GreedyVisibleGoldOpponent",
    "RandomOpponent",
    "StayOpponent",
    "build_scripted_opponent",
]
