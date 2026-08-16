from __future__ import annotations

from dataclasses import dataclass, field

from .constants import (
    BOMB_LOSS_DENOMINATOR,
    BOMB_LOSS_NUMERATOR,
    DEFAULT_VISION_RADIUS,
    MOVE_BUDGET,
    PICKUP_DENOMINATOR,
    PICKUP_NUMERATOR,
    ROUND_COUNT,
    SNAPSHOT_PERIOD,
    TRAMPLE_LOSS_DENOMINATOR,
    TRAMPLE_LOSS_NUMERATOR,
    TRAMPLE_NPC_THRESHOLD,
    VISION_COST_BY_VP,
    VISION_RADIUS_BY_VP,
)


@dataclass(frozen=True)
class RulesConfig:
    move_budget: int = MOVE_BUDGET
    round_count: int = ROUND_COUNT
    default_vision_radius: int = DEFAULT_VISION_RADIUS
    vision_radius_by_vp: dict[int, int] | None = None
    vision_cost_by_vp: dict[int, int] | None = None
    pickup_numerator: int = PICKUP_NUMERATOR
    pickup_denominator: int = PICKUP_DENOMINATOR
    bomb_loss_numerator: int = BOMB_LOSS_NUMERATOR
    bomb_loss_denominator: int = BOMB_LOSS_DENOMINATOR
    trample_loss_numerator: int = TRAMPLE_LOSS_NUMERATOR
    trample_loss_denominator: int = TRAMPLE_LOSS_DENOMINATOR
    trample_npc_threshold: int = TRAMPLE_NPC_THRESHOLD
    snapshot_period: int = SNAPSHOT_PERIOD

    def __post_init__(self) -> None:
        object.__setattr__(self, "vision_radius_by_vp", dict(VISION_RADIUS_BY_VP if self.vision_radius_by_vp is None else self.vision_radius_by_vp))
        object.__setattr__(self, "vision_cost_by_vp", dict(VISION_COST_BY_VP if self.vision_cost_by_vp is None else self.vision_cost_by_vp))


@dataclass(frozen=True)
class EpisodeConfig:
    rules: RulesConfig = field(default_factory=RulesConfig)
    seed: int | None = None
    map_id: int | None = None
    map_key: str | None = None
    mechanism_profile: str = "default"
