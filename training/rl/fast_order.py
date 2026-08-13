from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


FAST_ORDER_LATENT_FIRST_RATE_MIXTURE: tuple[tuple[float, float, float], ...] = (
    (0.40, 0.95, 1.00),
    (0.20, 0.70, 0.95),
    (0.20, 0.30, 0.70),
    (0.20, 0.02, 0.10),
)
FAST_ORDER_RNG_DOMAIN = "goldrush2/rl-fast-order-v1"


@dataclass(frozen=True)
class FastOrderConfig:
    latent_first_rate_mixture: tuple[tuple[float, float, float], ...] = FAST_ORDER_LATENT_FIRST_RATE_MIXTURE

    def __post_init__(self) -> None:
        _validate_mixture(self.latent_first_rate_mixture)


def sample_latent_first_rate(rng: random.Random, config: FastOrderConfig) -> float:
    draw = rng.random()
    cumulative = 0.0
    mixture = config.latent_first_rate_mixture
    for weight, low, high in mixture:
        cumulative += weight
        if draw <= cumulative:
            return rng.uniform(low, high)
    _weight, low, high = mixture[-1]
    return rng.uniform(low, high)


def make_latent_first_rate_rng(seed: int | None) -> random.Random:
    return random.Random(None if seed is None else _derive_seed(seed, "latent_first_rate"))


def make_fast_order_rng(seed: int | None, *, agent_player_id: int) -> random.Random:
    return random.Random(None if seed is None else _derive_seed(seed, f"fast_order_p{agent_player_id}"))


def _derive_seed(seed: int, stream: str) -> int:
    digest = hashlib.blake2b(
        f"{FAST_ORDER_RNG_DOMAIN}/{stream}/{seed}".encode("ascii"),
        digest_size=32,
    ).digest()
    return int.from_bytes(digest, "big")


def _validate_mixture(mixture: tuple[tuple[float, float, float], ...]) -> None:
    if not mixture:
        raise ValueError("latent_first_rate_mixture cannot be empty")
    total_weight = 0.0
    for item in mixture:
        if len(item) != 3:
            raise ValueError(f"latent_first_rate_mixture entries must be (weight, low, high), got {item!r}")
        weight, low, high = (float(item[0]), float(item[1]), float(item[2]))
        if weight <= 0.0:
            raise ValueError(f"latent_first_rate_mixture weights must be positive, got {weight}")
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(f"latent_first_rate bounds must satisfy 0 <= low <= high <= 1, got {(low, high)}")
        total_weight += weight
    if abs(total_weight - 1.0) > 1.0e-6:
        raise ValueError(f"latent_first_rate_mixture weights must sum to 1, got {total_weight}")
