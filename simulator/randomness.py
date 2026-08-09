from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SimulatorRngStreams:
    """Independent pseudo-random streams for simulator mechanism families."""

    environment: random.Random
    npc: random.Random
    bomb: random.Random
    center_gold: random.Random
    outer_gold: random.Random


def make_simulator_rng_streams(seed: int | None) -> SimulatorRngStreams:
    """Create stable named mechanism streams without consuming the environment stream."""

    return SimulatorRngStreams(
        environment=random.Random(seed),
        npc=random.Random(_derive_stream_seed(seed, "npc")),
        bomb=random.Random(_derive_stream_seed(seed, "bomb")),
        center_gold=random.Random(_derive_stream_seed(seed, "center_gold")),
        outer_gold=random.Random(_derive_stream_seed(seed, "outer_gold")),
    )


def _derive_stream_seed(seed: int | None, domain: str) -> int | None:
    if seed is None:
        return None
    digest = hashlib.blake2b(
        f"goldrush2/simulator-rng-v1/{domain}/{seed}".encode("ascii"),
        digest_size=32,
    ).digest()
    return int.from_bytes(digest, "big")


__all__ = ["SimulatorRngStreams", "make_simulator_rng_streams"]
