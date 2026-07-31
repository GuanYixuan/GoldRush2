from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SIMULATOR_FULL_REPLAY_FORMAT = "goldrush2_simulator_full_replay"
SIMULATOR_FULL_REPLAY_VERSION = 1

JsonObject = dict[str, Any]


@dataclass
class SimulatorReplayDocument:
    source: JsonObject
    players: dict[str, str]
    maps: list[list[int]]
    rounds: list[JsonObject] = field(default_factory=list)
    forfeit: JsonObject | None = None

    def to_json(self) -> JsonObject:
        payload: JsonObject = {
            "format": SIMULATOR_FULL_REPLAY_FORMAT,
            "format_version": SIMULATOR_FULL_REPLAY_VERSION,
            "source": self.source,
            "players": self.players,
            "maps": self.maps,
            "rounds": self.rounds,
        }
        if self.forfeit is not None:
            payload["forfeit"] = self.forfeit
        return payload


__all__ = [
    "JsonObject",
    "SIMULATOR_FULL_REPLAY_FORMAT",
    "SIMULATOR_FULL_REPLAY_VERSION",
    "SimulatorReplayDocument",
]
