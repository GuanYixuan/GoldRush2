from __future__ import annotations

import json
from pathlib import Path

from .recorder import SimulatorReplayRecorder
from .schema import JsonObject, SimulatorReplayDocument


def dumps_replay(replay: SimulatorReplayRecorder | SimulatorReplayDocument | JsonObject, *, indent: int | None = 2) -> str:
    return json.dumps(_as_payload(replay), ensure_ascii=False, indent=indent, sort_keys=True)


def write_replay(
    path: str | Path,
    replay: SimulatorReplayRecorder | SimulatorReplayDocument | JsonObject,
    *,
    indent: int | None = 2,
) -> None:
    Path(path).write_text(dumps_replay(replay, indent=indent) + "\n", encoding="utf-8")


def _as_payload(replay: SimulatorReplayRecorder | SimulatorReplayDocument | JsonObject) -> JsonObject:
    if isinstance(replay, SimulatorReplayRecorder):
        return replay.to_json()
    if isinstance(replay, SimulatorReplayDocument):
        return replay.to_json()
    if isinstance(replay, dict):
        return replay
    raise TypeError(f"unsupported replay payload type: {type(replay).__name__}")


__all__ = ["dumps_replay", "write_replay"]
