from __future__ import annotations

from dataclasses import dataclass

from simulator.observation.sdk import GameInput
from simulator.types import GameOutput

from .base import EpisodeContext, OpponentRunner, OpponentSpec
from .scripted import build_scripted_opponent


@dataclass
class PythonOpponentRunner:
    opponent: OpponentRunner

    def reset(self, seed: int, episode_context: EpisodeContext) -> None:
        self.opponent.reset(seed, episode_context)

    def act(self, game_input: GameInput) -> GameOutput:
        return self.opponent.act(game_input)

    def __call__(self, game_input: GameInput) -> GameOutput:
        return self.act(game_input)


def build_runner(spec: OpponentSpec) -> OpponentRunner:
    if spec.kind == "python":
        return PythonOpponentRunner(build_scripted_opponent(spec.name, spec.params))
    if spec.kind == "cpp_subprocess":
        raise NotImplementedError("cpp_subprocess opponent runner is reserved but not implemented yet")
    if spec.kind == "cpp_binding":
        raise NotImplementedError("cpp_binding opponent runner is reserved but not implemented yet")
    raise ValueError(f"unsupported opponent kind: {spec.kind!r}")
