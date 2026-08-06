from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from simulator.observation.sdk import GameInput
from simulator.types import GameOutput
from training.opponents import EpisodeContext, OpponentSpec, build_runner

from .schema import BcTeacherSpec


class TeacherPolicy(Protocol):
    def reset(self, seed: int, episode_context: EpisodeContext) -> None:
        ...

    def act(self, game_input: GameInput) -> GameOutput:
        ...


@dataclass
class OpponentTeacherAdapter:
    spec: OpponentSpec
    runner: TeacherPolicy

    def reset(self, seed: int, episode_context: EpisodeContext) -> None:
        self.runner.reset(seed, episode_context)

    def act(self, game_input: GameInput) -> GameOutput:
        return self.runner.act(game_input)


def build_teacher(spec: BcTeacherSpec) -> TeacherPolicy:
    if spec.kind == "scripted":
        opponent_spec = OpponentSpec(kind="python", name=spec.name, params=dict(spec.params))
        return OpponentTeacherAdapter(opponent_spec, build_runner(opponent_spec))
    raise ValueError(f"unsupported BC teacher kind: {spec.kind!r}")
