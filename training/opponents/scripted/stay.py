from __future__ import annotations

from dataclasses import dataclass

from simulator.observation.sdk import GameInput
from simulator.types import GameOutput

from ..base import EpisodeContext
from .common import stay_output


@dataclass
class StayOpponent:
    def reset(self, seed: int, episode_context: EpisodeContext) -> None:
        pass

    def act(self, game_input: GameInput) -> GameOutput:
        return stay_output()

    def __call__(self, game_input: GameInput) -> GameOutput:
        return self.act(game_input)
