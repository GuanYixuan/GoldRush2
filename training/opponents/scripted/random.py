from __future__ import annotations

import random
from dataclasses import dataclass, field

from simulator.constants import MOVE_BUDGET
from simulator.observation.sdk import GameInput
from simulator.types import Action, GameOutput

from ..base import EpisodeContext
from .common import validate_vp_policy, vp_from_policy


@dataclass
class RandomOpponent:
    stay_prob: float = 0.2
    vp_policy: str = "never"
    _rng: random.Random = field(default_factory=random.Random, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.stay_prob <= 1.0:
            raise ValueError(f"stay_prob must be in [0, 1], got {self.stay_prob}")
        validate_vp_policy(self.vp_policy)

    def reset(self, seed: int, episode_context: EpisodeContext) -> None:
        self._rng.seed(seed)

    def act(self, game_input: GameInput) -> GameOutput:
        actions = tuple(self._sample_action() for _ in range(MOVE_BUDGET))
        k = self._rng.randint(0, MOVE_BUDGET)
        order = self._rng.randint(0, 1)
        return GameOutput(actions=actions, k=k, order=order, vp=vp_from_policy(self.vp_policy, game_input.round))

    def __call__(self, game_input: GameInput) -> GameOutput:
        return self.act(game_input)

    def _sample_action(self) -> int:
        if self._rng.random() < self.stay_prob:
            return int(Action.STAY)
        return int(self._rng.choice((Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)))
