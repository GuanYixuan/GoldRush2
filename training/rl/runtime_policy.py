from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from policy_runtime import FeatureExtractor
from simulator.observation.sdk import GameInput
from simulator.types import GameOutput


FeatureDict = dict[str, Any]
FeaturePolicy = Callable[[FeatureDict], GameOutput | Sequence[int]]


@runtime_checkable
class RuntimeAwarePolicy(Protocol):
    def start_episode(self, *, player_id: int) -> None:
        ...

    def __call__(self, game_input: GameInput) -> GameOutput | Sequence[int]:
        ...


@dataclass
class RuntimePolicyWrapper:
    """Adapt a feature policy to the GameInput -> GameOutput rollout API."""

    feature_policy: FeaturePolicy
    player_id: int = 1

    def __post_init__(self) -> None:
        self.extractor = FeatureExtractor(player_id=self.player_id)

    def start_episode(self, *, player_id: int) -> None:
        self.player_id = int(player_id)
        self.extractor.reset(self.player_id)

    def __call__(self, game_input: GameInput) -> GameOutput:
        features = self.extractor.observe(game_input)
        action = _coerce_game_output(self.feature_policy(features))
        self.extractor.commit_action(action)
        return action


def _coerce_game_output(raw_output: GameOutput | Sequence[int]) -> GameOutput:
    if isinstance(raw_output, GameOutput):
        return raw_output
    values = tuple(int(value) for value in raw_output)
    if len(values) != 9:
        raise ValueError(f"policy sequence output must have 9 integers, got {len(values)}")
    return GameOutput(actions=values[:6], k=values[6], order=values[7], vp=values[8])
