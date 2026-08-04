from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from simulator.errors import SimulatorRuleError


@dataclass(frozen=True)
class PpoTransition:
    spatial_planes: Tensor
    scalars: Tensor
    actions: Tensor
    k: Tensor
    order: Tensor
    vp: Tensor
    old_logprob: Tensor
    value: Tensor
    reward: float
    done: bool
    episode_id: str
    round_index: int
    map_id: int | None
    agent_player_id: int
    info: dict[str, Any]


@dataclass(frozen=True)
class PpoMiniBatch:
    spatial_planes: Tensor
    scalars: Tensor
    actions: Tensor
    k: Tensor
    order: Tensor
    vp: Tensor
    old_logprob: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor

    @property
    def transition_count(self) -> int:
        return int(self.returns.shape[0])


@dataclass(frozen=True)
class PpoBatch:
    spatial_planes: Tensor
    scalars: Tensor
    actions: Tensor
    k: Tensor
    order: Tensor
    vp: Tensor
    old_logprob: Tensor
    old_values: Tensor
    rewards: Tensor
    dones: Tensor
    episode_ids: tuple[str, ...]
    round_indices: Tensor
    map_ids: tuple[int | None, ...]
    agent_player_ids: Tensor
    infos: tuple[dict[str, Any], ...]
    advantages: Tensor | None = None
    returns: Tensor | None = None

    @staticmethod
    def from_transitions(transitions: list[PpoTransition] | tuple[PpoTransition, ...]) -> PpoBatch:
        if not transitions:
            raise SimulatorRuleError("PpoBatch requires at least one transition")

        _validate_transitions(transitions)
        return PpoBatch(
            spatial_planes=torch.stack([transition.spatial_planes.float() for transition in transitions]),
            scalars=torch.stack([transition.scalars.float() for transition in transitions]),
            actions=torch.stack([transition.actions.long() for transition in transitions]),
            k=torch.stack([transition.k.long().reshape(()) for transition in transitions]),
            order=torch.stack([transition.order.long().reshape(()) for transition in transitions]),
            vp=torch.stack([transition.vp.long().reshape(()) for transition in transitions]),
            old_logprob=torch.stack([transition.old_logprob.float().reshape(()) for transition in transitions]),
            old_values=torch.stack([transition.value.float().reshape(()) for transition in transitions]),
            rewards=torch.tensor([transition.reward for transition in transitions], dtype=torch.float32),
            dones=torch.tensor([transition.done for transition in transitions], dtype=torch.bool),
            episode_ids=tuple(transition.episode_id for transition in transitions),
            round_indices=torch.tensor([transition.round_index for transition in transitions], dtype=torch.long),
            map_ids=tuple(transition.map_id for transition in transitions),
            agent_player_ids=torch.tensor([transition.agent_player_id for transition in transitions], dtype=torch.long),
            infos=tuple(transition.info for transition in transitions),
        )

    @property
    def transition_count(self) -> int:
        return int(self.rewards.shape[0])

    def compute_gae(
        self,
        *,
        gamma: float,
        gae_lambda: float,
        normalize_advantage: bool = True,
    ) -> PpoBatch:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")
        if not bool(self.dones[-1].item()):
            raise SimulatorRuleError("PPO v1 expects complete episodes ending in done=True")

        advantages = torch.zeros_like(self.rewards)
        next_advantage = torch.tensor(0.0, dtype=self.rewards.dtype, device=self.rewards.device)
        for idx in range(self.transition_count - 1, -1, -1):
            non_terminal = 0.0 if bool(self.dones[idx].item()) else 1.0
            next_value = torch.tensor(0.0, dtype=self.old_values.dtype, device=self.old_values.device)
            if idx + 1 < self.transition_count and non_terminal:
                next_value = self.old_values[idx + 1]
            delta = self.rewards[idx] + gamma * next_value * non_terminal - self.old_values[idx]
            next_advantage = delta + gamma * gae_lambda * non_terminal * next_advantage
            advantages[idx] = next_advantage

        returns = advantages + self.old_values
        normalized_advantages = advantages
        if normalize_advantage:
            normalized_advantages = advantages - advantages.mean()
            std = normalized_advantages.std(unbiased=False)
            if float(std.item()) > 1e-8:
                normalized_advantages = normalized_advantages / (std + 1e-8)

        return replace(self, advantages=normalized_advantages, returns=returns)

    def require_gae(self) -> None:
        if self.advantages is None or self.returns is None:
            raise SimulatorRuleError("PpoBatch.compute_gae() must be called before PPO update")

    def iter_minibatches(
        self,
        *,
        minibatch_size: int,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> Iterator[PpoMiniBatch]:
        self.require_gae()
        if minibatch_size <= 0:
            raise ValueError(f"minibatch_size must be positive, got {minibatch_size}")
        indices = torch.arange(self.transition_count, device=self.rewards.device)
        if shuffle:
            indices = indices[torch.randperm(self.transition_count, generator=generator, device=self.rewards.device)]
        for start in range(0, self.transition_count, minibatch_size):
            selected = indices[start : start + minibatch_size]
            yield self._select(selected)

    def to(self, device: torch.device | str) -> PpoBatch:
        return replace(
            self,
            spatial_planes=self.spatial_planes.to(device),
            scalars=self.scalars.to(device),
            actions=self.actions.to(device),
            k=self.k.to(device),
            order=self.order.to(device),
            vp=self.vp.to(device),
            old_logprob=self.old_logprob.to(device),
            old_values=self.old_values.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            round_indices=self.round_indices.to(device),
            agent_player_ids=self.agent_player_ids.to(device),
            advantages=None if self.advantages is None else self.advantages.to(device),
            returns=None if self.returns is None else self.returns.to(device),
        )

    def _select(self, indices: Tensor) -> PpoMiniBatch:
        assert self.advantages is not None
        assert self.returns is not None
        return PpoMiniBatch(
            spatial_planes=self.spatial_planes[indices],
            scalars=self.scalars[indices],
            actions=self.actions[indices],
            k=self.k[indices],
            order=self.order[indices],
            vp=self.vp[indices],
            old_logprob=self.old_logprob[indices],
            old_values=self.old_values[indices],
            advantages=self.advantages[indices],
            returns=self.returns[indices],
        )


def _validate_transitions(transitions: list[PpoTransition] | tuple[PpoTransition, ...]) -> None:
    for idx, transition in enumerate(transitions):
        if tuple(transition.spatial_planes.shape) != (38, 17, 17):
            raise ValueError(f"transition {idx} spatial_planes must have shape 38x17x17")
        if tuple(transition.scalars.shape) != (10,):
            raise ValueError(f"transition {idx} scalars must have shape 10")
        if tuple(transition.actions.shape) != (6,):
            raise ValueError(f"transition {idx} actions must have shape 6")
        if transition.agent_player_id not in (1, 2):
            raise ValueError(f"transition {idx} agent_player_id must be 1 or 2")
