from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from simulator.errors import SimulatorRuleError


ACTOR_SPATIAL_CHANNELS = 43
ACTOR_SCALAR_FEATURES = 10
CRITIC_SPATIAL_CHANNELS = 26
CRITIC_SCALAR_FEATURES = 17
GRID_SIZE = 17
MOVE_BUDGET = 6


@dataclass(frozen=True)
class PpoTransition:
    spatial_planes: Tensor
    scalars: Tensor
    critic_planes: Tensor
    critic_scalars: Tensor
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
    critic_planes: Tensor
    critic_scalars: Tensor
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
    critic_planes: Tensor
    critic_scalars: Tensor
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
            critic_planes=torch.stack([transition.critic_planes.float() for transition in transitions]),
            critic_scalars=torch.stack([transition.critic_scalars.float() for transition in transitions]),
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

    @staticmethod
    def from_arrays(
        *,
        spatial_planes: Any,
        scalars: Any,
        critic_planes: Any,
        critic_scalars: Any,
        actions: Any,
        k: Any,
        order: Any,
        vp: Any,
        old_logprob: Any,
        values: Any,
        rewards: Any,
        dones: Any,
        episode_ids: tuple[str, ...],
        round_indices: Any,
        map_ids: tuple[int | None, ...],
        agent_player_ids: Any,
        infos: tuple[dict[str, Any], ...],
    ) -> PpoBatch:
        batch = PpoBatch(
            spatial_planes=torch.as_tensor(spatial_planes, dtype=torch.float32),
            scalars=torch.as_tensor(scalars, dtype=torch.float32),
            critic_planes=torch.as_tensor(critic_planes, dtype=torch.float32),
            critic_scalars=torch.as_tensor(critic_scalars, dtype=torch.float32),
            actions=torch.as_tensor(actions, dtype=torch.long),
            k=torch.as_tensor(k, dtype=torch.long),
            order=torch.as_tensor(order, dtype=torch.long),
            vp=torch.as_tensor(vp, dtype=torch.long),
            old_logprob=torch.as_tensor(old_logprob, dtype=torch.float32),
            old_values=torch.as_tensor(values, dtype=torch.float32),
            rewards=torch.as_tensor(rewards, dtype=torch.float32),
            dones=torch.as_tensor(dones, dtype=torch.bool),
            episode_ids=tuple(episode_ids),
            round_indices=torch.as_tensor(round_indices, dtype=torch.long),
            map_ids=tuple(map_ids),
            agent_player_ids=torch.as_tensor(agent_player_ids, dtype=torch.long),
            infos=tuple(infos),
        )
        _validate_batch_arrays(batch)
        return batch

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
            critic_planes=self.critic_planes.to(device),
            critic_scalars=self.critic_scalars.to(device),
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
            critic_planes=self.critic_planes[indices],
            critic_scalars=self.critic_scalars[indices],
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
        if tuple(transition.spatial_planes.shape) != (ACTOR_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE):
            raise ValueError(f"transition {idx} spatial_planes must have shape 43x17x17")
        if tuple(transition.scalars.shape) != (ACTOR_SCALAR_FEATURES,):
            raise ValueError(f"transition {idx} scalars must have shape 10")
        if tuple(transition.critic_planes.shape) != (CRITIC_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE):
            raise ValueError(f"transition {idx} critic_planes must have shape 26x17x17")
        if tuple(transition.critic_scalars.shape) != (CRITIC_SCALAR_FEATURES,):
            raise ValueError(f"transition {idx} critic_scalars must have shape 17")
        if tuple(transition.actions.shape) != (MOVE_BUDGET,):
            raise ValueError(f"transition {idx} actions must have shape 6")
        if transition.agent_player_id not in (1, 2):
            raise ValueError(f"transition {idx} agent_player_id must be 1 or 2")


def _validate_batch_arrays(batch: PpoBatch) -> None:
    transition_count = int(batch.rewards.shape[0])
    if transition_count <= 0:
        raise SimulatorRuleError("PpoBatch requires at least one transition")
    if tuple(batch.spatial_planes.shape) != (transition_count, ACTOR_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE):
        raise ValueError(f"spatial_planes must have shape Nx43x17x17, got {tuple(batch.spatial_planes.shape)}")
    if tuple(batch.scalars.shape) != (transition_count, ACTOR_SCALAR_FEATURES):
        raise ValueError(f"scalars must have shape Nx10, got {tuple(batch.scalars.shape)}")
    if tuple(batch.critic_planes.shape) != (transition_count, CRITIC_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE):
        raise ValueError(f"critic_planes must have shape Nx26x17x17, got {tuple(batch.critic_planes.shape)}")
    if tuple(batch.critic_scalars.shape) != (transition_count, CRITIC_SCALAR_FEATURES):
        raise ValueError(f"critic_scalars must have shape Nx17, got {tuple(batch.critic_scalars.shape)}")
    if tuple(batch.actions.shape) != (transition_count, MOVE_BUDGET):
        raise ValueError(f"actions must have shape Nx6, got {tuple(batch.actions.shape)}")
    for name, tensor in (
        ("k", batch.k),
        ("order", batch.order),
        ("vp", batch.vp),
        ("old_logprob", batch.old_logprob),
        ("old_values", batch.old_values),
        ("rewards", batch.rewards),
        ("dones", batch.dones),
        ("round_indices", batch.round_indices),
        ("agent_player_ids", batch.agent_player_ids),
    ):
        if tuple(tensor.shape) != (transition_count,):
            raise ValueError(f"{name} must have shape N, got {tuple(tensor.shape)}")
    if len(batch.episode_ids) != transition_count:
        raise ValueError(f"episode_ids length must be {transition_count}, got {len(batch.episode_ids)}")
    if len(batch.map_ids) != transition_count:
        raise ValueError(f"map_ids length must be {transition_count}, got {len(batch.map_ids)}")
    if len(batch.infos) != transition_count:
        raise ValueError(f"infos length must be {transition_count}, got {len(batch.infos)}")
    invalid_player_ids = [int(value) for value in batch.agent_player_ids.tolist() if int(value) not in (1, 2)]
    if invalid_player_ids:
        raise ValueError(f"agent_player_ids must be 1 or 2, got {invalid_player_ids[0]}")
