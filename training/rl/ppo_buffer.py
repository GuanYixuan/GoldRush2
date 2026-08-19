from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor

from simulator.errors import SimulatorRuleError


ACTOR_SPATIAL_CHANNELS = 43
ACTOR_SCALAR_FEATURES = 10
FAST_SCALAR_FEATURES = 2
CRITIC_SPATIAL_CHANNELS = 39
CRITIC_SCALAR_FEATURES = 20
GRID_SIZE = 17
MOVE_BUDGET = 6

FAST_STATUS_NONE = 0
FAST_STATUS_SUCCESS = 1
FAST_STATUS_MISS_NO_TARGET = 2
FAST_STATUS_PATH_FAIL = 3


@dataclass(frozen=True)
class PpoTransition:
    spatial_planes: Tensor
    scalars: Tensor
    fast_scalars: Tensor
    critic_planes: Tensor
    critic_scalars: Tensor
    actions: Tensor
    k: Tensor
    order: Tensor
    vp: Tensor
    threshold_raw: Tensor
    threshold_int: Tensor
    old_logprob: Tensor
    value: Tensor
    reward: float
    reward_sum: float
    tau: int
    done: bool
    fast_success: bool
    fast_status: int
    episode_id: str
    round_index: int
    map_id: int | None
    agent_player_id: int
    info: dict[str, Any]


@dataclass(frozen=True)
class PpoMiniBatch:
    spatial_planes: Tensor
    scalars: Tensor
    fast_scalars: Tensor
    critic_planes: Tensor
    critic_scalars: Tensor
    actions: Tensor
    k: Tensor
    order: Tensor
    vp: Tensor
    threshold_raw: Tensor
    threshold_int: Tensor
    old_logprob: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor
    reward_sum: Tensor
    tau: Tensor
    fast_success: Tensor
    fast_status: Tensor

    @property
    def transition_count(self) -> int:
        return int(self.returns.shape[0])


@dataclass(frozen=True)
class PpoBatch:
    spatial_planes: Tensor
    scalars: Tensor
    fast_scalars: Tensor
    critic_planes: Tensor
    critic_scalars: Tensor
    actions: Tensor
    k: Tensor
    order: Tensor
    vp: Tensor
    threshold_raw: Tensor
    threshold_int: Tensor
    old_logprob: Tensor
    old_values: Tensor
    rewards: Tensor
    reward_sums: Tensor
    taus: Tensor
    fast_success: Tensor
    fast_status: Tensor
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
            fast_scalars=torch.stack([transition.fast_scalars.float() for transition in transitions]),
            critic_planes=torch.stack([transition.critic_planes.float() for transition in transitions]),
            critic_scalars=torch.stack([transition.critic_scalars.float() for transition in transitions]),
            actions=torch.stack([transition.actions.long() for transition in transitions]),
            k=torch.stack([transition.k.long().reshape(()) for transition in transitions]),
            order=torch.stack([transition.order.long().reshape(()) for transition in transitions]),
            vp=torch.stack([transition.vp.long().reshape(()) for transition in transitions]),
            threshold_raw=torch.stack([transition.threshold_raw.float().reshape(()) for transition in transitions]),
            threshold_int=torch.stack([transition.threshold_int.long().reshape(()) for transition in transitions]),
            old_logprob=torch.stack([transition.old_logprob.float().reshape(()) for transition in transitions]),
            old_values=torch.stack([transition.value.float().reshape(()) for transition in transitions]),
            rewards=torch.tensor([transition.reward for transition in transitions], dtype=torch.float32),
            reward_sums=torch.tensor([transition.reward_sum for transition in transitions], dtype=torch.float32),
            taus=torch.tensor([transition.tau for transition in transitions], dtype=torch.long),
            fast_success=torch.tensor([transition.fast_success for transition in transitions], dtype=torch.bool),
            fast_status=torch.tensor([transition.fast_status for transition in transitions], dtype=torch.long),
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
        fast_scalars: Any,
        critic_planes: Any,
        critic_scalars: Any,
        actions: Any,
        k: Any,
        order: Any,
        vp: Any,
        threshold_raw: Any,
        threshold_int: Any,
        old_logprob: Any,
        values: Any,
        rewards: Any,
        dones: Any,
        episode_ids: tuple[str, ...],
        round_indices: Any,
        map_ids: tuple[int | None, ...],
        agent_player_ids: Any,
        infos: tuple[dict[str, Any], ...],
        reward_sums: Any | None = None,
        taus: Any | None = None,
        fast_success: Any | None = None,
        fast_status: Any | None = None,
    ) -> PpoBatch:
        batch = PpoBatch(
            spatial_planes=torch.as_tensor(spatial_planes, dtype=torch.float32),
            scalars=torch.as_tensor(scalars, dtype=torch.float32),
            fast_scalars=torch.as_tensor(fast_scalars, dtype=torch.float32),
            critic_planes=torch.as_tensor(critic_planes, dtype=torch.float32),
            critic_scalars=torch.as_tensor(critic_scalars, dtype=torch.float32),
            actions=torch.as_tensor(actions, dtype=torch.long),
            k=torch.as_tensor(k, dtype=torch.long),
            order=torch.as_tensor(order, dtype=torch.long),
            vp=torch.as_tensor(vp, dtype=torch.long),
            threshold_raw=torch.as_tensor(threshold_raw, dtype=torch.float32),
            threshold_int=torch.as_tensor(threshold_int, dtype=torch.long),
            old_logprob=torch.as_tensor(old_logprob, dtype=torch.float32),
            old_values=torch.as_tensor(values, dtype=torch.float32),
            rewards=torch.as_tensor(rewards, dtype=torch.float32),
            reward_sums=torch.as_tensor(reward_sums if reward_sums is not None else rewards, dtype=torch.float32),
            taus=torch.as_tensor(taus if taus is not None else torch.ones_like(torch.as_tensor(rewards, dtype=torch.long)), dtype=torch.long),
            fast_success=torch.as_tensor(
                fast_success if fast_success is not None else torch.zeros_like(torch.as_tensor(rewards, dtype=torch.bool)),
                dtype=torch.bool,
            ),
            fast_status=torch.as_tensor(
                fast_status if fast_status is not None else torch.zeros_like(torch.as_tensor(rewards, dtype=torch.long)),
                dtype=torch.long,
            ),
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
            tau = int(self.taus[idx].item())
            if tau <= 0:
                raise SimulatorRuleError(f"tau must be positive, got {tau} at transition {idx}")
            same_episode_next = idx + 1 < self.transition_count and self.episode_ids[idx + 1] == self.episode_ids[idx]
            non_terminal = 0.0 if bool(self.dones[idx].item()) or not same_episode_next else 1.0
            next_value = torch.tensor(0.0, dtype=self.old_values.dtype, device=self.old_values.device)
            if non_terminal:
                next_value = self.old_values[idx + 1]
            reward_sum = self.reward_sums[idx]
            gamma_tau = gamma**tau
            lambda_tau = gae_lambda**tau
            delta = reward_sum + gamma_tau * next_value * non_terminal - self.old_values[idx]
            next_advantage = delta + gamma_tau * lambda_tau * non_terminal * next_advantage
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
            fast_scalars=self.fast_scalars.to(device),
            critic_planes=self.critic_planes.to(device),
            critic_scalars=self.critic_scalars.to(device),
            actions=self.actions.to(device),
            k=self.k.to(device),
            order=self.order.to(device),
            vp=self.vp.to(device),
            threshold_raw=self.threshold_raw.to(device),
            threshold_int=self.threshold_int.to(device),
            old_logprob=self.old_logprob.to(device),
            old_values=self.old_values.to(device),
            rewards=self.rewards.to(device),
            reward_sums=self.reward_sums.to(device),
            taus=self.taus.to(device),
            fast_success=self.fast_success.to(device),
            fast_status=self.fast_status.to(device),
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
            fast_scalars=self.fast_scalars[indices],
            critic_planes=self.critic_planes[indices],
            critic_scalars=self.critic_scalars[indices],
            actions=self.actions[indices],
            k=self.k[indices],
            order=self.order[indices],
            vp=self.vp[indices],
            threshold_raw=self.threshold_raw[indices],
            threshold_int=self.threshold_int[indices],
            old_logprob=self.old_logprob[indices],
            old_values=self.old_values[indices],
            advantages=self.advantages[indices],
            returns=self.returns[indices],
            reward_sum=self.reward_sums[indices],
            tau=self.taus[indices],
            fast_success=self.fast_success[indices],
            fast_status=self.fast_status[indices],
        )


def _validate_transitions(transitions: list[PpoTransition] | tuple[PpoTransition, ...]) -> None:
    for idx, transition in enumerate(transitions):
        if tuple(transition.spatial_planes.shape) != (ACTOR_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE):
            raise ValueError(f"transition {idx} spatial_planes must have shape 43x17x17")
        if tuple(transition.scalars.shape) != (ACTOR_SCALAR_FEATURES,):
            raise ValueError(f"transition {idx} scalars must have shape 10")
        if tuple(transition.fast_scalars.shape) != (FAST_SCALAR_FEATURES,):
            raise ValueError(f"transition {idx} fast_scalars must have shape 2")
        if tuple(transition.critic_planes.shape) != (CRITIC_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE):
            raise ValueError(f"transition {idx} critic_planes must have shape 39x17x17")
        if tuple(transition.critic_scalars.shape) != (CRITIC_SCALAR_FEATURES,):
            raise ValueError(f"transition {idx} critic_scalars must have shape 20")
        if tuple(transition.actions.shape) != (MOVE_BUDGET,):
            raise ValueError(f"transition {idx} actions must have shape 6")
        if transition.tau <= 0:
            raise ValueError(f"transition {idx} tau must be positive, got {transition.tau}")
        if transition.fast_success != (transition.fast_status == FAST_STATUS_SUCCESS):
            raise ValueError(f"transition {idx} fast_success must match fast_status={FAST_STATUS_SUCCESS}")
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
    if tuple(batch.fast_scalars.shape) != (transition_count, FAST_SCALAR_FEATURES):
        raise ValueError(f"fast_scalars must have shape Nx2, got {tuple(batch.fast_scalars.shape)}")
    if tuple(batch.critic_planes.shape) != (transition_count, CRITIC_SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE):
        raise ValueError(f"critic_planes must have shape Nx39x17x17, got {tuple(batch.critic_planes.shape)}")
    if tuple(batch.critic_scalars.shape) != (transition_count, CRITIC_SCALAR_FEATURES):
        raise ValueError(f"critic_scalars must have shape Nx20, got {tuple(batch.critic_scalars.shape)}")
    if tuple(batch.actions.shape) != (transition_count, MOVE_BUDGET):
        raise ValueError(f"actions must have shape Nx6, got {tuple(batch.actions.shape)}")
    if tuple(batch.reward_sums.shape) != (transition_count,):
        raise ValueError(f"reward_sums must have shape N, got {tuple(batch.reward_sums.shape)}")
    if tuple(batch.taus.shape) != (transition_count,):
        raise ValueError(f"taus must have shape N, got {tuple(batch.taus.shape)}")
    if tuple(batch.fast_success.shape) != (transition_count,):
        raise ValueError(f"fast_success must have shape N, got {tuple(batch.fast_success.shape)}")
    if tuple(batch.fast_status.shape) != (transition_count,):
        raise ValueError(f"fast_status must have shape N, got {tuple(batch.fast_status.shape)}")
    if bool((batch.taus <= 0).any().item()):
        raise ValueError("taus must all be positive")
    invalid_fast_status = (batch.fast_status < FAST_STATUS_NONE) | (batch.fast_status > FAST_STATUS_PATH_FAIL)
    if bool(invalid_fast_status.any().item()):
        raise ValueError("fast_status contains invalid status code")
    inconsistent_fast_success = batch.fast_success != (batch.fast_status == FAST_STATUS_SUCCESS)
    if bool(inconsistent_fast_success.any().item()):
        raise ValueError(f"fast_success must match fast_status={FAST_STATUS_SUCCESS}")
    for name, tensor in (
        ("k", batch.k),
        ("order", batch.order),
        ("vp", batch.vp),
        ("threshold_raw", batch.threshold_raw),
        ("threshold_int", batch.threshold_int),
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
