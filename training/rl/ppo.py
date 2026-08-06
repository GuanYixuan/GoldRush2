from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from policy_runtime import FeatureExtractor
from simulator.errors import SimulatorRuleError
from simulator.types import GameOutput
from training.models import (
    GoldRushPolicyNetwork,
    PolicyEvaluation,
    policy_action_to_game_output,
    policy_action_is_finite,
)
from training.opponents import OpponentSpec

from .env import SingleAgentGoldRushEnv
from .ppo_buffer import PpoBatch, PpoMiniBatch, PpoTransition
from .sampler import BatchRolloutSampler


@dataclass(frozen=True)
class PpoConfig:
    gamma: float = 0.9999
    gae_lambda: float = 0.995
    clip_range: float = 0.20
    value_clip_range: float = 0.20
    value_coef: float = 0.50
    huber_delta: float = 1.0
    max_grad_norm: float = 0.5
    target_joint_kl: float | None = 0.05
    update_epochs: int = 2
    minibatch_size: int = 1024
    normalize_advantage: bool = True


PpoEvaluation = PolicyEvaluation


@dataclass(frozen=True)
class PpoUpdateStats:
    policy_loss: float
    value_loss: float
    entropy_bonus: float
    loss: float
    approx_joint_kl: float
    clip_fraction: float
    explained_variance: float
    grad_norm: float
    update_count: int
    early_stopped: bool


def collect_ppo_rollouts(
    model: GoldRushPolicyNetwork,
    sampler: BatchRolloutSampler,
    *,
    pair_count: int,
    seed: int,
    map_ids: Sequence[int] | None = None,
    opponent_specs: Sequence[OpponentSpec] | None = None,
    device: torch.device | str | None = None,
) -> PpoBatch:
    if pair_count <= 0:
        raise SimulatorRuleError(f"pair_count must be positive, got {pair_count}")
    if map_ids is not None and not map_ids:
        raise SimulatorRuleError("map_ids cannot be empty when provided")
    if opponent_specs is not None and not opponent_specs:
        raise SimulatorRuleError("opponent_specs cannot be empty when provided")

    rollout_device = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    transitions: list[PpoTransition] = []
    try:
        for pair_index in range(pair_count):
            episode_seed = seed + pair_index
            requested_map_id = map_ids[pair_index % len(map_ids)] if map_ids is not None else None
            requested_opponent_spec = opponent_specs[pair_index % len(opponent_specs)] if opponent_specs is not None else None
            pair_id = f"pair-{pair_index:06d}-seed-{episode_seed}"

            first_map_id, first_opponent_spec = _collect_one_ppo_episode(
                model,
                sampler,
                transitions,
                pair_id=pair_id,
                pair_role="first",
                seed=episode_seed,
                map_id=requested_map_id,
                agent_player_id=1,
                opponent_spec=requested_opponent_spec,
                device=rollout_device,
            )
            _collect_one_ppo_episode(
                model,
                sampler,
                transitions,
                pair_id=pair_id,
                pair_role="second",
                seed=episode_seed,
                map_id=first_map_id,
                agent_player_id=2,
                opponent_spec=first_opponent_spec if requested_opponent_spec is None else requested_opponent_spec,
                device=rollout_device,
            )
    finally:
        if was_training:
            model.train()

    return PpoBatch.from_transitions(transitions)


def evaluate_actions(model: GoldRushPolicyNetwork, batch: PpoBatch | PpoMiniBatch) -> PpoEvaluation:
    return model.evaluate_actions(
        batch.spatial_planes,
        batch.scalars,
        batch.actions,
        batch.k,
        batch.order,
        batch.vp,
    )


def ppo_update(
    model: GoldRushPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    batch: PpoBatch,
    config: PpoConfig | None = None,
) -> PpoUpdateStats:
    config = PpoConfig() if config is None else config
    _validate_config(config)
    batch.require_gae()

    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropy_bonuses: list[float] = []
    losses: list[float] = []
    kls: list[float] = []
    clip_fractions: list[float] = []
    grad_norms: list[float] = []
    early_stopped = False

    for _epoch in range(config.update_epochs):
        for minibatch in batch.iter_minibatches(minibatch_size=config.minibatch_size, shuffle=True):
            evaluation = evaluate_actions(model, minibatch)
            ratio = torch.exp(evaluation.logprob - minibatch.old_logprob)
            clipped_ratio = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
            policy_loss = -torch.minimum(ratio * minibatch.advantages, clipped_ratio * minibatch.advantages).mean()

            unclipped_value_loss = F.huber_loss(
                evaluation.value,
                minibatch.returns,
                reduction="none",
                delta=config.huber_delta,
            )
            clipped_value = minibatch.old_values + torch.clamp(
                evaluation.value - minibatch.old_values,
                -config.value_clip_range,
                config.value_clip_range,
            )
            clipped_value_loss = F.huber_loss(
                clipped_value,
                minibatch.returns,
                reduction="none",
                delta=config.huber_delta,
            )
            value_loss = torch.maximum(unclipped_value_loss, clipped_value_loss).mean()
            entropy_bonus = evaluation.normalized_entropy.mean()
            loss = policy_loss + config.value_coef * value_loss - entropy_bonus

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_joint_kl = (minibatch.old_logprob - evaluation.logprob).mean()
                clip_fraction = (torch.abs(ratio - 1.0) > config.clip_range).float().mean()

            policy_losses.append(float(policy_loss.detach().cpu().item()))
            value_losses.append(float(value_loss.detach().cpu().item()))
            entropy_bonuses.append(float(entropy_bonus.detach().cpu().item()))
            losses.append(float(loss.detach().cpu().item()))
            kls.append(float(approx_joint_kl.detach().cpu().item()))
            clip_fractions.append(float(clip_fraction.detach().cpu().item()))
            grad_norms.append(float(torch.as_tensor(grad_norm).detach().cpu().item()))

            if config.target_joint_kl is not None and float(approx_joint_kl.item()) > config.target_joint_kl:
                early_stopped = True
                break
        if early_stopped:
            break

    return PpoUpdateStats(
        policy_loss=mean(policy_losses),
        value_loss=mean(value_losses),
        entropy_bonus=mean(entropy_bonuses),
        loss=mean(losses),
        approx_joint_kl=mean(kls),
        clip_fraction=mean(clip_fractions),
        explained_variance=explained_variance(batch.old_values, batch.returns),
        grad_norm=mean(grad_norms),
        update_count=len(losses),
        early_stopped=early_stopped,
    )


def explained_variance(values: Tensor, returns: Tensor | None) -> float:
    if returns is None:
        raise SimulatorRuleError("explained_variance requires returns")
    returns_var = torch.var(returns, unbiased=False)
    if float(returns_var.item()) < 1e-8:
        return 0.0
    residual_var = torch.var(returns - values, unbiased=False)
    return float((1.0 - residual_var / returns_var).detach().cpu().item())


def _collect_one_ppo_episode(
    model: GoldRushPolicyNetwork,
    sampler: BatchRolloutSampler,
    transitions: list[PpoTransition],
    *,
    pair_id: str,
    pair_role: str,
    seed: int,
    map_id: int | None,
    agent_player_id: int,
    opponent_spec: OpponentSpec | None,
    device: torch.device,
) -> tuple[int, OpponentSpec]:
    env = SingleAgentGoldRushEnv(
        config=sampler.env_config,
        mechanisms=sampler.mechanisms,
        map_pool=sampler.map_pool,
        spawn=sampler.spawn,
        reward_fn=sampler.reward_fn,
    )
    reset = env.reset(seed=seed, map_id=map_id, agent_player_id=agent_player_id, opponent_spec=opponent_spec)
    extractor = FeatureExtractor(player_id=agent_player_id)
    observation = reset.observation
    episode_id = f"{pair_id}-{pair_role}"

    while observation is not None:
        features = extractor.observe(observation)
        if features["feature_schema"] != "goldrush2_feature_v1":
            raise SimulatorRuleError(f"unexpected feature schema: {features['feature_schema']!r}")
        spatial_planes = torch.as_tensor(features["planes"], dtype=torch.float32, device=device).unsqueeze(0)
        scalars = torch.as_tensor(features["scalars"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            policy_action = model.act(spatial_planes, scalars)
            if not policy_action_is_finite(policy_action):
                raise SimulatorRuleError("PPO rollout model produced NaN or Inf")
        game_output = policy_action_to_game_output(policy_action)
        extractor.commit_action(game_output)
        step = env.step(game_output)
        transitions.append(
            PpoTransition(
                spatial_planes=spatial_planes.squeeze(0).detach().cpu(),
                scalars=scalars.squeeze(0).detach().cpu(),
                actions=policy_action.actions.squeeze(0).detach().cpu(),
                k=policy_action.k.squeeze(0).detach().cpu(),
                order=policy_action.order.squeeze(0).detach().cpu(),
                vp=policy_action.vp.squeeze(0).detach().cpu(),
                old_logprob=policy_action.logprob.squeeze(0).detach().cpu(),
                value=policy_action.value.squeeze(0).detach().cpu(),
                reward=float(step.reward),
                done=bool(step.terminated),
                episode_id=episode_id,
                round_index=int(observation.round),
                map_id=reset.info["map_id"],
                agent_player_id=agent_player_id,
                info=step.info,
            )
        )
        observation = step.observation

    return int(reset.info["map_id"]), reset.info["opponent_spec"]


def _validate_config(config: PpoConfig) -> None:
    if not 0.0 <= config.gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {config.gamma}")
    if not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {config.gae_lambda}")
    if config.clip_range <= 0.0:
        raise ValueError(f"clip_range must be positive, got {config.clip_range}")
    if config.value_clip_range <= 0.0:
        raise ValueError(f"value_clip_range must be positive, got {config.value_clip_range}")
    if config.huber_delta <= 0.0:
        raise ValueError(f"huber_delta must be positive, got {config.huber_delta}")
    if config.max_grad_norm <= 0.0:
        raise ValueError(f"max_grad_norm must be positive, got {config.max_grad_norm}")
    if config.update_epochs <= 0:
        raise ValueError(f"update_epochs must be positive, got {config.update_epochs}")
    if config.minibatch_size <= 0:
        raise ValueError(f"minibatch_size must be positive, got {config.minibatch_size}")
