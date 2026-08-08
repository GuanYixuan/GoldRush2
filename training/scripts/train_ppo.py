from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepMechanisms
from simulator.errors import SimulatorRuleError
from simulator.mechanisms.maps import SpawnConfig
from training.bc.schema import BC_CHECKPOINT_SCHEMA, FEATURE_SCHEMA as BC_FEATURE_SCHEMA
from training.models import GoldRushPolicyNetwork, PolicyNetworkConfig, TorchFeaturePolicy
from training.opponents import OpponentSpec
from training.rl import (
    BatchRolloutSampler,
    EvaluationCase,
    EvaluationConfig,
    MultiprocessRolloutConfig,
    MultiprocessRolloutPool,
    PpoConfig,
    RuntimePolicyWrapper,
    SingleAgentEnvConfig,
    TerminalWinPlusMarginPotentialReward,
    collect_ppo_rollouts,
    evaluate_policy,
    ppo_update,
)


@dataclass(frozen=True)
class TrainPpoConfig:
    seed: int = 20260804
    total_updates: int = 1
    pair_count: int = 1
    map_ids: tuple[int, ...] = (1,)
    opponent_specs: tuple[OpponentSpec, ...] = field(default_factory=lambda: (opponent_spec_from_name("stay"),))
    device: str = "cpu"
    output_dir: Path = Path("temp/ppo_runs/smoke")
    save_interval: int = 1
    eval_interval: int | None = None
    eval_cases: tuple[EvaluationCase, ...] = ()
    reward_beta: float = 0.2
    reward_margin_scale: float = 500.0
    learning_rate: float = 2.0e-4
    adam_eps: float = 1.0e-5
    rollout_mode: str = "serial"
    multiprocess_rollout: MultiprocessRolloutConfig = field(default_factory=MultiprocessRolloutConfig)
    ppo: PpoConfig = field(default_factory=PpoConfig)
    model: PolicyNetworkConfig = field(default_factory=PolicyNetworkConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    resume_checkpoint: Path | None = None
    init_model_checkpoint: Path | None = None


@dataclass(frozen=True)
class TrainPpoResult:
    output_dir: Path
    final_update: int
    latest_checkpoint: Path
    train_metrics: tuple[dict[str, Any], ...]
    eval_metrics: tuple[dict[str, Any], ...]


def run_training(config: TrainPpoConfig) -> TrainPpoResult:
    _validate_train_config(config)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", _config_to_jsonable(config))

    model = GoldRushPolicyNetwork(config.model).to(device)
    if config.init_model_checkpoint is not None:
        _load_init_model_checkpoint(Path(config.init_model_checkpoint), model=model, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=config.adam_eps)
    start_update = 0
    if config.resume_checkpoint is not None:
        start_update = _load_checkpoint(
            Path(config.resume_checkpoint),
            model=model,
            optimizer=optimizer,
            device=device,
        )

    train_metrics: list[dict[str, Any]] = []
    eval_metrics: list[dict[str, Any]] = []
    metrics_path = output_dir / "metrics.jsonl"
    sampler = _sampler(config)
    rollout_pool = MultiprocessRolloutPool(sampler, config.multiprocess_rollout) if config.rollout_mode == "multiprocess" else None

    try:
        for update_index in range(start_update + 1, config.total_updates + 1):
            batch, rollout_stats = _collect_training_rollouts(
                model,
                sampler,
                config=config,
                update_index=update_index,
                device=device,
                rollout_pool=rollout_pool,
            )
            batch = batch.to(device).compute_gae(
                gamma=config.ppo.gamma,
                gae_lambda=config.ppo.gae_lambda,
                normalize_advantage=config.ppo.normalize_advantage,
            )
            stats = ppo_update(model, optimizer, batch, config.ppo)
            train_record = {
                "kind": "train",
                "update": update_index,
                "transition_count": batch.transition_count,
                "mean_reward": float(batch.rewards.mean().detach().cpu().item()),
                "reward_sum": float(batch.rewards.sum().detach().cpu().item()),
                "policy_loss": stats.policy_loss,
                "value_loss": stats.value_loss,
                "entropy_bonus": stats.entropy_bonus,
                "loss": stats.loss,
                "approx_joint_kl": stats.approx_joint_kl,
                "clip_fraction": stats.clip_fraction,
                "explained_variance": stats.explained_variance,
                "grad_norm": stats.grad_norm,
                "ppo_update_count": stats.update_count,
                "early_stopped": stats.early_stopped,
                **rollout_stats,
                **_training_diagnostics(batch),
            }
            _append_jsonl(metrics_path, train_record)
            train_metrics.append(train_record)

            latest_checkpoint = _save_checkpoint(
                checkpoint_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                update_index=update_index,
                last_metrics=train_record,
            )
            if config.save_interval > 0 and update_index % config.save_interval == 0:
                _save_checkpoint(
                    checkpoint_dir / f"update_{update_index:06d}.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    update_index=update_index,
                    last_metrics=train_record,
                )

            if config.eval_interval is not None and update_index % config.eval_interval == 0:
                eval_record = _run_eval(model, config=config, update_index=update_index, device=device)
                _append_jsonl(metrics_path, eval_record)
                eval_metrics.append(eval_record)
    finally:
        if rollout_pool is not None:
            rollout_pool.close()

    latest_path = output_dir / "checkpoints" / "latest.pt"
    return TrainPpoResult(
        output_dir=output_dir,
        final_update=config.total_updates,
        latest_checkpoint=latest_path,
        train_metrics=tuple(train_metrics),
        eval_metrics=tuple(eval_metrics),
    )


def opponent_spec_from_name(name: str) -> OpponentSpec:
    if name == "stay":
        return OpponentSpec(kind="python", name="stay")
    if name == "random":
        return OpponentSpec(kind="python", name="random")
    if name == "fast_probe_v3_like":
        return OpponentSpec(kind="python", name="fast_probe_v3_like")
    if name == "greedy_visible_gold":
        return OpponentSpec(
            kind="python",
            name="greedy_visible_gold",
            params={
                "target_score": "value_per_step",
                "avoid_bombs": True,
                "risk_weight": 1.0,
                "vp_policy": "never",
            },
        )
    raise ValueError(f"unknown opponent name: {name}")


def load_checkpoint(
    checkpoint_path: Path,
    *,
    model_config: PolicyNetworkConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[GoldRushPolicyNetwork, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if checkpoint.get("schema") != "ppo_train_v1":
        raise SimulatorRuleError(f"unsupported checkpoint schema: {checkpoint.get('schema')!r}")
    config = model_config or _model_config_from_checkpoint(checkpoint)
    model = GoldRushPolicyNetwork(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal GoldRush2 PPO training loop.")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--total-updates", type=int, default=1)
    parser.add_argument("--pair-count", type=int, default=1)
    parser.add_argument("--map-ids", type=int, nargs="+", default=[1])
    parser.add_argument("--opponents", nargs="+", default=["stay"])
    parser.add_argument("--round-count", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=2026080401)
    parser.add_argument("--reward-beta", type=float, default=0.2)
    parser.add_argument("--reward-margin-scale", type=float, default=500.0)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--adam-eps", type=float, default=1.0e-5)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--init-model-checkpoint", type=Path, default=None)
    parser.add_argument("--rollout-mode", choices=["serial", "multiprocess"], default="serial")
    parser.add_argument("--rollout-workers", type=int, default=8)
    parser.add_argument("--rollout-max-inference-batch-size", type=int, default=64)
    parser.add_argument("--rollout-inference-timeout-ms", type=float, default=2.0)
    parser.add_argument("--model-width", type=int, default=96)
    parser.add_argument("--model-blocks", type=int, default=8)
    parser.add_argument("--scalar-hidden", type=int, nargs=2, default=[96, 96])
    parser.add_argument("--actor-hidden", type=int, default=256)
    parser.add_argument("--critic-hidden", type=int, nargs=2, default=[256, 128])
    parser.add_argument("--decoder-hidden", type=int, default=128)
    parser.add_argument("--decoder-embedding", type=int, default=16)
    parser.add_argument("--ppo-minibatch-size", type=int, default=1024)
    parser.add_argument("--ppo-update-epochs", type=int, default=2)
    parser.add_argument("--ppo-target-kl", type=float, default=0.05)
    return parser


def config_from_args(args: argparse.Namespace) -> TrainPpoConfig:
    episode = EpisodeConfig(rules=RulesConfig(round_count=int(args.round_count)))
    model = PolicyNetworkConfig(
        width=int(args.model_width),
        residual_blocks=int(args.model_blocks),
        scalar_hidden=tuple(int(value) for value in args.scalar_hidden),
        actor_hidden=int(args.actor_hidden),
        critic_hidden=tuple(int(value) for value in args.critic_hidden),
        decoder_hidden=int(args.decoder_hidden),
        decoder_embedding=int(args.decoder_embedding),
    )
    ppo = PpoConfig(
        minibatch_size=int(args.ppo_minibatch_size),
        update_epochs=int(args.ppo_update_epochs),
        target_joint_kl=None if args.ppo_target_kl < 0 else float(args.ppo_target_kl),
    )
    multiprocess_rollout = MultiprocessRolloutConfig(
        num_workers=int(args.rollout_workers),
        max_inference_batch_size=int(args.rollout_max_inference_batch_size),
        inference_timeout_ms=float(args.rollout_inference_timeout_ms),
    )
    opponent_specs = tuple(opponent_spec_from_name(name) for name in args.opponents)
    eval_cases: tuple[EvaluationCase, ...] = ()
    if args.eval_interval is not None:
        eval_cases = (EvaluationCase(seed=int(args.eval_seed), map_id=int(args.map_ids[0]), opponent_spec=opponent_specs[0], tag="train_eval"),)
    return TrainPpoConfig(
        seed=int(args.seed),
        total_updates=int(args.total_updates),
        pair_count=int(args.pair_count),
        map_ids=tuple(int(map_id) for map_id in args.map_ids),
        opponent_specs=opponent_specs,
        device=str(args.device),
        output_dir=Path(args.output_dir),
        save_interval=int(args.save_interval),
        eval_interval=None if args.eval_interval is None else int(args.eval_interval),
        eval_cases=eval_cases,
        reward_beta=float(args.reward_beta),
        reward_margin_scale=float(args.reward_margin_scale),
        learning_rate=float(args.learning_rate),
        adam_eps=float(args.adam_eps),
        rollout_mode=str(args.rollout_mode),
        multiprocess_rollout=multiprocess_rollout,
        ppo=ppo,
        model=model,
        episode=episode,
        resume_checkpoint=None if args.resume_checkpoint is None else Path(args.resume_checkpoint),
        init_model_checkpoint=None if args.init_model_checkpoint is None else Path(args.init_model_checkpoint),
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_training(config_from_args(args))
    print(json.dumps({"final_update": result.final_update, "latest_checkpoint": str(result.latest_checkpoint)}, ensure_ascii=True))


def _sampler(config: TrainPpoConfig) -> BatchRolloutSampler:
    return BatchRolloutSampler(
        env_config=SingleAgentEnvConfig(episode=config.episode, opponent_spec=config.opponent_specs[0]),
        mechanisms=RoundStepMechanisms(),
        spawn=SpawnConfig(),
        reward_fn=TerminalWinPlusMarginPotentialReward(
            beta=config.reward_beta,
            margin_scale=config.reward_margin_scale,
            gamma=config.ppo.gamma,
        ),
    )


def _collect_training_rollouts(
    model: GoldRushPolicyNetwork,
    sampler: BatchRolloutSampler,
    *,
    config: TrainPpoConfig,
    update_index: int,
    device: torch.device,
    rollout_pool: MultiprocessRolloutPool | None = None,
):
    seed = config.seed + (update_index - 1) * config.pair_count
    if config.rollout_mode == "serial":
        batch = collect_ppo_rollouts(
            model,
            sampler,
            pair_count=config.pair_count,
            seed=seed,
            map_ids=config.map_ids,
            opponent_specs=config.opponent_specs,
            device=device,
        )
        return batch, {"rollout_mode": "serial"}
    if config.rollout_mode == "multiprocess":
        if rollout_pool is None:
            raise ValueError("multiprocess rollout requires a MultiprocessRolloutPool")
        return rollout_pool.collect(
            model,
            pair_count=config.pair_count,
            seed=seed,
            map_ids=config.map_ids,
            opponent_specs=config.opponent_specs,
            device=device,
        )
    raise ValueError(f"unknown rollout_mode: {config.rollout_mode!r}")


def _run_eval(
    model: GoldRushPolicyNetwork,
    *,
    config: TrainPpoConfig,
    update_index: int,
    device: torch.device,
) -> dict[str, Any]:
    if not config.eval_cases:
        return {"kind": "eval", "update": update_index, "metrics": {}, "skipped": True}
    feature_policy = TorchFeaturePolicy(model, deterministic=True, device=device)
    runtime_policy = RuntimePolicyWrapper(feature_policy)
    eval_config = EvaluationConfig(
        env_config=SingleAgentEnvConfig(episode=config.episode, opponent_spec=None),
        cases=config.eval_cases,
        mechanisms=RoundStepMechanisms(),
        spawn=SpawnConfig(),
    )
    result = evaluate_policy(runtime_policy, eval_config)
    return {
        "kind": "eval",
        "update": update_index,
        "metrics": _jsonable(result.metrics),
        "fallback_error": feature_policy.last_fallback_error,
    }


def _save_checkpoint(
    path: Path,
    *,
    model: GoldRushPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    config: TrainPpoConfig,
    update_index: int,
    last_metrics: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "ppo_train_v1",
            "update_index": update_index,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_config": _config_to_jsonable(config),
            "last_metrics": _jsonable(last_metrics),
        },
        path,
    )
    return path


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    model: GoldRushPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("schema") != "ppo_train_v1":
        raise SimulatorRuleError(f"unsupported checkpoint schema: {checkpoint.get('schema')!r}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["update_index"])


def _load_init_model_checkpoint(
    checkpoint_path: Path,
    *,
    model: GoldRushPolicyNetwork,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("schema") != BC_CHECKPOINT_SCHEMA:
        raise SimulatorRuleError(f"init model checkpoint must be {BC_CHECKPOINT_SCHEMA}, got {checkpoint.get('schema')!r}")
    if checkpoint.get("feature_schema") != BC_FEATURE_SCHEMA:
        raise SimulatorRuleError(f"unsupported init feature schema: {checkpoint.get('feature_schema')!r}")
    raw_model_config = checkpoint.get("model_config")
    if raw_model_config is None:
        raise SimulatorRuleError("BC checkpoint missing model_config")
    expected_config = _actor_init_config(model.config)
    if _actor_init_config(raw_model_config) != expected_config:
        raise SimulatorRuleError("BC checkpoint model_config does not match PPO model config")

    current = model.state_dict()
    loaded = checkpoint["model_state_dict"]
    filtered = _actor_init_state_from_bc(current, loaded)
    missing = sorted(
        key
        for key in current
        if _is_actor_state_key(key) and key not in filtered
    )
    if missing:
        raise SimulatorRuleError(f"BC checkpoint missing actor model keys: {missing[:5]}")
    current.update(filtered)
    model.load_state_dict(current)


def _actor_init_config(config: PolicyNetworkConfig | dict[str, Any]) -> dict[str, Any]:
    raw = asdict(config) if isinstance(config, PolicyNetworkConfig) else dict(config)
    aliases = {
        "actor_spatial_channels": raw.get("actor_spatial_channels", raw.get("spatial_channels")),
        "actor_scalar_features": raw.get("actor_scalar_features", raw.get("scalar_features")),
    }
    return {
        "actor_spatial_channels": aliases["actor_spatial_channels"],
        "actor_scalar_features": aliases["actor_scalar_features"],
        "width": raw.get("width"),
        "residual_blocks": raw.get("residual_blocks"),
        "se_reduction": raw.get("se_reduction"),
        "scalar_hidden": tuple(raw.get("scalar_hidden", ())),
        "actor_hidden": raw.get("actor_hidden"),
        "decoder_hidden": raw.get("decoder_hidden"),
        "decoder_embedding": raw.get("decoder_embedding"),
        "activation": raw.get("activation"),
    }


def _actor_init_state_from_bc(
    current: dict[str, torch.Tensor],
    loaded: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    filtered: dict[str, torch.Tensor] = {}
    for key, value in loaded.items():
        mapped_key = _map_bc_actor_key(key)
        if mapped_key is not None and mapped_key in current:
            filtered[mapped_key] = value
    return filtered


def _map_bc_actor_key(key: str) -> str | None:
    if key.startswith("critic_encoder.") or key.startswith("critic_mlp."):
        return None
    if key.startswith("actor_encoder."):
        return key
    if key.startswith("stem.") or key.startswith("scalar_tower.") or key.startswith("blocks."):
        return f"actor_encoder.{key}"
    return key


def _is_actor_state_key(key: str) -> bool:
    return not (key.startswith("critic_encoder.") or key.startswith("critic_mlp."))


def _model_config_from_checkpoint(checkpoint: dict[str, Any]) -> PolicyNetworkConfig:
    raw = checkpoint["train_config"]["model"]
    missing = {"decoder_hidden", "decoder_embedding"} - raw.keys()
    if missing:
        raise SimulatorRuleError(
            "checkpoint uses the retired factorized action head and cannot be loaded by the autoregressive model; "
            f"missing config fields: {sorted(missing)}"
        )
    return PolicyNetworkConfig(
        width=int(raw["width"]),
        residual_blocks=int(raw["residual_blocks"]),
        se_reduction=int(raw["se_reduction"]),
        scalar_hidden=tuple(int(value) for value in raw["scalar_hidden"]),
        actor_hidden=int(raw["actor_hidden"]),
        critic_hidden=tuple(int(value) for value in raw["critic_hidden"]),
        decoder_hidden=int(raw["decoder_hidden"]),
        decoder_embedding=int(raw["decoder_embedding"]),
        activation=str(raw["activation"]),
    )


def _validate_train_config(config: TrainPpoConfig) -> None:
    if config.total_updates <= 0:
        raise ValueError(f"total_updates must be positive, got {config.total_updates}")
    if config.pair_count <= 0:
        raise ValueError(f"pair_count must be positive, got {config.pair_count}")
    if not config.map_ids:
        raise ValueError("map_ids cannot be empty")
    if not config.opponent_specs:
        raise ValueError("opponent_specs cannot be empty")
    if config.save_interval < 0:
        raise ValueError(f"save_interval must be non-negative, got {config.save_interval}")
    if config.eval_interval is not None and config.eval_interval <= 0:
        raise ValueError(f"eval_interval must be positive when set, got {config.eval_interval}")
    if config.rollout_mode not in ("serial", "multiprocess"):
        raise ValueError(f"rollout_mode must be serial or multiprocess, got {config.rollout_mode!r}")
    if config.resume_checkpoint is not None and config.init_model_checkpoint is not None:
        raise ValueError("resume_checkpoint and init_model_checkpoint cannot both be set")


def _training_diagnostics(batch) -> dict[str, float]:
    batch.require_gae()
    diagnostics: dict[str, float] = {}

    actions = batch.actions.detach().cpu()
    action_total = float(actions.numel())
    action_names = ("up", "down", "left", "right", "stay")
    for action_value, name in enumerate(action_names):
        diagnostics[f"action_fraction_{name}"] = float((actions == action_value).sum().item()) / action_total
    diagnostics["stay_action_fraction"] = diagnostics["action_fraction_stay"]

    k = batch.k.detach().cpu()
    order = batch.order.detach().cpu()
    vp = batch.vp.detach().cpu()
    ko = 2 * k + order
    for value in range(7):
        diagnostics[f"k_fraction_{value}"] = _fraction(k, value)
    for value in range(2):
        diagnostics[f"order_fraction_{value}"] = _fraction(order, value)
    for value in range(14):
        diagnostics[f"ko_fraction_{value}"] = _fraction(ko, value)
    for value in range(3):
        diagnostics[f"vp_fraction_{value}"] = _fraction(vp, value)
    diagnostics["vp_nonzero_fraction"] = float((vp != 0).sum().item()) / float(vp.numel())
    diagnostics["k_edge_fraction"] = float(((k == 0) | (k == 6)).sum().item()) / float(k.numel())
    diagnostics["same_role_reverse_fraction"] = _same_role_reverse_fraction(actions, k)

    terminal_indices = [idx for idx, done in enumerate(batch.dones.detach().cpu().tolist()) if done]
    diagnostics.update(_terminal_diagnostics(batch, terminal_indices))

    assert batch.returns is not None
    assert batch.advantages is not None
    diagnostics["return_mean"] = _tensor_mean(batch.returns)
    diagnostics["return_std"] = _tensor_std(batch.returns)
    diagnostics["value_mean"] = _tensor_mean(batch.old_values)
    diagnostics["value_std"] = _tensor_std(batch.old_values)
    diagnostics["advantage_mean"] = _tensor_mean(batch.advantages)
    diagnostics["advantage_std"] = _tensor_std(batch.advantages)
    return diagnostics


def _terminal_diagnostics(batch, terminal_indices: list[int]) -> dict[str, float]:
    if not terminal_indices:
        return {
            "train_win_rate": 0.0,
            "train_net_gold_margin_mean": 0.0,
            "train_agent_net_gold_mean": 0.0,
            "train_opponent_net_gold_mean": 0.0,
            "train_agent_pickups_mean": 0.0,
            "train_opponent_pickups_mean": 0.0,
            "train_agent_vision_spent_mean": 0.0,
        }

    wins: list[float] = []
    margins: list[float] = []
    agent_net_gold: list[float] = []
    opponent_net_gold: list[float] = []
    agent_pickups: list[float] = []
    opponent_pickups: list[float] = []
    agent_vision_spent: list[float] = []
    agent_player_ids = batch.agent_player_ids.detach().cpu().tolist()
    for idx in terminal_indices:
        agent_player_id = int(agent_player_ids[idx])
        opponent_player_id = 2 if agent_player_id == 1 else 1
        info = batch.infos[idx]
        scores = info["scores"]
        events = info["events"]
        wins.append(1.0 if info["game_result"].winner_id == agent_player_id else 0.0)
        agent_score = float(scores["net_gold"][agent_player_id])
        opponent_score = float(scores["net_gold"][opponent_player_id])
        agent_net_gold.append(agent_score)
        opponent_net_gold.append(opponent_score)
        margins.append(agent_score - opponent_score)
        agent_pickups.append(float(events["pickups"][agent_player_id]))
        opponent_pickups.append(float(events["pickups"][opponent_player_id]))
        agent_vision_spent.append(float(scores["vision_spent"][agent_player_id]))

    return {
        "train_win_rate": _mean(wins),
        "train_net_gold_margin_mean": _mean(margins),
        "train_agent_net_gold_mean": _mean(agent_net_gold),
        "train_opponent_net_gold_mean": _mean(opponent_net_gold),
        "train_agent_pickups_mean": _mean(agent_pickups),
        "train_opponent_pickups_mean": _mean(opponent_pickups),
        "train_agent_vision_spent_mean": _mean(agent_vision_spent),
    }


def _fraction(values: torch.Tensor, target: int) -> float:
    values = values.detach().cpu()
    return float((values == target).sum().item()) / float(values.numel())


def _same_role_reverse_fraction(actions: torch.Tensor, k: torch.Tensor) -> float:
    slot = torch.arange(actions.shape[1] - 1).unsqueeze(0)
    same_role = ((slot + 1) < k.unsqueeze(1)) | (slot >= k.unsqueeze(1))
    first = actions[:, :-1]
    second = actions[:, 1:]
    reverse = (
        ((first == 0) & (second == 1))
        | ((first == 1) & (second == 0))
        | ((first == 2) & (second == 3))
        | ((first == 3) & (second == 2))
    )
    denominator = int(same_role.sum().item())
    return 0.0 if denominator == 0 else float((reverse & same_role).sum().item()) / denominator


def _tensor_mean(values: torch.Tensor) -> float:
    return float(values.detach().float().mean().cpu().item())


def _tensor_std(values: torch.Tensor) -> float:
    return float(values.detach().float().std(unbiased=False).cpu().item())


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _config_to_jsonable(config: TrainPpoConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["resume_checkpoint"] = None if config.resume_checkpoint is None else str(config.resume_checkpoint)
    payload["init_model_checkpoint"] = None if config.init_model_checkpoint is None else str(config.init_model_checkpoint)
    payload["opponent_specs"] = [_opponent_to_jsonable(spec) for spec in config.opponent_specs]
    payload["eval_cases"] = [_eval_case_to_jsonable(case) for case in config.eval_cases]
    return _jsonable(payload)


def _opponent_to_jsonable(spec: OpponentSpec) -> dict[str, Any]:
    return {"kind": spec.kind, "name": spec.name, "params": dict(spec.params)}


def _eval_case_to_jsonable(case: EvaluationCase) -> dict[str, Any]:
    return {
        "seed": case.seed,
        "map_id": case.map_id,
        "opponent_spec": None if case.opponent_spec is None else _opponent_to_jsonable(case.opponent_spec),
        "tag": case.tag,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
