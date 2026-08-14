from __future__ import annotations

import copy
import time
import traceback
from dataclasses import replace
from typing import Any

from policy_runtime import FastRuntimeState, FeatureExtractor
from simulator.errors import SimulatorRuleError
from simulator.types import GameOutput
from training.models import INITIAL_FAST_SCALARS
from training.rl.env import AGENT_DECISION_FAST, SingleAgentEnvConfig, SingleAgentGoldRushEnv
from training.rl.privileged_critic_features import extract_privileged_critic_features
from training.rl.rollout_mp.shared_memory import FeatureSharedMemory, attach_feature_shared_memory
from training.rl.rollout_mp.worker import _add_event_counts, _empty_event_counts

from .types import EvalEpisodeSummary, EvalFeatureRequest, EvalTask, opponent_key


def worker_loop(worker_id: int, command_queue: Any, result_queue: Any, static_config: dict[str, Any]) -> None:
    feature_shared: FeatureSharedMemory | None = None
    try:
        feature_shared = attach_feature_shared_memory(static_config["feature_shared_memory"])
        result_queue.put({"type": "worker_ready", "worker_id": worker_id})
        while True:
            msg = command_queue.get()
            msg_type = msg["type"]
            if msg_type == "stop":
                return
            if msg_type != "start_eval_episode":
                raise RuntimeError(f"eval worker expected start_eval_episode, got {msg_type!r}")
            run_worker_episode(
                worker_id,
                str(msg["eval_id"]),
                msg["task"],
                command_queue,
                result_queue,
                static_config,
                feature_shared,
            )
    except Exception as exc:
        result_queue.put(
            {
                "type": "worker_error",
                "worker_id": worker_id,
                "task_id": None,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if feature_shared is not None:
            feature_shared.close()


def run_worker_episode(
    worker_id: int,
    eval_id: str,
    task: EvalTask,
    command_queue: Any,
    result_queue: Any,
    static_config: dict[str, Any],
    feature_shared: FeatureSharedMemory,
) -> None:
    import numpy as np

    profile_stats: dict[str, int] = {"steps": 0}
    episode_wall_start = time.perf_counter_ns()
    env_config = _env_config_for_task(static_config, task)
    env = SingleAgentGoldRushEnv(
        config=env_config,
        mechanisms=copy.deepcopy(static_config["mechanisms"]),
        map_pool=static_config["map_pool"],
        spawn=static_config["spawn"],
        reward_fn=copy.deepcopy(static_config["reward_fn"]),
    )
    reset = env.reset(seed=task.seed, map_id=task.map_id, agent_player_id=task.agent_player_id, opponent_spec=task.opponent_spec)
    result_queue.put(
        {
            "type": "episode_started",
            "worker_id": worker_id,
            "eval_id": eval_id,
            "task_id": task.task_id,
            "seed": task.seed,
            "map_id": int(reset.info["map_id"]),
            "agent_player_id": int(task.agent_player_id),
            "opponent_spec": reset.info["opponent_spec"],
        }
    )
    use_fast_runtime_features = bool(static_config.get("enable_fast_runtime_features", False))
    extractor = None if use_fast_runtime_features else FeatureExtractor(player_id=task.agent_player_id)
    fast_runtime = FastRuntimeState(player_id=task.agent_player_id) if use_fast_runtime_features else None
    observation = reset.observation
    episode_events = _empty_event_counts()
    fast_events = _empty_event_counts()
    fast_infos: list[dict[str, Any]] = []
    threshold_raw_values: list[float] = []
    threshold_int_values: list[int] = []
    threshold_log_std_values: list[float] = []
    threshold_entropy_values: list[float] = []
    fast_scalar_values: list[tuple[float, float]] = []
    request_index = 0
    pending_output: GameOutput | None = None

    while True:
        if pending_output is not None:
            if fast_runtime is not None and observation is not None:
                fast_try = fast_runtime.try_fast(observation)
                if fast_try["status"] == "success":
                    output = _game_output_from_payload(fast_try["output"])
                    step = env.step(output, agent_decision_mode=AGENT_DECISION_FAST)
                    _add_event_counts(episode_events, step.info["events"])
                    _add_event_counts(fast_events, step.info["events"])
                    fast_infos.append(step.info)
                    pending_output = None
                    observation = step.observation
                    request_index += 1
                    profile_stats["steps"] = request_index
                    continue
            pending_output = None
        if observation is None:
            break
        if fast_runtime is None:
            assert extractor is not None
            actor_features = extractor.observe(observation)
            fast_scalars = np.asarray(INITIAL_FAST_SCALARS, dtype=np.float32)
        else:
            prepared = fast_runtime.prepare_neural(observation)
            actor_features = prepared["actor_features"]
            fast_scalars = np.asarray(prepared["fast_scalars"], dtype=np.float32)
        fast_scalar_values.append((float(fast_scalars[0]), float(fast_scalars[1])))
        if actor_features["feature_schema"] != "goldrush2_feature_v2":
            raise SimulatorRuleError(f"unexpected feature schema: {actor_features['feature_schema']!r}")
        critic_features = _extract_critic_features(env, task.agent_player_id, task.round_count)
        feature_shared.actor_planes[worker_id, ...] = np.asarray(actor_features["planes"], dtype=np.float32)
        feature_shared.actor_scalars[worker_id, ...] = np.asarray(actor_features["scalars"], dtype=np.float32)
        feature_shared.fast_scalars[worker_id, ...] = fast_scalars
        feature_shared.critic_planes[worker_id, ...] = np.asarray(critic_features["planes"], dtype=np.float32)
        feature_shared.critic_scalars[worker_id, ...] = np.asarray(critic_features["scalars"], dtype=np.float32)

        request_id = f"{task.task_id}-round-{request_index:04d}"
        result_queue.put(
            {
                "type": "feature_request",
                "eval_id": eval_id,
                "request": EvalFeatureRequest(
                    worker_id=worker_id,
                    eval_id=eval_id,
                    task_id=task.task_id,
                    request_id=request_id,
                    round_index=int(observation.round),
                    feature_slot=worker_id,
                    policy_sample_key=task.policy_sample_key,
                    policy_sample_seed=task.policy_sample_seed,
                ),
            }
        )
        action_wait_start = time.perf_counter_ns()
        action_msg = command_queue.get()
        profile_stats["action_wait_ns"] = profile_stats.get("action_wait_ns", 0) + (
            time.perf_counter_ns() - action_wait_start
        )
        if action_msg["type"] == "stop":
            return
        if action_msg["type"] != "action_result" or action_msg["eval_id"] != eval_id or action_msg["request_id"] != request_id:
            raise RuntimeError(f"eval worker received unexpected action message: {action_msg}")
        payload = action_msg["action"]
        output = GameOutput(
            actions=tuple(int(value) for value in payload["actions"]),
            k=int(payload["k"]),
            order=int(payload["order"]),
            vp=int(payload["vp"]),
        )
        if fast_runtime is None:
            assert extractor is not None
            extractor.commit_action(output)
        else:
            fast_runtime.commit_neural(output)
            fast_runtime.set_next_threshold(int(action_msg["threshold_int"]))
        threshold_raw_values.append(float(action_msg["threshold_raw"]))
        threshold_int_values.append(int(action_msg["threshold_int"]))
        threshold_log_std_values.append(float(action_msg["threshold_log_std"]))
        threshold_entropy_values.append(float(action_msg["threshold_entropy"]))
        env_step_start = time.perf_counter_ns()
        step = env.step(output)
        profile_stats["env_step_ns"] = profile_stats.get("env_step_ns", 0) + (time.perf_counter_ns() - env_step_start)
        _add_event_counts(episode_events, step.info["events"])
        observation = step.observation
        request_index += 1
        profile_stats["steps"] = request_index
        pending_output = output

    if request_index <= 0:
        raise SimulatorRuleError(f"eval task {task.task_id!r} produced no transitions")
    profile_stats["episode_wall_ns"] = time.perf_counter_ns() - episode_wall_start
    final_fast_diagnostics = _empty_fast_diagnostics() if fast_runtime is None else dict(fast_runtime.diagnostics())
    result_queue.put(
        {
            "type": "episode_done",
            "worker_id": worker_id,
            "eval_id": eval_id,
            "task_id": task.task_id,
            "summary": _episode_summary(
                task,
                reset_map_id=int(reset.info["map_id"]),
                opponent_spec=reset.info["opponent_spec"],
                episode_length=request_index,
                terminal_info=step.info,
                episode_events=episode_events,
                fast_events=fast_events,
                fast_infos=fast_infos,
                fast_diagnostics=final_fast_diagnostics,
                threshold_raw_values=threshold_raw_values,
                threshold_int_values=threshold_int_values,
                threshold_log_std_values=threshold_log_std_values,
                threshold_entropy_values=threshold_entropy_values,
                fast_scalar_values=fast_scalar_values,
            ),
            "worker_stats": profile_stats,
        }
    )


def _env_config_for_task(static_config: dict[str, Any], task: EvalTask) -> SingleAgentEnvConfig:
    base: SingleAgentEnvConfig = static_config["env_config"]
    rules = base.episode.rules
    if rules.round_count == task.round_count:
        return SingleAgentEnvConfig(
            episode=base.episode,
            agent_player_id=task.agent_player_id,
            opponent_spec=task.opponent_spec if task.opponent_spec is not None else base.opponent_spec,
            opponent_league=base.opponent_league,
            league_split=base.league_split,
            record_replay=False,
        )
    episode = replace(
        base.episode,
        rules=replace(rules, round_count=int(task.round_count)),
    )
    return SingleAgentEnvConfig(
        episode=episode,
        agent_player_id=task.agent_player_id,
        opponent_spec=task.opponent_spec if task.opponent_spec is not None else base.opponent_spec,
        opponent_league=base.opponent_league,
        league_split=base.league_split,
        record_replay=False,
    )


def _episode_summary(
    task: EvalTask,
    *,
    reset_map_id: int,
    opponent_spec: Any,
    episode_length: int,
    terminal_info: dict[str, Any],
    episode_events: dict[str, dict[int, int]],
    fast_events: dict[str, dict[int, int]],
    fast_infos: list[dict[str, Any]],
    fast_diagnostics: dict[str, Any],
    threshold_raw_values: list[float],
    threshold_int_values: list[int],
    threshold_log_std_values: list[float],
    threshold_entropy_values: list[float],
    fast_scalar_values: list[tuple[float, float]],
) -> EvalEpisodeSummary:
    agent = int(task.agent_player_id)
    opponent = 2 if agent == 1 else 1
    scores = terminal_info["scores"]
    game_result = terminal_info["game_result"]
    agent_net = int(scores["net_gold"][agent])
    opponent_net = int(scores["net_gold"][opponent])
    return EvalEpisodeSummary(
        task_id=task.task_id,
        setting=task.setting,
        seed=int(task.seed),
        map_id=reset_map_id,
        opponent=opponent_key(opponent_spec),
        agent_player_id=agent,
        round_count=int(task.round_count),
        episode_length=int(episode_length),
        agent_net_gold=agent_net,
        opponent_net_gold=opponent_net,
        margin=agent_net - opponent_net,
        agent_won=game_result.winner_id == agent,
        agent_pickup_gold=int(episode_events["pickup_gold"][agent]),
        agent_pickups=int(episode_events["pickups"][agent]),
        agent_bomb_lost_gold=int(episode_events["bomb_lost_gold"][agent]),
        agent_bomb_triggers=int(episode_events["bomb_triggers"][agent]),
        agent_trample_penalty=int(episode_events["trample_penalty"][agent]),
        agent_tramples=int(episode_events["tramples"][agent]),
        agent_vision_spent=int(scores["vision_spent"][agent]),
        opponent_pickup_gold=int(episode_events["pickup_gold"][opponent]),
        opponent_pickups=int(episode_events["pickups"][opponent]),
        opponent_bomb_lost_gold=int(episode_events["bomb_lost_gold"][opponent]),
        opponent_bomb_triggers=int(episode_events["bomb_triggers"][opponent]),
        opponent_trample_penalty=int(episode_events["trample_penalty"][opponent]),
        opponent_tramples=int(episode_events["tramples"][opponent]),
        opponent_vision_spent=int(scores["vision_spent"][opponent]),
        extra=_eval_extra_metrics(
            agent_player_id=agent,
            fast_events=fast_events,
            fast_infos=fast_infos,
            fast_diagnostics=fast_diagnostics,
            threshold_raw_values=threshold_raw_values,
            threshold_int_values=threshold_int_values,
            threshold_log_std_values=threshold_log_std_values,
            threshold_entropy_values=threshold_entropy_values,
            fast_scalar_values=fast_scalar_values,
        ),
    )


def _eval_extra_metrics(
    *,
    agent_player_id: int,
    fast_events: dict[str, dict[int, int]],
    fast_infos: list[dict[str, Any]],
    fast_diagnostics: dict[str, Any],
    threshold_raw_values: list[float],
    threshold_int_values: list[int],
    threshold_log_std_values: list[float],
    threshold_entropy_values: list[float],
    fast_scalar_values: list[tuple[float, float]],
) -> dict[str, float]:
    p_fast_values = [value[0] for value in fast_scalar_values]
    confidence_values = [value[1] for value in fast_scalar_values]
    fast_success = _diagnostic_float(fast_diagnostics, "fast_success")
    effective_updates = _diagnostic_float(fast_diagnostics, "fast_effective_updates")
    fast_samples = [info for info in fast_infos if bool(info.get("fast_order_sampled", False))]
    agent_first = [info for info in fast_samples if bool(info.get("agent_first", False))]
    latent_rates = [float(info["latent_first_rate"]) for info in fast_infos if "latent_first_rate" in info]
    return {
        "threshold_raw_mean": _mean(threshold_raw_values),
        "threshold_raw_std": _std(threshold_raw_values),
        "threshold_int_mean": _mean([float(value) for value in threshold_int_values]),
        "threshold_int_p10": _percentile([float(value) for value in threshold_int_values], 0.10),
        "threshold_int_p50": _percentile([float(value) for value in threshold_int_values], 0.50),
        "threshold_int_p90": _percentile([float(value) for value in threshold_int_values], 0.90),
        "threshold_log_std": _mean(threshold_log_std_values),
        "threshold_entropy": _mean(threshold_entropy_values),
        "threshold_approx_kl": 0.0,
        "fast_success_per_episode": fast_success,
        "fast_miss_no_target_per_episode": _diagnostic_float(fast_diagnostics, "fast_miss_no_target"),
        "fast_path_fail_per_episode": _diagnostic_float(fast_diagnostics, "fast_path_fail"),
        "neural_fallback_per_episode": _diagnostic_float(fast_diagnostics, "neural_fallback"),
        "fast_nonstay_per_episode": _diagnostic_float(fast_diagnostics, "fast_nonstay"),
        "latent_first_rate_mean": _mean(latent_rates),
        "actual_fast_first_rate": 0.0 if not fast_samples else float(len(agent_first)) / float(len(fast_samples)),
        "fast_order_samples_per_episode": float(len(fast_samples)),
        "p_fast_effective_mean": _mean(p_fast_values),
        "p_fast_effective_p50": _percentile(p_fast_values, 0.50),
        "fast_effective_confidence_mean": _mean(confidence_values),
        "fast_effective_updates_per_episode": effective_updates,
        "fast_effective_score_mean": _diagnostic_ratio(fast_diagnostics, "fast_effective_score_sum", effective_updates),
        "fast_expected_gain_mean": _diagnostic_ratio(fast_diagnostics, "fast_expected_gain_sum", effective_updates),
        "fast_actual_delta_mean": _diagnostic_ratio(fast_diagnostics, "fast_actual_delta_sum", effective_updates),
        "fast_pickup_gold_per_episode": float(fast_events["pickup_gold"][agent_player_id]),
        "fast_bomb_lost_gold_per_episode": float(fast_events["bomb_lost_gold"][agent_player_id]),
        "fast_trample_penalty_per_episode": float(fast_events["trample_penalty"][agent_player_id]),
        "one_step_fast_delta_per_episode": _diagnostic_float(fast_diagnostics, "one_step_fast_delta_sum"),
    }


def _empty_fast_diagnostics() -> dict[str, float]:
    return {
        "fast_success": 0.0,
        "fast_miss_no_target": 0.0,
        "fast_path_fail": 0.0,
        "neural_fallback": 0.0,
        "fast_nonstay": 0.0,
        "fast_effective_updates": 0.0,
        "fast_effective_score_sum": 0.0,
        "fast_expected_gain_sum": 0.0,
        "fast_actual_delta_sum": 0.0,
        "one_step_fast_delta_sum": 0.0,
    }


def _diagnostic_float(diagnostics: dict[str, Any], key: str) -> float:
    return float(diagnostics.get(key, 0.0))


def _diagnostic_ratio(diagnostics: dict[str, Any], numerator_key: str, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return _diagnostic_float(diagnostics, numerator_key) / float(denominator)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = _mean(values)
    return float((sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return float(ordered[index])


def _extract_critic_features(env: SingleAgentGoldRushEnv, agent_player_id: int, round_count: int) -> dict[str, Any]:
    if env.round_env is None:
        raise SimulatorRuleError("single-agent env has no round_env while extracting critic features")
    if env.round_env.state is None or env.round_env.template is None or env.round_env.outer_state is None:
        raise SimulatorRuleError("round_env missing state/template/outer_state while extracting critic features")
    return extract_privileged_critic_features(
        state=env.round_env.state,
        template=env.round_env.template,
        outer_state=env.round_env.outer_state,
        agent_player_id=agent_player_id,
        round_count=round_count,
    )
