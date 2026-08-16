from __future__ import annotations

import copy
import time
import traceback
from collections.abc import Sequence
from typing import Any

from policy_runtime import FastRuntimeState, FeatureExtractor
from simulator.errors import SimulatorRuleError
from simulator.types import GameOutput
from training.rl.ppo_buffer import (
    FAST_STATUS_MISS_NO_TARGET,
    FAST_STATUS_NONE,
    FAST_STATUS_PATH_FAIL,
    FAST_STATUS_SUCCESS,
)
from training.models import INITIAL_FAST_SCALARS
from training.rl.env import AGENT_DECISION_FAST, SingleAgentGoldRushEnv
from training.rl.privileged_critic_features import extract_privileged_critic_features

from .shared_memory import FeatureSharedMemory, TransitionSharedMemory, attach_feature_shared_memory, attach_transition_shared_memory
from .types import EpisodeTask, FeatureRequest


def worker_loop(worker_id: int, command_queue: Any, result_queue: Any, static_config: dict[str, Any]) -> None:
    feature_shared: FeatureSharedMemory | None = None
    transition_shared: TransitionSharedMemory | None = None
    current_rollout_id: str | None = None
    try:
        feature_shared = attach_feature_shared_memory(static_config["feature_shared_memory"])
        result_queue.put({"type": "worker_ready", "worker_id": worker_id})
        while True:
            msg = command_queue.get()
            msg_type = msg["type"]
            if msg_type == "stop":
                return
            if msg_type == "configure_rollout":
                if transition_shared is not None:
                    transition_shared.close()
                current_rollout_id = str(msg["rollout_id"])
                transition_shared = attach_transition_shared_memory(msg["transition_shared_memory"])
                result_queue.put({"type": "configure_ready", "worker_id": worker_id, "rollout_id": current_rollout_id})
                continue
            if msg_type == "release_rollout":
                rollout_id = str(msg["rollout_id"])
                if current_rollout_id != rollout_id:
                    raise RuntimeError(f"worker {worker_id} release rollout mismatch: {rollout_id!r} != {current_rollout_id!r}")
                if transition_shared is not None:
                    transition_shared.close()
                    transition_shared = None
                current_rollout_id = None
                result_queue.put({"type": "release_ready", "worker_id": worker_id, "rollout_id": rollout_id})
                continue
            if msg_type != "start_episode":
                raise RuntimeError(f"worker expected start_episode, got {msg_type!r}")
            if transition_shared is None or current_rollout_id is None:
                raise RuntimeError(f"worker {worker_id} received start_episode before configure_rollout")
            if msg.get("rollout_id") != current_rollout_id:
                raise RuntimeError(
                    f"worker {worker_id} start_episode rollout mismatch: {msg.get('rollout_id')!r} != {current_rollout_id!r}"
                )
            run_worker_episode(
                worker_id,
                current_rollout_id,
                msg["task"],
                command_queue,
                result_queue,
                static_config,
                feature_shared,
                transition_shared,
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
        if transition_shared is not None:
            transition_shared.close()


def run_worker_episode(
    worker_id: int,
    rollout_id: str,
    task: EpisodeTask,
    command_queue: Any,
    result_queue: Any,
    static_config: dict[str, Any],
    feature_shared: FeatureSharedMemory,
    transition_shared: TransitionSharedMemory,
) -> None:
    import numpy as np

    profile_stats: dict[str, int] = {"steps": 0}
    episode_policy_steps = 0
    episode_env_steps = 0
    episode_wall_start = time.perf_counter_ns()
    env = SingleAgentGoldRushEnv(
        config=static_config["env_config"],
        mechanisms=copy.deepcopy(static_config["mechanisms"]),
        map_pool=static_config["map_pool"],
        spawn=static_config["spawn"],
        reward_fn=copy.deepcopy(static_config["reward_fn"]),
    )
    reset = env.reset(
        seed=task.seed,
        map_id=task.map_id,
        map_key=task.map_key,
        agent_player_id=task.agent_player_id,
        opponent_spec=task.opponent_spec,
    )
    result_queue.put(
        {
            "type": "episode_started",
            "worker_id": worker_id,
            "rollout_id": rollout_id,
            "task_id": task.task_id,
            "pair_id": task.pair_id,
            "pair_role": task.pair_role,
            "seed": task.seed,
            "map_id": int(reset.info["map_id"]),
            "map_key": reset.info["map_key"],
            "agent_player_id": int(task.agent_player_id),
            "opponent_spec": reset.info["opponent_spec"],
        }
    )
    use_fast_runtime_features = bool(static_config.get("enable_fast_runtime_features", False))
    extractor = None if use_fast_runtime_features else FeatureExtractor(player_id=task.agent_player_id)
    fast_runtime = FastRuntimeState(player_id=task.agent_player_id) if use_fast_runtime_features else None
    observation = reset.observation
    info_items: list[dict[str, Any]] = []
    episode_events = _empty_event_counts()
    request_index = 0
    transition_slot = int(task.transition_slot)
    max_episode_length = int(transition_shared.done.shape[1])
    pending_transition: dict[str, Any] | None = None
    reward_fold_gamma = float(static_config.get("reward_fold_gamma", 0.97))

    while True:
        if pending_transition is not None:
            if fast_runtime is not None and observation is not None:
                fast_try = fast_runtime.try_fast(observation)
                if fast_try["status"] == "success":
                    fast_output = _game_output_from_payload(fast_try["output"])
                    env_step_start = time.perf_counter_ns()
                    step = env.step(fast_output, agent_decision_mode=AGENT_DECISION_FAST)
                    profile_stats["env_step_ns"] = profile_stats.get("env_step_ns", 0) + (
                        time.perf_counter_ns() - env_step_start
                    )
                    _add_event_counts(episode_events, step.info["events"])
                    tau = int(pending_transition["tau"])
                    pending_transition["reward_sum"] = float(pending_transition["reward_sum"]) + (reward_fold_gamma**tau) * float(step.reward)
                    pending_transition["tau"] = tau + 1
                    pending_transition["fast_success"] = True
                    pending_transition["fast_status"] = FAST_STATUS_SUCCESS
                    pending_transition["done"] = bool(step.terminated)
                    pending_transition["info"] = transition_info(
                        step.info,
                        done=bool(step.terminated),
                        mode=str(static_config["transition_info_mode"]),
                        episode_events=episode_events,
                    )
                    _write_pending_transition(
                        transition_shared,
                        transition_slot=transition_slot,
                        index=request_index,
                        pending_transition=pending_transition,
                    )
                    info_items.append(pending_transition["info"])
                    request_index += 1
                    profile_stats["steps"] = request_index
                    episode_policy_steps += 1
                    episode_env_steps += 1
                    pending_transition = None
                    observation = step.observation
                    continue
                pending_transition["fast_success"] = False
                pending_transition["fast_status"] = _fast_status_code(str(fast_try["status"]))
            pending_transition["info"] = transition_info(
                pending_transition["info"],
                done=bool(pending_transition["done"]),
                mode=str(static_config["transition_info_mode"]),
                episode_events=episode_events,
            )
            _write_pending_transition(
                transition_shared,
                transition_slot=transition_slot,
                index=request_index,
                pending_transition=pending_transition,
            )
            info_items.append(pending_transition["info"])
            request_index += 1
            profile_stats["steps"] = request_index
            episode_policy_steps += 1
            pending_transition = None
            continue

        if observation is None:
            break
        if request_index >= max_episode_length:
            raise SimulatorRuleError(
                f"episode {task.task_id!r} exceeded transition shared memory length {max_episode_length}"
            )

        if fast_runtime is None:
            assert extractor is not None
            actor_features = extractor.observe(observation)
            fast_scalars = np.asarray(INITIAL_FAST_SCALARS, dtype=np.float32)
        else:
            prepared = fast_runtime.prepare_neural(observation)
            actor_features = prepared["actor_features"]
            fast_scalars = np.asarray(prepared["fast_scalars"], dtype=np.float32)
        if actor_features["feature_schema"] != "goldrush2_feature_v2":
            raise SimulatorRuleError(f"unexpected feature schema: {actor_features['feature_schema']!r}")
        critic_features = _extract_critic_features(env, task.agent_player_id, int(static_config["round_count"]))
        actor_planes = np.asarray(actor_features["planes"], dtype=np.float32)
        actor_scalars = np.asarray(actor_features["scalars"], dtype=np.float32)
        critic_planes = np.asarray(critic_features["planes"], dtype=np.float32)
        critic_scalars = np.asarray(critic_features["scalars"], dtype=np.float32)
        feature_shared.actor_planes[worker_id, ...] = actor_planes
        feature_shared.actor_scalars[worker_id, ...] = actor_scalars
        feature_shared.fast_scalars[worker_id, ...] = fast_scalars
        feature_shared.critic_planes[worker_id, ...] = critic_planes
        feature_shared.critic_scalars[worker_id, ...] = critic_scalars
        request_id = f"{task.task_id}-round-{request_index:04d}"
        result_queue.put(
            {
                "type": "feature_request",
                "rollout_id": rollout_id,
                "request": FeatureRequest(
                    worker_id=worker_id,
                    rollout_id=rollout_id,
                    task_id=task.task_id,
                    request_id=request_id,
                    round_index=int(observation.round),
                    feature_slot=worker_id,
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
        if (
            action_msg["type"] != "action_result"
            or action_msg["rollout_id"] != rollout_id
            or action_msg["request_id"] != request_id
        ):
            raise RuntimeError(f"worker received unexpected action message: {action_msg}")

        action_payload = action_msg["action"]
        game_output = GameOutput(
            actions=tuple(int(value) for value in action_payload["actions"]),
            k=int(action_payload["k"]),
            order=int(action_payload["order"]),
            vp=int(action_payload["vp"]),
        )
        if fast_runtime is None:
            assert extractor is not None
            extractor.commit_action(game_output)
        else:
            fast_runtime.commit_neural(game_output)
            fast_runtime.set_next_threshold(int(action_msg["threshold_int"]))
        env_step_start = time.perf_counter_ns()
        step = env.step(game_output)
        profile_stats["env_step_ns"] = profile_stats.get("env_step_ns", 0) + (time.perf_counter_ns() - env_step_start)
        _add_event_counts(episode_events, step.info["events"])
        pending_transition = {
            "actor_planes": actor_planes,
            "actor_scalars": actor_scalars,
            "fast_scalars": fast_scalars,
            "critic_planes": critic_planes,
            "critic_scalars": critic_scalars,
            "actions": tuple(int(value) for value in game_output.actions),
            "k": int(game_output.k),
            "order": int(game_output.order),
            "vp": int(game_output.vp),
            "threshold_raw": float(action_msg["threshold_raw"]),
            "threshold_int": int(action_msg["threshold_int"]),
            "old_logprob": float(action_msg["old_logprob"]),
            "value": float(action_msg["value"]),
            "reward": float(step.reward),
            "reward_sum": float(step.reward),
            "tau": 1,
            "fast_success": False,
            "fast_status": FAST_STATUS_NONE,
            "done": bool(step.terminated),
            "round_index": int(observation.round),
            "info": step.info,
        }
        pending_transition["info"]["threshold_mu_raw"] = float(action_msg["threshold_mu_raw"])
        pending_transition["info"]["threshold_base_raw"] = float(action_msg["threshold_base_raw"])
        pending_transition["info"]["threshold_residual_raw"] = float(action_msg["threshold_residual_raw"])
        observation = step.observation
        episode_env_steps += 1
        if fast_runtime is None or observation is None:
            pending_transition["info"] = transition_info(
                pending_transition["info"],
                done=bool(pending_transition["done"]),
                mode=str(static_config["transition_info_mode"]),
                episode_events=episode_events,
            )
            _write_pending_transition(
                transition_shared,
                transition_slot=transition_slot,
                index=request_index,
                pending_transition=pending_transition,
            )
            info_items.append(pending_transition["info"])
            request_index += 1
            profile_stats["steps"] = request_index
            episode_policy_steps += 1
            pending_transition = None

    if request_index <= 0:
        raise SimulatorRuleError(f"episode {task.task_id!r} produced no transitions")
    profile_stats["episode_wall_ns"] = time.perf_counter_ns() - episode_wall_start
    result_queue.put(
        {
            "type": "episode_done",
            "worker_id": worker_id,
            "rollout_id": rollout_id,
            "task_id": task.task_id,
            "pair_id": task.pair_id,
            "pair_role": task.pair_role,
            "seed": task.seed,
            "map_id": int(reset.info["map_id"]),
            "map_key": reset.info["map_key"],
            "agent_player_id": int(task.agent_player_id),
            "opponent_spec": reset.info["opponent_spec"],
            "transition_slot": transition_slot,
            "episode_length": request_index,
            "policy_step_count": episode_policy_steps,
            "env_step_count": episode_env_steps,
            "infos": tuple(info_items),
            "worker_stats": profile_stats,
        }
    )


def transition_info(
    step_info: dict[str, Any],
    *,
    done: bool,
    mode: str,
    episode_events: dict[str, dict[int, int]] | None = None,
) -> dict[str, Any]:
    if mode == "debug":
        return step_info
    order_payload = _order_info(step_info)
    if mode == "training":
        reward_components = step_info.get("reward_components")
        if not done:
            if reward_components is not None:
                order_payload["reward_components"] = reward_components
            return order_payload
        if episode_events is None:
            raise SimulatorRuleError("training transition info requires episode event totals on terminal step")
        payload = {
            **order_payload,
            "scores": step_info["scores"],
            "events": _copy_event_counts(episode_events),
            "game_result": step_info["game_result"],
        }
        if reward_components is not None:
            payload["reward_components"] = reward_components
        return payload
    raise SimulatorRuleError(f"unknown transition_info_mode: {mode!r}")


def _write_pending_transition(
    transition_shared: TransitionSharedMemory,
    *,
    transition_slot: int,
    index: int,
    pending_transition: dict[str, Any],
) -> None:
    transition_shared.actor_planes[transition_slot, index, ...] = pending_transition["actor_planes"]
    transition_shared.actor_scalars[transition_slot, index, ...] = pending_transition["actor_scalars"]
    transition_shared.fast_scalars[transition_slot, index, ...] = pending_transition["fast_scalars"]
    transition_shared.critic_planes[transition_slot, index, ...] = pending_transition["critic_planes"]
    transition_shared.critic_scalars[transition_slot, index, ...] = pending_transition["critic_scalars"]
    transition_shared.actions[transition_slot, index, :] = tuple(int(value) for value in pending_transition["actions"])
    transition_shared.k[transition_slot, index] = int(pending_transition["k"])
    transition_shared.order[transition_slot, index] = int(pending_transition["order"])
    transition_shared.vp[transition_slot, index] = int(pending_transition["vp"])
    transition_shared.threshold_raw[transition_slot, index] = float(pending_transition["threshold_raw"])
    transition_shared.threshold_int[transition_slot, index] = int(pending_transition["threshold_int"])
    transition_shared.old_logprob[transition_slot, index] = float(pending_transition["old_logprob"])
    transition_shared.value[transition_slot, index] = float(pending_transition["value"])
    transition_shared.reward[transition_slot, index] = float(pending_transition["reward"])
    transition_shared.reward_sum[transition_slot, index] = float(pending_transition["reward_sum"])
    transition_shared.tau[transition_slot, index] = int(pending_transition["tau"])
    transition_shared.fast_success[transition_slot, index] = bool(pending_transition["fast_success"])
    transition_shared.fast_status[transition_slot, index] = int(pending_transition["fast_status"])
    transition_shared.done[transition_slot, index] = bool(pending_transition["done"])
    transition_shared.round_index[transition_slot, index] = int(pending_transition["round_index"])


def _game_output_from_payload(payload: Any) -> GameOutput:
    if isinstance(payload, GameOutput):
        return payload
    return GameOutput(
        actions=tuple(int(value) for value in payload["actions"]),
        k=int(payload["k"]),
        order=int(payload["order"]),
        vp=int(payload["vp"]),
    )


def _fast_status_code(status: str) -> int:
    if status == "miss_no_target":
        return FAST_STATUS_MISS_NO_TARGET
    if status == "path_fail":
        return FAST_STATUS_PATH_FAIL
    raise SimulatorRuleError(f"unexpected fast status while folding transition: {status!r}")


def _order_info(step_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "first_player_id": int(step_info["first_player_id"]),
        "agent_decision_mode": str(step_info.get("agent_decision_mode", "neural")),
        "latent_first_rate": float(step_info.get("latent_first_rate", 0.0)),
        "fast_order_sampled": bool(step_info.get("fast_order_sampled", False)),
        "agent_first": bool(step_info.get("agent_first", False)),
    }


def _empty_event_counts() -> dict[str, dict[int, int]]:
    return {
        "pickups": {1: 0, 2: 0},
        "pickup_gold": {1: 0, 2: 0},
        "bomb_triggers": {1: 0, 2: 0},
        "bomb_lost_gold": {1: 0, 2: 0},
        "tramples": {1: 0, 2: 0},
        "trample_penalty": {1: 0, 2: 0},
    }


def _add_event_counts(total: dict[str, dict[int, int]], step_events: dict[str, Any]) -> None:
    for field, per_player_total in total.items():
        per_player_step = step_events.get(field, {})
        if not isinstance(per_player_step, dict):
            continue
        for player_id in (1, 2):
            per_player_total[player_id] += int(per_player_step.get(player_id, per_player_step.get(str(player_id), 0)))


def _copy_event_counts(events: dict[str, dict[int, int]]) -> dict[str, dict[int, int]]:
    return {field: dict(per_player) for field, per_player in events.items()}


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
