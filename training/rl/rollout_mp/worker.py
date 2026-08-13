from __future__ import annotations

import copy
import time
import traceback
from collections.abc import Sequence
from typing import Any

from policy_runtime import FeatureExtractor
from simulator.errors import SimulatorRuleError
from simulator.types import GameOutput
from training.models import INITIAL_FAST_SCALARS
from training.rl.env import SingleAgentGoldRushEnv
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
    episode_wall_start = time.perf_counter_ns()
    env = SingleAgentGoldRushEnv(
        config=static_config["env_config"],
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
            "rollout_id": rollout_id,
            "task_id": task.task_id,
            "pair_id": task.pair_id,
            "pair_role": task.pair_role,
            "seed": task.seed,
            "map_id": int(reset.info["map_id"]),
            "agent_player_id": int(task.agent_player_id),
            "opponent_spec": reset.info["opponent_spec"],
        }
    )
    extractor = FeatureExtractor(player_id=task.agent_player_id)
    observation = reset.observation
    info_items: list[dict[str, Any]] = []
    episode_events = _empty_event_counts()
    request_index = 0
    transition_slot = int(task.transition_slot)
    max_episode_length = int(transition_shared.done.shape[1])

    while observation is not None:
        if request_index >= max_episode_length:
            raise SimulatorRuleError(
                f"episode {task.task_id!r} exceeded transition shared memory length {max_episode_length}"
            )
        actor_features = extractor.observe(observation)
        if actor_features["feature_schema"] != "goldrush2_feature_v2":
            raise SimulatorRuleError(f"unexpected feature schema: {actor_features['feature_schema']!r}")
        critic_features = _extract_critic_features(env, task.agent_player_id, int(static_config["round_count"]))
        actor_planes = np.asarray(actor_features["planes"], dtype=np.float32)
        actor_scalars = np.asarray(actor_features["scalars"], dtype=np.float32)
        fast_scalars = np.asarray(INITIAL_FAST_SCALARS, dtype=np.float32)
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
        extractor.commit_action(game_output)
        env_step_start = time.perf_counter_ns()
        step = env.step(game_output)
        profile_stats["env_step_ns"] = profile_stats.get("env_step_ns", 0) + (time.perf_counter_ns() - env_step_start)
        done = bool(step.terminated)
        _add_event_counts(episode_events, step.info["events"])
        transition_shared.actor_planes[transition_slot, request_index, ...] = actor_planes
        transition_shared.actor_scalars[transition_slot, request_index, ...] = actor_scalars
        transition_shared.fast_scalars[transition_slot, request_index, ...] = fast_scalars
        transition_shared.critic_planes[transition_slot, request_index, ...] = critic_planes
        transition_shared.critic_scalars[transition_slot, request_index, ...] = critic_scalars
        transition_shared.actions[transition_slot, request_index, :] = tuple(int(value) for value in game_output.actions)
        transition_shared.k[transition_slot, request_index] = int(game_output.k)
        transition_shared.order[transition_slot, request_index] = int(game_output.order)
        transition_shared.vp[transition_slot, request_index] = int(game_output.vp)
        transition_shared.threshold_raw[transition_slot, request_index] = float(action_msg["threshold_raw"])
        transition_shared.threshold_int[transition_slot, request_index] = int(action_msg["threshold_int"])
        transition_shared.old_logprob[transition_slot, request_index] = float(action_msg["old_logprob"])
        transition_shared.value[transition_slot, request_index] = float(action_msg["value"])
        transition_shared.reward[transition_slot, request_index] = float(step.reward)
        transition_shared.done[transition_slot, request_index] = done
        transition_shared.round_index[transition_slot, request_index] = int(observation.round)
        info_items.append(
            transition_info(
                step.info,
                done=done,
                mode=str(static_config["transition_info_mode"]),
                episode_events=episode_events,
            )
        )
        observation = step.observation
        request_index += 1
        profile_stats["steps"] = request_index

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
            "agent_player_id": int(task.agent_player_id),
            "opponent_spec": reset.info["opponent_spec"],
            "transition_slot": transition_slot,
            "episode_length": request_index,
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
    if mode == "training":
        reward_components = step_info.get("reward_components")
        if not done:
            return {} if reward_components is None else {"reward_components": reward_components}
        if episode_events is None:
            raise SimulatorRuleError("training transition info requires episode event totals on terminal step")
        payload = {
            "scores": step_info["scores"],
            "events": _copy_event_counts(episode_events),
            "game_result": step_info["game_result"],
        }
        if reward_components is not None:
            payload["reward_components"] = reward_components
        return payload
    raise SimulatorRuleError(f"unknown transition_info_mode: {mode!r}")


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
