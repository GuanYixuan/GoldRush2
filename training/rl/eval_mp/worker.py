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
from training.rl.env import SingleAgentEnvConfig, SingleAgentGoldRushEnv
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
    request_index = 0

    while observation is not None:
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
        env_step_start = time.perf_counter_ns()
        step = env.step(output)
        profile_stats["env_step_ns"] = profile_stats.get("env_step_ns", 0) + (time.perf_counter_ns() - env_step_start)
        _add_event_counts(episode_events, step.info["events"])
        observation = step.observation
        request_index += 1
        profile_stats["steps"] = request_index

    if request_index <= 0:
        raise SimulatorRuleError(f"eval task {task.task_id!r} produced no transitions")
    profile_stats["episode_wall_ns"] = time.perf_counter_ns() - episode_wall_start
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
    )


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
