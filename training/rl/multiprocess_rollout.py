from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from policy_runtime import FeatureExtractor
from simulator.errors import SimulatorRuleError
from simulator.types import GameOutput
from training.models import (
    GoldRushPolicyNetwork,
    policy_output_is_finite,
    sample_action,
)
from training.opponents import OpponentSpec

from .env import SingleAgentGoldRushEnv
from .ppo_buffer import PpoBatch, PpoTransition
from .sampler import BatchRolloutSampler


PairRole = Literal["first", "second"]


@dataclass(frozen=True)
class MultiprocessRolloutConfig:
    num_workers: int = 8
    max_inference_batch_size: int = 64
    inference_timeout_ms: float = 2.0
    worker_join_timeout_s: float = 5.0
    transition_info_mode: Literal["training", "debug"] = "training"


@dataclass(frozen=True)
class EpisodeTask:
    task_id: str
    pair_id: str
    pair_role: PairRole
    seed: int
    map_id: int | None
    agent_player_id: int
    opponent_spec: OpponentSpec | None


@dataclass(frozen=True)
class FeatureRequest:
    worker_id: int
    task_id: str
    request_id: str
    round_index: int
    spatial_planes: Any
    scalars: Any


def collect_multiprocess_ppo_rollouts(
    model: GoldRushPolicyNetwork,
    sampler: BatchRolloutSampler,
    *,
    pair_count: int,
    seed: int,
    map_ids: Sequence[int] | None = None,
    opponent_specs: Sequence[OpponentSpec] | None = None,
    device: torch.device | str | None = None,
    config: MultiprocessRolloutConfig | None = None,
) -> tuple[PpoBatch, dict[str, Any]]:
    config = MultiprocessRolloutConfig() if config is None else config
    _validate_inputs(pair_count=pair_count, map_ids=map_ids, opponent_specs=opponent_specs, config=config)

    rollout_device = torch.device(device) if device is not None else next(model.parameters()).device
    was_training = model.training
    model.eval()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    command_queues = [ctx.Queue() for _ in range(config.num_workers)]
    worker_config = _worker_static_config(sampler, config)
    processes = [
        ctx.Process(
            target=_worker_loop,
            args=(worker_id, command_queues[worker_id], result_queue, worker_config),
            daemon=True,
        )
        for worker_id in range(config.num_workers)
    ]
    worker_startup_start = time.perf_counter_ns()
    for process in processes:
        process.start()
    worker_startup_ns = time.perf_counter_ns() - worker_startup_start

    try:
        scheduler_start = time.perf_counter_ns()
        payloads, stats = _scheduler_loop(
            model,
            command_queues,
            result_queue,
            pair_count=pair_count,
            seed=seed,
            map_ids=map_ids,
            opponent_specs=opponent_specs,
            device=rollout_device,
            config=config,
            worker_startup_start_ns=worker_startup_start,
        )
        scheduler_ns = time.perf_counter_ns() - scheduler_start
    finally:
        if was_training:
            model.train()
        for command_queue in command_queues:
            command_queue.put({"type": "stop"})
        for process in processes:
            process.join(timeout=config.worker_join_timeout_s)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

    batch_assembly_start = time.perf_counter_ns()
    transitions = [_payload_to_transition(payload) for payload in payloads]
    batch = PpoBatch.from_transitions(transitions)
    batch_assembly_ns = time.perf_counter_ns() - batch_assembly_start
    stats = {
        **stats,
        "worker_startup_ms": worker_startup_ns / 1_000_000,
        "scheduler_ms": scheduler_ns / 1_000_000,
        "batch_assembly_ms": batch_assembly_ns / 1_000_000,
    }
    return batch, stats


def _scheduler_loop(
    model: GoldRushPolicyNetwork,
    command_queues: list[Any],
    result_queue: Any,
    *,
    pair_count: int,
    seed: int,
    map_ids: Sequence[int] | None,
    opponent_specs: Sequence[OpponentSpec] | None,
    device: torch.device,
    config: MultiprocessRolloutConfig,
    worker_startup_start_ns: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pending_tasks = _initial_tasks(seed=seed, pair_count=pair_count, map_ids=map_ids, opponent_specs=opponent_specs)
    idle_workers = list(range(len(command_queues)))
    active_tasks: dict[str, EpisodeTask] = {}
    payloads: list[dict[str, Any]] = []
    pending_requests: list[FeatureRequest] = []
    feature_batches = 0
    feature_batch_sizes: list[int] = []
    inference_ns = 0
    first_started = 0
    second_started = 0
    first_done = 0
    second_done = 0
    second_enqueued_pairs: set[str] = set()
    expected_episodes = pair_count * 2
    completed_episodes = 0
    ready_workers = 0
    first_worker_ready_ns: int | None = None
    all_workers_ready_ns: int | None = None
    timeout_s = config.inference_timeout_ms / 1000.0
    queue_get_ns = 0
    queue_get_empty_ns = 0
    queue_get_by_type_ns: dict[str, int] = {}
    message_handle_ns = 0
    message_handle_by_type_ns: dict[str, int] = {}
    task_dispatch_ns = 0
    payload_extend_ns = 0
    inference_stack_ns = 0
    inference_model_sample_ns = 0
    inference_action_send_ns = 0
    inference_total_with_action_send_ns = 0

    while completed_episodes < expected_episodes:
        while idle_workers and pending_tasks:
            dispatch_start = time.perf_counter_ns()
            worker_id = idle_workers.pop(0)
            task = pending_tasks.pop(0)
            active_tasks[task.task_id] = task
            command_queues[worker_id].put({"type": "start_episode", "task": task})
            task_dispatch_ns += time.perf_counter_ns() - dispatch_start

        try:
            queue_get_start = time.perf_counter_ns()
            msg = result_queue.get(timeout=timeout_s)
            queue_get_elapsed = time.perf_counter_ns() - queue_get_start
            queue_get_ns += queue_get_elapsed
        except queue.Empty:
            queue_get_elapsed = time.perf_counter_ns() - queue_get_start
            queue_get_ns += queue_get_elapsed
            queue_get_empty_ns += queue_get_elapsed
            msg = None

        if msg is not None:
            msg_type = msg["type"]
            queue_get_by_type_ns[msg_type] = queue_get_by_type_ns.get(msg_type, 0) + queue_get_elapsed
            message_handle_start = time.perf_counter_ns()
            if msg_type == "worker_ready":
                ready_workers += 1
                now_ns = time.perf_counter_ns()
                if first_worker_ready_ns is None:
                    first_worker_ready_ns = now_ns
                if ready_workers == len(command_queues):
                    all_workers_ready_ns = now_ns
            elif msg_type == "feature_request":
                pending_requests.append(msg["request"])
            elif msg_type == "episode_started":
                task = active_tasks[msg["task_id"]]
                if task.pair_role == "first":
                    first_started += 1
                    if task.pair_id not in second_enqueued_pairs:
                        second_enqueued_pairs.add(task.pair_id)
                        pending_tasks.append(
                            EpisodeTask(
                                task_id=f"{task.pair_id}-second",
                                pair_id=task.pair_id,
                                pair_role="second",
                                seed=task.seed,
                                map_id=int(msg["map_id"]),
                                agent_player_id=2,
                                opponent_spec=msg["opponent_spec"],
                            )
                        )
                else:
                    second_started += 1
            elif msg_type == "episode_done":
                completed_episodes += 1
                worker_id = int(msg["worker_id"])
                idle_workers.append(worker_id)
                task = active_tasks.pop(msg["task_id"])
                payload_extend_start = time.perf_counter_ns()
                payloads.extend(msg["transitions"])
                payload_extend_ns += time.perf_counter_ns() - payload_extend_start
                if task.pair_role == "first":
                    first_done += 1
                else:
                    second_done += 1
            elif msg_type == "worker_error":
                raise RuntimeError(f"worker {msg['worker_id']} failed: {msg['error']}\n{msg['traceback']}")
            else:
                raise RuntimeError(f"unexpected worker message type: {msg_type!r}")
            handle_elapsed = time.perf_counter_ns() - message_handle_start
            message_handle_ns += handle_elapsed
            message_handle_by_type_ns[msg_type] = message_handle_by_type_ns.get(msg_type, 0) + handle_elapsed

        if pending_requests and (len(pending_requests) >= config.max_inference_batch_size or msg is None):
            inference_stats = _run_inference_batch(model, pending_requests, command_queues, device)
            elapsed_ns = inference_stats["elapsed_ns"]
            batch_size = inference_stats["batch_size"]
            inference_ns += elapsed_ns
            inference_stack_ns += inference_stats["stack_ns"]
            inference_model_sample_ns += inference_stats["model_sample_ns"]
            inference_action_send_ns += inference_stats["action_send_ns"]
            inference_total_with_action_send_ns += inference_stats["total_with_action_send_ns"]
            feature_batches += 1
            feature_batch_sizes.append(batch_size)
            pending_requests.clear()

    if pending_requests:
        inference_stats = _run_inference_batch(model, pending_requests, command_queues, device)
        elapsed_ns = inference_stats["elapsed_ns"]
        batch_size = inference_stats["batch_size"]
        inference_ns += elapsed_ns
        inference_stack_ns += inference_stats["stack_ns"]
        inference_model_sample_ns += inference_stats["model_sample_ns"]
        inference_action_send_ns += inference_stats["action_send_ns"]
        inference_total_with_action_send_ns += inference_stats["total_with_action_send_ns"]
        feature_batches += 1
        feature_batch_sizes.append(batch_size)

    payload_sort_start = time.perf_counter_ns()
    payloads.sort(key=lambda item: (item["pair_id"], 0 if item["pair_role"] == "first" else 1, item["round_index"]))
    payload_sort_ns = time.perf_counter_ns() - payload_sort_start
    all_ready_elapsed_ns = None if all_workers_ready_ns is None else all_workers_ready_ns - worker_startup_start_ns
    first_ready_elapsed_ns = None if first_worker_ready_ns is None else first_worker_ready_ns - worker_startup_start_ns
    return payloads, {
        "rollout_mode": "multiprocess",
        "rollout_num_workers": len(command_queues),
        "worker_ready_count": ready_workers,
        "worker_first_ready_ms": None if first_ready_elapsed_ns is None else first_ready_elapsed_ns / 1_000_000,
        "worker_all_ready_ms": None if all_ready_elapsed_ns is None else all_ready_elapsed_ns / 1_000_000,
        "first_started": first_started,
        "second_started": second_started,
        "first_episodes": first_done,
        "second_episodes": second_done,
        "feature_batches": feature_batches,
        "mean_feature_batch_size": _mean(feature_batch_sizes),
        "max_feature_batch_size": max(feature_batch_sizes) if feature_batch_sizes else 0,
        "inference_ms": inference_ns / 1_000_000,
        "inference_mean_ms": inference_ns / max(feature_batches, 1) / 1_000_000,
        "inference_stack_ms": inference_stack_ns / 1_000_000,
        "inference_model_sample_ms": inference_model_sample_ns / 1_000_000,
        "inference_action_send_ms": inference_action_send_ns / 1_000_000,
        "inference_total_with_action_send_ms": inference_total_with_action_send_ns / 1_000_000,
        "scheduler_queue_get_ms": queue_get_ns / 1_000_000,
        "scheduler_queue_get_empty_ms": queue_get_empty_ns / 1_000_000,
        "scheduler_queue_get_worker_ready_ms": queue_get_by_type_ns.get("worker_ready", 0) / 1_000_000,
        "scheduler_queue_get_feature_request_ms": queue_get_by_type_ns.get("feature_request", 0) / 1_000_000,
        "scheduler_queue_get_episode_started_ms": queue_get_by_type_ns.get("episode_started", 0) / 1_000_000,
        "scheduler_queue_get_episode_done_ms": queue_get_by_type_ns.get("episode_done", 0) / 1_000_000,
        "scheduler_message_handle_ms": message_handle_ns / 1_000_000,
        "scheduler_message_handle_worker_ready_ms": message_handle_by_type_ns.get("worker_ready", 0) / 1_000_000,
        "scheduler_message_handle_feature_request_ms": message_handle_by_type_ns.get("feature_request", 0) / 1_000_000,
        "scheduler_message_handle_episode_started_ms": message_handle_by_type_ns.get("episode_started", 0) / 1_000_000,
        "scheduler_message_handle_episode_done_ms": message_handle_by_type_ns.get("episode_done", 0) / 1_000_000,
        "scheduler_task_dispatch_ms": task_dispatch_ns / 1_000_000,
        "scheduler_payload_extend_ms": payload_extend_ns / 1_000_000,
        "scheduler_payload_sort_ms": payload_sort_ns / 1_000_000,
    }


def _run_inference_batch(
    model: GoldRushPolicyNetwork,
    requests: list[FeatureRequest],
    command_queues: list[Any],
    device: torch.device,
) -> dict[str, int]:
    import numpy as np

    _sync(device)
    start = time.perf_counter_ns()
    stack_start = time.perf_counter_ns()
    spatial = torch.as_tensor(np.stack([request.spatial_planes for request in requests]), dtype=torch.float32, device=device)
    scalars = torch.as_tensor(np.stack([request.scalars for request in requests]), dtype=torch.float32, device=device)
    stack_ns = time.perf_counter_ns() - stack_start
    model_sample_start = time.perf_counter_ns()
    with torch.no_grad():
        output = model(spatial, scalars)
        if not policy_output_is_finite(output):
            raise SimulatorRuleError("multiprocess rollout model produced NaN or Inf")
        action = sample_action(output)
    model_sample_ns = time.perf_counter_ns() - model_sample_start
    _sync(device)
    elapsed_ns = time.perf_counter_ns() - start
    action_send_start = time.perf_counter_ns()
    actions_cpu = action.actions.detach().cpu().tolist()
    k_cpu = action.k.detach().cpu().tolist()
    order_cpu = action.order.detach().cpu().tolist()
    vp_cpu = action.vp.detach().cpu().tolist()
    logprob_cpu = action.logprob.detach().cpu().tolist()
    value_cpu = action.value.detach().cpu().tolist()
    for batch_index, request in enumerate(requests):
        command_queues[request.worker_id].put(
            {
                "type": "action_result",
                "request_id": request.request_id,
                "action": {
                    "actions": tuple(int(value) for value in actions_cpu[batch_index]),
                    "k": int(k_cpu[batch_index]),
                    "order": int(order_cpu[batch_index]),
                    "vp": int(vp_cpu[batch_index]),
                },
                "old_logprob": float(logprob_cpu[batch_index]),
                "value": float(value_cpu[batch_index]),
            }
        )
    action_send_ns = time.perf_counter_ns() - action_send_start
    total_with_action_send_ns = time.perf_counter_ns() - start
    return {
        "elapsed_ns": elapsed_ns,
        "batch_size": len(requests),
        "stack_ns": stack_ns,
        "model_sample_ns": model_sample_ns,
        "action_send_ns": action_send_ns,
        "total_with_action_send_ns": total_with_action_send_ns,
    }


def _transition_info(step_info: dict[str, Any], *, done: bool, mode: str) -> dict[str, Any]:
    if mode == "debug":
        return step_info
    if mode == "training":
        if not done:
            return {}
        return {
            "scores": step_info["scores"],
            "events": step_info["events"],
            "game_result": step_info["game_result"],
        }
    raise SimulatorRuleError(f"unknown transition_info_mode: {mode!r}")


def _worker_loop(worker_id: int, command_queue: Any, result_queue: Any, static_config: dict[str, Any]) -> None:
    try:
        result_queue.put({"type": "worker_ready", "worker_id": worker_id})
        while True:
            msg = command_queue.get()
            msg_type = msg["type"]
            if msg_type == "stop":
                return
            if msg_type != "start_episode":
                raise RuntimeError(f"worker expected start_episode, got {msg_type!r}")
            _run_worker_episode(worker_id, msg["task"], command_queue, result_queue, static_config)
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


def _run_worker_episode(worker_id: int, task: EpisodeTask, command_queue: Any, result_queue: Any, static_config: dict[str, Any]) -> None:
    import numpy as np

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
    transitions: list[dict[str, Any]] = []
    request_index = 0

    while observation is not None:
        features = extractor.observe(observation)
        if features["feature_schema"] != "goldrush2_feature_v1":
            raise SimulatorRuleError(f"unexpected feature schema: {features['feature_schema']!r}")
        request_id = f"{task.task_id}-round-{request_index:04d}"
        spatial_planes = np.asarray(features["planes"], dtype=np.float32)
        scalars = np.asarray(features["scalars"], dtype=np.float32)
        result_queue.put(
            {
                "type": "feature_request",
                "request": FeatureRequest(
                    worker_id=worker_id,
                    task_id=task.task_id,
                    request_id=request_id,
                    round_index=int(observation.round),
                    spatial_planes=spatial_planes,
                    scalars=scalars,
                ),
            }
        )
        action_msg = command_queue.get()
        if action_msg["type"] == "stop":
            return
        if action_msg["type"] != "action_result" or action_msg["request_id"] != request_id:
            raise RuntimeError(f"worker received unexpected action message: {action_msg}")

        action_payload = action_msg["action"]
        game_output = GameOutput(
            actions=tuple(int(value) for value in action_payload["actions"]),
            k=int(action_payload["k"]),
            order=int(action_payload["order"]),
            vp=int(action_payload["vp"]),
        )
        extractor.commit_action(game_output)
        step = env.step(game_output)
        done = bool(step.terminated)
        transitions.append(
            {
                "spatial_planes": spatial_planes,
                "scalars": scalars,
                "actions": tuple(int(value) for value in game_output.actions),
                "k": int(game_output.k),
                "order": int(game_output.order),
                "vp": int(game_output.vp),
                "old_logprob": float(action_msg["old_logprob"]),
                "value": float(action_msg["value"]),
                "reward": float(step.reward),
                "done": done,
                "episode_id": f"{task.pair_id}-{task.pair_role}",
                "pair_id": task.pair_id,
                "pair_role": task.pair_role,
                "round_index": int(observation.round),
                "map_id": int(reset.info["map_id"]),
                "agent_player_id": int(task.agent_player_id),
                "info": _transition_info(step.info, done=done, mode=str(static_config["transition_info_mode"])),
            }
        )
        observation = step.observation
        request_index += 1

    result_queue.put(
        {
            "type": "episode_done",
            "worker_id": worker_id,
            "task_id": task.task_id,
            "pair_id": task.pair_id,
            "pair_role": task.pair_role,
            "seed": task.seed,
            "map_id": int(reset.info["map_id"]),
            "agent_player_id": int(task.agent_player_id),
            "opponent_spec": reset.info["opponent_spec"],
            "transitions": transitions,
        }
    )


def _initial_tasks(
    *,
    seed: int,
    pair_count: int,
    map_ids: Sequence[int] | None,
    opponent_specs: Sequence[OpponentSpec] | None,
) -> list[EpisodeTask]:
    tasks: list[EpisodeTask] = []
    for pair_index in range(pair_count):
        episode_seed = seed + pair_index
        pair_id = f"pair-{pair_index:06d}-seed-{episode_seed}"
        tasks.append(
            EpisodeTask(
                task_id=f"{pair_id}-first",
                pair_id=pair_id,
                pair_role="first",
                seed=episode_seed,
                map_id=map_ids[pair_index % len(map_ids)] if map_ids is not None else None,
                agent_player_id=1,
                opponent_spec=opponent_specs[pair_index % len(opponent_specs)] if opponent_specs is not None else None,
            )
        )
    return tasks


def _payload_to_transition(payload: dict[str, Any]) -> PpoTransition:
    return PpoTransition(
        spatial_planes=torch.as_tensor(payload["spatial_planes"], dtype=torch.float32),
        scalars=torch.as_tensor(payload["scalars"], dtype=torch.float32),
        actions=torch.as_tensor(payload["actions"], dtype=torch.long),
        k=torch.as_tensor(payload["k"], dtype=torch.long),
        order=torch.as_tensor(payload["order"], dtype=torch.long),
        vp=torch.as_tensor(payload["vp"], dtype=torch.long),
        old_logprob=torch.as_tensor(payload["old_logprob"], dtype=torch.float32),
        value=torch.as_tensor(payload["value"], dtype=torch.float32),
        reward=float(payload["reward"]),
        done=bool(payload["done"]),
        episode_id=str(payload["episode_id"]),
        round_index=int(payload["round_index"]),
        map_id=int(payload["map_id"]),
        agent_player_id=int(payload["agent_player_id"]),
        info=payload["info"],
    )


def _worker_static_config(sampler: BatchRolloutSampler, config: MultiprocessRolloutConfig) -> dict[str, Any]:
    return {
        "env_config": sampler.env_config,
        "mechanisms": sampler.mechanisms,
        "map_pool": sampler.map_pool,
        "spawn": sampler.spawn,
        "reward_fn": sampler.reward_fn,
        "transition_info_mode": config.transition_info_mode,
    }


def _validate_inputs(
    *,
    pair_count: int,
    map_ids: Sequence[int] | None,
    opponent_specs: Sequence[OpponentSpec] | None,
    config: MultiprocessRolloutConfig,
) -> None:
    if pair_count <= 0:
        raise SimulatorRuleError(f"pair_count must be positive, got {pair_count}")
    if map_ids is not None and not map_ids:
        raise SimulatorRuleError("map_ids cannot be empty when provided")
    if opponent_specs is not None and not opponent_specs:
        raise SimulatorRuleError("opponent_specs cannot be empty when provided")
    if config.num_workers <= 0:
        raise SimulatorRuleError(f"num_workers must be positive, got {config.num_workers}")
    if config.max_inference_batch_size <= 0:
        raise SimulatorRuleError(f"max_inference_batch_size must be positive, got {config.max_inference_batch_size}")
    if config.inference_timeout_ms < 0.0:
        raise SimulatorRuleError(f"inference_timeout_ms must be non-negative, got {config.inference_timeout_ms}")
    if config.worker_join_timeout_s <= 0.0:
        raise SimulatorRuleError(f"worker_join_timeout_s must be positive, got {config.worker_join_timeout_s}")
    if config.transition_info_mode not in ("training", "debug"):
        raise SimulatorRuleError(f"transition_info_mode must be training or debug, got {config.transition_info_mode!r}")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


__all__ = ["MultiprocessRolloutConfig", "collect_multiprocess_ppo_rollouts"]
