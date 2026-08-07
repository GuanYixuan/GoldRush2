from __future__ import annotations

import queue
import time
from collections.abc import Sequence
from typing import Any

import torch

from simulator.errors import SimulatorRuleError
from training.models import GoldRushPolicyNetwork, policy_action_is_finite
from training.opponents import OpponentSpec

from .batch_assembly import initial_tasks
from .shared_memory import FeatureSharedMemory
from .stats import mean, sync, worker_profile_metrics
from .types import EpisodeTask, FeatureRequest, MultiprocessRolloutConfig


def scheduler_loop(
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
    feature_shared: FeatureSharedMemory,
    rollout_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pending_tasks = initial_tasks(seed=seed, pair_count=pair_count, map_ids=map_ids, opponent_specs=opponent_specs)
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
    worker_stat_totals: dict[str, int] = {}
    worker_episode_count = 0

    while completed_episodes < expected_episodes:
        while idle_workers and pending_tasks:
            dispatch_start = time.perf_counter_ns()
            worker_id = idle_workers.pop(0)
            task = pending_tasks.pop(0)
            active_tasks[task.task_id] = task
            command_queues[worker_id].put({"type": "start_episode", "rollout_id": rollout_id, "task": task})
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
            if msg_type == "feature_request":
                if msg.get("rollout_id") != rollout_id:
                    raise RuntimeError(f"feature_request rollout_id mismatch: {msg}")
                pending_requests.append(msg["request"])
            elif msg_type == "episode_started":
                if msg.get("rollout_id") != rollout_id:
                    raise RuntimeError(f"episode_started rollout_id mismatch: {msg}")
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
                                transition_slot=task.transition_slot + 1,
                            )
                        )
                else:
                    second_started += 1
            elif msg_type == "episode_done":
                if msg.get("rollout_id") != rollout_id:
                    raise RuntimeError(f"episode_done rollout_id mismatch: {msg}")
                completed_episodes += 1
                worker_id = int(msg["worker_id"])
                idle_workers.append(worker_id)
                task = active_tasks.pop(msg["task_id"])
                payload_extend_start = time.perf_counter_ns()
                payloads.append(msg)
                payload_extend_ns += time.perf_counter_ns() - payload_extend_start
                worker_stats = msg.get("worker_stats", {})
                if worker_stats:
                    worker_episode_count += 1
                    for key, value in worker_stats.items():
                        if isinstance(value, int):
                            worker_stat_totals[key] = worker_stat_totals.get(key, 0) + value
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
            inference_stats = run_inference_batch(model, pending_requests, command_queues, device, feature_shared)
            inference_ns += inference_stats["elapsed_ns"]
            inference_stack_ns += inference_stats["stack_ns"]
            inference_model_sample_ns += inference_stats["model_sample_ns"]
            inference_action_send_ns += inference_stats["action_send_ns"]
            inference_total_with_action_send_ns += inference_stats["total_with_action_send_ns"]
            feature_batches += 1
            feature_batch_sizes.append(inference_stats["batch_size"])
            pending_requests.clear()

    if pending_requests:
        inference_stats = run_inference_batch(model, pending_requests, command_queues, device, feature_shared)
        inference_ns += inference_stats["elapsed_ns"]
        inference_stack_ns += inference_stats["stack_ns"]
        inference_model_sample_ns += inference_stats["model_sample_ns"]
        inference_action_send_ns += inference_stats["action_send_ns"]
        inference_total_with_action_send_ns += inference_stats["total_with_action_send_ns"]
        feature_batches += 1
        feature_batch_sizes.append(inference_stats["batch_size"])

    payload_sort_start = time.perf_counter_ns()
    payloads.sort(key=lambda item: (item["pair_id"], 0 if item["pair_role"] == "first" else 1))
    payload_sort_ns = time.perf_counter_ns() - payload_sort_start
    stats = {
        "rollout_mode": "multiprocess",
        "rollout_num_workers": len(command_queues),
        "first_started": first_started,
        "second_started": second_started,
        "first_episodes": first_done,
        "second_episodes": second_done,
        "feature_batches": feature_batches,
        "mean_feature_batch_size": mean(feature_batch_sizes),
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
        "scheduler_message_handle_feature_request_ms": message_handle_by_type_ns.get("feature_request", 0) / 1_000_000,
        "scheduler_message_handle_episode_started_ms": message_handle_by_type_ns.get("episode_started", 0) / 1_000_000,
        "scheduler_message_handle_episode_done_ms": message_handle_by_type_ns.get("episode_done", 0) / 1_000_000,
        "scheduler_task_dispatch_ms": task_dispatch_ns / 1_000_000,
        "scheduler_payload_extend_ms": payload_extend_ns / 1_000_000,
        "scheduler_payload_sort_ms": payload_sort_ns / 1_000_000,
    }
    stats.update(worker_profile_metrics(worker_stat_totals, worker_episode_count))
    return payloads, stats


def run_inference_batch(
    model: GoldRushPolicyNetwork,
    requests: list[FeatureRequest],
    command_queues: list[Any],
    device: torch.device,
    feature_shared: FeatureSharedMemory,
) -> dict[str, int]:
    import numpy as np

    sync(device)
    start = time.perf_counter_ns()
    stack_start = time.perf_counter_ns()
    feature_slots = [request.feature_slot for request in requests]
    spatial = torch.as_tensor(np.asarray(feature_shared.actor_planes[feature_slots]), dtype=torch.float32, device=device)
    scalars = torch.as_tensor(np.asarray(feature_shared.actor_scalars[feature_slots]), dtype=torch.float32, device=device)
    stack_ns = time.perf_counter_ns() - stack_start
    model_sample_start = time.perf_counter_ns()
    with torch.no_grad():
        action = model.act(spatial, scalars)
        if not policy_action_is_finite(action):
            raise SimulatorRuleError("multiprocess rollout model produced NaN or Inf")
    sync(device)
    model_sample_ns = time.perf_counter_ns() - model_sample_start
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
                "rollout_id": request.rollout_id,
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
