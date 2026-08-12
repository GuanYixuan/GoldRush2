from __future__ import annotations

import hashlib
import queue
import time
from collections.abc import Sequence
from typing import Any

import torch

from training.models import GoldRushPolicyNetwork, policy_action_is_finite
from training.rl.rollout_mp.shared_memory import FeatureSharedMemory
from training.rl.rollout_mp.stats import mean, sync, worker_profile_metrics

from .types import EvalEpisodeSummary, EvalFeatureRequest, EvalTask, ParallelEvalConfig


def scheduler_loop(
    model: GoldRushPolicyNetwork,
    command_queues: list[Any],
    result_queue: Any,
    *,
    tasks: Sequence[EvalTask],
    device: torch.device,
    config: ParallelEvalConfig,
    feature_shared: FeatureSharedMemory,
    eval_id: str,
) -> tuple[list[EvalEpisodeSummary], dict[str, Any]]:
    pending_tasks = list(tasks)
    idle_workers = list(range(len(command_queues)))
    active_tasks: dict[str, EvalTask] = {}
    summaries: list[EvalEpisodeSummary] = []
    pending_requests: list[EvalFeatureRequest] = []
    expected_episodes = len(pending_tasks)
    completed_episodes = 0
    timeout_s = config.inference_timeout_ms / 1000.0

    feature_batches = 0
    feature_batch_sizes: list[int] = []
    inference_ns = 0
    inference_stack_ns = 0
    inference_model_sample_ns = 0
    inference_action_send_ns = 0
    inference_total_with_action_send_ns = 0
    queue_get_ns = 0
    queue_get_empty_ns = 0
    queue_get_by_type_ns: dict[str, int] = {}
    message_handle_ns = 0
    message_handle_by_type_ns: dict[str, int] = {}
    task_dispatch_ns = 0
    summary_extend_ns = 0
    summary_sort_ns = 0
    worker_stat_totals: dict[str, int] = {}
    worker_episode_count = 0
    episode_started = 0

    while completed_episodes < expected_episodes:
        while idle_workers and pending_tasks:
            dispatch_start = time.perf_counter_ns()
            worker_id = idle_workers.pop(0)
            task = pending_tasks.pop(0)
            active_tasks[task.task_id] = task
            command_queues[worker_id].put({"type": "start_eval_episode", "eval_id": eval_id, "task": task})
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
                if msg.get("eval_id") != eval_id:
                    raise RuntimeError(f"feature_request eval_id mismatch: {msg}")
                pending_requests.append(msg["request"])
            elif msg_type == "episode_started":
                if msg.get("eval_id") != eval_id:
                    raise RuntimeError(f"episode_started eval_id mismatch: {msg}")
                episode_started += 1
            elif msg_type == "episode_done":
                if msg.get("eval_id") != eval_id:
                    raise RuntimeError(f"episode_done eval_id mismatch: {msg}")
                completed_episodes += 1
                worker_id = int(msg["worker_id"])
                idle_workers.append(worker_id)
                active_tasks.pop(msg["task_id"])
                summary_extend_start = time.perf_counter_ns()
                summaries.append(msg["summary"])
                summary_extend_ns += time.perf_counter_ns() - summary_extend_start
                worker_stats = msg.get("worker_stats", {})
                if worker_stats:
                    worker_episode_count += 1
                    for key, value in worker_stats.items():
                        if isinstance(value, int):
                            worker_stat_totals[key] = worker_stat_totals.get(key, 0) + value
            elif msg_type == "worker_error":
                raise RuntimeError(f"worker {msg['worker_id']} failed: {msg['error']}\n{msg['traceback']}")
            else:
                raise RuntimeError(f"unexpected eval worker message type: {msg_type!r}")
            handle_elapsed = time.perf_counter_ns() - message_handle_start
            message_handle_ns += handle_elapsed
            message_handle_by_type_ns[msg_type] = message_handle_by_type_ns.get(msg_type, 0) + handle_elapsed

        if pending_requests and (len(pending_requests) >= config.max_inference_batch_size or msg is None):
            inference_stats = run_eval_inference_batch(
                model,
                pending_requests,
                command_queues,
                device,
                feature_shared,
                deterministic=config.deterministic,
            )
            inference_ns += inference_stats["elapsed_ns"]
            inference_stack_ns += inference_stats["stack_ns"]
            inference_model_sample_ns += inference_stats["model_sample_ns"]
            inference_action_send_ns += inference_stats["action_send_ns"]
            inference_total_with_action_send_ns += inference_stats["total_with_action_send_ns"]
            feature_batches += 1
            feature_batch_sizes.append(inference_stats["batch_size"])
            pending_requests.clear()

    if pending_requests:
        inference_stats = run_eval_inference_batch(
            model,
            pending_requests,
            command_queues,
            device,
            feature_shared,
            deterministic=config.deterministic,
        )
        inference_ns += inference_stats["elapsed_ns"]
        inference_stack_ns += inference_stats["stack_ns"]
        inference_model_sample_ns += inference_stats["model_sample_ns"]
        inference_action_send_ns += inference_stats["action_send_ns"]
        inference_total_with_action_send_ns += inference_stats["total_with_action_send_ns"]
        feature_batches += 1
        feature_batch_sizes.append(inference_stats["batch_size"])

    summary_sort_start = time.perf_counter_ns()
    summaries.sort(key=lambda item: item.task_id)
    summary_sort_ns = time.perf_counter_ns() - summary_sort_start
    stats = {
        "eval_mode": "parallel",
        "eval_deterministic": config.deterministic,
        "eval_num_workers": len(command_queues),
        "eval_task_count": len(tasks),
        "eval_episode_started": episode_started,
        "eval_episode_count": completed_episodes,
        "eval_feature_batches": feature_batches,
        "eval_mean_feature_batch_size": mean(feature_batch_sizes),
        "eval_max_feature_batch_size": max(feature_batch_sizes) if feature_batch_sizes else 0,
        "eval_inference_ms": inference_ns / 1_000_000,
        "eval_inference_mean_ms": inference_ns / max(feature_batches, 1) / 1_000_000,
        "eval_inference_stack_ms": inference_stack_ns / 1_000_000,
        "eval_inference_model_sample_ms": inference_model_sample_ns / 1_000_000,
        "eval_inference_action_send_ms": inference_action_send_ns / 1_000_000,
        "eval_inference_total_with_action_send_ms": inference_total_with_action_send_ns / 1_000_000,
        "eval_scheduler_queue_get_ms": queue_get_ns / 1_000_000,
        "eval_scheduler_queue_get_empty_ms": queue_get_empty_ns / 1_000_000,
        "eval_scheduler_queue_get_feature_request_ms": queue_get_by_type_ns.get("feature_request", 0) / 1_000_000,
        "eval_scheduler_queue_get_episode_started_ms": queue_get_by_type_ns.get("episode_started", 0) / 1_000_000,
        "eval_scheduler_queue_get_episode_done_ms": queue_get_by_type_ns.get("episode_done", 0) / 1_000_000,
        "eval_scheduler_message_handle_ms": message_handle_ns / 1_000_000,
        "eval_scheduler_message_handle_feature_request_ms": message_handle_by_type_ns.get("feature_request", 0) / 1_000_000,
        "eval_scheduler_message_handle_episode_started_ms": message_handle_by_type_ns.get("episode_started", 0) / 1_000_000,
        "eval_scheduler_message_handle_episode_done_ms": message_handle_by_type_ns.get("episode_done", 0) / 1_000_000,
        "eval_scheduler_task_dispatch_ms": task_dispatch_ns / 1_000_000,
        "eval_scheduler_summary_extend_ms": summary_extend_ns / 1_000_000,
        "eval_scheduler_summary_sort_ms": summary_sort_ns / 1_000_000,
    }
    stats.update({f"eval_{key}": value for key, value in worker_profile_metrics(worker_stat_totals, worker_episode_count).items()})
    return summaries, stats


def run_eval_inference_batch(
    model: GoldRushPolicyNetwork,
    requests: list[EvalFeatureRequest],
    command_queues: list[Any],
    device: torch.device,
    feature_shared: FeatureSharedMemory,
    *,
    deterministic: bool,
) -> dict[str, int]:
    import numpy as np

    sync(device)
    start = time.perf_counter_ns()
    stack_start = time.perf_counter_ns()
    slots = [request.feature_slot for request in requests]
    spatial = torch.as_tensor(np.asarray(feature_shared.actor_planes[slots]), dtype=torch.float32, device=device)
    scalars = torch.as_tensor(np.asarray(feature_shared.actor_scalars[slots]), dtype=torch.float32, device=device)
    critic_spatial = torch.as_tensor(np.asarray(feature_shared.critic_planes[slots]), dtype=torch.float32, device=device)
    critic_scalars = torch.as_tensor(np.asarray(feature_shared.critic_scalars[slots]), dtype=torch.float32, device=device)
    stack_ns = time.perf_counter_ns() - stack_start
    model_sample_start = time.perf_counter_ns()
    with torch.no_grad():
        action = model.act(
            spatial,
            scalars,
            critic_spatial,
            critic_scalars,
            deterministic=deterministic,
            sample_uniforms=None if deterministic else _request_sample_uniforms(requests, device=device),
        )
        if not policy_action_is_finite(action):
            raise RuntimeError("parallel eval model produced NaN or Inf")
        actions_cpu = action.actions.detach().cpu().tolist()
        k_cpu = action.k.detach().cpu().tolist()
        order_cpu = action.order.detach().cpu().tolist()
        vp_cpu = action.vp.detach().cpu().tolist()
    sync(device)
    model_sample_ns = time.perf_counter_ns() - model_sample_start
    elapsed_ns = time.perf_counter_ns() - start
    action_send_start = time.perf_counter_ns()
    for batch_index, request in enumerate(requests):
        command_queues[request.worker_id].put(
            {
                "type": "action_result",
                "eval_id": request.eval_id,
                "request_id": request.request_id,
                "action": {
                    "actions": tuple(int(value) for value in actions_cpu[batch_index]),
                    "k": int(k_cpu[batch_index]),
                    "order": int(order_cpu[batch_index]),
                    "vp": int(vp_cpu[batch_index]),
                },
            }
        )
    action_send_ns = time.perf_counter_ns() - action_send_start
    return {
        "elapsed_ns": elapsed_ns,
        "batch_size": len(requests),
        "stack_ns": stack_ns,
        "model_sample_ns": model_sample_ns,
        "action_send_ns": action_send_ns,
        "total_with_action_send_ns": time.perf_counter_ns() - start,
    }


def _request_sample_uniforms(requests: list[EvalFeatureRequest], *, device: torch.device) -> torch.Tensor | None:
    if not all(request.policy_sample_key is not None for request in requests):
        return None
    values = [_request_uniform_row(request) for request in requests]
    return torch.tensor(values, dtype=torch.float32, device=device)


def _request_uniform_row(request: EvalFeatureRequest) -> list[float]:
    if request.policy_sample_key is None:
        return []
    seed = _request_seed(request)
    digest = hashlib.blake2b(str(seed).encode("utf-8"), digest_size=64).digest()
    return [
        (int.from_bytes(digest[index * 8 : (index + 1) * 8], byteorder="little", signed=False) + 0.5) / 2**64
        for index in range(8)
    ]


def _request_seed(request: EvalFeatureRequest) -> int:
    base_seed = 0 if request.policy_sample_seed is None else int(request.policy_sample_seed)
    payload = f"{request.policy_sample_key}|{request.round_index}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    offset = int.from_bytes(digest, byteorder="little", signed=False)
    return (base_seed + offset) % (2**63 - 1)
