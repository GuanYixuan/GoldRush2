from __future__ import annotations

import copy
import multiprocessing as mp
import queue
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any, Literal

import torch

from policy_runtime import FeatureExtractor
from simulator.errors import SimulatorRuleError
from simulator.types import GameOutput
from training.models import (
    GoldRushPolicyNetwork,
    policy_action_is_finite,
)
from training.opponents import OpponentSpec

from .env import SingleAgentGoldRushEnv
from .ppo_buffer import PpoBatch
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
    transition_slot: int


@dataclass(frozen=True)
class FeatureRequest:
    worker_id: int
    rollout_id: str
    task_id: str
    request_id: str
    round_index: int
    feature_slot: int


@dataclass
class FeatureSharedMemory:
    planes_shm: shared_memory.SharedMemory
    scalars_shm: shared_memory.SharedMemory
    planes: Any
    scalars: Any

    def config(self) -> dict[str, Any]:
        return {
            "planes": {
                "name": self.planes_shm.name,
                "shape": tuple(int(value) for value in self.planes.shape),
                "dtype": str(self.planes.dtype),
            },
            "scalars": {
                "name": self.scalars_shm.name,
                "shape": tuple(int(value) for value in self.scalars.shape),
                "dtype": str(self.scalars.dtype),
            },
        }

    def close(self) -> None:
        self.planes_shm.close()
        self.scalars_shm.close()

    def unlink(self) -> None:
        self.planes_shm.unlink()
        self.scalars_shm.unlink()


@dataclass
class TransitionSharedMemory:
    planes_shm: shared_memory.SharedMemory
    scalars_shm: shared_memory.SharedMemory
    actions_shm: shared_memory.SharedMemory
    k_shm: shared_memory.SharedMemory
    order_shm: shared_memory.SharedMemory
    vp_shm: shared_memory.SharedMemory
    old_logprob_shm: shared_memory.SharedMemory
    value_shm: shared_memory.SharedMemory
    reward_shm: shared_memory.SharedMemory
    done_shm: shared_memory.SharedMemory
    round_index_shm: shared_memory.SharedMemory
    planes: Any
    scalars: Any
    actions: Any
    k: Any
    order: Any
    vp: Any
    old_logprob: Any
    value: Any
    reward: Any
    done: Any
    round_index: Any

    def config(self) -> dict[str, Any]:
        return {
            "planes": _shared_array_config(self.planes_shm, self.planes),
            "scalars": _shared_array_config(self.scalars_shm, self.scalars),
            "actions": _shared_array_config(self.actions_shm, self.actions),
            "k": _shared_array_config(self.k_shm, self.k),
            "order": _shared_array_config(self.order_shm, self.order),
            "vp": _shared_array_config(self.vp_shm, self.vp),
            "old_logprob": _shared_array_config(self.old_logprob_shm, self.old_logprob),
            "value": _shared_array_config(self.value_shm, self.value),
            "reward": _shared_array_config(self.reward_shm, self.reward),
            "done": _shared_array_config(self.done_shm, self.done),
            "round_index": _shared_array_config(self.round_index_shm, self.round_index),
        }

    def close(self) -> None:
        for shm in self._shms():
            shm.close()

    def unlink(self) -> None:
        for shm in self._shms():
            shm.unlink()

    def _shms(self) -> tuple[shared_memory.SharedMemory, ...]:
        return (
            self.planes_shm,
            self.scalars_shm,
            self.actions_shm,
            self.k_shm,
            self.order_shm,
            self.vp_shm,
            self.old_logprob_shm,
            self.value_shm,
            self.reward_shm,
            self.done_shm,
            self.round_index_shm,
        )


class MultiprocessRolloutPool:
    def __init__(self, sampler: BatchRolloutSampler, config: MultiprocessRolloutConfig | None = None) -> None:
        self.sampler = sampler
        self.config = MultiprocessRolloutConfig() if config is None else config
        _validate_pool_config(self.config)
        self.ctx = mp.get_context("spawn")
        self.result_queue = self.ctx.Queue()
        self.command_queues = [self.ctx.Queue() for _ in range(self.config.num_workers)]
        self.feature_shared: FeatureSharedMemory | None = None
        self.processes: list[mp.Process] = []
        self.closed = False
        self.broken = False
        self.collect_count = 0
        self.startup_stats: dict[str, Any] = {}
        self._start_workers()

    def __enter__(self) -> MultiprocessRolloutPool:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def collect(
        self,
        model: GoldRushPolicyNetwork,
        *,
        pair_count: int,
        seed: int,
        map_ids: Sequence[int] | None = None,
        opponent_specs: Sequence[OpponentSpec] | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[PpoBatch, dict[str, Any]]:
        self._ensure_usable()
        _validate_inputs(pair_count=pair_count, map_ids=map_ids, opponent_specs=opponent_specs, config=self.config)
        assert self.feature_shared is not None

        rollout_id = f"rollout-{self.collect_count + 1:06d}"
        rollout_device = torch.device(device) if device is not None else next(model.parameters()).device
        was_training = model.training
        model.eval()
        transition_shared: TransitionSharedMemory | None = None
        released = False

        try:
            transition_shared = _create_transition_shared_memory(
                episode_count=pair_count * 2,
                round_count=self.sampler.env_config.episode.rules.round_count,
            )
            configure_start = time.perf_counter_ns()
            self._broadcast(
                {
                    "type": "configure_rollout",
                    "rollout_id": rollout_id,
                    "transition_shared_memory": transition_shared.config(),
                }
            )
            self._wait_for_worker_messages("configure_ready", rollout_id=rollout_id)
            configure_ms = (time.perf_counter_ns() - configure_start) / 1_000_000

            scheduler_start = time.perf_counter_ns()
            payloads, stats = _scheduler_loop(
                model,
                self.command_queues,
                self.result_queue,
                pair_count=pair_count,
                seed=seed,
                map_ids=map_ids,
                opponent_specs=opponent_specs,
                device=rollout_device,
                config=self.config,
                feature_shared=self.feature_shared,
                rollout_id=rollout_id,
            )
            scheduler_ns = time.perf_counter_ns() - scheduler_start
            batch_assembly_start = time.perf_counter_ns()
            batch = _episode_payloads_to_batch(payloads, transition_shared)
            batch_assembly_ns = time.perf_counter_ns() - batch_assembly_start
            release_ms = self._release_rollout(rollout_id)
            released = True
            stats = self._finalize_collect_stats(
                stats,
                scheduler_ns=scheduler_ns,
                batch_assembly_ns=batch_assembly_ns,
                configure_ms=configure_ms,
                release_ms=release_ms,
            )
            self.collect_count += 1
            return batch, stats
        except Exception:
            self.broken = True
            raise
        finally:
            if was_training:
                model.train()
            if transition_shared is not None:
                if not released and not self.closed:
                    try:
                        self._release_rollout(rollout_id)
                    except Exception:
                        self.broken = True
                transition_shared.close()
                transition_shared.unlink()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for command_queue in self.command_queues:
            command_queue.put({"type": "stop"})
        for process in self.processes:
            process.join(timeout=self.config.worker_join_timeout_s)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        if self.feature_shared is not None:
            self.feature_shared.close()
            self.feature_shared.unlink()
            self.feature_shared = None

    def _start_workers(self) -> None:
        started_processes: list[mp.Process] = []
        startup_start = time.perf_counter_ns()
        try:
            self.feature_shared = _create_feature_shared_memory(self.config.num_workers)
            worker_config = _worker_static_config(self.sampler, self.config, self.feature_shared)
            self.processes = [
                self.ctx.Process(
                    target=_worker_loop,
                    args=(worker_id, self.command_queues[worker_id], self.result_queue, worker_config),
                    daemon=True,
                )
                for worker_id in range(self.config.num_workers)
            ]
            for process in self.processes:
                process.start()
                started_processes.append(process)
            worker_startup_ns = time.perf_counter_ns() - startup_start
            ready_count, first_ready_ns, all_ready_ns = self._wait_for_worker_ready(startup_start)
            self.startup_stats = {
                "worker_ready_count": ready_count,
                "worker_startup_ms": worker_startup_ns / 1_000_000,
                "worker_first_ready_ms": None if first_ready_ns is None else first_ready_ns / 1_000_000,
                "worker_all_ready_ms": None if all_ready_ns is None else all_ready_ns / 1_000_000,
            }
        except Exception:
            self.processes = started_processes
            self.broken = True
            self.close()
            raise

    def _wait_for_worker_ready(self, startup_start: int) -> tuple[int, int | None, int | None]:
        ready_workers = 0
        first_ready_ns: int | None = None
        all_ready_ns: int | None = None
        while ready_workers < self.config.num_workers:
            msg = self.result_queue.get(timeout=self.config.worker_join_timeout_s)
            msg_type = msg["type"]
            if msg_type == "worker_ready":
                ready_workers += 1
                now_ns = time.perf_counter_ns()
                if first_ready_ns is None:
                    first_ready_ns = now_ns - startup_start
                if ready_workers == self.config.num_workers:
                    all_ready_ns = now_ns - startup_start
            elif msg_type == "worker_error":
                raise RuntimeError(f"worker {msg['worker_id']} failed: {msg['error']}\n{msg['traceback']}")
            else:
                raise RuntimeError(f"unexpected worker startup message type: {msg_type!r}")
        return ready_workers, first_ready_ns, all_ready_ns

    def _broadcast(self, msg: dict[str, Any]) -> None:
        for command_queue in self.command_queues:
            command_queue.put(msg)

    def _wait_for_worker_messages(self, expected_type: str, *, rollout_id: str) -> None:
        ready_workers = 0
        while ready_workers < self.config.num_workers:
            msg = self.result_queue.get(timeout=self.config.worker_join_timeout_s)
            msg_type = msg["type"]
            if msg_type == expected_type and msg.get("rollout_id") == rollout_id:
                ready_workers += 1
            elif msg_type == "worker_error":
                raise RuntimeError(f"worker {msg['worker_id']} failed: {msg['error']}\n{msg['traceback']}")
            else:
                raise RuntimeError(f"unexpected worker message while waiting for {expected_type}: {msg}")

    def _release_rollout(self, rollout_id: str) -> float:
        release_start = time.perf_counter_ns()
        self._broadcast({"type": "release_rollout", "rollout_id": rollout_id})
        self._wait_for_worker_messages("release_ready", rollout_id=rollout_id)
        return (time.perf_counter_ns() - release_start) / 1_000_000

    def _finalize_collect_stats(
        self,
        stats: dict[str, Any],
        *,
        scheduler_ns: int,
        batch_assembly_ns: int,
        configure_ms: float,
        release_ms: float,
    ) -> dict[str, Any]:
        if self.collect_count == 0:
            startup_stats = dict(self.startup_stats)
            worker_pool_reused = False
        else:
            startup_stats = {
                "worker_ready_count": self.config.num_workers,
                "worker_startup_ms": 0.0,
                "worker_first_ready_ms": 0.0,
                "worker_all_ready_ms": 0.0,
            }
            worker_pool_reused = True
        stats = {
            **stats,
            **startup_stats,
            "worker_pool_reused": worker_pool_reused,
            "worker_configure_ms": configure_ms,
            "worker_release_ms": release_ms,
            "scheduler_ms": scheduler_ns / 1_000_000,
            "batch_assembly_ms": batch_assembly_ns / 1_000_000,
        }
        scheduler_accounted_ms = (
            float(stats["scheduler_queue_get_ms"])
            + float(stats["inference_total_with_action_send_ms"])
            + float(stats["scheduler_message_handle_ms"])
            + float(stats["scheduler_task_dispatch_ms"])
            + float(stats["scheduler_payload_sort_ms"])
        )
        stats["scheduler_accounted_ms"] = scheduler_accounted_ms
        stats["scheduler_unaccounted_ms"] = float(stats["scheduler_ms"]) - scheduler_accounted_ms
        return stats

    def _ensure_usable(self) -> None:
        if self.closed:
            raise RuntimeError("multiprocess rollout pool is closed")
        if self.broken:
            raise RuntimeError("multiprocess rollout pool is broken after a previous failure")


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
    with MultiprocessRolloutPool(sampler, config) as pool:
        return pool.collect(
            model,
            pair_count=pair_count,
            seed=seed,
            map_ids=map_ids,
            opponent_specs=opponent_specs,
            device=device,
        )


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
    feature_shared: FeatureSharedMemory,
    rollout_id: str,
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
            inference_stats = _run_inference_batch(model, pending_requests, command_queues, device, feature_shared)
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
        inference_stats = _run_inference_batch(model, pending_requests, command_queues, device, feature_shared)
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
        "scheduler_message_handle_feature_request_ms": message_handle_by_type_ns.get("feature_request", 0) / 1_000_000,
        "scheduler_message_handle_episode_started_ms": message_handle_by_type_ns.get("episode_started", 0) / 1_000_000,
        "scheduler_message_handle_episode_done_ms": message_handle_by_type_ns.get("episode_done", 0) / 1_000_000,
        "scheduler_task_dispatch_ms": task_dispatch_ns / 1_000_000,
        "scheduler_payload_extend_ms": payload_extend_ns / 1_000_000,
        "scheduler_payload_sort_ms": payload_sort_ns / 1_000_000,
    }
    stats.update(_worker_profile_metrics(worker_stat_totals, worker_episode_count))
    return payloads, stats


def _run_inference_batch(
    model: GoldRushPolicyNetwork,
    requests: list[FeatureRequest],
    command_queues: list[Any],
    device: torch.device,
    feature_shared: FeatureSharedMemory,
) -> dict[str, int]:
    import numpy as np

    _sync(device)
    start = time.perf_counter_ns()
    stack_start = time.perf_counter_ns()
    feature_slots = [request.feature_slot for request in requests]
    spatial = torch.as_tensor(np.asarray(feature_shared.planes[feature_slots]), dtype=torch.float32, device=device)
    scalars = torch.as_tensor(np.asarray(feature_shared.scalars[feature_slots]), dtype=torch.float32, device=device)
    stack_ns = time.perf_counter_ns() - stack_start
    model_sample_start = time.perf_counter_ns()
    with torch.no_grad():
        action = model.act(spatial, scalars)
        if not policy_action_is_finite(action):
            raise SimulatorRuleError("multiprocess rollout model produced NaN or Inf")
    _sync(device)
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


def _create_feature_shared_memory(num_workers: int) -> FeatureSharedMemory:
    import numpy as np

    planes_shape = (num_workers, 38, 17, 17)
    scalars_shape = (num_workers, 10)
    planes_nbytes = int(np.prod(planes_shape)) * np.dtype(np.float32).itemsize
    scalars_nbytes = int(np.prod(scalars_shape)) * np.dtype(np.float32).itemsize
    planes_shm = shared_memory.SharedMemory(create=True, size=planes_nbytes)
    try:
        scalars_shm = shared_memory.SharedMemory(create=True, size=scalars_nbytes)
    except Exception:
        planes_shm.close()
        planes_shm.unlink()
        raise
    return FeatureSharedMemory(
        planes_shm=planes_shm,
        scalars_shm=scalars_shm,
        planes=np.ndarray(planes_shape, dtype=np.float32, buffer=planes_shm.buf),
        scalars=np.ndarray(scalars_shape, dtype=np.float32, buffer=scalars_shm.buf),
    )


def _attach_feature_shared_memory(config: dict[str, Any]) -> FeatureSharedMemory:
    import numpy as np

    planes = config["planes"]
    scalars = config["scalars"]
    planes_shm = shared_memory.SharedMemory(name=str(planes["name"]))
    try:
        scalars_shm = shared_memory.SharedMemory(name=str(scalars["name"]))
    except Exception:
        planes_shm.close()
        raise
    planes_shape = tuple(int(value) for value in planes["shape"])
    scalars_shape = tuple(int(value) for value in scalars["shape"])
    return FeatureSharedMemory(
        planes_shm=planes_shm,
        scalars_shm=scalars_shm,
        planes=np.ndarray(planes_shape, dtype=np.dtype(str(planes["dtype"])), buffer=planes_shm.buf),
        scalars=np.ndarray(scalars_shape, dtype=np.dtype(str(scalars["dtype"])), buffer=scalars_shm.buf),
    )


def _create_transition_shared_memory(*, episode_count: int, round_count: int) -> TransitionSharedMemory:
    import numpy as np

    created: list[shared_memory.SharedMemory] = []

    def create_array(shape: tuple[int, ...], dtype: Any) -> tuple[shared_memory.SharedMemory, Any]:
        np_dtype = np.dtype(dtype)
        shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * np_dtype.itemsize)
        created.append(shm)
        return shm, np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)

    try:
        planes_shm, planes = create_array((episode_count, round_count, 38, 17, 17), np.float32)
        scalars_shm, scalars = create_array((episode_count, round_count, 10), np.float32)
        actions_shm, actions = create_array((episode_count, round_count, 6), np.int64)
        k_shm, k = create_array((episode_count, round_count), np.int64)
        order_shm, order = create_array((episode_count, round_count), np.int64)
        vp_shm, vp = create_array((episode_count, round_count), np.int64)
        old_logprob_shm, old_logprob = create_array((episode_count, round_count), np.float32)
        value_shm, value = create_array((episode_count, round_count), np.float32)
        reward_shm, reward = create_array((episode_count, round_count), np.float32)
        done_shm, done = create_array((episode_count, round_count), np.bool_)
        round_index_shm, round_index = create_array((episode_count, round_count), np.int64)
    except Exception:
        for shm in created:
            shm.close()
            shm.unlink()
        raise

    return TransitionSharedMemory(
        planes_shm=planes_shm,
        scalars_shm=scalars_shm,
        actions_shm=actions_shm,
        k_shm=k_shm,
        order_shm=order_shm,
        vp_shm=vp_shm,
        old_logprob_shm=old_logprob_shm,
        value_shm=value_shm,
        reward_shm=reward_shm,
        done_shm=done_shm,
        round_index_shm=round_index_shm,
        planes=planes,
        scalars=scalars,
        actions=actions,
        k=k,
        order=order,
        vp=vp,
        old_logprob=old_logprob,
        value=value,
        reward=reward,
        done=done,
        round_index=round_index,
    )


def _attach_transition_shared_memory(config: dict[str, Any]) -> TransitionSharedMemory:
    attached: list[shared_memory.SharedMemory] = []

    def attach_array(key: str) -> tuple[shared_memory.SharedMemory, Any]:
        import numpy as np

        spec = config[key]
        shm = shared_memory.SharedMemory(name=str(spec["name"]))
        attached.append(shm)
        shape = tuple(int(value) for value in spec["shape"])
        array = np.ndarray(shape, dtype=np.dtype(str(spec["dtype"])), buffer=shm.buf)
        return shm, array

    try:
        planes_shm, planes = attach_array("planes")
        scalars_shm, scalars = attach_array("scalars")
        actions_shm, actions = attach_array("actions")
        k_shm, k = attach_array("k")
        order_shm, order = attach_array("order")
        vp_shm, vp = attach_array("vp")
        old_logprob_shm, old_logprob = attach_array("old_logprob")
        value_shm, value = attach_array("value")
        reward_shm, reward = attach_array("reward")
        done_shm, done = attach_array("done")
        round_index_shm, round_index = attach_array("round_index")
    except Exception:
        for shm in attached:
            shm.close()
        raise

    return TransitionSharedMemory(
        planes_shm=planes_shm,
        scalars_shm=scalars_shm,
        actions_shm=actions_shm,
        k_shm=k_shm,
        order_shm=order_shm,
        vp_shm=vp_shm,
        old_logprob_shm=old_logprob_shm,
        value_shm=value_shm,
        reward_shm=reward_shm,
        done_shm=done_shm,
        round_index_shm=round_index_shm,
        planes=planes,
        scalars=scalars,
        actions=actions,
        k=k,
        order=order,
        vp=vp,
        old_logprob=old_logprob,
        value=value,
        reward=reward,
        done=done,
        round_index=round_index,
    )


def _shared_array_config(shm: shared_memory.SharedMemory, array: Any) -> dict[str, Any]:
    return {
        "name": shm.name,
        "shape": tuple(int(value) for value in array.shape),
        "dtype": str(array.dtype),
    }


def _worker_loop(worker_id: int, command_queue: Any, result_queue: Any, static_config: dict[str, Any]) -> None:
    feature_shared: FeatureSharedMemory | None = None
    transition_shared: TransitionSharedMemory | None = None
    current_rollout_id: str | None = None
    try:
        feature_shared = _attach_feature_shared_memory(static_config["feature_shared_memory"])
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
                transition_shared = _attach_transition_shared_memory(msg["transition_shared_memory"])
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
            _run_worker_episode(
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


def _run_worker_episode(
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
    request_index = 0
    transition_slot = int(task.transition_slot)
    max_episode_length = int(transition_shared.done.shape[1])

    while observation is not None:
        if request_index >= max_episode_length:
            raise SimulatorRuleError(
                f"episode {task.task_id!r} exceeded transition shared memory length {max_episode_length}"
            )
        features = extractor.observe(observation)
        if features["feature_schema"] != "goldrush2_feature_v1":
            raise SimulatorRuleError(f"unexpected feature schema: {features['feature_schema']!r}")
        request_id = f"{task.task_id}-round-{request_index:04d}"
        spatial_planes = np.asarray(features["planes"], dtype=np.float32)
        scalars = np.asarray(features["scalars"], dtype=np.float32)
        feature_shared.planes[worker_id, ...] = spatial_planes
        feature_shared.scalars[worker_id, ...] = scalars
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
        transition_shared.planes[transition_slot, request_index, ...] = spatial_planes
        transition_shared.scalars[transition_slot, request_index, ...] = scalars
        transition_shared.actions[transition_slot, request_index, :] = tuple(int(value) for value in game_output.actions)
        transition_shared.k[transition_slot, request_index] = int(game_output.k)
        transition_shared.order[transition_slot, request_index] = int(game_output.order)
        transition_shared.vp[transition_slot, request_index] = int(game_output.vp)
        transition_shared.old_logprob[transition_slot, request_index] = float(action_msg["old_logprob"])
        transition_shared.value[transition_slot, request_index] = float(action_msg["value"])
        transition_shared.reward[transition_slot, request_index] = float(step.reward)
        transition_shared.done[transition_slot, request_index] = done
        transition_shared.round_index[transition_slot, request_index] = int(observation.round)
        # 已测 w64/p256：feature/extractor/action decode/info/SHM writes 通常 <=0.12ms/transition；
        # env.step 内部几乎全在 RoundStepEnv.step，opponent/reward/info 均 <0.04ms/transition。
        info_items.append(_transition_info(step.info, done=done, mode=str(static_config["transition_info_mode"])))
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
                transition_slot=pair_index * 2,
            )
        )
    return tasks


def _episode_payloads_to_batch(payloads: list[dict[str, Any]], transition_shared: TransitionSharedMemory) -> PpoBatch:
    import numpy as np

    if not payloads:
        raise SimulatorRuleError("multiprocess rollout produced no episode payloads")

    spatial_planes: list[Any] = []
    scalars: list[Any] = []
    actions: list[Any] = []
    k: list[Any] = []
    order: list[Any] = []
    vp: list[Any] = []
    old_logprob: list[Any] = []
    values: list[Any] = []
    rewards: list[Any] = []
    dones: list[Any] = []
    round_indices: list[Any] = []
    episode_ids: list[str] = []
    map_ids: list[int] = []
    agent_player_ids: list[int] = []
    infos: list[dict[str, Any]] = []

    for payload in payloads:
        transition_slot = int(payload["transition_slot"])
        episode_length = int(payload["episode_length"])
        if episode_length <= 0:
            raise SimulatorRuleError(f"episode payload {payload['task_id']!r} has no transitions")
        if not 0 <= transition_slot < int(transition_shared.done.shape[0]):
            raise SimulatorRuleError(f"transition slot out of range for {payload['task_id']!r}: {transition_slot}")
        if episode_length > int(transition_shared.done.shape[1]):
            raise SimulatorRuleError(
                f"episode payload {payload['task_id']!r} length {episode_length} exceeds shared transition length "
                f"{int(transition_shared.done.shape[1])}"
            )
        spatial_planes.append(transition_shared.planes[transition_slot, :episode_length])
        scalars.append(transition_shared.scalars[transition_slot, :episode_length])
        actions.append(transition_shared.actions[transition_slot, :episode_length])
        k.append(transition_shared.k[transition_slot, :episode_length])
        order.append(transition_shared.order[transition_slot, :episode_length])
        vp.append(transition_shared.vp[transition_slot, :episode_length])
        old_logprob.append(transition_shared.old_logprob[transition_slot, :episode_length])
        values.append(transition_shared.value[transition_slot, :episode_length])
        rewards.append(transition_shared.reward[transition_slot, :episode_length])
        dones.append(transition_shared.done[transition_slot, :episode_length])
        round_indices.append(transition_shared.round_index[transition_slot, :episode_length])
        episode_ids.extend((f"{payload['pair_id']}-{payload['pair_role']}",) * episode_length)
        map_ids.extend((int(payload["map_id"]),) * episode_length)
        agent_player_ids.extend((int(payload["agent_player_id"]),) * episode_length)
        episode_infos = tuple(payload["infos"])
        if len(episode_infos) != episode_length:
            raise SimulatorRuleError(
                f"episode payload {payload['task_id']!r} infos length {len(episode_infos)} != {episode_length}"
            )
        infos.extend(episode_infos)

    return PpoBatch.from_arrays(
        spatial_planes=np.concatenate(spatial_planes, axis=0),
        scalars=np.concatenate(scalars, axis=0),
        actions=np.concatenate(actions, axis=0),
        k=np.concatenate(k, axis=0),
        order=np.concatenate(order, axis=0),
        vp=np.concatenate(vp, axis=0),
        old_logprob=np.concatenate(old_logprob, axis=0),
        values=np.concatenate(values, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        dones=np.concatenate(dones, axis=0),
        episode_ids=tuple(episode_ids),
        round_indices=np.concatenate(round_indices, axis=0),
        map_ids=tuple(map_ids),
        agent_player_ids=np.asarray(agent_player_ids, dtype=np.int64),
        infos=tuple(infos),
    )


def _worker_static_config(
    sampler: BatchRolloutSampler,
    config: MultiprocessRolloutConfig,
    feature_shared: FeatureSharedMemory,
) -> dict[str, Any]:
    return {
        "env_config": sampler.env_config,
        "mechanisms": sampler.mechanisms,
        "map_pool": sampler.map_pool,
        "spawn": sampler.spawn,
        "reward_fn": sampler.reward_fn,
        "transition_info_mode": config.transition_info_mode,
        "feature_shared_memory": feature_shared.config(),
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


def _validate_pool_config(config: MultiprocessRolloutConfig) -> None:
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


def _worker_profile_metrics(totals: dict[str, int], episode_count: int) -> dict[str, float | int]:
    if not totals:
        return {
            "worker_profile_episode_count": 0,
            "worker_profile_transition_count": 0,
        }
    transition_count = int(totals.get("steps", 0))
    metrics: dict[str, float | int] = {
        "worker_profile_episode_count": episode_count,
        "worker_profile_transition_count": transition_count,
    }
    for key, value in sorted(totals.items()):
        if key == "steps":
            metrics["worker_sum_steps"] = int(value)
            metrics["worker_mean_steps_per_episode"] = int(value) / max(episode_count, 1)
            continue
        if not key.endswith("_ns"):
            continue
        name = key[:-3]
        total_ms = int(value) / 1_000_000
        metrics[f"worker_sum_{name}_ms"] = total_ms
        metrics[f"worker_mean_episode_{name}_ms"] = total_ms / max(episode_count, 1)
        metrics[f"worker_per_transition_{name}_ms"] = total_ms / max(transition_count, 1)
    return metrics


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


__all__ = ["MultiprocessRolloutConfig", "MultiprocessRolloutPool", "collect_multiprocess_ppo_rollouts"]
