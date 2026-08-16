from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Sequence
from typing import Any

import torch

from simulator.errors import SimulatorRuleError
from training.models import GoldRushPolicyNetwork
from training.opponents import OpponentSpec
from training.rl.ppo_buffer import PpoBatch
from training.rl.sampler import BatchRolloutSampler

from .batch_assembly import episode_payloads_to_batch
from .scheduler import scheduler_loop
from .shared_memory import FeatureSharedMemory, TransitionSharedMemory, create_feature_shared_memory, create_transition_shared_memory
from .types import MultiprocessRolloutConfig
from .worker import worker_loop


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
            transition_shared = create_transition_shared_memory(
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
            payloads, stats = scheduler_loop(
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
                map_pool=self.sampler.map_pool,
            )
            scheduler_ns = time.perf_counter_ns() - scheduler_start
            batch_assembly_start = time.perf_counter_ns()
            batch = episode_payloads_to_batch(payloads, transition_shared)
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
            self.feature_shared = create_feature_shared_memory(self.config.num_workers)
            worker_config = _worker_static_config(self.sampler, self.config, self.feature_shared)
            self.processes = [
                self.ctx.Process(
                    target=worker_loop,
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
        "enable_fast_runtime_features": bool(config.enable_fast_runtime_features),
        "reward_fold_gamma": float(config.reward_fold_gamma),
        "feature_shared_memory": feature_shared.config(),
        "round_count": sampler.env_config.episode.rules.round_count,
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
    if config.num_workers < 1:
        raise SimulatorRuleError(f"num_workers must be positive, got {config.num_workers}")
    if pair_count * 2 < config.num_workers:
        # Valid but wasteful; keep this explicit in stats rather than failing.
        return


def _validate_pool_config(config: MultiprocessRolloutConfig) -> None:
    if config.num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {config.num_workers}")
    if config.max_inference_batch_size <= 0:
        raise ValueError(f"max_inference_batch_size must be positive, got {config.max_inference_batch_size}")
    if config.inference_timeout_ms <= 0:
        raise ValueError(f"inference_timeout_ms must be positive, got {config.inference_timeout_ms}")
    if config.worker_join_timeout_s <= 0:
        raise ValueError(f"worker_join_timeout_s must be positive, got {config.worker_join_timeout_s}")
    if config.transition_info_mode not in ("training", "debug"):
        raise ValueError(f"unknown transition_info_mode: {config.transition_info_mode!r}")
    if not 0.0 <= config.reward_fold_gamma <= 1.0:
        raise ValueError(f"reward_fold_gamma must be in [0, 1], got {config.reward_fold_gamma}")
