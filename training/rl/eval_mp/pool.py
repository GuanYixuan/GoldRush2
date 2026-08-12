from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Sequence
from typing import Any

import torch

from simulator.errors import SimulatorRuleError
from training.models import GoldRushPolicyNetwork
from training.rl.env import SingleAgentEnvConfig
from training.rl.rollout_mp.shared_memory import FeatureSharedMemory, create_feature_shared_memory

from .scheduler import scheduler_loop
from .summary import summarize_eval
from .types import EvalEpisodeSummary, EvalTask, ParallelEvalConfig
from .worker import worker_loop


class ParallelEvalPool:
    def __init__(
        self,
        *,
        env_config: SingleAgentEnvConfig,
        mechanisms: Any = None,
        map_pool: Any = None,
        spawn: Any = None,
        reward_fn: Any = None,
        config: ParallelEvalConfig | None = None,
    ) -> None:
        self.env_config = env_config
        self.mechanisms = mechanisms
        self.map_pool = map_pool
        self.spawn = spawn
        self.reward_fn = reward_fn
        self.config = ParallelEvalConfig() if config is None else config
        _validate_config(self.config)
        self.ctx = mp.get_context("spawn")
        self.result_queue = self.ctx.Queue()
        self.command_queues = [self.ctx.Queue() for _ in range(self.config.num_workers)]
        self.feature_shared: FeatureSharedMemory | None = None
        self.processes: list[mp.Process] = []
        self.closed = False
        self.broken = False
        self.eval_count = 0
        self.startup_stats: dict[str, Any] = {}
        self._start_workers()

    def __enter__(self) -> ParallelEvalPool:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def evaluate(
        self,
        model: GoldRushPolicyNetwork,
        tasks: Sequence[EvalTask],
        *,
        device: torch.device | str | None = None,
    ) -> tuple[list[EvalEpisodeSummary], dict[str, Any]]:
        self._ensure_usable()
        if not tasks:
            raise SimulatorRuleError("parallel eval requires at least one task")
        assert self.feature_shared is not None
        eval_id = f"eval-{self.eval_count + 1:06d}"
        eval_device = torch.device(device) if device is not None else next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            start = time.perf_counter_ns()
            summaries, stats = scheduler_loop(
                model,
                self.command_queues,
                self.result_queue,
                tasks=tasks,
                device=eval_device,
                config=self.config,
                feature_shared=self.feature_shared,
                eval_id=eval_id,
            )
            scheduler_ns = time.perf_counter_ns() - start
            stats = self._finalize_stats(stats, scheduler_ns=scheduler_ns)
            stats.update({f"eval_{key}": value for key, value in summarize_eval(summaries).items()})
            self.eval_count += 1
            return summaries, stats
        except Exception:
            self.broken = True
            raise
        finally:
            if was_training:
                model.train()

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
        started: list[mp.Process] = []
        startup_start = time.perf_counter_ns()
        try:
            self.feature_shared = create_feature_shared_memory(self.config.num_workers)
            worker_config = {
                "env_config": self.env_config,
                "mechanisms": self.mechanisms,
                "map_pool": self.map_pool,
                "spawn": self.spawn,
                "reward_fn": self.reward_fn,
                "feature_shared_memory": self.feature_shared.config(),
            }
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
                started.append(process)
            worker_startup_ns = time.perf_counter_ns() - startup_start
            ready_count, first_ready_ns, all_ready_ns = self._wait_for_worker_ready(startup_start)
            self.startup_stats = {
                "eval_worker_ready_count": ready_count,
                "eval_worker_startup_ms": worker_startup_ns / 1_000_000,
                "eval_worker_first_ready_ms": None if first_ready_ns is None else first_ready_ns / 1_000_000,
                "eval_worker_all_ready_ms": None if all_ready_ns is None else all_ready_ns / 1_000_000,
            }
        except Exception:
            self.processes = started
            self.broken = True
            self.close()
            raise

    def _wait_for_worker_ready(self, startup_start: int) -> tuple[int, int | None, int | None]:
        ready = 0
        first_ready_ns: int | None = None
        all_ready_ns: int | None = None
        while ready < self.config.num_workers:
            msg = self.result_queue.get(timeout=self.config.worker_join_timeout_s)
            msg_type = msg["type"]
            if msg_type == "worker_ready":
                ready += 1
                now = time.perf_counter_ns()
                if first_ready_ns is None:
                    first_ready_ns = now - startup_start
                if ready == self.config.num_workers:
                    all_ready_ns = now - startup_start
            elif msg_type == "worker_error":
                raise RuntimeError(f"worker {msg['worker_id']} failed: {msg['error']}\n{msg['traceback']}")
            else:
                raise RuntimeError(f"unexpected eval worker startup message type: {msg_type!r}")
        return ready, first_ready_ns, all_ready_ns

    def _finalize_stats(self, stats: dict[str, Any], *, scheduler_ns: int) -> dict[str, Any]:
        if self.eval_count == 0:
            startup_stats = dict(self.startup_stats)
            worker_pool_reused = False
        else:
            startup_stats = {
                "eval_worker_ready_count": self.config.num_workers,
                "eval_worker_startup_ms": 0.0,
                "eval_worker_first_ready_ms": 0.0,
                "eval_worker_all_ready_ms": 0.0,
            }
            worker_pool_reused = True
        stats = {
            **stats,
            **startup_stats,
            "eval_worker_pool_reused": worker_pool_reused,
            "eval_scheduler_ms": scheduler_ns / 1_000_000,
        }
        accounted = (
            float(stats["eval_scheduler_queue_get_ms"])
            + float(stats["eval_inference_total_with_action_send_ms"])
            + float(stats["eval_scheduler_message_handle_ms"])
            + float(stats["eval_scheduler_task_dispatch_ms"])
            + float(stats["eval_scheduler_summary_sort_ms"])
        )
        stats["eval_scheduler_accounted_ms"] = accounted
        stats["eval_scheduler_unaccounted_ms"] = float(stats["eval_scheduler_ms"]) - accounted
        return stats

    def _ensure_usable(self) -> None:
        if self.closed:
            raise RuntimeError("parallel eval pool is closed")
        if self.broken:
            raise RuntimeError("parallel eval pool is broken after a previous failure")


def evaluate_parallel(
    model: GoldRushPolicyNetwork,
    tasks: Sequence[EvalTask],
    *,
    env_config: SingleAgentEnvConfig,
    mechanisms: Any = None,
    map_pool: Any = None,
    spawn: Any = None,
    reward_fn: Any = None,
    device: torch.device | str | None = None,
    config: ParallelEvalConfig | None = None,
) -> tuple[list[EvalEpisodeSummary], dict[str, Any]]:
    with ParallelEvalPool(
        env_config=env_config,
        mechanisms=mechanisms,
        map_pool=map_pool,
        spawn=spawn,
        reward_fn=reward_fn,
        config=config,
    ) as pool:
        return pool.evaluate(model, tasks, device=device)


def _validate_config(config: ParallelEvalConfig) -> None:
    if config.num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {config.num_workers}")
    if config.max_inference_batch_size <= 0:
        raise ValueError(f"max_inference_batch_size must be positive, got {config.max_inference_batch_size}")
    if config.inference_timeout_ms <= 0:
        raise ValueError(f"inference_timeout_ms must be positive, got {config.inference_timeout_ms}")
    if config.worker_join_timeout_s <= 0:
        raise ValueError(f"worker_join_timeout_s must be positive, got {config.worker_join_timeout_s}")
