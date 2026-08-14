from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import torch

from simulator.errors import SimulatorRuleError
from training.models import GoldRushPolicyNetwork
from training.rl.env import SingleAgentEnvConfig
from training.rl.rollout_mp.shared_memory import FeatureSharedMemory, create_feature_shared_memory

from .scheduler import scheduler_loop
from .summary import summarize_eval
from .types import EvalEpisodeSummary, EvalTask, PairedEvalResult, ParallelEvalConfig
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
                "enable_fast_runtime_features": bool(self.config.enable_fast_runtime_features),
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


def evaluate_fast_runtime_crn_pair(
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
    policy_sample_seed: int = 0,
    policy_sample_key_prefix: str = "fast_crn",
) -> PairedEvalResult:
    paired_tasks = _tasks_with_crn(
        tasks,
        policy_sample_seed=policy_sample_seed,
        policy_sample_key_prefix=policy_sample_key_prefix,
    )
    base_config = ParallelEvalConfig() if config is None else config
    off_config = replace(base_config, enable_fast_runtime_features=False)
    on_config = replace(base_config, enable_fast_runtime_features=True)
    off_summaries, off_stats = evaluate_parallel(
        model,
        paired_tasks,
        env_config=env_config,
        mechanisms=mechanisms,
        map_pool=map_pool,
        spawn=spawn,
        reward_fn=reward_fn,
        device=device,
        config=off_config,
    )
    on_summaries, on_stats = evaluate_parallel(
        model,
        paired_tasks,
        env_config=env_config,
        mechanisms=mechanisms,
        map_pool=map_pool,
        spawn=spawn,
        reward_fn=reward_fn,
        device=device,
        config=on_config,
    )
    return PairedEvalResult(
        fast_off_summaries=tuple(off_summaries),
        fast_on_summaries=tuple(on_summaries),
        fast_off_stats=off_stats,
        fast_on_stats=on_stats,
        paired_stats=_paired_delta_stats(off_summaries, on_summaries),
    )


def _validate_config(config: ParallelEvalConfig) -> None:
    if config.num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {config.num_workers}")
    if config.max_inference_batch_size <= 0:
        raise ValueError(f"max_inference_batch_size must be positive, got {config.max_inference_batch_size}")
    if config.inference_timeout_ms <= 0:
        raise ValueError(f"inference_timeout_ms must be positive, got {config.inference_timeout_ms}")
    if config.worker_join_timeout_s <= 0:
        raise ValueError(f"worker_join_timeout_s must be positive, got {config.worker_join_timeout_s}")


def _tasks_with_crn(
    tasks: Sequence[EvalTask],
    *,
    policy_sample_seed: int,
    policy_sample_key_prefix: str,
) -> tuple[EvalTask, ...]:
    if not tasks:
        raise SimulatorRuleError("paired parallel eval requires at least one task")
    seen: set[str] = set()
    paired: list[EvalTask] = []
    for task in tasks:
        if task.task_id in seen:
            raise SimulatorRuleError(f"paired parallel eval requires unique task_id, got duplicate {task.task_id!r}")
        seen.add(task.task_id)
        paired.append(
            replace(
                task,
                policy_sample_key=task.policy_sample_key or f"{policy_sample_key_prefix}:{task.task_id}",
                policy_sample_seed=task.policy_sample_seed if task.policy_sample_seed is not None else int(policy_sample_seed),
            )
        )
    return tuple(paired)


def _paired_delta_stats(
    off_summaries: Sequence[EvalEpisodeSummary],
    on_summaries: Sequence[EvalEpisodeSummary],
) -> dict[str, float | int]:
    off_by_id = {summary.task_id: summary for summary in off_summaries}
    on_by_id = {summary.task_id: summary for summary in on_summaries}
    if set(off_by_id) != set(on_by_id):
        raise SimulatorRuleError("paired eval summary task ids differ between fast off and fast on")
    deltas = [_episode_delta(off_by_id[task_id], on_by_id[task_id]) for task_id in sorted(off_by_id)]
    if not deltas:
        return {"paired_episode_count": 0}
    return {
        "paired_episode_count": len(deltas),
        "paired_mean_agent_net_gold_delta": _mean_float(item["agent_net_gold_delta"] for item in deltas),
        "paired_mean_margin_delta": _mean_float(item["margin_delta"] for item in deltas),
        "paired_win_rate_delta": _mean_float(item["agent_won_delta"] for item in deltas),
        "paired_mean_agent_pickup_gold_delta": _mean_float(item["agent_pickup_gold_delta"] for item in deltas),
        "paired_mean_agent_bomb_lost_gold_delta": _mean_float(item["agent_bomb_lost_gold_delta"] for item in deltas),
        "paired_mean_agent_trample_penalty_delta": _mean_float(item["agent_trample_penalty_delta"] for item in deltas),
        "paired_mean_agent_vision_spent_delta": _mean_float(item["agent_vision_spent_delta"] for item in deltas),
        "paired_mean_episode_length_delta": _mean_float(item["episode_length_delta"] for item in deltas),
    }


def _episode_delta(off: EvalEpisodeSummary, on: EvalEpisodeSummary) -> dict[str, float]:
    return {
        "agent_net_gold_delta": float(on.agent_net_gold - off.agent_net_gold),
        "margin_delta": float(on.margin - off.margin),
        "agent_won_delta": (1.0 if on.agent_won else 0.0) - (1.0 if off.agent_won else 0.0),
        "agent_pickup_gold_delta": float(on.agent_pickup_gold - off.agent_pickup_gold),
        "agent_bomb_lost_gold_delta": float(on.agent_bomb_lost_gold - off.agent_bomb_lost_gold),
        "agent_trample_penalty_delta": float(on.agent_trample_penalty - off.agent_trample_penalty),
        "agent_vision_spent_delta": float(on.agent_vision_spent - off.agent_vision_spent),
        "episode_length_delta": float(on.episode_length - off.episode_length),
    }


def _mean_float(values: Any) -> float:
    items = [float(value) for value in values]
    return float(sum(items) / len(items)) if items else 0.0
