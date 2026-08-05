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
    policy_action_to_game_output,
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
    worker_config = _worker_static_config(sampler)
    processes = [
        ctx.Process(
            target=_worker_loop,
            args=(worker_id, command_queues[worker_id], result_queue, worker_config),
            daemon=True,
        )
        for worker_id in range(config.num_workers)
    ]
    for process in processes:
        process.start()

    try:
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
        )
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

    transitions = [_payload_to_transition(payload) for payload in payloads]
    return PpoBatch.from_transitions(transitions), stats


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

    while completed_episodes < expected_episodes:
        while idle_workers and pending_tasks:
            worker_id = idle_workers.pop(0)
            task = pending_tasks.pop(0)
            active_tasks[task.task_id] = task
            command_queues[worker_id].put({"type": "start_episode", "task": task})

        try:
            msg = result_queue.get(timeout=timeout_s)
        except queue.Empty:
            msg = None

        if msg is not None:
            msg_type = msg["type"]
            if msg_type == "feature_request":
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
                payloads.extend(msg["transitions"])
                if task.pair_role == "first":
                    first_done += 1
                else:
                    second_done += 1
            elif msg_type == "worker_error":
                raise RuntimeError(f"worker {msg['worker_id']} failed: {msg['error']}\n{msg['traceback']}")
            else:
                raise RuntimeError(f"unexpected worker message type: {msg_type!r}")

        if pending_requests and (len(pending_requests) >= config.max_inference_batch_size or msg is None):
            elapsed_ns, batch_size = _run_inference_batch(model, pending_requests, command_queues, device)
            inference_ns += elapsed_ns
            feature_batches += 1
            feature_batch_sizes.append(batch_size)
            pending_requests.clear()

    if pending_requests:
        elapsed_ns, batch_size = _run_inference_batch(model, pending_requests, command_queues, device)
        inference_ns += elapsed_ns
        feature_batches += 1
        feature_batch_sizes.append(batch_size)

    payloads.sort(key=lambda item: (item["pair_id"], 0 if item["pair_role"] == "first" else 1, item["round_index"]))
    return payloads, {
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
    }


def _run_inference_batch(
    model: GoldRushPolicyNetwork,
    requests: list[FeatureRequest],
    command_queues: list[Any],
    device: torch.device,
) -> tuple[int, int]:
    import numpy as np

    _sync(device)
    start = time.perf_counter_ns()
    spatial = torch.as_tensor(np.stack([request.spatial_planes for request in requests]), dtype=torch.float32, device=device)
    scalars = torch.as_tensor(np.stack([request.scalars for request in requests]), dtype=torch.float32, device=device)
    with torch.no_grad():
        output = model(spatial, scalars)
        if not policy_output_is_finite(output):
            raise SimulatorRuleError("multiprocess rollout model produced NaN or Inf")
        action = sample_action(output)
    _sync(device)
    elapsed_ns = time.perf_counter_ns() - start
    for batch_index, request in enumerate(requests):
        game_output = policy_action_to_game_output(action, batch_index=batch_index)
        command_queues[request.worker_id].put(
            {
                "type": "action_result",
                "request_id": request.request_id,
                "action": {
                    "actions": tuple(int(value) for value in game_output.actions),
                    "k": int(game_output.k),
                    "order": int(game_output.order),
                    "vp": int(game_output.vp),
                },
                "old_logprob": float(action.logprob[batch_index].detach().cpu().item()),
                "value": float(action.value[batch_index].detach().cpu().item()),
            }
        )
    return elapsed_ns, len(requests)


def _worker_loop(worker_id: int, command_queue: Any, result_queue: Any, static_config: dict[str, Any]) -> None:
    try:
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
                "done": bool(step.terminated),
                "episode_id": f"{task.pair_id}-{task.pair_role}",
                "pair_id": task.pair_id,
                "pair_role": task.pair_role,
                "round_index": int(observation.round),
                "map_id": int(reset.info["map_id"]),
                "agent_player_id": int(task.agent_player_id),
                "info": step.info,
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


def _worker_static_config(sampler: BatchRolloutSampler) -> dict[str, Any]:
    return {
        "env_config": sampler.env_config,
        "mechanisms": sampler.mechanisms,
        "map_pool": sampler.map_pool,
        "spawn": sampler.spawn,
        "reward_fn": sampler.reward_fn,
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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


__all__ = ["MultiprocessRolloutConfig", "collect_multiprocess_ppo_rollouts"]
