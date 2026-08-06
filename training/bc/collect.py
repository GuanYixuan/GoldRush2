from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from policy_runtime import FeatureExtractor
from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepConfig, RoundStepEnv, RoundStepMechanisms, RoundStepResult
from simulator.errors import SimulatorRuleError
from simulator.mechanisms.maps import SpawnConfig
from simulator.types import ActorKind, GameOutput, MoveStatus, MovementEvent, Position
from training.opponents import EpisodeContext

from .schema import (
    BC_DATASET_SCHEMA,
    BC_SHARD_SCHEMA,
    FEATURE_SCHEMA,
    BcCollectionConfig,
    BcDatasetManifest,
    BcDatasetSplit,
    BcShardSummary,
    BcTask,
    write_json,
)
from .teachers import build_teacher


@dataclass(frozen=True)
class BcCollectionResult:
    manifest: BcDatasetManifest
    output_dir: Path


def build_tasks(config: BcCollectionConfig) -> tuple[BcTask, ...]:
    _validate_collection_config(config)
    tasks: list[BcTask] = []
    task_id = 0
    for split, seeds in (
        ("train", config.train_seeds),
        ("val", config.val_seeds),
        ("holdout", config.holdout_seeds),
    ):
        for seed in seeds:
            for map_id in config.map_ids:
                tasks.append(BcTask(task_id=task_id, split=split, paired_seed=int(seed), map_id=int(map_id)))
                task_id += 1
    return tuple(tasks)


def collect_dataset(output_dir: Path, config: BcCollectionConfig, *, resume: bool = False) -> BcCollectionResult:
    output_dir = Path(output_dir)
    tasks = build_tasks(config)
    _prepare_output_dir(output_dir, config, tasks, resume=resume)
    chunks = _chunk_tasks(tasks, config)
    max_workers = min(config.num_workers, len(chunks)) if chunks else 1

    summaries: list[BcShardSummary] = []
    if max_workers <= 1:
        for worker_index, chunk in enumerate(chunks):
            summary = _collect_task_chunk(output_dir, config, chunk, worker_index=worker_index)
            summaries.append(summary)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_collect_task_chunk, output_dir, config, chunk, worker_index=index): index
                for index, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                summaries.append(future.result())

    summaries = sorted(summaries, key=lambda item: min(item.task_ids) if item.task_ids else -1)
    manifest = BcDatasetManifest(
        schema=BC_DATASET_SCHEMA,
        feature_schema=FEATURE_SCHEMA,
        config_hash=config.config_hash,
        config=config.to_jsonable(),
        shards=tuple(summaries),
    )
    write_json(output_dir / "manifest.json", manifest.to_jsonable())
    return BcCollectionResult(manifest=manifest, output_dir=output_dir)


def collect_task(task: BcTask, config: BcCollectionConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for first_player_id in (1, 2):
        records.extend(_collect_episode(task, config, first_player_id=first_player_id))
    return records


def _collect_episode(task: BcTask, config: BcCollectionConfig, *, first_player_id: int) -> list[dict[str, Any]]:
    late_player_id = _other_player(first_player_id)
    episode_id = f"task-{task.task_id:06d}-seed-{task.paired_seed}-map-{task.map_id}-late-p{late_player_id}"
    env = RoundStepEnv(
        config=RoundStepConfig(EpisodeConfig(seed=task.paired_seed, map_id=task.map_id, rules=RulesConfig(round_count=config.round_count))),
        mechanisms=RoundStepMechanisms(),
        spawn=SpawnConfig(),
        p90_latency_ns={first_player_id: 1, late_player_id: 2},
    )
    observations = env.reset(seed=task.paired_seed, map_id=task.map_id)
    teacher_by_player = {1: build_teacher(config.teacher), 2: build_teacher(config.teacher)}
    for player_id, teacher in teacher_by_player.items():
        teacher.reset(
            _teacher_seed(task.paired_seed, player_id, first_player_id),
            EpisodeContext(
                player_id=player_id,
                opponent_id=_other_player(player_id),
                map_id=task.map_id,
                seed=task.paired_seed,
                tags=("bc_collect", task.split, f"first_p{first_player_id}"),
            ),
        )
    extractor = FeatureExtractor(player_id=late_player_id)
    records: list[dict[str, Any]] = []

    while observations:
        round_index = observations[late_player_id].round
        outputs = {
            1: teacher_by_player[1].act(observations[1]),
            2: teacher_by_player[2].act(observations[2]),
        }
        late_output = outputs[late_player_id]
        features = extractor.observe(observations[late_player_id])
        if features["feature_schema"] != FEATURE_SCHEMA:
            raise SimulatorRuleError(f"unexpected feature schema: {features['feature_schema']!r}")
        extractor.commit_action(late_output)
        result = env.step(outputs, first_player_id=first_player_id)
        records.append(
            {
                "planes": torch.as_tensor(features["planes"], dtype=torch.float32),
                "scalars": torch.as_tensor(features["scalars"], dtype=torch.float32),
                "actions": torch.tensor(late_output.actions, dtype=torch.long),
                "k": int(late_output.k),
                "order": int(late_output.order),
                "vp": int(late_output.vp),
                "episode_id": episode_id,
                "paired_seed": int(task.paired_seed),
                "episode_seed": int(task.paired_seed),
                "split": task.split,
                "map_id": int(task.map_id),
                "player_id": int(late_player_id),
                "first_player_id": int(first_player_id),
                "round_index": int(round_index),
                "teacher_spec": config.teacher.to_jsonable(),
                **_diagnostics(result, late_player_id=late_player_id),
            }
        )
        observations = result.observations

    if len(records) != config.round_count:
        raise SimulatorRuleError(f"{episode_id} expected {config.round_count} transitions, got {len(records)}")
    return records


def records_to_shard(records: list[dict[str, Any]], *, config: BcCollectionConfig, task_ids: tuple[int, ...]) -> dict[str, Any]:
    if not records:
        raise SimulatorRuleError("cannot write empty BC shard")
    return {
        "schema": BC_SHARD_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "config_hash": config.config_hash,
        "task_ids": list(task_ids),
        "teacher_spec": config.teacher.to_jsonable(),
        "planes": torch.stack([record["planes"] for record in records]),
        "scalars": torch.stack([record["scalars"] for record in records]),
        "actions": torch.stack([record["actions"] for record in records]),
        "k": torch.tensor([record["k"] for record in records], dtype=torch.long),
        "order": torch.tensor([record["order"] for record in records], dtype=torch.long),
        "vp": torch.tensor([record["vp"] for record in records], dtype=torch.long),
        "episode_id": [record["episode_id"] for record in records],
        "paired_seed": torch.tensor([record["paired_seed"] for record in records], dtype=torch.long),
        "episode_seed": torch.tensor([record["episode_seed"] for record in records], dtype=torch.long),
        "split": [record["split"] for record in records],
        "map_id": torch.tensor([record["map_id"] for record in records], dtype=torch.long),
        "player_id": torch.tensor([record["player_id"] for record in records], dtype=torch.long),
        "first_player_id": torch.tensor([record["first_player_id"] for record in records], dtype=torch.long),
        "round_index": torch.tensor([record["round_index"] for record in records], dtype=torch.long),
        "teacher_diagnostic_tag": [record["teacher_diagnostic_tag"] for record in records],
        "gross_gold": torch.tensor([record["gross_gold"] for record in records], dtype=torch.long),
        "net_gold": torch.tensor([record["net_gold"] for record in records], dtype=torch.long),
        "vision_spent": torch.tensor([record["vision_spent"] for record in records], dtype=torch.long),
        "margin": torch.tensor([record["margin"] for record in records], dtype=torch.long),
        "center_reached": torch.tensor([record["center_reached"] for record in records], dtype=torch.bool),
        "pickup_count": torch.tensor([record["pickup_count"] for record in records], dtype=torch.long),
        "blocked_count": torch.tensor([record["blocked_count"] for record in records], dtype=torch.long),
        "effective_move_count": torch.tensor([record["effective_move_count"] for record in records], dtype=torch.long),
        "reverse_count": torch.tensor([record["reverse_count"] for record in records], dtype=torch.long),
        "aba_count": torch.tensor([record["aba_count"] for record in records], dtype=torch.long),
        "zero_displacement": torch.tensor([record["zero_displacement"] for record in records], dtype=torch.bool),
    }


def _collect_task_chunk(output_dir: Path, config: BcCollectionConfig, tasks: tuple[BcTask, ...], *, worker_index: int) -> BcShardSummary:
    if not tasks:
        raise SimulatorRuleError("worker received empty BC task chunk")
    split = tasks[0].split
    if any(task.split != split for task in tasks):
        raise SimulatorRuleError("task chunk cannot mix splits")
    task_ids = tuple(task.task_id for task in tasks)
    shard_relpath = f"{split}/shard_{min(task_ids):06d}_{max(task_ids):06d}.pt"
    shard_path = output_dir / shard_relpath
    if shard_path.exists():
        return _summary_from_existing_shard(shard_path, shard_relpath=shard_relpath, split=split, task_ids=task_ids, config=config)
    records: list[dict[str, Any]] = []
    for task in tasks:
        records.extend(collect_task(task, config))
    shard = records_to_shard(records, config=config, task_ids=task_ids)
    _atomic_torch_save(shard, shard_path)
    _append_worker_log(
        output_dir / "logs" / f"worker_{worker_index:03d}.jsonl",
        {
            "worker_index": worker_index,
            "task_ids": list(task_ids),
            "split": split,
            "transition_count": len(records),
            "shard": shard_relpath,
        },
    )
    return BcShardSummary(
        path=shard_relpath,
        split=split,
        task_ids=task_ids,
        transition_count=len(records),
        episode_count=len(tasks) * 2,
        paired_seed_count=len(tasks),
    )


def _summary_from_existing_shard(
    shard_path: Path,
    *,
    shard_relpath: str,
    split: BcDatasetSplit,
    task_ids: tuple[int, ...],
    config: BcCollectionConfig,
) -> BcShardSummary:
    shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    if shard.get("schema") != BC_SHARD_SCHEMA:
        raise SimulatorRuleError(f"cannot resume from unsupported BC shard schema in {shard_path}: {shard.get('schema')!r}")
    if shard.get("config_hash") != config.config_hash:
        raise SimulatorRuleError(f"cannot resume from BC shard with different config hash: {shard_path}")
    if tuple(int(value) for value in shard.get("task_ids", ())) != task_ids:
        raise SimulatorRuleError(f"cannot resume from BC shard with unexpected task ids: {shard_path}")
    transition_count = int(shard["k"].shape[0])
    return BcShardSummary(
        path=shard_relpath,
        split=split,
        task_ids=task_ids,
        transition_count=transition_count,
        episode_count=len(set(str(value) for value in shard["episode_id"])),
        paired_seed_count=len(set(int(value) for value in shard["paired_seed"].tolist())),
    )


def _diagnostics(result: RoundStepResult, *, late_player_id: int) -> dict[str, Any]:
    state = result.state
    opponent_id = _other_player(late_player_id)
    player = state.players[late_player_id]
    opponent = state.players[opponent_id]
    movement_events = _player_movement_events(result.trace.transition_result.movement_events, late_player_id)
    blocked_count = sum(1 for event in movement_events if event.status not in (MoveStatus.MOVED, MoveStatus.STAYED))
    effective_move_count = sum(1 for event in movement_events if event.status == MoveStatus.MOVED)
    pickup_count = 0
    for interaction in result.trace.transition_result.interaction_events:
        for pickup in interaction.pickups:
            if pickup.actor.kind == ActorKind.PLAYER_UNIT and pickup.actor.player_id == late_player_id:
                pickup_count += 1
    unit_positions = [unit.position for unit in player.units]
    center_reached = any(_in_center(position) for position in unit_positions)
    reverse_count, aba_count = _path_oscillation_counts(movement_events)
    zero_displacement = all(event.from_pos == event.to_pos for event in movement_events)
    return {
        "teacher_diagnostic_tag": _teacher_diagnostic_tag(center_reached, pickup_count, blocked_count, reverse_count, aba_count, zero_displacement),
        "gross_gold": player.gross_gold,
        "net_gold": player.net_gold,
        "vision_spent": player.vision_spent,
        "margin": player.net_gold - opponent.net_gold,
        "center_reached": center_reached,
        "pickup_count": pickup_count,
        "blocked_count": blocked_count,
        "effective_move_count": effective_move_count,
        "reverse_count": reverse_count,
        "aba_count": aba_count,
        "zero_displacement": zero_displacement,
    }


def _player_movement_events(events: tuple[MovementEvent, ...], player_id: int) -> tuple[MovementEvent, ...]:
    return tuple(event for event in events if event.player_id == player_id)


def _path_oscillation_counts(events: tuple[MovementEvent, ...]) -> tuple[int, int]:
    reverse_count = 0
    aba_count = 0
    by_unit: dict[int, list[Position]] = {0: [], 1: []}
    for event in events:
        path = by_unit[event.unit_id]
        if not path:
            path.append(event.from_pos)
        path.append(event.to_pos)
    for path in by_unit.values():
        for idx in range(2, len(path)):
            if path[idx] == path[idx - 2] and path[idx] != path[idx - 1]:
                aba_count += 1
                reverse_count += 1
    return reverse_count, aba_count


def _teacher_diagnostic_tag(center_reached: bool, pickup_count: int, blocked_count: int, reverse_count: int, aba_count: int, zero_displacement: bool) -> str:
    if pickup_count > 0:
        return "harvest"
    if blocked_count > 0 or zero_displacement or (reverse_count > 0 and aba_count > 0):
        return "failure"
    if center_reached:
        return "approach"
    return "early_navigation"


def _prepare_output_dir(output_dir: Path, config: BcCollectionConfig, tasks: tuple[BcTask, ...], *, resume: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        existing_plan = output_dir / "plan.json"
        if not resume:
            raise FileExistsError(f"output dir already exists and is not empty: {output_dir}")
        if not existing_plan.exists():
            raise FileExistsError(f"cannot resume without existing plan.json: {output_dir}")
        raw = json.loads(existing_plan.read_text(encoding="utf-8"))
        if raw.get("config_hash") != config.config_hash:
            raise SimulatorRuleError("cannot resume BC collection with a different config hash")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "holdout", "logs", "audit"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "plan.json",
        {
            "schema": "goldrush2_bc_collection_plan_v1",
            "config_hash": config.config_hash,
            "config": config.to_jsonable(),
            "tasks": [task.to_jsonable() for task in tasks],
        },
    )


def _chunk_tasks(tasks: tuple[BcTask, ...], config: BcCollectionConfig) -> tuple[tuple[BcTask, ...], ...]:
    chunks: list[tuple[BcTask, ...]] = []
    transitions_per_task = config.round_count * 2
    max_tasks_by_limit = max(1, config.shard_transition_limit // transitions_per_task)
    chunk_size = min(config.task_chunk_size, max_tasks_by_limit)
    for split in ("train", "val", "holdout"):
        split_tasks = [task for task in tasks if task.split == split]
        for start in range(0, len(split_tasks), chunk_size):
            chunks.append(tuple(split_tasks[start : start + chunk_size]))
    return tuple(chunks)


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(value, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _append_worker_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")


def _teacher_seed(paired_seed: int, player_id: int, first_player_id: int) -> int:
    return int(paired_seed) * 100 + int(player_id) * 10 + int(first_player_id)


def _other_player(player_id: int) -> int:
    if player_id == 1:
        return 2
    if player_id == 2:
        return 1
    raise ValueError(f"player_id must be 1 or 2, got {player_id}")


def _in_center(position: Position) -> bool:
    return 4 <= position.row <= 12 and 4 <= position.col <= 12


def _validate_collection_config(config: BcCollectionConfig) -> None:
    if not config.map_ids:
        raise ValueError("map_ids cannot be empty")
    if not (config.train_seeds or config.val_seeds or config.holdout_seeds):
        raise ValueError("at least one BC split must contain seeds")
    if config.round_count <= 0:
        raise ValueError(f"round_count must be positive, got {config.round_count}")
    if config.num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {config.num_workers}")
    if config.task_chunk_size <= 0:
        raise ValueError(f"task_chunk_size must be positive, got {config.task_chunk_size}")
    if config.shard_transition_limit <= 0:
        raise ValueError(f"shard_transition_limit must be positive, got {config.shard_transition_limit}")
    seen: dict[int, str] = {}
    seen_in_split: set[tuple[str, int]] = set()
    for split, seeds in (("train", config.train_seeds), ("val", config.val_seeds), ("holdout", config.holdout_seeds)):
        for seed in seeds:
            key = (split, int(seed))
            if key in seen_in_split:
                raise ValueError(f"paired seed {seed} appears more than once in {split}")
            seen_in_split.add(key)
            previous = seen.setdefault(int(seed), split)
            if previous != split:
                raise ValueError(f"paired seed {seed} appears in both {previous} and {split}")
