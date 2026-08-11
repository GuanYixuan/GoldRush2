#!/usr/bin/env python3
"""将选定实验目录内的 metrics.jsonl 导出为 TensorBoard 标量事件。"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_METRICS = (
    "actor_lr",
    "critic_lr",
    "approx_joint_kl",
    "clip_fraction",
    "policy_loss",
    "value_loss",
    "explained_variance",
    "entropy_bonus",
    "grad_norm",
    "actor_grad_norm",
    "critic_grad_norm",
    "early_stopped",
    "loss",
    "mean_reward",
    "return_mean",
    "return_std",
    "reward_sum",
    "reward_clipped_net_gold_gain_mean",
    "reward_net_gold_gain_mean",
    "reward_net_gold_gain_reward_mean",
    "reward_vision_info_gold_mean",
    "reward_vision_info_cells_mean",
    "reward_vision_info_reward_mean",
    "advantage_mean",
    "advantage_std",
    "value_mean",
    "value_std",
    "train_win_rate",
    "train_net_gold_margin_mean",
    "train_agent_net_gold_mean",
    "train_opponent_net_gold_mean",
    "train_agent_pickups_mean",
    "train_opponent_pickups_mean",
    "train_agent_vision_spent_mean",
    "train_agent_bomb_lost_gold_per_episode",
    "train_agent_bomb_triggers_per_episode",
    "train_agent_pickup_gold_per_episode",
    "train_agent_trample_penalty_per_episode",
    "train_agent_tramples_per_episode",
    "train_opponent_bomb_lost_gold_per_episode",
    "train_opponent_bomb_triggers_per_episode",
    "train_opponent_pickup_gold_per_episode",
    "train_opponent_trample_penalty_per_episode",
    "train_opponent_tramples_per_episode",
    "batch_assembly_ms",
    "inference_total_with_action_send_ms",
    "scheduler_ms",
    "worker_per_transition_episode_wall_ms",
    "update_wall_ms",
)

POLICY_METRICS = {
    "approx_joint_kl",
    "clip_fraction",
    "policy_loss",
    "entropy_bonus",
    "grad_norm",
    "actor_grad_norm",
    "critic_grad_norm",
}
CRITIC_METRICS = {"value_loss", "explained_variance", "value_mean", "value_std"}
RETURN_METRICS = {
    "mean_reward",
    "reward_sum",
    "reward_clipped_net_gold_gain_mean",
    "reward_net_gold_gain_mean",
    "reward_net_gold_gain_reward_mean",
    "reward_vision_info_gold_mean",
    "reward_vision_info_cells_mean",
    "reward_vision_info_reward_mean",
    "return_mean",
    "return_std",
    "advantage_mean",
    "advantage_std",
}
BEHAVIOR_PREFIXES = ("action_fraction_", "k_fraction_", "ko_fraction_", "order_fraction_", "vp_fraction_")
BEHAVIOR_METRICS = {
    "stay_action_fraction",
    "same_role_reverse_fraction",
    "k_edge_fraction",
    "vp_nonzero_fraction",
}
PERFORMANCE_PREFIXES = ("worker_", "scheduler_", "inference_")
SYSTEM_METRICS = {
    "transition_count",
    "feature_batches",
    "max_feature_batch_size",
    "mean_feature_batch_size",
    "first_episodes",
    "second_episodes",
    "first_started",
    "second_started",
    "ppo_update_count",
    "trainable_param_count",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Source:
    path: Path
    series: str
    offset: int = 0


@dataclass(frozen=True)
class Experiment:
    root: Path
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将选定实验目录内的 metrics.jsonl 导出为 TensorBoard。"
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        type=Path,
        help="递归扫描的一个或多个实验目录",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help="要新建的 TensorBoard event 目录；默认写入 temp/tensorboards_logs/时间戳",
    )
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="要导出的 metric 字段；可重复传入，并替代默认集合",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="导出每个有限数值字段，而非默认的核心指标集合",
    )
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=None,
        metavar="SECONDS",
        help="以此间隔持续读取追加的 JSONL 行",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="导出前删除已有 --logdir",
    )
    args = parser.parse_args()
    if args.all_metrics and args.metric:
        parser.error("--all-metrics 和 --metric 不能同时使用")
    if args.watch_seconds is not None and args.watch_seconds <= 0:
        parser.error("--watch-seconds 必须为正数")
    if args.overwrite and args.logdir is None:
        parser.error("--overwrite 必须与显式 --logdir 一起使用")
    return args


def main() -> None:
    args = parse_args()
    try:
        from torch.utils.tensorboard import SummaryWriter
        import numpy as np
        from tensorboard.compat.tensorflow_stub import dtypes

        # TensorBoard 2.14 accepts SummaryWriter with NumPy 2, but its CLI then
        # crashes while loading hparams. Exercise the affected dtype path first.
        dtypes.as_dtype(np.dtype("S1"))
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "当前 TensorBoard 与环境依赖不兼容。请安装兼容 NumPy 2 和 protobuf 3 的版本："
            "conda run --no-capture-output -n goldrush python -m pip install "
            "'tensorboard==2.19.0' 'setuptools<81'"
        ) from exc

    logdir = default_logdir() if args.logdir is None else args.logdir.resolve()
    experiments = prepare_experiments(args.experiments, logdir)
    sources = discover_sources(experiments, require_any=args.watch_seconds is None)
    prepare_logdir(logdir, overwrite=args.overwrite)
    print(f"TensorBoard event 输出目录：{logdir}")
    selected_metrics = None if args.all_metrics else set(args.metric or DEFAULT_METRICS)
    writers = {source.path: SummaryWriter(log_dir=str(logdir / source.series)) for source in sources}
    try:
        export_once(sources, writers, selected_metrics)
        if args.watch_seconds is None:
            return
        print(f"每 {args.watch_seconds:g} 秒跟随 {len(sources)} 个 metrics 文件；Ctrl-C 停止。")
        while True:
            time.sleep(args.watch_seconds)
            register_new_sources(experiments, sources, writers, SummaryWriter, logdir)
            export_once(sources, writers, selected_metrics)
    except KeyboardInterrupt:
        print("已停止。")
    finally:
        for writer in writers.values():
            writer.flush()
            writer.close()


def prepare_experiments(experiment_paths: list[Path], logdir: Path) -> list[Experiment]:
    experiments: list[Experiment] = []
    used_series: set[str] = set()
    for experiment_path in experiment_paths:
        root = experiment_path.resolve()
        if not root.is_dir():
            raise RuntimeError(f"实验目录不存在：{root}")
        if logdir.is_relative_to(root):
            raise RuntimeError(f"--logdir 不能位于输入实验目录内：{logdir}")
        label = unique_series_label(root.name, used_series)
        experiments.append(Experiment(root=root, label=label))
    return experiments


def discover_sources(experiments: list[Experiment], *, require_any: bool = True) -> list[Source]:
    sources = discover_sources_silently(experiments)
    if not sources and require_any:
        roots = ", ".join(str(experiment.root) for experiment in experiments)
        raise RuntimeError(f"目录下未找到 metrics.jsonl：{roots}")
    print("输入日志：")
    for source in sources:
        print(f"  {source.series}: {source.path}")
    return sources


def register_new_sources(
    experiments: list[Experiment],
    sources: list[Source],
    writers: dict[Path, Any],
    summary_writer: Any,
    logdir: Path,
) -> None:
    known_paths = {source.path for source in sources}
    for source in discover_sources_silently(experiments):
        if source.path in known_paths:
            continue
        sources.append(source)
        writers[source.path] = summary_writer(log_dir=str(logdir / source.series))
        print(f"发现新 metrics 文件：{source.series}: {source.path}")


def discover_sources_silently(experiments: list[Experiment]) -> list[Source]:
    sources: list[Source] = []
    for experiment in experiments:
        for metrics_path in sorted(experiment.root.rglob("metrics.jsonl")):
            relative_parent = metrics_path.parent.relative_to(experiment.root)
            series = experiment.label if relative_parent == Path(".") else f"{experiment.label}/{relative_parent.as_posix()}"
            sources.append(Source(path=metrics_path, series=series))
    return sources


def unique_series_label(label: str, used: set[str]) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._") or "experiment"
    candidate = sanitized
    index = 2
    while candidate in used:
        candidate = f"{sanitized}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def default_logdir() -> Path:
    root = REPO_ROOT / "temp/tensorboards_logs"
    stem = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / stem
    index = 2
    while candidate.exists():
        candidate = root / f"{stem}_{index}"
        index += 1
    return candidate


def prepare_logdir(logdir: Path, *, overwrite: bool) -> None:
    if logdir.exists():
        if not overwrite:
            raise RuntimeError(f"--logdir 已存在：{logdir}；请新建目录或显式传入 --overwrite")
        shutil.rmtree(logdir)
    logdir.mkdir(parents=True)


def export_once(sources: list[Source], writers: dict[Path, Any], selected_metrics: set[str] | None) -> None:
    total_rows = 0
    for source in sources:
        if source.path.stat().st_size < source.offset:
            raise RuntimeError(f"跟随期间 metrics 文件被截断：{source.path}")
        rows = read_complete_rows(source)
        for row in rows:
            write_row(writers[source.path], row, selected_metrics)
        total_rows += len(rows)
    if total_rows:
        for writer in writers.values():
            writer.flush()
        print(f"已导出 {total_rows} 条新记录。")


def read_complete_rows(source: Source) -> list[dict[str, Any]]:
    with source.path.open("rb") as handle:
        handle.seek(source.offset)
        payload = handle.read()
    rows: list[dict[str, Any]] = []
    consumed = 0
    for raw_line in payload.splitlines(keepends=True):
        if not raw_line.endswith(b"\n"):
            break
        consumed += len(raw_line)
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"JSONL 行无效，文件为 {source.path}，字节位置为 {source.offset + consumed}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"JSONL 行必须是对象：{source.path}")
        if "update" not in row:
            raise RuntimeError(f"JSONL 行缺少 update 字段：{source.path}")
        rows.append(row)
    source.offset += consumed
    return rows


def write_row(writer: Any, row: dict[str, Any], selected_metrics: set[str] | None) -> None:
    step = int(row["update"])
    for field, value in row.items():
        if selected_metrics is not None and field not in selected_metrics:
            continue
        if isinstance(value, bool):
            scalar = float(value)
        elif isinstance(value, (int, float)):
            scalar = float(value)
        else:
            continue
        if not math.isfinite(scalar):
            continue
        writer.add_scalar(metric_tag(field), scalar, global_step=step)


def metric_tag(field: str) -> str:
    if field in {"actor_lr", "critic_lr"}:
        return f"optimization/{field}"
    if field == "loss":
        return f"optimization/{field}"
    if field in POLICY_METRICS:
        return f"policy/{field}"
    if field == "early_stopped":
        return f"policy/{field}"
    if field in CRITIC_METRICS:
        return f"critic/{field}"
    if field in RETURN_METRICS:
        return f"return/{field}"
    if field.startswith("train_agent_"):
        return f"agent_outcome/{field.removeprefix('train_agent_')}"
    if field.startswith("train_opponent_"):
        return f"opponent_outcome/{field.removeprefix('train_opponent_')}"
    if field.startswith("train_"):
        return f"match_outcome/{field.removeprefix('train_')}"
    if field.startswith(BEHAVIOR_PREFIXES) or field in BEHAVIOR_METRICS:
        return f"behavior/{field}"
    if field.endswith("_ms") or field.startswith(PERFORMANCE_PREFIXES):
        return f"performance/{field}"
    if field in SYSTEM_METRICS:
        return f"system/{field}"
    return f"other/{field}"


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(2) from exc
