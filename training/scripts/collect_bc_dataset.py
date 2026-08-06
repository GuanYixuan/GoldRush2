from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.bc.collect import collect_dataset
from training.bc.schema import BcCollectionConfig, BcTeacherSpec


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect GoldRush2 behavior cloning dataset.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher", default="fast_probe_v3_like")
    parser.add_argument("--teacher-kind", default="scripted")
    parser.add_argument("--map-ids", type=int, nargs="+", default=[1])
    parser.add_argument("--train-seeds", default="")
    parser.add_argument("--val-seeds", default="")
    parser.add_argument("--holdout-seeds", default="")
    parser.add_argument("--round-count", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--task-chunk-size", type=int, default=1)
    parser.add_argument("--shard-transition-limit", type=int, default=20000)
    parser.add_argument("--base-seed", type=int, default=20260806)
    parser.add_argument("--resume", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> BcCollectionConfig:
    return BcCollectionConfig(
        teacher=BcTeacherSpec(kind=str(args.teacher_kind), name=str(args.teacher), params={}),
        map_ids=tuple(int(value) for value in args.map_ids),
        train_seeds=_parse_seed_spec(args.train_seeds),
        val_seeds=_parse_seed_spec(args.val_seeds),
        holdout_seeds=_parse_seed_spec(args.holdout_seeds),
        round_count=int(args.round_count),
        num_workers=int(args.num_workers),
        task_chunk_size=int(args.task_chunk_size),
        shard_transition_limit=int(args.shard_transition_limit),
        base_seed=int(args.base_seed),
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = collect_dataset(args.output_dir, config_from_args(args), resume=bool(args.resume))
    print(json.dumps({"output_dir": str(result.output_dir), "shard_count": len(result.manifest.shards)}, ensure_ascii=True))


def _parse_seed_spec(raw: str) -> tuple[int, ...]:
    if raw == "":
        return ()
    seeds: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            pieces = token.split(":")
            if len(pieces) != 2:
                raise ValueError(f"seed range must be start:end, got {token!r}")
            start, end = (int(value) for value in pieces)
            if end < start:
                raise ValueError(f"seed range end must be >= start, got {token!r}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    return tuple(seeds)


if __name__ == "__main__":
    main()
