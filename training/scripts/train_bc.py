from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.bc.losses import BcLossConfig
from training.bc.train_bc import TrainBcConfig, run_bc_training
from training.models import PolicyNetworkConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GoldRush2 actor with behavior cloning.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--adam-eps", type=float, default=1.0e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--w-ko", type=float, default=1.0)
    parser.add_argument("--w-action", type=float, default=1.0)
    parser.add_argument("--w-vp", type=float, default=0.2)
    parser.add_argument("--model-width", type=int, default=96)
    parser.add_argument("--model-blocks", type=int, default=8)
    parser.add_argument("--scalar-hidden", type=int, nargs=2, default=[96, 96])
    parser.add_argument("--actor-hidden", type=int, default=256)
    parser.add_argument("--critic-hidden", type=int, nargs=2, default=[256, 128])
    parser.add_argument("--decoder-hidden", type=int, default=128)
    parser.add_argument("--decoder-embedding", type=int, default=16)
    return parser


def config_from_args(args: argparse.Namespace) -> TrainBcConfig:
    return TrainBcConfig(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        seed=int(args.seed),
        device=str(args.device),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        adam_eps=float(args.adam_eps),
        num_workers=int(args.num_workers),
        loss=BcLossConfig(w_ko=float(args.w_ko), w_action=float(args.w_action), w_vp=float(args.w_vp)),
        model=PolicyNetworkConfig(
            width=int(args.model_width),
            residual_blocks=int(args.model_blocks),
            scalar_hidden=tuple(int(value) for value in args.scalar_hidden),
            actor_hidden=int(args.actor_hidden),
            critic_hidden=tuple(int(value) for value in args.critic_hidden),
            decoder_hidden=int(args.decoder_hidden),
            decoder_embedding=int(args.decoder_embedding),
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = run_bc_training(config_from_args(args))
    print(json.dumps({"output_dir": str(result.output_dir), "latest_checkpoint": str(result.latest_checkpoint)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
