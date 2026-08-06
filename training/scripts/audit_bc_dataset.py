from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.bc.audit import audit_dataset


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a GoldRush2 BC dataset.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-limit", type=int, default=-1)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result = audit_dataset(
        args.dataset_dir,
        device=args.device,
        sample_limit=None if args.sample_limit < 0 else int(args.sample_limit),
        output_dir=args.output_dir,
    )
    print(json.dumps(result.summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
