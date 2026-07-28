from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualizer.app import VisualizerBootstrapError, build_window


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GoldRush2 replay visualizer")
    parser.add_argument("replay_path", nargs="?", help="Optional official NDJSON or merged JSON replay path.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        app, window = build_window(args.replay_path, argv=[sys.argv[0]])
    except VisualizerBootstrapError as exc:
        print(exc, file=sys.stderr)
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

