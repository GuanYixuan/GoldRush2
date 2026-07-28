from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from mechanism.scripts.analysis import extract_gold_features as features


def grid(fill: int) -> list[list[int]]:
    return [[fill for _ in range(17)] for _ in range(17)]


def visible(fill: int) -> list[list[int]]:
    return [[fill for _ in range(17)] for _ in range(17)]


def write_merged(path: Path, curr_value: int) -> None:
    maps = grid(0)
    round0_end = grid(0)
    round1_start = grid(0)
    round1_start[4][4] = curr_value
    data = {
        "format": "goldrush2_merged_replay",
        "format_version": 1,
        "source": {"game_id": 100, "map_id": 1, "run_id": "run-a", "views": {}},
        "players": {"player1": "A", "player2": "B"},
        "maps": maps,
        "rounds": [
            {
                "round": 0,
                "merged": {
                    "start": {"grid": grid(0), "visible_by": visible(3), "players": [], "npcs": []},
                    "end": {"grid": round0_end, "visible_by": visible(3), "players": [], "npcs": []},
                },
            },
            {
                "round": 1,
                "merged": {
                    "start": {"grid": round1_start, "visible_by": visible(3), "players": [], "npcs": []},
                    "end": {"grid": round1_start, "visible_by": visible(3), "players": [], "npcs": []},
                    "snapshot": {
                        "round": 1,
                        "window": [0, 4],
                        "regions": [
                            {
                                "id": rid,
                                "enter": 0,
                                "leave": 0,
                                "gold_generated": 5 if rid == 1 else 0,
                                "gold_collected": 0,
                                "gold_remaining": 0,
                                "occupants": 0,
                            }
                            for rid in range(1, 6)
                        ],
                    },
                },
            },
        ],
        "forfeit": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class ExtractGoldFeaturesTest(unittest.TestCase):
    def test_extract_run_writes_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            merged_root = temp / "merged"
            output_root = temp / "features"
            write_merged(merged_root / "run-a" / "100.json", 5)

            summary = features.extract_run("run-a", merged_root=merged_root, output_root=output_root)
            out = output_root / "run-a"
            self.assertEqual(289, summary["cell_transitions"])
            self.assertEqual(1, summary["cell_events"])

            with (out / "window_observed.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            region1 = [row for row in rows if row["region"] == "1"][0]
            self.assertEqual("0", region1["window_begin"])
            self.assertEqual("1", region1["transition_rounds"])
            self.assertEqual("1", region1["events"])
            self.assertEqual("5", region1["delta_sum"])

            with (out / "cell_summary.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            cell = [row for row in rows if row["row"] == "4" and row["col"] == "4"][0]
            self.assertEqual("1", cell["events"])
            self.assertEqual("5", cell["delta_sum"])
            self.assertEqual('{"5":1}', cell["amount_counts_json"])

    def test_negative_delta_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            merged_root = temp / "merged"
            output_root = temp / "features"
            write_merged(merged_root / "run-a" / "100.json", -1)
            # -1 is excluded rather than negative delta; overwrite with valid lower gold.
            data = json.loads((merged_root / "run-a" / "100.json").read_text(encoding="utf-8"))
            data["rounds"][0]["merged"]["end"]["grid"][4][4] = 5
            data["rounds"][1]["merged"]["start"]["grid"][4][4] = 0
            (merged_root / "run-a" / "100.json").write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(features.FeatureError, "负增量"):
                features.extract_run("run-a", merged_root=merged_root, output_root=output_root)


if __name__ == "__main__":
    unittest.main()
