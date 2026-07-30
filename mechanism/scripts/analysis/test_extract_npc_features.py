from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from mechanism.scripts.analysis import extract_npc_features as features


def grid(fill: int) -> list[list[int]]:
    return [[fill for _ in range(17)] for _ in range(17)]


def frame(
    grid_values: list[list[int]],
    npc_pos: list[int],
    npc_actions: list[int] | None = None,
    dispatch_order: list[int] | None = None,
) -> dict:
    npc = {
        "id": -1,
        "position": npc_pos,
        "visible_by": 3,
        "position_source": 3,
        "cost": 0,
        "pickup": 0,
    }
    if npc_actions is not None:
        npc["actions"] = npc_actions
        npc["actions_source"] = 3
    out = {
        "grid": grid_values,
        "visible_by": grid(3),
        "players": [
            {
                "id": 1,
                "order": 0,
                "units": [
                    {"position": [0, 0], "actions": [4], "gold": 0},
                    {"position": [0, 1], "actions": [4], "gold": 0},
                ],
            },
            {
                "id": 2,
                "order": 0,
                "units": [
                    {"position": [16, 16], "actions": [4], "gold": 0},
                    {"position": [16, 15], "actions": [4], "gold": 0},
                ],
            },
        ],
        "npcs": [npc],
    }
    if dispatch_order is not None:
        out["dispatch_order"] = dispatch_order
    return out


def write_merged(path: Path) -> None:
    maps = grid(0)
    start_grid = grid(0)
    start_grid[8][9] = -3
    start_grid[7][9] = 5
    end_grid = grid(0)
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
                    "start": frame(start_grid, [8, 8]),
                    "end": frame(start_grid, [7, 9], [3, 4, 0], [1, -1, 2]),
                },
            }
        ],
        "forfeit": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class ExtractNpcFeaturesTest(unittest.TestCase):
    def test_extract_run_writes_npc_tables_and_bomb_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            merged_root = temp / "merged"
            output_root = temp / "features"
            write_merged(merged_root / "run-a" / "100.json")

            summary = features.extract_run("run-a", merged_root=merged_root, output_root=output_root)

            self.assertEqual(1, summary["continuous_trajectories"])
            self.assertEqual(3, summary["step_rows"])
            self.assertEqual({"0": 1, "3": 1, "4": 1}, summary["action_counts"])
            self.assertEqual(1, summary["interaction_counts"]["bomb_trigger"])

            out = output_root / "run-a"
            with (out / "npc_round_observations.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(1, len(rows))
            self.assertEqual("[3,4,0]", rows[0]["actions_json"])
            self.assertEqual("8", rows[0]["start_row"])
            self.assertEqual("7", rows[0]["end_row"])
            self.assertEqual("1", rows[0]["nearest_bomb_dist"])

            with (out / "npc_step_events.csv").open(encoding="utf-8", newline="") as f:
                step_rows = list(csv.DictReader(f))
            self.assertEqual(3, len(step_rows))
            self.assertEqual("1", step_rows[0]["target_has_bomb_at_round_start"])
            self.assertEqual("1", step_rows[2]["target_has_gold_at_round_start"])

            with (out / "npc_interactions.csv").open(encoding="utf-8", newline="") as f:
                interactions = list(csv.DictReader(f))
            bomb_events = [row for row in interactions if row["event_type"] == "bomb_trigger"]
            self.assertEqual(1, len(bomb_events))
            self.assertEqual("8", bomb_events[0]["row"])
            self.assertEqual("9", bomb_events[0]["col"])
            self.assertEqual("0", bomb_events[0]["step_idx"])

    def test_replay_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            merged_root = temp / "merged"
            output_root = temp / "features"
            path = merged_root / "run-a" / "100.json"
            write_merged(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["rounds"][0]["merged"]["end"]["npcs"][0]["position"] = [8, 8]
            path.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(features.NpcFeatureError, "replay mismatch"):
                features.extract_run("run-a", merged_root=merged_root, output_root=output_root)


if __name__ == "__main__":
    unittest.main()
