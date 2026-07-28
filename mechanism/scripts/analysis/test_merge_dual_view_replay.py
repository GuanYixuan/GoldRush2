from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mechanism.scripts.analysis import merge_dual_view_replay as merge


def grid(fill: int) -> list[list[int]]:
    return [[fill for _ in range(17)] for _ in range(17)]


def write_replay(path: Path, players: dict, maps: list[list[int]], rounds: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [players, [[str(v) for v in row] for row in maps], *rounds]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def player(player_id: int, position: list[int] | None, actions: list[int] | None = None) -> dict:
    unit = {"gold": 0, "position": position}
    if actions is not None:
        unit["actions"] = actions
        unit["pickup"] = 0
    return {
        "id": player_id,
        "cost": 10,
        "gold": 0,
        "vision_spent": 0,
        "order": 0,
        "units": [unit, unit.copy()],
    }


def sample_round(viewer_side: int, common_value: int = 1) -> dict:
    start_grid = grid(-5)
    end_grid = grid(-5)
    if viewer_side == 1:
        start_grid[0][0] = 0
        start_grid[0][1] = common_value
        end_grid[0][0] = 0
        end_grid[0][1] = common_value
        players = [player(1, [0, 0], []), player(2, None)]
    else:
        start_grid[0][1] = common_value
        start_grid[0][2] = 2
        end_grid[0][1] = common_value
        end_grid[0][2] = 2
        players = [player(1, None), player(2, [0, 2], [])]
    npc = {"id": -1, "position": [0, 1], "actions": [4], "pickup": 0, "cost": 0}
    return {
        "round": 0,
        "start": {
            "grid": start_grid,
            "players": players,
            "npcs": [npc],
            "overlap_events": [],
            "vision_r": {str(viewer_side): 2},
        },
        "end": {
            "grid": end_grid,
            "players": players,
            "npcs": [npc],
            "overlap_events": [],
            "dispatch_order": [1, -1, 2],
            "trample_events": [],
            "burned": 0,
        },
        "snapshot": {"regions": []},
    }


class MergeDualViewReplayTest(unittest.TestCase):
    def test_process_run_merges_default_raw_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            raw_root = temp / "raw" / "observer_runs"
            output_root = temp / "processed" / "merged_replays"
            run_id = "run-a"
            game_id = 100
            players = {"player1": "A", "player2": "B"}
            maps = grid(0)

            write_replay(raw_root / run_id / "observer" / "replays" / f"{game_id}.ndjson", players, maps, [sample_round(1)])
            write_replay(raw_root / run_id / "main" / "replays" / f"{game_id}.ndjson", players, maps, [sample_round(2)])

            written = merge.process_run(run_id, None, raw_run_root=raw_root, output_root=output_root)
            self.assertEqual([output_root / run_id / f"{game_id}.json"], written)

            merged = json.loads(written[0].read_text(encoding="utf-8"))
            self.assertEqual("goldrush2_merged_replay", merged["format"])
            self.assertEqual("observer", merged["source"]["views"]["1"]["account_dir"])
            self.assertEqual("main", merged["source"]["views"]["2"]["account_dir"])
            self.assertNotIn("views", merged["rounds"][0])

            frame = merged["rounds"][0]["merged"]["start"]
            self.assertEqual(0, frame["grid"][0][0])
            self.assertEqual(1, frame["grid"][0][1])
            self.assertEqual(2, frame["grid"][0][2])
            self.assertEqual(1, frame["visible_by"][0][0])
            self.assertEqual(3, frame["visible_by"][0][1])
            self.assertEqual(2, frame["visible_by"][0][2])
            self.assertEqual({"1": 2, "2": 2}, frame["vision_r"])
            self.assertEqual(3, frame["npcs"][0]["visible_by"])
            self.assertFalse(frame["npcs"][0]["is_complete"])

    def test_common_visible_grid_conflict_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            raw_root = temp / "raw" / "observer_runs"
            run_id = "run-a"
            game_id = 100
            players = {"player1": "A", "player2": "B"}
            maps = grid(0)

            write_replay(raw_root / run_id / "observer" / "replays" / f"{game_id}.ndjson", players, maps, [sample_round(1, 1)])
            write_replay(raw_root / run_id / "main" / "replays" / f"{game_id}.ndjson", players, maps, [sample_round(2, 0)])

            with self.assertRaisesRegex(merge.MergeError, "共同可见 grid 冲突"):
                merge.merge_game(run_id, game_id, raw_run_root=raw_root)


if __name__ == "__main__":
    unittest.main()
