from __future__ import annotations

import json
import copy
import tempfile
import unittest
from pathlib import Path

from visualizer.model import FramePhase
from visualizer.replay_io import ReplayKind, ReplayLoader
from visualizer.replay_open import open_replay
from visualizer.ui import PlaybackController


class ReplayVisualizerTests(unittest.TestCase):
    def test_official_replay_loads_and_derives_frames(self) -> None:
        document, bundles = open_replay("official_sdk/data/user/g3.txt")

        self.assertEqual(document.kind, ReplayKind.OFFICIAL)
        self.assertEqual(len(bundles), 500)
        self.assertEqual(bundles[0].round_index, 0)
        self.assertEqual(len(bundles[0].end.entities), 4)
        self.assertTrue(bundles[0].motions)

    def test_merged_replay_loads_auto_merged_layer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "merged.json"
            path.write_text(json.dumps(_merged_payload()), encoding="utf-8")

            document, bundles = open_replay(path)

        self.assertEqual(document.kind, ReplayKind.MERGED)
        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0].start.grid[0][0], 0)
        self.assertEqual(bundles[0].start.visible_by[0][0], 3)
        self.assertEqual(bundles[0].end.players[0].net_gold, 5)

    def test_simulator_replay_loads_full_state_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "simulator.json"
            path.write_text(json.dumps(_simulator_payload()), encoding="utf-8")

            document, bundles = open_replay(path)

        self.assertEqual(document.kind, ReplayKind.SIMULATOR_FULL)
        self.assertEqual(len(bundles), 1)
        self.assertEqual(len(bundles[0].end.entities), 5)
        self.assertTrue(any(motion.ref.owner_id == 1 and motion.ref.entity_id == 0 for motion in bundles[0].motions))
        self.assertTrue(any(label.text == "+7" for label in bundles[0].end.floating_labels))
        self.assertTrue(any(label.text == "-1" for label in bundles[0].end.floating_labels))
        self.assertTrue(any(annotation.marker == "!" for annotation in bundles[0].end.cell_annotations))
        self.assertTrue(any(annotation.marker == "T" for annotation in bundles[0].end.cell_annotations))

    def test_simulator_smoke_replay_loads(self) -> None:
        path = Path("temp/simulator_smoke_replays/stay_players_default_mechanisms_map1_seed2026073101.json")
        if not path.exists():
            self.skipTest("simulator smoke replay not present")

        document, bundles = open_replay(path)

        self.assertEqual(document.kind, ReplayKind.SIMULATOR_FULL)
        self.assertEqual(len(bundles), 500)

    def test_playback_steps_and_clamps(self) -> None:
        _, bundles = open_replay("official_sdk/data/user/g3.txt")
        controller = PlaybackController()
        controller.load_bundles(bundles)

        controller.set_round(9999)
        self.assertEqual(controller.state().current_round, len(bundles) - 1)
        controller.step_backward()
        self.assertEqual(controller.current_bundle().round_index, len(bundles) - 2)

    def test_loader_normalizes_static_map_strings(self) -> None:
        document = ReplayLoader.load("official_sdk/data/user/g3.txt")

        self.assertIsInstance(document.static_map[0][0], int)

    def test_bomb_moving_out_of_view_is_not_trigger(self) -> None:
        payload = _merged_payload()
        merged = payload["rounds"][0]["merged"]
        merged["start"] = copy.deepcopy(merged["start"])
        merged["end"] = copy.deepcopy(merged["end"])
        merged["start"]["grid"][3][4] = -3
        merged["end"]["grid"][3][4] = -5

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "merged.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            _, bundles = open_replay(path)

        self.assertFalse(any(annotation.marker == "!" for annotation in bundles[0].end.cell_annotations))

    def test_visible_bomb_disappearance_is_trigger(self) -> None:
        payload = _merged_payload()
        merged = payload["rounds"][0]["merged"]
        merged["start"] = copy.deepcopy(merged["start"])
        merged["end"] = copy.deepcopy(merged["end"])
        merged["start"]["grid"][3][4] = -3
        merged["end"]["grid"][3][4] = 0

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "merged.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            _, bundles = open_replay(path)

        self.assertTrue(any(annotation.marker == "!" for annotation in bundles[0].end.cell_annotations))


def _merged_payload() -> dict:
    grid = [[0 for _ in range(17)] for _ in range(17)]
    visible_by = [[3 for _ in range(17)] for _ in range(17)]
    frame = {
        "grid": grid,
        "visible_by": visible_by,
        "players": [
            {
                "id": 1,
                "gold": 5,
                "vision_spent": 0,
                "units": [
                    {"position": [0, 0], "gold": 5, "actions": [3], "pickup": 5, "is_complete": True},
                    {"position": [16, 16], "gold": 0, "actions": [], "pickup": 0, "is_complete": True},
                ],
            },
            {
                "id": 2,
                "gold": 0,
                "vision_spent": 0,
                "units": [
                    {"position": [0, 16], "gold": 0, "actions": [], "pickup": 0, "is_complete": True},
                    {"position": [16, 0], "gold": 0, "actions": [], "pickup": 0, "is_complete": True},
                ],
            },
        ],
        "npcs": [],
        "overlap_events": [],
    }
    end = dict(frame)
    end["dispatch_order"] = [1, 2]
    end["trample_events"] = []
    end["burned"] = 0
    return {
        "format": "goldrush2_merged_replay",
        "format_version": 1,
        "source": {"views": {}},
        "players": {"player1": "A", "player2": "B"},
        "maps": grid,
        "rounds": [{"round": 0, "merged": {"start": frame, "end": end}}],
        "forfeit": None,
    }


def _simulator_payload() -> dict:
    maps = [[0 for _ in range(17)] for _ in range(17)]
    start_grid = [[0 for _ in range(17)] for _ in range(17)]
    end_grid = copy.deepcopy(start_grid)
    start_grid[0][1] = 10
    start_grid[2][2] = -3
    end_grid[0][1] = 3
    end_grid[2][2] = 0
    start = {
        "round_index": 0,
        "grid": start_grid,
        "players": [
            {"id": 1, "gold": 0, "vision_spent": 0, "units": [{"id": 0, "position": [0, 0], "gold": 0}, {"id": 1, "position": [16, 16], "gold": 0}]},
            {"id": 2, "gold": 0, "vision_spent": 0, "units": [{"id": 0, "position": [0, 16], "gold": 0}, {"id": 1, "position": [16, 0], "gold": 0}]},
        ],
        "npcs": [{"id": -1, "position": [2, 1]}],
        "gold": [{"position": [0, 1], "amount": 10}],
        "bombs": [[2, 2]],
    }
    end = {
        "round_index": 1,
        "grid": end_grid,
        "players": [
            {"id": 1, "gold": 6, "vision_spent": 0, "units": [{"id": 0, "position": [0, 1], "gold": 6}, {"id": 1, "position": [16, 16], "gold": 0}]},
            {"id": 2, "gold": 0, "vision_spent": 0, "units": [{"id": 0, "position": [0, 16], "gold": 0}, {"id": 1, "position": [16, 0], "gold": 0}]},
        ],
        "npcs": [{"id": -1, "position": [2, 2]}],
        "gold": [{"position": [0, 1], "amount": 3}],
        "bombs": [],
    }
    return {
        "format": "goldrush2_simulator_full_replay",
        "format_version": 1,
        "source": {"seed": 1, "map_id": 1, "map_name": "test", "mechanisms": {}},
        "players": {"player1": "A", "player2": "B"},
        "maps": maps,
        "rounds": [
            {
                "round": 0,
                "start": start,
                "end": end,
                "actions": {"players": {"1": {"actions": [3, 4, 4, 4, 4, 4], "k": 1, "order": 0, "vp": 0}}, "npcs": {"-1": [3]}, "npc_order": [-1]},
                "events": {
                    "gold_generated": [],
                    "movement": [
                        {"player_id": 1, "unit_id": 0, "action": 3, "from_pos": [0, 0], "to_pos": [0, 1], "status": "moved", "blocked_by": None},
                        {"player_id": 0, "unit_id": -1, "action": 3, "from_pos": [2, 1], "to_pos": [2, 2], "status": "moved", "blocked_by": None},
                    ],
                    "interactions": [
                        {
                            "pickups": [
                                {"actor": {"kind": "player_unit", "player_id": 1, "unit_id": 0, "npc_id": None}, "position": [0, 1], "available_gold": 10, "picked_gold": 7, "remaining_gold": 3}
                            ],
                            "bomb_triggers": [
                                {"actor": {"kind": "npc", "player_id": None, "unit_id": None, "npc_id": -1}, "position": [2, 2], "lost_gold": 0}
                            ],
                            "tramples": [
                                {"actor": {"kind": "player_unit", "player_id": 1, "unit_id": 0, "npc_id": None}, "position": [0, 1], "npc_count": 3, "penalty": 1}
                            ],
                        }
                    ],
                    "dispatch_order": [1, -1, 2],
                    "bomb_refresh": {"cleared": [], "spawned": []},
                    "mechanisms": {},
                },
                "snapshot": None,
            }
        ],
        "forfeit": None,
    }


if __name__ == "__main__":
    unittest.main()
