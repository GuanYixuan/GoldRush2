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


if __name__ == "__main__":
    unittest.main()
