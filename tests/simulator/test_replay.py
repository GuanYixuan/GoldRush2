from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from simulator.constants import GRID_BOMB, GRID_EMPTY, GRID_OBSTACLE, STATIC_EMPTY
from simulator.config import RulesConfig
from simulator.errors import SimulatorRuleError
from simulator.mechanisms.bombs import BombRefreshEvent
from simulator.mechanisms.maps import MapTemplate
from simulator.mechanisms.scripted import ScriptedMechanisms, ScriptedRound
from simulator.replay import (
    SIMULATOR_FULL_REPLAY_FORMAT,
    SIMULATOR_FULL_REPLAY_VERSION,
    SimulatorReplayRecorder,
    dumps_replay,
    serialize_state,
    write_replay,
)
from simulator.rules.snapshot import SnapshotAccumulator
from simulator.rules.transition import transition_one_round
from simulator.state import GameState, NpcState, PlayerState, UnitState
from simulator.types import Action, GameOutput, GoldGenerationEvent, Position


class ReplayTests(unittest.TestCase):
    def test_serialize_state_uses_resource_grid_and_no_npc_gold(self) -> None:
        state = GameState(
            round_index=3,
            players={
                1: PlayerState(1, [UnitState(0, Position(0, 0), 5), UnitState(1, Position(16, 16))]),
                2: PlayerState(2, [UnitState(0, Position(0, 16)), UnitState(1, Position(16, 0), 2)]),
            },
            obstacles=frozenset({Position(1, 1)}),
            gold={Position(2, 2): 9},
            bombs={Position(3, 3)},
            npcs={-1: NpcState(-1, Position(4, 4))},
        )

        payload = serialize_state(state)

        self.assertEqual(payload["grid"][1][1], GRID_OBSTACLE)
        self.assertEqual(payload["grid"][2][2], 9)
        self.assertEqual(payload["grid"][3][3], GRID_BOMB)
        self.assertEqual(payload["grid"][0][0], GRID_EMPTY)
        grid_values = {value for row in payload["grid"] for value in row}
        self.assertNotIn(-2, grid_values)
        self.assertNotIn(-4, grid_values)
        self.assertNotIn(-5, grid_values)
        self.assertEqual(payload["npcs"], [{"id": -1, "position": [4, 4]}])
        self.assertNotIn("gold", payload["npcs"][0])
        json.dumps(payload, ensure_ascii=False, sort_keys=True)

    def test_recorder_exports_stable_full_replay_document(self) -> None:
        template = _empty_template()
        state = _transition_state()
        start_state = copy.deepcopy(state)
        mechanisms = ScriptedMechanisms(
            {
                0: ScriptedRound(
                    gold_generated=(GoldGenerationEvent(Position(5, 6), 10),),
                    npc_actions={-1: (Action.LEFT,)},
                )
            }
        )
        p1 = GameOutput(actions=(Action.RIGHT, Action.STAY, Action.STAY, Action.STAY, Action.STAY, Action.STAY), k=1, order=0, vp=1)
        p2 = GameOutput(actions=(Action.STAY,) * 6, k=3, order=0, vp=0)
        result = transition_one_round(
            state,
            player_outputs={1: p1, 2: p2},
            mechanisms=mechanisms,
            first_player_id=1,
            snapshot_accumulator=SnapshotAccumulator(RulesConfig(snapshot_period=1)),
        )
        recorder = SimulatorReplayRecorder(
            map_template=template,
            seed=123,
            players={1: "policy_a", 2: "policy_b"},
            mechanisms={"gold": "scripted"},
        )
        recorder.record_round(
            start_state=start_state,
            end_state=state,
            player_outputs={1: p1, 2: p2},
            transition_result=result,
            npc_actions={-1: (Action.LEFT,)},
            npc_order=(-1,),
            bomb_refresh_event=BombRefreshEvent(cleared=frozenset(), spawned=frozenset()),
            mechanism_events={"npc_profile": {"model": "scripted"}},
        )

        payload = recorder.to_json()
        self.assertEqual(payload["format"], SIMULATOR_FULL_REPLAY_FORMAT)
        self.assertEqual(payload["format_version"], SIMULATOR_FULL_REPLAY_VERSION)
        self.assertEqual(payload["source"]["seed"], 123)
        self.assertEqual(payload["source"]["map_id"], 99)
        self.assertEqual(payload["source"]["map_key"], "empty_test_map")
        self.assertEqual(payload["players"], {"player1": "policy_a", "player2": "policy_b"})
        self.assertEqual(len(payload["rounds"]), 1)

        round0 = payload["rounds"][0]
        self.assertEqual(round0["round"], 0)
        self.assertEqual(round0["start"]["round_index"], 0)
        self.assertEqual(round0["end"]["round_index"], 1)
        self.assertEqual(round0["actions"]["players"]["1"]["actions"][0], int(Action.RIGHT))
        self.assertEqual(round0["actions"]["npcs"], {"-1": [int(Action.LEFT)]})
        self.assertEqual(round0["actions"]["npc_order"], [-1])
        self.assertEqual(round0["events"]["dispatch_order"], [1, -1, 2])
        self.assertEqual(round0["events"]["bomb_refresh"], {"cleared": [], "spawned": []})
        self.assertIsNotNone(round0["snapshot"])
        self.assertNotIn("gold", round0["end"]["npcs"][0])

        serialized = dumps_replay(recorder)
        self.assertEqual(json.loads(serialized), payload)

    def test_write_replay_writes_utf8_json(self) -> None:
        recorder = SimulatorReplayRecorder(map_template=_empty_template(), seed=7)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sim_replay.json"
            write_replay(path, recorder, indent=None)
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["format"], SIMULATOR_FULL_REPLAY_FORMAT)
        self.assertEqual(payload["source"]["seed"], 7)

    def test_serialize_state_rejects_invalid_resource_overlap(self) -> None:
        state = _minimal_state()
        state.gold[Position(1, 1)] = 5
        state.bombs.add(Position(1, 1))
        with self.assertRaisesRegex(SimulatorRuleError, "bomb overlaps gold"):
            serialize_state(state)

    def test_serialize_state_rejects_gold_on_obstacle(self) -> None:
        state = _minimal_state()
        state.obstacles = frozenset({Position(2, 2)})
        state.gold[Position(2, 2)] = 5
        with self.assertRaisesRegex(SimulatorRuleError, "gold overlaps obstacle"):
            serialize_state(state)

    def test_record_round_requires_sequential_round_index(self) -> None:
        recorder = SimulatorReplayRecorder(map_template=_empty_template())
        state = _minimal_state()
        state.round_index = 2
        result = transition_one_round(
            state,
            player_outputs={1: _stay_output(), 2: _stay_output()},
            mechanisms=ScriptedMechanisms(),
            first_player_id=1,
        )
        with self.assertRaisesRegex(SimulatorRuleError, "expected round 0"):
            recorder.record_round(
                start_state=state,
                end_state=state,
                player_outputs={1: _stay_output(), 2: _stay_output()},
                transition_result=result,
            )


def _empty_template() -> MapTemplate:
    return MapTemplate.from_static_grid(99, "empty_test_map", [[STATIC_EMPTY for _ in range(17)] for _ in range(17)])


def _minimal_state() -> GameState:
    return GameState(
        round_index=0,
        players={
            1: PlayerState(1, [UnitState(0, Position(0, 0)), UnitState(1, Position(16, 16))]),
            2: PlayerState(2, [UnitState(0, Position(0, 16)), UnitState(1, Position(16, 0))]),
        },
    )


def _transition_state() -> GameState:
    state = _minimal_state()
    state.players[1].units[0].position = Position(5, 5)
    state.players[1].units[1].position = Position(10, 10)
    state.npcs[-1] = NpcState(-1, Position(5, 7))
    return state


def _stay_output() -> GameOutput:
    return GameOutput(actions=(Action.STAY,) * 6, k=3, order=0, vp=0)


if __name__ == "__main__":
    unittest.main()
