from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepConfig, RoundStepEnv
from simulator.types import Action, GameOutput


EXPECTED_DEFAULT_ROLLOUT_DIGEST = "26abf12758c1fa85c26a85c77c6d547af0c85f4bb5bbc2791a4e698d3705b519"


class SimulatorEquivalenceHarnessTests(unittest.TestCase):
    def test_default_mechanisms_rollout_fingerprint_stays_stable(self) -> None:
        payload = []
        for seed in (2026080501, 2026080502):
            for map_id in (1, 2, 3):
                payload.append({"seed": seed, "map_id": map_id, "records": _run_rollout(seed, map_id)})

        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(), EXPECTED_DEFAULT_ROLLOUT_DIGEST)


def _run_rollout(seed: int, map_id: int, rounds: int = 18) -> list[dict[str, Any]]:
    env = RoundStepEnv(
        config=RoundStepConfig(
            EpisodeConfig(
                seed=seed,
                map_id=map_id,
                rules=RulesConfig(round_count=rounds, snapshot_period=5),
            )
        ),
        p90_latency_ns={1: seed, 2: seed + 1},
    )
    observations = env.reset()
    records = [{"reset_obs": _observations_sig(observations), "state": _state_sig(env.state)}]

    for round_index in range(rounds):
        result = env.step(
            {
                1: _player_output(seed, 1, round_index),
                2: _player_output(seed, 2, round_index),
            },
            first_player_id=1 if (seed + round_index) % 2 == 0 else 2,
        )
        trace = result.trace
        records.append(
            {
                "round": round_index,
                "first": trace.first_player_id,
                "npc_order": trace.npc_order,
                "npc_actions": {npc_id: [int(action) for action in actions] for npc_id, actions in sorted(trace.npc_actions.items())},
                "gold_generated": _canon(trace.gold_generated),
                "bomb_refresh": _canon(trace.bomb_refresh_event),
                "dispatch": trace.transition_result.dispatch_order,
                "movement": _canon(trace.transition_result.movement_events),
                "interactions": _canon(trace.transition_result.interaction_events),
                "snapshot": _canon(trace.transition_result.snapshot),
                "state": _state_sig(result.state),
                "obs": _observations_sig(result.observations),
                "terminated": result.terminated,
                "game_result": _canon(result.game_result),
            }
        )
    return records


def _player_output(seed: int, player_id: int, round_index: int) -> GameOutput:
    actions_source = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY)
    value = seed * 31 + player_id * 17 + round_index * 13
    actions = tuple(int(actions_source[(value + index * 7) % len(actions_source)]) for index in range(6))
    return GameOutput(actions=actions, k=value % 7, order=(value // 3) % 2, vp=(value // 5) % 3)


def _state_sig(state: Any) -> dict[str, Any]:
    return {
        "round_index": state.round_index,
        "players": {
            player_id: {
                "units": [[unit.position.row, unit.position.col, unit.gold] for unit in player.units],
                "vision_spent": player.vision_spent,
                "active_vision_radius": player.active_vision_radius,
                "next_vision_radius": player.next_vision_radius,
                "gross_gold": player.gross_gold,
                "net_gold": player.net_gold,
            }
            for player_id, player in sorted(state.players.items())
        },
        "gold": sorted([position.row, position.col, amount] for position, amount in state.gold.items()),
        "bombs": sorted([position.row, position.col] for position in state.bombs),
        "npcs": {npc_id: [npc.position.row, npc.position.col] for npc_id, npc in sorted(state.npcs.items())},
    }


def _observations_sig(observations: dict[int, Any]) -> dict[int, Any]:
    return {
        player_id: {
            "round": game_input.round,
            "grid": game_input.grid,
            "my_units": game_input.my_units,
            "my_units_gold": game_input.my_units_gold,
            "gold_opp": game_input.gold_opp,
            "visible_enemies": game_input.visible_enemies,
            "visible_npcs": [[npc.id, npc.row, npc.col] for npc in game_input.visible_npcs],
            "snapshot_valid": game_input.snapshot_valid,
            "snapshot": _canon(game_input.snapshot),
        }
        for player_id, game_input in sorted(observations.items())
    }


def _canon(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _canon(asdict(value))
    if isinstance(value, dict):
        return {str(key): _canon(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (set, frozenset)):
        return [_canon(item) for item in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [_canon(item) for item in value]
    return value


if __name__ == "__main__":
    unittest.main()
