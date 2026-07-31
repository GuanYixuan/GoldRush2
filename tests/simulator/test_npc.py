from __future__ import annotations

import random
import unittest

from simulator.errors import SimulatorRuleError
from simulator.mechanisms.maps import MapTemplate
from simulator.mechanisms.npc import (
    M4aNpcPolicy,
    M4aWeights,
    NpcEpisodeProfile,
    NpcPath,
    NpcPolicyConfig,
    enumerate_legal_paths,
    path_features,
)
from simulator.state import GameState, NpcState, PlayerState, UnitState
from simulator.types import Action, Position


class NpcPolicyTests(unittest.TestCase):
    def test_enumerate_legal_paths_avoids_bounds_and_static_obstacles(self) -> None:
        template = _template(obstacles={Position(1, 2)})

        paths = enumerate_legal_paths(template, Position(1, 1))

        self.assertTrue(paths)
        self.assertTrue(all(len(path.actions) == 3 for path in paths))
        self.assertTrue(all(all(pos.in_bounds() for pos in path.positions) for path in paths))
        self.assertTrue(all(Position(1, 2) not in path.positions for path in paths))

    def test_enumerate_from_invalid_start_fails_fast(self) -> None:
        with self.assertRaises(SimulatorRuleError):
            enumerate_legal_paths(_template(), Position(-1, 0))
        with self.assertRaises(SimulatorRuleError):
            enumerate_legal_paths(_template(obstacles={Position(1, 1)}), Position(1, 1))

    def test_path_features_use_local_dynamic_pickup_and_bomb_state(self) -> None:
        state = _state(npcs={-1: Position(1, 1)}, gold={Position(1, 2): 10}, bombs={Position(1, 2)})
        path = NpcPath(
            actions=(Action.RIGHT, Action.LEFT, Action.RIGHT),
            positions=(Position(1, 2), Position(1, 1), Position(1, 2)),
        )

        features = path_features(state, Position(1, 1), path)

        self.assertEqual(features["dynamic_reward_div10"], 0.9)
        self.assertEqual(features["enter_bomb_count"], 1.0)
        self.assertEqual(features["backtrack"], 1.0)
        self.assertEqual(state.gold[Position(1, 2)], 10)
        self.assertEqual(state.bombs, {Position(1, 2)})

    def test_stay_does_not_pick_up_gold_underfoot(self) -> None:
        state = _state(npcs={-1: Position(1, 1)}, gold={Position(1, 1): 10})
        path = NpcPath(
            actions=(Action.STAY, Action.STAY, Action.STAY),
            positions=(Position(1, 1), Position(1, 1), Position(1, 1)),
        )

        features = path_features(state, Position(1, 1), path)

        self.assertEqual(features["dynamic_reward_div10"], 0.0)
        self.assertEqual(features["stay_count"], 3.0)

    def test_bomb_blindness_removes_bomb_risk_features_only(self) -> None:
        state = _state(
            npcs={-1: Position(1, 1)},
            bombs={Position(0, 1), Position(2, 1), Position(1, 0), Position(1, 2)},
        )
        stay_path = NpcPath(
            actions=(Action.STAY, Action.STAY, Action.STAY),
            positions=(Position(1, 1), Position(1, 1), Position(1, 1)),
        )
        bomb_path = NpcPath(
            actions=(Action.RIGHT, Action.LEFT, Action.RIGHT),
            positions=(Position(1, 2), Position(1, 1), Position(1, 2)),
        )

        self.assertEqual(path_features(state, Position(1, 1), stay_path)["bomb_trapped_stay"], 1.0)
        self.assertEqual(path_features(state, Position(1, 1), stay_path, bombs_visible=False)["bomb_trapped_stay"], 0.0)
        self.assertEqual(path_features(state, Position(1, 1), bomb_path)["enter_bomb_count"], 1.0)
        self.assertEqual(path_features(state, Position(1, 1), bomb_path, bombs_visible=False)["enter_bomb_count"], 0.0)

    def test_sample_order_is_seeded_permutation(self) -> None:
        state = _state(npcs={npc_id: Position(8, 8) for npc_id in range(-1, -8, -1)})
        policy = M4aNpcPolicy()

        order1 = policy.sample_order(state, random.Random(20260730))
        order2 = policy.sample_order(state, random.Random(20260730))

        self.assertEqual(order1, order2)
        self.assertEqual(set(order1), set(state.npcs))
        self.assertNotEqual(order1, tuple(sorted(state.npcs)))

    def test_decide_all_uses_same_snapshot_for_all_npcs_and_does_not_mutate_state(self) -> None:
        template = _template()
        state = _state(
            npcs={-1: Position(1, 1), -2: Position(1, 1)},
            gold={Position(1, 2): 10},
        )
        profile = NpcEpisodeProfile(
            weights=M4aWeights(gold=1000.0, enter_bomb=0.0, stay=0.0, straight3=0.0, backtrack=0.0, bomb_trapped_stay=0.0),
            temperature=1.0,
            bomb_blind_p=0.0,
        )
        policy = M4aNpcPolicy()

        decisions = policy.decide_all(state, template, (-1, -2), random.Random(1), profile)

        self.assertEqual(_reward_for_actions(state, Position(1, 1), decisions[-1]), 0.9)
        self.assertEqual(_reward_for_actions(state, Position(1, 1), decisions[-2]), 0.9)
        self.assertEqual(state.npcs[-1].position, Position(1, 1))
        self.assertEqual(state.gold[Position(1, 2)], 10)

    def test_invalid_config_fails_fast(self) -> None:
        with self.assertRaises(SimulatorRuleError):
            NpcPolicyConfig(temperature=0.0)
        with self.assertRaises(SimulatorRuleError):
            NpcPolicyConfig(bomb_blind_p=-0.1)
        with self.assertRaises(SimulatorRuleError):
            NpcPolicyConfig(npc_jitter_scale=-0.1)


def _template(obstacles: set[Position] | None = None) -> MapTemplate:
    grid = [[0 for _ in range(17)] for _ in range(17)]
    for pos in obstacles or set():
        grid[pos.row][pos.col] = 1
    return MapTemplate.from_static_grid(99, "test_map", grid)


def _state(
    *,
    npcs: dict[int, Position],
    gold: dict[Position, int] | None = None,
    bombs: set[Position] | None = None,
    obstacles: set[Position] | None = None,
) -> GameState:
    return GameState(
        round_index=0,
        players={
            1: PlayerState(1, [UnitState(0, Position(0, 0)), UnitState(1, Position(16, 16))]),
            2: PlayerState(2, [UnitState(0, Position(0, 16)), UnitState(1, Position(16, 0))]),
        },
        obstacles=frozenset(obstacles or set()),
        gold=dict(gold or {}),
        bombs=set(bombs or set()),
        npcs={npc_id: NpcState(npc_id, position) for npc_id, position in npcs.items()},
    )


def _reward_for_actions(state: GameState, start: Position, actions: tuple[Action, Action, Action]) -> float:
    positions = []
    position = start
    for action in actions:
        position = position.moved(action)
        positions.append(position)
    return path_features(state, start, NpcPath(actions=actions, positions=(positions[0], positions[1], positions[2])))["dynamic_reward_div10"]


if __name__ == "__main__":
    unittest.main()
