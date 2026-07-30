from __future__ import annotations

import unittest

from simulator.errors import SimulatorRuleError
from simulator.rules.movement import apply_npc_step, apply_player_step, apply_player_turn
from simulator.state import GameState, NpcState, PlayerState, UnitState
from simulator.types import Action, GameOutput, MoveStatus, PlayerUnitRef, Position


class MovementTests(unittest.TestCase):
    def test_out_of_bounds_step_is_ignored(self) -> None:
        state = _state(p1=(Position(0, 0), Position(5, 5)))

        event = apply_player_step(state, 1, 0, Action.UP)

        self.assertEqual(event.status, MoveStatus.BLOCKED_OUT_OF_BOUNDS)
        self.assertEqual(state.player_unit(1, 0).position, Position(0, 0))

    def test_obstacle_step_is_ignored(self) -> None:
        state = _state(p1=(Position(1, 1), Position(5, 5)), obstacles={Position(1, 2)})

        event = apply_player_step(state, 1, 0, Action.RIGHT)

        self.assertEqual(event.status, MoveStatus.BLOCKED_OBSTACLE)
        self.assertEqual(state.player_unit(1, 0).position, Position(1, 1))

    def test_enemy_player_blocks_movement(self) -> None:
        state = _state(p1=(Position(1, 1), Position(5, 5)), p2=(Position(1, 2), Position(16, 0)))

        event = apply_player_step(state, 1, 0, Action.RIGHT)

        self.assertEqual(event.status, MoveStatus.BLOCKED_PLAYER)
        self.assertEqual(event.blocked_by, PlayerUnitRef(2, 0))
        self.assertEqual(state.player_unit(1, 0).position, Position(1, 1))

    def test_npc_does_not_block_player_movement(self) -> None:
        state = _state(p1=(Position(1, 1), Position(5, 5)), npcs={-1: Position(1, 2)})

        event = apply_player_step(state, 1, 0, Action.RIGHT)

        self.assertEqual(event.status, MoveStatus.MOVED)
        self.assertEqual(state.player_unit(1, 0).position, Position(1, 2))

    def test_teammate_current_position_blocks_movement(self) -> None:
        state = _state(p1=(Position(1, 1), Position(1, 2)))

        event = apply_player_step(state, 1, 0, Action.RIGHT)

        self.assertEqual(event.status, MoveStatus.BLOCKED_PLAYER)
        self.assertEqual(event.blocked_by, PlayerUnitRef(1, 1))
        self.assertEqual(state.player_unit(1, 0).position, Position(1, 1))

    def test_unit_can_enter_teammate_vacated_cell_later_in_turn(self) -> None:
        state = _state(p1=(Position(1, 1), Position(1, 2)))
        output = GameOutput(actions=(0, 2, 4, 4, 4, 4), k=1, order=0, vp=0)

        events = apply_player_turn(state, 1, output)

        self.assertEqual([event.status for event in events[:2]], [MoveStatus.MOVED, MoveStatus.MOVED])
        self.assertEqual(state.player_unit(1, 1).position, Position(1, 1))
        self.assertEqual(state.player_unit(1, 0).position, Position(0, 1))

    def test_teammates_cannot_directly_swap_positions(self) -> None:
        state = _state(p1=(Position(1, 1), Position(1, 2)))
        output = GameOutput(actions=(3, 2, 4, 4, 4, 4), k=1, order=0, vp=0)

        events = apply_player_turn(state, 1, output)

        self.assertEqual(events[0].status, MoveStatus.BLOCKED_PLAYER)
        self.assertEqual(events[1].status, MoveStatus.BLOCKED_PLAYER)
        self.assertEqual(state.player_unit(1, 0).position, Position(1, 1))
        self.assertEqual(state.player_unit(1, 1).position, Position(1, 2))

    def test_npc_valid_step_can_overlap_player(self) -> None:
        state = _state(p1=(Position(1, 2), Position(5, 5)), npcs={-1: Position(1, 1)})

        event = apply_npc_step(state, -1, Action.RIGHT)

        self.assertEqual(event.status, MoveStatus.MOVED)
        self.assertEqual(state.npcs[-1].position, Position(1, 2))

    def test_npc_out_of_bounds_attempt_fails_fast(self) -> None:
        state = _state(p1=(Position(5, 5), Position(6, 6)), npcs={-1: Position(0, 0)})

        with self.assertRaises(SimulatorRuleError):
            apply_npc_step(state, -1, Action.UP)

        self.assertEqual(state.npcs[-1].position, Position(0, 0))

    def test_npc_obstacle_attempt_fails_fast(self) -> None:
        state = _state(
            p1=(Position(5, 5), Position(6, 6)),
            obstacles={Position(1, 2)},
            npcs={-1: Position(1, 1)},
        )

        with self.assertRaises(SimulatorRuleError):
            apply_npc_step(state, -1, Action.RIGHT)

        self.assertEqual(state.npcs[-1].position, Position(1, 1))


def _state(
    *,
    p1: tuple[Position, Position],
    p2: tuple[Position, Position] = (Position(0, 16), Position(16, 0)),
    obstacles: set[Position] | None = None,
    npcs: dict[int, Position] | None = None,
) -> GameState:
    return GameState(
        round_index=0,
        obstacles=frozenset(obstacles or set()),
        players={
            1: PlayerState(1, [UnitState(0, p1[0]), UnitState(1, p1[1])]),
            2: PlayerState(2, [UnitState(0, p2[0]), UnitState(1, p2[1])]),
        },
        npcs={npc_id: NpcState(npc_id, position) for npc_id, position in (npcs or {}).items()},
    )


if __name__ == "__main__":
    unittest.main()
