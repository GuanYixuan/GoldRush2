from __future__ import annotations

from dataclasses import dataclass

from simulator.constants import MOVE_BUDGET
from simulator.observation.sdk import GameInput
from simulator.types import Action, GameOutput, Position

from .common import ACTION_ORDER, CENTER
from .fast_probe_v3_like import (
    FastProbeV3LikeOpponent,
    _clamp,
    _compose_output,
    _find_bounce,
    _safe_cell,
    _useful_bounce_pairs,
)


@dataclass(frozen=True)
class _BfsTargetChoice:
    role: int
    target: Position
    gold: int
    path_actions: tuple[int, ...]


@dataclass(frozen=True)
class _BfsActionPlan:
    actions: tuple[int, ...]
    reached: bool
    end: Position


@dataclass
class FastProbeV3BfsOpponent(FastProbeV3LikeOpponent):
    """fast_probe_v3_like with BFS-reachable gold target selection and BFS paths."""

    def act(self, game_input: GameInput) -> GameOutput:
        if game_input.round == 0:
            self._bad_visible_rounds = 0
            self._current_vision_radius = 2

        choice = _choose_bfs_gold_target(
            game_input,
            self._current_vision_radius,
            self.max_target_distance,
        )
        if choice is None:
            role = self._fast_rand(game_input) & 1
            target = CENTER
            target_gold = 0
            main_plan = self._build_main_actions(game_input, role, target, False, target_gold)
        else:
            role = choice.role
            target_gold = choice.gold
            main_plan = self._build_bfs_main_actions(game_input, choice)

        side_role = 1 - role
        spare_steps = MOVE_BUDGET - len(main_plan.actions)
        side_actions = (
            self._build_center_actions(game_input, side_role, spare_steps, main_plan.end)
            if spare_steps > 0
            else ()
        )
        output = _compose_output(role, main_plan.actions, side_actions)

        good_gold_seen = (
            choice is not None
            and target_gold >= self.good_gold_threshold
            and main_plan.reached
        )
        if good_gold_seen:
            self._bad_visible_rounds = 0
        else:
            self._bad_visible_rounds += 1

        vp = 0
        if self.enable_vision and self._bad_visible_rounds >= self.bad_rounds_before_vp:
            vp = 2
            self._bad_visible_rounds = 0
        self._current_vision_radius = 4 if vp == 2 else 2
        return GameOutput(actions=output.actions, k=output.k, order=output.order, vp=vp)

    def _build_bfs_main_actions(
        self,
        game_input: GameInput,
        choice: _BfsTargetChoice,
    ) -> _BfsActionPlan:
        actions = list(choice.path_actions[:MOVE_BUDGET])
        pos = Position(*game_input.my_units[choice.role])
        for action in actions:
            pos = pos.moved(action)

        reached = pos == choice.target
        entered_by_move = reached and len(actions) > 0
        if reached:
            max_bounces = _useful_bounce_pairs(choice.gold, entered_by_move)
            bounce = (
                _find_bounce(game_input, choice.role, choice.target)
                if max_bounces > 0
                else None
            )
            used_bounces = 0
            while (
                bounce is not None
                and len(actions) + 1 < MOVE_BUDGET
                and used_bounces < max_bounces
            ):
                actions.append(int(bounce))
                actions.append(_opposite_action(int(bounce)))
                used_bounces += 1

        return _BfsActionPlan(tuple(actions), reached, pos)


def _choose_bfs_gold_target(
    game_input: GameInput,
    vision_radius: int,
    max_distance: int,
) -> _BfsTargetChoice | None:
    reachability = tuple(_run_bfs(game_input, role, max_distance) for role in range(2))
    best_choice: _BfsTargetChoice | None = None
    best_key: tuple[int, int, int, int, int] | None = None

    for target, gold in _visible_gold_in_windows(game_input, vision_radius):
        role_choice: tuple[int, int, tuple[int, ...]] | None = None
        for role, bfs in enumerate(reachability):
            path = bfs.get(target)
            if path is None:
                continue
            candidate = (len(path), role, path)
            if role_choice is None or candidate[:2] < role_choice[:2]:
                role_choice = candidate
        if role_choice is None:
            continue

        distance, role, path_actions = role_choice
        key = (gold, -distance, -role, -target.row, -target.col)
        if best_key is None or key > best_key:
            best_key = key
            best_choice = _BfsTargetChoice(
                role=role,
                target=target,
                gold=gold,
                path_actions=path_actions,
            )

    return best_choice


def _run_bfs(game_input: GameInput, role: int, max_distance: int) -> dict[Position, tuple[int, ...]]:
    start = Position(*game_input.my_units[role])
    if not start.in_bounds():
        return {}

    paths: dict[Position, tuple[int, ...]] = {start: ()}
    queue: list[Position] = [start]
    head = 0
    while head < len(queue):
        pos = queue[head]
        head += 1
        path = paths[pos]
        if len(path) >= max_distance:
            continue

        for action in ACTION_ORDER:
            next_pos = pos.moved(action)
            if next_pos in paths:
                continue
            if not _safe_cell(game_input, role, next_pos):
                continue
            paths[next_pos] = path + (int(action),)
            queue.append(next_pos)
    return paths


def _visible_gold_in_windows(
    game_input: GameInput,
    vision_radius: int,
) -> tuple[tuple[Position, int], ...]:
    targets: dict[Position, int] = {}
    for row, col in game_input.my_units:
        r0 = _clamp(row - vision_radius)
        r1 = _clamp(row + vision_radius)
        c0 = _clamp(col - vision_radius)
        c1 = _clamp(col + vision_radius)
        for target_row in range(r0, r1 + 1):
            for target_col in range(c0, c1 + 1):
                gold = game_input.grid[target_row][target_col]
                if gold > 0:
                    targets[Position(target_row, target_col)] = gold
    return tuple(targets.items())


def _opposite_action(action: int) -> int:
    if action == int(Action.UP):
        return int(Action.DOWN)
    if action == int(Action.DOWN):
        return int(Action.UP)
    if action == int(Action.LEFT):
        return int(Action.RIGHT)
    if action == int(Action.RIGHT):
        return int(Action.LEFT)
    raise ValueError(f"action has no opposite: {action}")
