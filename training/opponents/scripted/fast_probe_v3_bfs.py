from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from simulator.constants import GRID_BOMB, GRID_OBSTACLE, GRID_SIZE, MOVE_BUDGET, STATIC_OBSTACLE
from simulator.mechanisms.maps import MapTemplate, built_in_training_map_pool
from simulator.observation.sdk import GameInput
from simulator.types import Action, GameOutput, Position

from ..base import EpisodeContext
from .common import ACTION_ORDER, CENTER
from .fast_probe_v3_like import (
    FastProbeV3LikeOpponent,
    NO_BLOCK,
    _clamp,
    _compose_output,
    _find_bounce,
    _safe_cell,
    _crowded_npc_at,
    _visible_enemy_at,
    _useful_bounce_pairs,
)


CENTER_FALLBACK_MIN_ROW = 6
CENTER_FALLBACK_MAX_ROW = 10
CENTER_FALLBACK_MIN_COL = 6
CENTER_FALLBACK_MAX_COL = 10
CENTER_FALLBACK_COMMIT_ROUNDS = 8


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

    _bfs_template: MapTemplate | None = None
    _center_fallback_targets: tuple[Position | None, Position | None] = (None, None)
    _center_fallback_until_rounds: tuple[int, int] = (-1, -1)

    def reset(self, seed: int, episode_context: EpisodeContext) -> None:
        super().reset(seed, episode_context)
        self._bfs_template = _template_for_context(episode_context)
        self._center_fallback_targets = (None, None)
        self._center_fallback_until_rounds = (-1, -1)

    def act(self, game_input: GameInput) -> GameOutput:
        if game_input.round == 0:
            self._bad_visible_rounds = 0
            self._current_vision_radius = 2
            self._center_fallback_targets = (None, None)
            self._center_fallback_until_rounds = (-1, -1)

        choice = _choose_bfs_gold_target(
            game_input,
            self._current_vision_radius,
            self.max_target_distance,
        )
        if choice is None:
            role = self._fast_rand(game_input) & 1
            target_gold = 0
            main_plan = self._build_center_fallback_plan(game_input, role, MOVE_BUDGET, None)
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

    def _build_center_actions(self, game_input: GameInput, role: int, max_steps: int, extra_block: Position) -> tuple[int, ...]:
        return self._build_center_fallback_plan(game_input, role, max_steps, extra_block).actions

    def _build_center_fallback_plan(
        self,
        game_input: GameInput,
        role: int,
        max_steps: int,
        extra_block: Position | None,
    ) -> _BfsActionPlan:
        if max_steps <= 0:
            pos = Position(*game_input.my_units[role])
            return _BfsActionPlan((), pos == CENTER, pos)
        if self._bfs_template is None:
            return _greedy_center_fallback_plan(self, game_input, role, max_steps, extra_block)

        target, path = self._center_target_and_path(game_input, role, extra_block)
        if target is None or path is None:
            return _greedy_center_fallback_plan(self, game_input, role, max_steps, extra_block)

        actions = tuple(path[:max_steps])
        pos = Position(*game_input.my_units[role])
        for action in actions:
            pos = pos.moved(action)
        return _BfsActionPlan(actions, pos == target, pos)

    def _center_target_and_path(
        self,
        game_input: GameInput,
        role: int,
        extra_block: Position | None,
    ) -> tuple[Position | None, tuple[int, ...] | None]:
        assert self._bfs_template is not None
        current_target = self._center_fallback_targets[role]
        current_until = self._center_fallback_until_rounds[role]
        current_path = (
            _shortest_static_path(game_input, self._bfs_template, role, current_target, extra_block)
            if current_target is not None and game_input.round <= current_until
            else None
        )
        if current_target is not None and current_path is not None:
            if Position(*game_input.my_units[role]) != current_target:
                return current_target, current_path

        candidates = _reachable_center_targets(game_input, self._bfs_template, role, extra_block)
        if not candidates:
            self._set_center_fallback_target(role, None, -1)
            return None, None

        non_current_candidates = tuple(
            (target, path)
            for target, path in candidates
            if target != Position(*game_input.my_units[role])
        )
        if non_current_candidates:
            candidates = non_current_candidates

        index = self._fast_rand(game_input) % len(candidates)
        target, path = candidates[index]
        self._set_center_fallback_target(role, target, game_input.round + CENTER_FALLBACK_COMMIT_ROUNDS)
        return target, path

    def _set_center_fallback_target(self, role: int, target: Position | None, until_round: int) -> None:
        targets = list(self._center_fallback_targets)
        until_rounds = list(self._center_fallback_until_rounds)
        targets[role] = target
        until_rounds[role] = int(until_round)
        self._center_fallback_targets = (targets[0], targets[1])
        self._center_fallback_until_rounds = (until_rounds[0], until_rounds[1])

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


def _template_for_context(episode_context: EpisodeContext) -> MapTemplate | None:
    pool = built_in_training_map_pool()
    if episode_context.map_key is not None:
        try:
            return pool.get_by_key(episode_context.map_key)
        except KeyError:
            return None
    if episode_context.map_id is None:
        return None
    try:
        return pool.get(int(episode_context.map_id))
    except KeyError:
        return None


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


def _reachable_center_targets(
    game_input: GameInput,
    template: MapTemplate,
    role: int,
    extra_block: Position | None,
) -> tuple[tuple[Position, tuple[int, ...]], ...]:
    center_targets = _center_fallback_candidates(template)
    paths = _run_static_bfs(game_input, template, role, GRID_SIZE * GRID_SIZE, extra_block)
    reachable = [(target, path) for target, path in paths.items() if target in center_targets]
    reachable.sort(
        key=lambda item: (
            len(item[1]),
            abs(item[0].row - CENTER.row) + abs(item[0].col - CENTER.col),
            item[0].row,
            item[0].col,
        )
    )
    return tuple(reachable)


def _center_fallback_candidates(template: MapTemplate) -> frozenset[Position]:
    return frozenset(
        Position(row, col)
        for row in range(CENTER_FALLBACK_MIN_ROW, CENTER_FALLBACK_MAX_ROW + 1)
        for col in range(CENTER_FALLBACK_MIN_COL, CENTER_FALLBACK_MAX_COL + 1)
        if template.static_grid[row][col] != STATIC_OBSTACLE
    )


def _shortest_static_path(
    game_input: GameInput,
    template: MapTemplate,
    role: int,
    target: Position,
    extra_block: Position | None,
) -> tuple[int, ...] | None:
    return _run_static_bfs(game_input, template, role, GRID_SIZE * GRID_SIZE, extra_block).get(target)


def _run_static_bfs(
    game_input: GameInput,
    template: MapTemplate,
    role: int,
    max_distance: int,
    extra_block: Position | None,
) -> dict[Position, tuple[int, ...]]:
    start = Position(*game_input.my_units[role])
    if not start.in_bounds():
        return {}

    paths: dict[Position, tuple[int, ...]] = {start: ()}
    queue: deque[Position] = deque([start])
    while queue:
        pos = queue.popleft()
        path = paths[pos]
        if len(path) >= max_distance:
            continue
        for action in ACTION_ORDER:
            next_pos = pos.moved(action)
            if next_pos in paths:
                continue
            if not _static_safe_cell(game_input, template, role, next_pos, extra_block):
                continue
            paths[next_pos] = path + (int(action),)
            queue.append(next_pos)
    return paths


def _static_safe_cell(
    game_input: GameInput,
    template: MapTemplate,
    role: int,
    pos: Position,
    extra_block: Position | None,
) -> bool:
    if not pos.in_bounds():
        return False
    if template.static_grid[pos.row][pos.col] == STATIC_OBSTACLE:
        return False
    if extra_block is not None and pos == extra_block:
        return False
    if game_input.grid[pos.row][pos.col] in (GRID_BOMB, GRID_OBSTACLE):
        return False
    other = Position(*game_input.my_units[1 - role])
    if pos == other:
        return False
    if _visible_enemy_at(game_input, pos):
        return False
    return not _crowded_npc_at(game_input, pos)


def _greedy_center_fallback_plan(
    opponent: FastProbeV3LikeOpponent,
    game_input: GameInput,
    role: int,
    max_steps: int,
    extra_block: Position | None,
) -> _BfsActionPlan:
    actions = FastProbeV3LikeOpponent._build_center_actions(
        opponent,
        game_input,
        role,
        max_steps,
        NO_BLOCK if extra_block is None else extra_block,
    )
    pos = Position(*game_input.my_units[role])
    for action in actions:
        pos = pos.moved(action)
    return _BfsActionPlan(actions, pos == CENTER, pos)


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
