from __future__ import annotations

from dataclasses import dataclass

from simulator.constants import GRID_BOMB, GRID_OBSTACLE, GRID_SIZE, MOVE_BUDGET, STATIC_OBSTACLE
from simulator.mechanisms.gold import OUTER_REGIONS, outer_static2_candidate_cells
from simulator.mechanisms.maps import MapTemplate, built_in_training_map_pool
from simulator.observation.sdk import GameInput
from simulator.types import GameOutput, Position

from ..base import EpisodeContext
from .common import ACTION_ORDER
from .fast_probe_v3_bfs import (
    FastProbeV3BfsOpponent,
    _BfsActionPlan,
    _BfsTargetChoice,
    _choose_bfs_gold_target,
)
from .fast_probe_v3_like import _compose_output, _find_bounce, _useful_bounce_pairs

VISIBLE_GOLD_OVERRIDE = 12
SNAPSHOT_REMAINING_TRIGGER = 60
SNAPSHOT_GENERATED_TRIGGER = 70
COMMIT_ROUNDS = 10


@dataclass
class FastProbeOuterStatic2MapAwareOpponent(FastProbeV3BfsOpponent):
    """Map-aware outer static2 responder for training opponent leagues."""

    _template: MapTemplate | None = None
    _commit_region: int | None = None
    _commit_until_round: int = -1

    def reset(self, seed: int, episode_context: EpisodeContext) -> None:
        super().reset(seed, episode_context)
        self._template = _template_for_map_id(episode_context.map_id)
        self._commit_region = None
        self._commit_until_round = -1

    def act(self, game_input: GameInput) -> GameOutput:
        if game_input.round == 0:
            self._bad_visible_rounds = 0
            self._current_vision_radius = 2
            self._commit_region = None
            self._commit_until_round = -1

        visible_choice = _choose_bfs_gold_target(
            game_input,
            self._current_vision_radius,
            self.max_target_distance,
        )
        if visible_choice is not None and visible_choice.gold >= VISIBLE_GOLD_OVERRIDE:
            return self._act_on_bfs_choice(game_input, visible_choice)

        snapshot_region = _snapshot_outer_region(game_input)
        if snapshot_region is not None:
            self._commit_region = snapshot_region
            self._commit_until_round = game_input.round + COMMIT_ROUNDS

        if self._template is not None and self._commit_region is not None:
            if game_input.round <= self._commit_until_round:
                static2_choice = _choose_static2_target(
                    game_input,
                    self._template,
                    self._commit_region,
                )
                if static2_choice is not None:
                    return self._act_on_static2_choice(game_input, static2_choice)
            else:
                self._commit_region = None

        return super().act(game_input)

    def _act_on_bfs_choice(self, game_input: GameInput, choice: _BfsTargetChoice) -> GameOutput:
        main_plan = self._build_bfs_main_actions(game_input, choice)
        return self._compose_with_v3_vision(
            game_input,
            main_role=choice.role,
            main_plan=main_plan,
            target_gold=choice.gold,
            choice_was_gold=True,
        )

    def _act_on_static2_choice(self, game_input: GameInput, choice: _BfsTargetChoice) -> GameOutput:
        main_plan = _build_static2_main_actions(game_input, choice)
        reached_static2_target = main_plan.reached
        if reached_static2_target:
            self._commit_region = None
            self._commit_until_round = -1
        return self._compose_with_v3_vision(
            game_input,
            main_role=choice.role,
            main_plan=main_plan,
            target_gold=choice.gold,
            choice_was_gold=choice.gold > 0,
            force_vp2=reached_static2_target,
        )

    def _compose_with_v3_vision(
        self,
        game_input: GameInput,
        *,
        main_role: int,
        main_plan: _BfsActionPlan,
        target_gold: int,
        choice_was_gold: bool,
        force_vp2: bool = False,
    ) -> GameOutput:
        side_role = 1 - main_role
        spare_steps = MOVE_BUDGET - len(main_plan.actions)
        side_actions = (
            self._build_center_actions(game_input, side_role, spare_steps, main_plan.end)
            if spare_steps > 0
            else ()
        )
        output = _compose_output(main_role, main_plan.actions, side_actions)

        good_gold_seen = (
            choice_was_gold
            and target_gold >= self.good_gold_threshold
            and main_plan.reached
        )
        if good_gold_seen:
            self._bad_visible_rounds = 0
        else:
            self._bad_visible_rounds += 1

        if force_vp2:
            vp = 2
            self._bad_visible_rounds = 0
        elif self.enable_vision and self._bad_visible_rounds >= self.bad_rounds_before_vp:
            vp = 2
            self._bad_visible_rounds = 0
        else:
            vp = 0
        self._current_vision_radius = 4 if vp == 2 else 2
        return GameOutput(actions=output.actions, k=output.k, order=output.order, vp=vp)


def _template_for_map_id(map_id: int | None) -> MapTemplate | None:
    if map_id is None:
        return None
    try:
        return built_in_training_map_pool().get(int(map_id))
    except KeyError:
        return None


def _snapshot_outer_region(game_input: GameInput) -> int | None:
    if not game_input.snapshot_valid or game_input.snapshot is None:
        return None

    best_region: int | None = None
    best_key: tuple[int, int, int] | None = None
    for stat in game_input.snapshot.regions:
        if stat.id not in OUTER_REGIONS:
            continue
        if stat.gold_remaining < SNAPSHOT_REMAINING_TRIGGER and stat.gold_generated < SNAPSHOT_GENERATED_TRIGGER:
            continue
        key = (int(stat.gold_remaining), int(stat.gold_generated), -int(stat.id))
        if best_key is None or key > best_key:
            best_key = key
            best_region = int(stat.id)
    return best_region


def _choose_static2_target(
    game_input: GameInput,
    template: MapTemplate,
    region: int,
) -> _BfsTargetChoice | None:
    candidates = outer_static2_candidate_cells(template, region)
    reachability = tuple(_run_map_bfs(game_input, template, role) for role in range(2))
    best_choice: _BfsTargetChoice | None = None
    best_key: tuple[int, int, int, int] | None = None

    for target in candidates:
        gold = _visible_gold_at(game_input, target)
        for role, paths in enumerate(reachability):
            path = paths.get(target)
            if path is None:
                continue
            key = (-len(path), gold, -role, -target.row * GRID_SIZE - target.col)
            if best_key is None or key > best_key:
                best_key = key
                best_choice = _BfsTargetChoice(role=role, target=target, gold=gold, path_actions=path)
    return best_choice


def _run_map_bfs(
    game_input: GameInput,
    template: MapTemplate,
    role: int,
) -> dict[Position, tuple[int, ...]]:
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
        if len(path) >= MOVE_BUDGET:
            continue
        for action in ACTION_ORDER:
            next_pos = pos.moved(action)
            if next_pos in paths:
                continue
            if not _map_safe_cell(game_input, template, role, next_pos):
                continue
            paths[next_pos] = path + (int(action),)
            queue.append(next_pos)
    return paths


def _map_safe_cell(
    game_input: GameInput,
    template: MapTemplate,
    role: int,
    pos: Position,
) -> bool:
    if not pos.in_bounds():
        return False
    if template.static_grid[pos.row][pos.col] == STATIC_OBSTACLE:
        return False
    cell = game_input.grid[pos.row][pos.col]
    if cell in (GRID_OBSTACLE, GRID_BOMB):
        return False
    if pos == Position(*game_input.my_units[1 - role]):
        return False
    if any(Position(row, col) == pos for row, col in game_input.visible_enemies if Position(row, col).in_bounds()):
        return False
    return not _crowded_npc_at(game_input, pos)


def _crowded_npc_at(game_input: GameInput, pos: Position) -> bool:
    if len(game_input.visible_npcs) < 3:
        return False
    count = 0
    for npc in game_input.visible_npcs:
        npc_pos = Position(npc.row, npc.col)
        if npc_pos.in_bounds() and npc_pos == pos:
            count += 1
            if count >= 3:
                return True
    return False


def _visible_gold_at(game_input: GameInput, pos: Position) -> int:
    value = game_input.grid[pos.row][pos.col]
    return int(value) if value > 0 else 0


def _build_static2_main_actions(game_input: GameInput, choice: _BfsTargetChoice) -> _BfsActionPlan:
    actions = list(choice.path_actions[:MOVE_BUDGET])
    pos = Position(*game_input.my_units[choice.role])
    for action in actions:
        pos = pos.moved(action)

    reached = pos == choice.target
    entered_by_move = reached and len(actions) > 0
    if reached and choice.gold > 0:
        max_bounces = _useful_bounce_pairs(choice.gold, entered_by_move)
        bounce = _find_bounce(game_input, choice.role, choice.target) if max_bounces > 0 else None
        used_bounces = 0
        while bounce is not None and len(actions) + 1 < MOVE_BUDGET and used_bounces < max_bounces:
            actions.append(int(bounce))
            actions.append(_opposite_action(int(bounce)))
            used_bounces += 1
    return _BfsActionPlan(tuple(actions), reached, pos)


def _opposite_action(action: int) -> int:
    if action == 0:
        return 1
    if action == 1:
        return 0
    if action == 2:
        return 3
    if action == 3:
        return 2
    raise ValueError(f"action has no opposite: {action}")
