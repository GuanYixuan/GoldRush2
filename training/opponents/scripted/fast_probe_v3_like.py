from __future__ import annotations

from dataclasses import dataclass

from simulator.constants import GRID_BOMB, GRID_FOG, GRID_OBSTACLE, GRID_SIZE, MOVE_BUDGET
from simulator.observation.sdk import GameInput
from simulator.types import Action, GameOutput, Position

from ..base import EpisodeContext
from .common import CENTER, DC, DR, OPPOSITE, manhattan

NO_BLOCK = Position(-1, -1)


@dataclass(frozen=True)
class _ActionPlan:
    actions: tuple[int, ...]
    reached: bool
    end: Position


@dataclass(frozen=True)
class _TargetChoice:
    role: int
    target: Position
    gold: int


@dataclass
class FastProbeV3LikeOpponent:
    """Python 复刻 fast_probing/v3 的轻量 scripted opponent。

    该策略用于 opponent league，不追求逐位复现 C++ 全局状态，只保留其可见金币选择、
    贪心移动、剩余步数向中心和连续坏轮买视野的行为骨架。
    """

    bad_rounds_before_vp: int = 2
    enable_vision: bool = True
    max_target_distance: int = MOVE_BUDGET
    good_gold_threshold: int = 2
    _rng_state: int = 0x9E3779B9
    _bad_visible_rounds: int = 0
    _current_vision_radius: int = 2

    def __post_init__(self) -> None:
        if self.bad_rounds_before_vp <= 0:
            raise ValueError(f"bad_rounds_before_vp must be > 0, got {self.bad_rounds_before_vp}")
        if not 0 <= self.max_target_distance <= MOVE_BUDGET:
            raise ValueError(f"max_target_distance must be in [0, {MOVE_BUDGET}], got {self.max_target_distance}")
        if self.good_gold_threshold <= 0:
            raise ValueError(f"good_gold_threshold must be > 0, got {self.good_gold_threshold}")

    def reset(self, seed: int, episode_context: EpisodeContext) -> None:
        del episode_context
        self._rng_state = (0x9E3779B9 ^ int(seed)) & 0xFFFFFFFF
        self._bad_visible_rounds = 0
        self._current_vision_radius = 2

    def act(self, game_input: GameInput) -> GameOutput:
        if game_input.round == 0:
            self._bad_visible_rounds = 0
            self._current_vision_radius = 2

        choice = _choose_gold_target(game_input, self._current_vision_radius, self.max_target_distance)
        enable_bounce = choice is not None
        if choice is None:
            role = self._fast_rand(game_input) & 1
            target = CENTER
            target_gold = 0
        else:
            role = choice.role
            target = choice.target
            target_gold = choice.gold

        main_plan = self._build_main_actions(game_input, role, target, enable_bounce, target_gold)
        side_role = 1 - role
        spare_steps = MOVE_BUDGET - len(main_plan.actions)
        side_actions = self._build_center_actions(game_input, side_role, spare_steps, main_plan.end) if spare_steps > 0 else ()
        output = _compose_output(role, main_plan.actions, side_actions)

        good_gold_seen = choice is not None and target_gold >= self.good_gold_threshold and main_plan.reached
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

    def __call__(self, game_input: GameInput) -> GameOutput:
        return self.act(game_input)

    def _build_main_actions(self, game_input: GameInput, role: int, target: Position, enable_bounce: bool, target_gold: int) -> _ActionPlan:
        pos = Position(*game_input.my_units[role])
        actions: list[int] = []
        rnd = self._fast_rand(game_input)
        reached = False
        entered_by_move = False

        while len(actions) < MOVE_BUDGET:
            if pos == target:
                if not enable_bounce:
                    break
                max_bounces = _useful_bounce_pairs(target_gold, entered_by_move)
                if max_bounces <= 0:
                    break
                bounce = _find_bounce(game_input, role, target)
                if bounce is None:
                    break
                reached = True
                used_bounces = 0
                while len(actions) + 1 < MOVE_BUDGET and used_bounces < max_bounces:
                    actions.append(int(bounce))
                    actions.append(int(OPPOSITE[int(bounce)]))
                    used_bounces += 1
                break

            action = _choose_step_toward(game_input, role, pos, target, rnd + len(actions))
            if action == Action.STAY:
                break
            actions.append(int(action))
            pos = pos.moved(action)
            if pos == target:
                reached = True
                entered_by_move = True

        return _ActionPlan(tuple(actions), reached, pos)

    def _build_center_actions(self, game_input: GameInput, role: int, max_steps: int, extra_block: Position) -> tuple[int, ...]:
        pos = Position(*game_input.my_units[role])
        rnd = self._fast_rand(game_input)
        actions: list[int] = []
        while len(actions) < max_steps:
            action = _choose_step_toward(game_input, role, pos, CENTER, rnd + len(actions), extra_block)
            if action == Action.STAY:
                break
            actions.append(int(action))
            pos = pos.moved(action)
        return tuple(actions)

    def _fast_rand(self, game_input: GameInput) -> int:
        u0 = Position(*game_input.my_units[0])
        u1 = Position(*game_input.my_units[1])
        mix = ((game_input.round + 1) * 0x85EBCA6B) & 0xFFFFFFFF
        mix ^= ((u0.row + 1) * 31 + u0.col) & 0xFFFFFFFF
        mix ^= (((u1.row + 1) * 131 + u1.col) << 1) & 0xFFFFFFFF
        state = self._rng_state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        self._rng_state = state & 0xFFFFFFFF
        return (self._rng_state ^ mix) & 0xFFFFFFFF


def _compose_output(main_role: int, main_actions: tuple[int, ...], side_actions: tuple[int, ...]) -> GameOutput:
    actions = [int(Action.STAY)] * MOVE_BUDGET
    if main_role == 0:
        for idx, action in enumerate(main_actions[:MOVE_BUDGET]):
            actions[idx] = action
        for idx, action in enumerate(side_actions[: MOVE_BUDGET - len(main_actions)]):
            actions[len(main_actions) + idx] = action
        return GameOutput(actions=tuple(actions), k=len(main_actions), order=0, vp=0)

    for idx, action in enumerate(side_actions[:MOVE_BUDGET]):
        actions[idx] = action
    for idx, action in enumerate(main_actions[: MOVE_BUDGET - len(side_actions)]):
        actions[len(side_actions) + idx] = action
    return GameOutput(actions=tuple(actions), k=len(side_actions), order=1, vp=0)


def _choose_gold_target(game_input: GameInput, vision_radius: int, max_distance: int) -> _TargetChoice | None:
    best_gold = -1
    best_distance = 1_000_000
    best_choice: _TargetChoice | None = None

    for unit in (Position(*game_input.my_units[0]), Position(*game_input.my_units[1])):
        r0 = _clamp(unit.row - vision_radius)
        r1 = _clamp(unit.row + vision_radius)
        c0 = _clamp(unit.col - vision_radius)
        c1 = _clamp(unit.col + vision_radius)
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                gold = game_input.grid[row][col]
                if gold <= 0:
                    continue
                target = Position(row, col)
                d0 = manhattan(Position(*game_input.my_units[0]), target)
                d1 = manhattan(Position(*game_input.my_units[1]), target)
                role = 1 if d1 < d0 else 0
                distance = min(d0, d1)
                if distance > max_distance:
                    continue
                if gold > best_gold or (gold == best_gold and distance < best_distance):
                    best_gold = gold
                    best_distance = distance
                    best_choice = _TargetChoice(role=role, target=target, gold=gold)
    return best_choice


def _choose_step_toward(
    game_input: GameInput,
    role: int,
    pos: Position,
    target: Position,
    rnd: int,
    extra_block: Position = NO_BLOCK,
) -> Action:
    current_distance = manhattan(pos, target)
    fallback_action = Action.STAY

    for idx in range(4):
        action = Action((rnd + idx) & 3)
        next_pos = pos.moved(action)
        if not _safe_cell(game_input, role, next_pos, extra_block):
            continue
        if fallback_action == Action.STAY:
            fallback_action = action
        if manhattan(next_pos, target) < current_distance:
            return action
    return fallback_action


def _find_bounce(game_input: GameInput, role: int, target: Position) -> Action | None:
    for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
        next_pos = target.moved(action)
        if _safe_cell(game_input, role, next_pos):
            return action
    return None


def _safe_cell(game_input: GameInput, role: int, pos: Position, extra_block: Position = NO_BLOCK) -> bool:
    if not pos.in_bounds():
        return False
    cell = game_input.grid[pos.row][pos.col]
    if cell in (GRID_FOG, GRID_OBSTACLE, GRID_BOMB):
        return False
    other = Position(*game_input.my_units[1 - role])
    if pos == other or pos == extra_block:
        return False
    if _visible_enemy_at(game_input, pos):
        return False
    return not _crowded_npc_at(game_input, pos)


def _visible_enemy_at(game_input: GameInput, pos: Position) -> bool:
    return any(Position(row, col) == pos for row, col in game_input.visible_enemies if Position(row, col).in_bounds())


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


def _useful_entries_for_gold(gold: int) -> int:
    if gold <= 0:
        return 0
    if gold <= 2:
        return 1
    if gold <= 8:
        return 2
    return 3


def _useful_bounce_pairs(gold: int, already_entered_target: bool) -> int:
    entries = _useful_entries_for_gold(gold)
    if already_entered_target and entries > 0:
        entries -= 1
    return entries


def _clamp(value: int) -> int:
    if value < 0:
        return 0
    if value >= GRID_SIZE:
        return GRID_SIZE - 1
    return value
