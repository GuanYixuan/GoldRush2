from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..constants import STATIC_OBSTACLE
from ..errors import SimulatorRuleError
from ..state import GameState
from ..types import Action, Position
from .maps import MapTemplate, StaticGrid

NPC_PATH_LENGTH = 3
NPC_CENTER = Position(8, 8)
OPPOSITE_ACTIONS = {
    Action.UP: Action.DOWN,
    Action.DOWN: Action.UP,
    Action.LEFT: Action.RIGHT,
    Action.RIGHT: Action.LEFT,
}


@dataclass(frozen=True)
class NpcPath:
    actions: tuple[Action, Action, Action]
    positions: tuple[Position, Position, Position]


@dataclass(frozen=True)
class M4aWeights:
    gold: float = 2.106687
    enter_bomb: float = -2.796592
    stay: float = -3.126171
    straight3: float = 1.212827
    backtrack: float = -0.580930
    bomb_trapped_stay: float = 10.882944
    center: float = 0.180587


@dataclass(frozen=True)
class NpcPolicyConfig:
    weights: M4aWeights = M4aWeights()
    temperature: float = 1.0
    bomb_blind_p: float = 0.0
    randomize_profile: bool = False
    npc_jitter_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise SimulatorRuleError(f"NPC temperature must be positive, got {self.temperature}")
        if not 0.0 <= self.bomb_blind_p <= 1.0:
            raise SimulatorRuleError(f"bomb_blind_p must be in [0, 1], got {self.bomb_blind_p}")
        if self.npc_jitter_scale < 0.0:
            raise SimulatorRuleError(f"npc_jitter_scale must be non-negative, got {self.npc_jitter_scale}")


@dataclass(frozen=True)
class NpcEpisodeProfile:
    weights: M4aWeights
    temperature: float
    bomb_blind_p: float
    npc_overrides: dict[int, tuple[M4aWeights, float]] = field(default_factory=dict)

    def profile_for_npc(self, npc_id: int) -> tuple[M4aWeights, float]:
        return self.npc_overrides.get(npc_id, (self.weights, self.temperature))


@dataclass(frozen=True)
class NpcDecisionBatch:
    order: tuple[int, ...]
    actions: dict[int, tuple[Action, Action, Action]]


@dataclass
class M4aNpcPolicy:
    config: NpcPolicyConfig = field(default_factory=NpcPolicyConfig)
    _path_cache: dict[tuple[int, StaticGrid, Position], tuple[NpcPath, ...]] = field(default_factory=dict, init=False)

    def sample_profile(self, state: GameState, rng: random.Random) -> NpcEpisodeProfile:
        weights = self.config.weights
        temperature = self.config.temperature
        bomb_blind_p = self.config.bomb_blind_p

        if self.config.randomize_profile:
            weights = M4aWeights(
                gold=weights.gold * _log_uniform(rng, 0.5, 2.0),
                enter_bomb=weights.enter_bomb * _log_uniform(rng, 0.5, 2.0),
                stay=weights.stay + rng.uniform(-0.8, 0.8),
                straight3=weights.straight3 + rng.uniform(-0.8, 0.8),
                backtrack=weights.backtrack + rng.uniform(-0.5, 0.5),
                bomb_trapped_stay=weights.bomb_trapped_stay + rng.uniform(-3.0, 3.0),
                center=weights.center + rng.uniform(-0.14, 0.14),
            )
            temperature *= _log_uniform(rng, 0.7, 1.5)
            bomb_blind_p = rng.uniform(0.0, 0.15)

        overrides: dict[int, tuple[M4aWeights, float]] = {}
        if self.config.npc_jitter_scale > 0.0:
            for npc_id in sorted(state.npcs):
                overrides[npc_id] = _jitter_profile(weights, temperature, self.config.npc_jitter_scale, rng)
        return NpcEpisodeProfile(weights=weights, temperature=temperature, bomb_blind_p=bomb_blind_p, npc_overrides=overrides)

    def sample_order(self, state: GameState, rng: random.Random) -> tuple[int, ...]:
        order = sorted(state.npcs)
        rng.shuffle(order)
        return tuple(order)

    def decide_round(
        self,
        state: GameState,
        template: MapTemplate,
        rng: random.Random,
        profile: NpcEpisodeProfile | None = None,
    ) -> NpcDecisionBatch:
        profile = self.sample_profile(state, rng) if profile is None else profile
        order = self.sample_order(state, rng)
        return NpcDecisionBatch(order=order, actions=self.decide_all(state, template, order, rng, profile))

    def decide_all(
        self,
        decision_state: GameState,
        template: MapTemplate,
        npc_order: tuple[int, ...],
        rng: random.Random,
        profile: NpcEpisodeProfile | None = None,
    ) -> dict[int, tuple[Action, Action, Action]]:
        profile = self.sample_profile(decision_state, rng) if profile is None else profile
        if set(npc_order) != set(decision_state.npcs):
            raise SimulatorRuleError(f"npc_order must contain exactly current NPC ids, got {npc_order}")

        decisions: dict[int, tuple[Action, Action, Action]] = {}
        for npc_id in npc_order:
            start = decision_state.npcs[npc_id].position
            paths = self.paths_from(template, start)
            weights, temperature = profile.profile_for_npc(npc_id)
            bombs_visible = profile.bomb_blind_p <= 0.0 or rng.random() >= profile.bomb_blind_p
            chosen = sample_path(decision_state, start, paths, weights, temperature, bombs_visible, rng)
            decisions[npc_id] = chosen.actions
        return decisions

    def paths_from(self, template: MapTemplate, start: Position) -> tuple[NpcPath, ...]:
        cache_key = (template.map_id, template.static_grid, start)
        cached = self._path_cache.get(cache_key)
        if cached is not None:
            return cached
        paths = enumerate_legal_paths(template, start)
        self._path_cache[cache_key] = paths
        return paths


def enumerate_legal_paths(template: MapTemplate, start: Position) -> tuple[NpcPath, ...]:
    if not start.in_bounds():
        raise SimulatorRuleError(f"NPC path start out of bounds: {start}")
    if template.static_grid[start.row][start.col] == STATIC_OBSTACLE:
        raise SimulatorRuleError(f"NPC path start overlaps obstacle: {start}")

    paths: list[NpcPath] = []

    def visit(position: Position, actions: list[Action], positions: list[Position]) -> None:
        if len(actions) == NPC_PATH_LENGTH:
            paths.append(NpcPath(actions=(actions[0], actions[1], actions[2]), positions=(positions[0], positions[1], positions[2])))
            return
        for action in Action:
            next_pos = position.moved(action)
            if not _is_legal_npc_position(template, next_pos):
                continue
            actions.append(action)
            positions.append(next_pos)
            visit(next_pos, actions, positions)
            actions.pop()
            positions.pop()

    visit(start, [], [])
    if not paths:
        raise SimulatorRuleError(f"no legal NPC paths from {start}")
    return tuple(paths)


def score_path(
    state: GameState,
    start: Position,
    path: NpcPath,
    weights: M4aWeights = M4aWeights(),
    *,
    bombs_visible: bool = True,
) -> float:
    features = path_features(state, start, path, bombs_visible=bombs_visible)
    return (
        weights.gold * features["dynamic_reward_div10"]
        + weights.enter_bomb * features["enter_bomb_count"]
        + weights.stay * features["stay_count"]
        + weights.straight3 * features["straight3"]
        + weights.backtrack * features["backtrack"]
        + weights.bomb_trapped_stay * features["bomb_trapped_stay"]
        + weights.center * features["center_delta_chebyshev"]
    )


def path_features(state: GameState, start: Position, path: NpcPath, *, bombs_visible: bool = True) -> dict[str, float]:
    local_gold = dict(state.gold)
    local_bombs = set(state.bombs) if bombs_visible else set()
    reward = 0
    enter_bomb_count = 0

    for action, position in zip(path.actions, path.positions):
        if action == Action.STAY:
            continue
        available = local_gold.get(position)
        if available is not None:
            picked = _ceil_pickup(available)
            reward += picked
            remaining = available - picked
            if remaining > 0:
                local_gold[position] = remaining
            else:
                del local_gold[position]
        if position in local_bombs:
            enter_bomb_count += 1
            local_bombs.remove(position)

    return {
        "dynamic_reward_div10": reward / 10.0,
        "enter_bomb_count": float(enter_bomb_count),
        "stay_count": float(sum(1 for action in path.actions if action == Action.STAY)),
        "straight3": float(_is_straight3(path.actions)),
        "backtrack": float(_has_backtrack(path.actions)),
        "bomb_trapped_stay": float(bombs_visible and _is_bomb_trapped_stay(state, start, path)),
        "center_delta_chebyshev": float(_center_chebyshev(start) - _center_chebyshev(path.positions[-1])),
    }


def sample_path(
    state: GameState,
    start: Position,
    paths: tuple[NpcPath, ...],
    weights: M4aWeights,
    temperature: float,
    bombs_visible: bool,
    rng: random.Random,
) -> NpcPath:
    if temperature <= 0.0:
        raise SimulatorRuleError(f"NPC temperature must be positive, got {temperature}")
    scaled_scores = [score_path(state, start, path, weights, bombs_visible=bombs_visible) / temperature for path in paths]
    max_score = max(scaled_scores)
    exp_scores = [math.exp(score - max_score) for score in scaled_scores]
    total = sum(exp_scores)
    ticket = rng.random() * total
    running = 0.0
    for path, weight in zip(paths, exp_scores):
        running += weight
        if ticket <= running:
            return path
    return paths[-1]


def _is_legal_npc_position(template: MapTemplate, position: Position) -> bool:
    return position.in_bounds() and template.static_grid[position.row][position.col] != STATIC_OBSTACLE


def _ceil_pickup(value: int) -> int:
    if value <= 0:
        raise SimulatorRuleError(f"gold amount must be positive, got {value}")
    return (value * 65 + 99) // 100


def _is_straight3(actions: tuple[Action, Action, Action]) -> bool:
    return actions[0] == actions[1] == actions[2] and actions[0] != Action.STAY


def _has_backtrack(actions: tuple[Action, Action, Action]) -> bool:
    return any(a != Action.STAY and OPPOSITE_ACTIONS.get(a) == b for a, b in zip(actions, actions[1:]))


def _is_bomb_trapped_stay(state: GameState, start: Position, path: NpcPath) -> bool:
    if path.actions != (Action.STAY, Action.STAY, Action.STAY):
        return False
    legal_neighbors = tuple(
        start.moved(action)
        for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)
        if start.moved(action).in_bounds() and start.moved(action) not in state.obstacles
    )
    return bool(legal_neighbors) and all(position in state.bombs for position in legal_neighbors)


def _center_chebyshev(position: Position) -> int:
    return max(abs(position.row - NPC_CENTER.row), abs(position.col - NPC_CENTER.col))


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    if low <= 0.0 or high <= low:
        raise SimulatorRuleError(f"invalid log-uniform bounds: low={low}, high={high}")
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _jitter_profile(weights: M4aWeights, temperature: float, scale: float, rng: random.Random) -> tuple[M4aWeights, float]:
    return (
        M4aWeights(
            gold=weights.gold + rng.gauss(0.0, 0.1 * scale * abs(weights.gold)),
            enter_bomb=weights.enter_bomb + rng.gauss(0.0, 0.1 * scale * abs(weights.enter_bomb)),
            stay=weights.stay + rng.gauss(0.0, 0.1 * scale * abs(weights.stay)),
            straight3=weights.straight3 + rng.gauss(0.0, 0.1 * scale * abs(weights.straight3)),
            backtrack=weights.backtrack + rng.gauss(0.0, 0.1 * scale * abs(weights.backtrack)),
            bomb_trapped_stay=weights.bomb_trapped_stay + rng.gauss(0.0, 0.1 * scale * abs(weights.bomb_trapped_stay)),
            center=weights.center + rng.gauss(0.0, 0.1 * scale * abs(weights.center)),
        ),
        temperature * _log_uniform(rng, 0.9, 1.1),
    )


__all__ = [
    "M4aNpcPolicy",
    "M4aWeights",
    "NpcDecisionBatch",
    "NpcEpisodeProfile",
    "NpcPath",
    "NpcPolicyConfig",
    "enumerate_legal_paths",
    "path_features",
    "sample_path",
    "score_path",
]
