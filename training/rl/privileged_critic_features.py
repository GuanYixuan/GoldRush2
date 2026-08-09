from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from simulator.constants import BOMB_REFRESH_PERIOD, GRID_SIZE, ROUND_COUNT, STATIC_EMPTY, STATIC_SPECIAL_NON_BLOCKING
from simulator.errors import SimulatorRuleError
from simulator.mechanisms.maps import MapTemplate
from simulator.rules.snapshot import region_id
from simulator.state import GameState
from simulator.types import Position


FEATURE_SCHEMA = "goldrush2_privileged_critic_feature_v1"
SPATIAL_CHANNELS = 26
SCALAR_FEATURES = 17
CENTER_GOLD_B = 0.06735

_CHANNEL_NAMES = (
    "gold_count",
    "bomb_mask",
    "obstacle_mask",
    "npc_count",
    "npc_crowded_mask",
    "static_2_mask",
    "static_2_available_mask",
    "next_static2_region_mask",
    "center_gold_spawn_prob",
    "own_unit0_mask",
    "own_unit1_mask",
    "enemy_unit0_mask",
    "enemy_unit1_mask",
    "unit0_manhattan_distance",
    "unit1_manhattan_distance",
    "unit0_obstacle_distance",
    "unit1_obstacle_distance",
    "enemy_unit0_manhattan_distance",
    "enemy_unit1_manhattan_distance",
    "enemy_unit0_obstacle_distance",
    "enemy_unit1_obstacle_distance",
    "region_1_center_mask",
    "region_2_up_mask",
    "region_3_left_mask",
    "region_4_down_mask",
    "region_5_right_mask",
)

_SCALAR_NAMES = (
    "game_phase_sin",
    "game_phase_cos",
    "remaining_rounds_scaled",
    "bomb_refresh_phase_sin",
    "bomb_refresh_phase_cos",
    "static2_high_imminence",
    "own_gross_gold_scaled",
    "own_net_gold_scaled",
    "opp_gross_gold_scaled",
    "opp_net_gold_scaled",
    "net_gold_margin_scaled",
    "own_unit0_gold_scaled",
    "own_unit1_gold_scaled",
    "enemy_unit0_gold_scaled",
    "enemy_unit1_gold_scaled",
    "total_gold_on_map_scaled",
    "center_gold_on_map_scaled",
)


def feature_schema() -> str:
    return FEATURE_SCHEMA


def channel_names() -> tuple[str, ...]:
    return _CHANNEL_NAMES


def scalar_names() -> tuple[str, ...]:
    return _SCALAR_NAMES


def privileged_critic_feature_schema() -> str:
    return feature_schema()


def privileged_critic_channel_names() -> tuple[str, ...]:
    return channel_names()


def privileged_critic_scalar_names() -> tuple[str, ...]:
    return scalar_names()


def extract_privileged_critic_features(
    *,
    state: GameState,
    template: MapTemplate,
    outer_state: Any,
    agent_player_id: int,
    round_count: int = ROUND_COUNT,
) -> dict[str, Any]:
    _validate_inputs(state=state, template=template, outer_state=outer_state, agent_player_id=agent_player_id, round_count=round_count)

    planes = np.zeros((SPATIAL_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    scalars = np.zeros((SCALAR_FEATURES,), dtype=np.float32)
    channel_index = {name: idx for idx, name in enumerate(_CHANNEL_NAMES)}
    scalar_index = {name: idx for idx, name in enumerate(_SCALAR_NAMES)}

    own_player = state.players[agent_player_id]
    enemy_player_id = 3 - agent_player_id
    enemy_player = state.players[enemy_player_id]
    next_static2_region = int(outer_state.next_static2_region)
    next_static2_round = int(outer_state.next_static2_round)

    player_occupancy = {unit.position for _, unit in state.iter_player_units()}
    npc_occupancy = {npc.position for npc in state.npcs.values()}
    blocked_generation = player_occupancy | npc_occupancy | set(state.bombs)

    for position, amount in state.gold.items():
        planes[channel_index["gold_count"], position.row, position.col] = _clip(float(amount) / 20.0, 0.0, 3.0)
    for position in state.bombs:
        planes[channel_index["bomb_mask"], position.row, position.col] = 1.0
    for position in state.obstacles:
        planes[channel_index["obstacle_mask"], position.row, position.col] = 1.0

    npc_counts: dict[Position, int] = {}
    for npc in state.npcs.values():
        npc_counts[npc.position] = npc_counts.get(npc.position, 0) + 1
    for position, count in npc_counts.items():
        planes[channel_index["npc_count"], position.row, position.col] = _clip(float(count) / 3.0, 0.0, 1.0)
        if count >= 3:
            planes[channel_index["npc_crowded_mask"], position.row, position.col] = 1.0

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            pos = Position(row, col)
            pos_region = region_id(pos)
            planes[channel_index[f"region_{pos_region}_{_region_suffix(pos_region)}"], row, col] = 1.0
            if pos_region == next_static2_region:
                planes[channel_index["next_static2_region_mask"], row, col] = 1.0
            if template.static_grid[row][col] == STATIC_SPECIAL_NON_BLOCKING:
                planes[channel_index["static_2_mask"], row, col] = 1.0
                if pos not in blocked_generation:
                    planes[channel_index["static_2_available_mask"], row, col] = 1.0
            if _is_center_gold_candidate(template, pos):
                sq = (row - 8) * (row - 8) + (col - 8) * (col - 8)
                planes[channel_index["center_gold_spawn_prob"], row, col] = math.exp(-CENTER_GOLD_B * float(sq))

    _write_unit_planes(planes, channel_index, own_player.units[0].position, "own_unit0_mask", "unit0_manhattan_distance", "unit0_obstacle_distance", state.obstacles)
    _write_unit_planes(planes, channel_index, own_player.units[1].position, "own_unit1_mask", "unit1_manhattan_distance", "unit1_obstacle_distance", state.obstacles)
    _write_unit_planes(
        planes,
        channel_index,
        enemy_player.units[0].position,
        "enemy_unit0_mask",
        "enemy_unit0_manhattan_distance",
        "enemy_unit0_obstacle_distance",
        state.obstacles,
    )
    _write_unit_planes(
        planes,
        channel_index,
        enemy_player.units[1].position,
        "enemy_unit1_mask",
        "enemy_unit1_manhattan_distance",
        "enemy_unit1_obstacle_distance",
        state.obstacles,
    )

    game_phase_denominator = max(round_count - 1, 1)
    game_phase = -math.pi / 2.0 + math.pi * float(state.round_index) / float(game_phase_denominator)
    bomb_phase = float(state.round_index % BOMB_REFRESH_PERIOD) / float(BOMB_REFRESH_PERIOD)
    rounds_to_next_static2_high = next_static2_round - state.round_index

    own_gross = own_player.gross_gold
    own_net = own_player.net_gold
    enemy_gross = enemy_player.gross_gold
    enemy_net = enemy_player.net_gold
    total_map_gold = sum(state.gold.values())
    center_map_gold = sum(amount for position, amount in state.gold.items() if region_id(position) == 1)

    scalars[scalar_index["game_phase_sin"]] = math.sin(game_phase)
    scalars[scalar_index["game_phase_cos"]] = math.cos(game_phase)
    scalars[scalar_index["remaining_rounds_scaled"]] = float(round_count - state.round_index) / float(round_count)
    scalars[scalar_index["bomb_refresh_phase_sin"]] = math.sin(2.0 * math.pi * bomb_phase)
    scalars[scalar_index["bomb_refresh_phase_cos"]] = math.cos(2.0 * math.pi * bomb_phase)
    scalars[scalar_index["static2_high_imminence"]] = math.exp(-(float(rounds_to_next_static2_high) - 1.0) / 8.0)
    scalars[scalar_index["own_gross_gold_scaled"]] = _clip(float(own_gross) / 2000.0, 0.0, 3.0)
    scalars[scalar_index["own_net_gold_scaled"]] = _clip(float(own_net) / 2000.0, -1.0, 3.0)
    scalars[scalar_index["opp_gross_gold_scaled"]] = _clip(float(enemy_gross) / 2000.0, 0.0, 3.0)
    scalars[scalar_index["opp_net_gold_scaled"]] = _clip(float(enemy_net) / 2000.0, -1.0, 3.0)
    scalars[scalar_index["net_gold_margin_scaled"]] = _clip(float(own_net - enemy_net) / 500.0, -3.0, 3.0)
    scalars[scalar_index["own_unit0_gold_scaled"]] = _clip(float(own_player.units[0].gold) / 1000.0, 0.0, 3.0)
    scalars[scalar_index["own_unit1_gold_scaled"]] = _clip(float(own_player.units[1].gold) / 1000.0, 0.0, 3.0)
    scalars[scalar_index["enemy_unit0_gold_scaled"]] = _clip(float(enemy_player.units[0].gold) / 1000.0, 0.0, 3.0)
    scalars[scalar_index["enemy_unit1_gold_scaled"]] = _clip(float(enemy_player.units[1].gold) / 1000.0, 0.0, 3.0)
    scalars[scalar_index["total_gold_on_map_scaled"]] = _clip(float(total_map_gold) / 2000.0, 0.0, 3.0)
    scalars[scalar_index["center_gold_on_map_scaled"]] = _clip(float(center_map_gold) / 1000.0, 0.0, 3.0)

    return {
        "feature_schema": FEATURE_SCHEMA,
        "planes": planes,
        "scalars": scalars,
        "channel_names": _CHANNEL_NAMES,
        "scalar_names": _SCALAR_NAMES,
    }


def _validate_inputs(
    *,
    state: GameState,
    template: MapTemplate,
    outer_state: Any,
    agent_player_id: int,
    round_count: int,
) -> None:
    if agent_player_id not in (1, 2):
        raise SimulatorRuleError(f"agent_player_id must be 1 or 2, got {agent_player_id}")
    if set(state.players) != {1, 2}:
        raise SimulatorRuleError(f"state.players must contain ids {{1, 2}}, got {sorted(state.players)}")
    if state.obstacles != template.obstacles:
        raise SimulatorRuleError("state obstacles do not match map template obstacles")
    if round_count <= 0:
        raise SimulatorRuleError(f"round_count must be positive, got {round_count}")
    if state.round_index < 0:
        raise SimulatorRuleError(f"round_index must be non-negative, got {state.round_index}")
    if state.round_index >= round_count:
        raise SimulatorRuleError(f"round_index {state.round_index} must be less than round_count {round_count}")
    if not hasattr(outer_state, "next_static2_round"):
        raise SimulatorRuleError("outer_state must expose next_static2_round")
    if not hasattr(outer_state, "next_static2_region"):
        raise SimulatorRuleError("outer_state must expose next_static2_region")
    next_static2_round = int(outer_state.next_static2_round)
    next_static2_region = int(outer_state.next_static2_region)
    if next_static2_round <= state.round_index:
        raise SimulatorRuleError(
            f"next_static2_round must point to a future round after generation; got {next_static2_round} at round {state.round_index}"
        )
    if next_static2_region not in (2, 3, 4, 5):
        raise SimulatorRuleError(f"next_static2_region must be one of {{2, 3, 4, 5}}, got {next_static2_region}")


def _write_unit_planes(
    planes: np.ndarray,
    channel_index: dict[str, int],
    unit_pos: Position,
    mask_name: str,
    manhattan_name: str,
    obstacle_name: str,
    obstacles: frozenset[Position],
) -> None:
    planes[channel_index[mask_name], unit_pos.row, unit_pos.col] = 1.0
    bfs = _obstacle_distances(unit_pos, obstacles)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            flat = row * GRID_SIZE + col
            planes[channel_index[manhattan_name], row, col] = float(abs(row - unit_pos.row) + abs(col - unit_pos.col)) / 32.0
            planes[channel_index[obstacle_name], row, col] = 2.0 if bfs[flat] < 0 else _clip(float(bfs[flat]) / 32.0, 0.0, 2.0)


def _obstacle_distances(start: Position, obstacles: frozenset[Position]) -> list[int]:
    distances = [-1] * (GRID_SIZE * GRID_SIZE)
    if start in obstacles:
        return distances
    start_flat = start.row * GRID_SIZE + start.col
    distances[start_flat] = 0
    queue: deque[Position] = deque((start,))
    while queue:
        current = queue.popleft()
        current_distance = distances[current.row * GRID_SIZE + current.col]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = Position(current.row + dr, current.col + dc)
            if not nxt.in_bounds() or nxt in obstacles:
                continue
            flat = nxt.row * GRID_SIZE + nxt.col
            if distances[flat] >= 0:
                continue
            distances[flat] = current_distance + 1
            queue.append(nxt)
    return distances


def _is_center_gold_candidate(template: MapTemplate, position: Position) -> bool:
    return region_id(position) == 1 and template.static_grid[position.row][position.col] == STATIC_EMPTY and position not in template.obstacles


def _region_suffix(id_: int) -> str:
    if id_ == 1:
        return "center_mask"
    if id_ == 2:
        return "up_mask"
    if id_ == 3:
        return "left_mask"
    if id_ == 4:
        return "down_mask"
    if id_ == 5:
        return "right_mask"
    raise SimulatorRuleError(f"unknown region id: {id_}")


def _clip(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


__all__ = [
    "CENTER_GOLD_B",
    "FEATURE_SCHEMA",
    "SCALAR_FEATURES",
    "SPATIAL_CHANNELS",
    "channel_names",
    "extract_privileged_critic_features",
    "feature_schema",
    "privileged_critic_channel_names",
    "privileged_critic_feature_schema",
    "privileged_critic_scalar_names",
    "scalar_names",
]
