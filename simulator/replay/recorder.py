from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum, IntEnum
from typing import Any

from ..constants import GRID_BOMB, GRID_EMPTY, GRID_OBSTACLE, GRID_SIZE
from ..errors import SimulatorRuleError
from ..mechanisms.bombs import BombRefreshEvent
from ..mechanisms.maps import MapTemplate
from ..rules.transition import TransitionResult
from ..state import GameState
from ..types import Action, GameOutput, Position
from .schema import JsonObject, SimulatorReplayDocument


class SimulatorReplayRecorder:
    def __init__(
        self,
        *,
        map_template: MapTemplate,
        seed: int | None = None,
        players: dict[int, str] | None = None,
        mechanisms: JsonObject | None = None,
    ) -> None:
        self.document = SimulatorReplayDocument(
            source={
                "seed": seed,
                "map_id": map_template.map_id,
                "map_key": map_template.map_key,
                "map_name": map_template.name,
                "mechanisms": mechanisms or {},
            },
            players={
                "player1": (players or {}).get(1, "player1"),
                "player2": (players or {}).get(2, "player2"),
            },
            maps=[list(row) for row in map_template.static_grid],
        )

    def record_round(
        self,
        *,
        start_state: GameState,
        end_state: GameState,
        player_outputs: dict[int, GameOutput],
        transition_result: TransitionResult,
        npc_actions: dict[int, tuple[int | Action, ...]] | None = None,
        npc_order: tuple[int, ...] | None = None,
        bomb_refresh_event: BombRefreshEvent | None = None,
        mechanism_events: JsonObject | None = None,
    ) -> None:
        round_index = start_state.round_index
        if round_index != len(self.document.rounds):
            raise SimulatorRuleError(f"simulator replay expected round {len(self.document.rounds)}, got {round_index}")
        if set(player_outputs) != {1, 2}:
            raise SimulatorRuleError(f"player_outputs must contain player ids 1 and 2, got {sorted(player_outputs)}")

        events: JsonObject = {
            "gold_generated": _to_json(transition_result.gold_generated),
            "movement": _to_json(transition_result.movement_events),
            "interactions": _to_json(transition_result.interaction_events),
            "dispatch_order": list(transition_result.dispatch_order),
        }
        if bomb_refresh_event is not None:
            events["bomb_refresh"] = _to_json(bomb_refresh_event)
        if mechanism_events:
            events["mechanisms"] = _to_json(mechanism_events)

        actions: JsonObject = {
            "players": {str(player_id): _game_output_to_json(output) for player_id, output in sorted(player_outputs.items())},
            "npcs": {str(npc_id): [int(Action(action)) for action in actions] for npc_id, actions in sorted((npc_actions or {}).items())},
        }
        if npc_order is not None:
            actions["npc_order"] = list(npc_order)

        self.document.rounds.append(
            {
                "round": round_index,
                "start": serialize_state(start_state),
                "end": serialize_state(end_state),
                "actions": actions,
                "events": events,
                "snapshot": _to_json(transition_result.snapshot),
            }
        )

    def to_json(self) -> JsonObject:
        return self.document.to_json()


def serialize_state(state: GameState) -> JsonObject:
    return {
        "round_index": state.round_index,
        "grid": _resource_grid(state),
        "players": [_player_to_json(player_id, state) for player_id in sorted(state.players)],
        "npcs": [{"id": npc_id, "position": _position_to_json(state.npcs[npc_id].position)} for npc_id in sorted(state.npcs)],
        "gold": [{"position": _position_to_json(position), "amount": state.gold[position]} for position in sorted(state.gold)],
        "bombs": [_position_to_json(position) for position in sorted(state.bombs)],
    }


def _resource_grid(state: GameState) -> list[list[int]]:
    grid = [[GRID_EMPTY for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for position in sorted(state.obstacles):
        _validate_position(position, "obstacle")
        grid[position.row][position.col] = GRID_OBSTACLE

    for position, amount in sorted(state.gold.items()):
        _validate_position(position, "gold")
        if amount <= 0:
            raise SimulatorRuleError(f"gold amount must be positive at {position}: {amount}")
        if grid[position.row][position.col] == GRID_OBSTACLE:
            raise SimulatorRuleError(f"gold overlaps obstacle at {position}")
        grid[position.row][position.col] = amount

    for position in sorted(state.bombs):
        _validate_position(position, "bomb")
        if grid[position.row][position.col] == GRID_OBSTACLE:
            raise SimulatorRuleError(f"bomb overlaps obstacle at {position}")
        if grid[position.row][position.col] > 0:
            raise SimulatorRuleError(f"bomb overlaps gold at {position}")
        grid[position.row][position.col] = GRID_BOMB
    return grid


def _player_to_json(player_id: int, state: GameState) -> JsonObject:
    player = state.players[player_id]
    return {
        "id": player.id,
        "gold": player.gross_gold,
        "vision_spent": player.vision_spent,
        "active_vision_radius": player.active_vision_radius,
        "next_vision_radius": player.next_vision_radius,
        "units": [
            {
                "id": unit.id,
                "position": _position_to_json(unit.position),
                "gold": unit.gold,
            }
            for unit in player.units
        ],
    }


def _game_output_to_json(output: GameOutput) -> JsonObject:
    return {
        "actions": [int(Action(action)) for action in output.actions],
        "k": output.k,
        "order": output.order,
        "vp": output.vp,
    }


def _to_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Position):
        return _position_to_json(value)
    if isinstance(value, IntEnum):
        return int(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _to_json(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(_to_json(key)): _to_json(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_to_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_to_json(item) for item in sorted(value)]
    return value


def _position_to_json(position: Position) -> list[int]:
    _validate_position(position, "position")
    return [position.row, position.col]


def _validate_position(position: Position, label: str) -> None:
    if not position.in_bounds():
        raise SimulatorRuleError(f"{label} position out of bounds: {position}")


__all__ = ["SimulatorReplayRecorder", "serialize_state"]
