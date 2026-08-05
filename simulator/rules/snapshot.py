from __future__ import annotations

from dataclasses import dataclass, field

from ..config import RulesConfig
from ..constants import REGION_COUNT
from ..errors import SimulatorRuleError
from ..state import GameState
from ..types import GoldGenerationEvent, InteractionEvents, Position, RegionStat, Snapshot

OccupantPositionKey = tuple[str, int, int | None]
OccupantPositions = dict[OccupantPositionKey, Position]


@dataclass
class _MutableRegionStat:
    enter: int = 0
    leave: int = 0
    gold_generated: int = 0
    gold_collected: int = 0


@dataclass
class SnapshotAccumulator:
    rules: RulesConfig = field(default_factory=RulesConfig)
    window_begin: int | None = None
    _regions: dict[int, _MutableRegionStat] = field(default_factory=lambda: {region_id: _MutableRegionStat() for region_id in range(1, REGION_COUNT + 1)})
    _recorded_rounds: int = field(default=0, init=False)

    def record_round(
        self,
        round_index: int,
        start_state: GameState,
        end_state: GameState,
        gold_generated: tuple[GoldGenerationEvent, ...] = (),
        interactions: tuple[InteractionEvents, ...] = (),
    ) -> Snapshot | None:
        return self.record_round_from_positions(
            round_index,
            capture_occupant_positions(start_state),
            end_state,
            gold_generated=gold_generated,
            interactions=interactions,
        )

    def record_round_from_positions(
        self,
        round_index: int,
        start_positions: OccupantPositions,
        end_state: GameState,
        gold_generated: tuple[GoldGenerationEvent, ...] = (),
        interactions: tuple[InteractionEvents, ...] = (),
    ) -> Snapshot | None:
        if self.window_begin is None:
            self.window_begin = round_index
        expected_round = self.window_begin + self._recorded_rounds
        if round_index != expected_round:
            raise SimulatorRuleError(f"snapshot window expected round {expected_round}, got {round_index}")

        self._record_occupant_transitions(start_positions, end_state)
        self._record_gold_generated(gold_generated)
        self._record_gold_collected(interactions)
        self._recorded_rounds += 1

        if self._recorded_rounds < self.rules.snapshot_period:
            return None
        if self._recorded_rounds > self.rules.snapshot_period:
            raise SimulatorRuleError("snapshot window exceeded configured period")

        snapshot = self._build_snapshot(end_state, round_index)
        self.window_begin = round_index + 1
        self._regions = {region_id: _MutableRegionStat() for region_id in range(1, REGION_COUNT + 1)}
        self._recorded_rounds = 0
        return snapshot

    def _record_occupant_transitions(self, start_positions: OccupantPositions, end_state: GameState) -> None:
        end_positions = capture_occupant_positions(end_state)
        if set(start_positions) != set(end_positions):
            missing = sorted(set(start_positions) ^ set(end_positions))
            raise SimulatorRuleError(f"occupant set changed during snapshot window: {missing}")

        for actor_id, start_pos in start_positions.items():
            start_region = region_id(start_pos)
            end_region = region_id(end_positions[actor_id])
            if start_region == end_region:
                continue
            self._regions[start_region].leave += 1
            self._regions[end_region].enter += 1

    def _record_gold_generated(self, gold_generated: tuple[GoldGenerationEvent, ...]) -> None:
        for event in gold_generated:
            if event.amount <= 0:
                raise SimulatorRuleError(f"gold generation amount must be positive: {event}")
            self._regions[region_id(event.position)].gold_generated += event.amount

    def _record_gold_collected(self, interactions: tuple[InteractionEvents, ...]) -> None:
        for interaction in interactions:
            for pickup in interaction.pickups:
                self._regions[region_id(pickup.position)].gold_collected += pickup.picked_gold

    def _build_snapshot(self, state: GameState, window_end: int) -> Snapshot:
        if self.window_begin is None:
            raise SimulatorRuleError("cannot build snapshot before window starts")
        regions = []
        for id_ in range(1, REGION_COUNT + 1):
            stat = self._regions[id_]
            regions.append(
                RegionStat(
                    id=id_,
                    enter=stat.enter,
                    leave=stat.leave,
                    gold_generated=stat.gold_generated,
                    gold_collected=stat.gold_collected,
                    gold_remaining=_gold_remaining(state, id_),
                    occupants=_occupants(state, id_),
                )
            )
        snapshot = Snapshot(window_begin=self.window_begin, window_end=window_end, regions=regions)
        return snapshot


def region_id(position: Position) -> int:
    if not position.in_bounds():
        raise SimulatorRuleError(f"position out of bounds for region lookup: {position}")
    row, col = position.row, position.col
    if 4 <= row <= 12 and 4 <= col <= 12:
        return 1
    if 0 <= row <= 3 and 0 <= col <= 12:
        return 2
    if 4 <= row <= 16 and 0 <= col <= 3:
        return 3
    if 13 <= row <= 16 and 4 <= col <= 16:
        return 4
    if 0 <= row <= 12 and 13 <= col <= 16:
        return 5
    raise SimulatorRuleError(f"position does not belong to any region: {position}")


def record_snapshot_round(
    accumulator: SnapshotAccumulator,
    round_index: int,
    start_state: GameState,
    end_state: GameState,
    gold_generated: tuple[GoldGenerationEvent, ...] = (),
    interactions: tuple[InteractionEvents, ...] = (),
) -> Snapshot | None:
    return accumulator.record_round(round_index, start_state, end_state, gold_generated, interactions)


def capture_occupant_positions(state: GameState) -> OccupantPositions:
    positions: OccupantPositions = {}
    for player_id, player in state.players.items():
        for unit in player.units:
            positions[("player", player_id, unit.id)] = unit.position
    for npc_id, npc in state.npcs.items():
        positions[("npc", npc_id, None)] = npc.position
    return positions


def _gold_remaining(state: GameState, id_: int) -> int:
    return sum(amount for position, amount in state.gold.items() if region_id(position) == id_)


def _occupants(state: GameState, id_: int) -> int:
    return sum(1 for position in capture_occupant_positions(state).values() if region_id(position) == id_)


__all__ = ["SnapshotAccumulator", "capture_occupant_positions", "record_snapshot_round", "region_id"]
