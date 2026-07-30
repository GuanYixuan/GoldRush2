from __future__ import annotations

from dataclasses import dataclass

from ..config import RulesConfig
from ..errors import SimulatorRuleError
from ..state import GameState


@dataclass(frozen=True)
class GameResult:
    winner_id: int
    loser_id: int
    reason: str
    net_gold: dict[int, int]


def gross_gold(state: GameState, player_id: int) -> int:
    return state.players[player_id].gross_gold


def net_gold(state: GameState, player_id: int) -> int:
    return state.players[player_id].net_gold


def apply_vision_purchase(
    state: GameState,
    player_id: int,
    vp: int,
    rules: RulesConfig | None = None,
) -> None:
    """Record a player's vision purchase for the next round.

    Vision spending is settled after the game and does not require current
    gross gold to be sufficient.
    """
    rules = RulesConfig() if rules is None else rules
    if vp not in rules.vision_cost_by_vp or vp not in rules.vision_radius_by_vp:
        raise ValueError(f"unknown vision purchase option: {vp}")
    player = state.players[player_id]
    player.vision_spent += rules.vision_cost_by_vp[vp]
    player.next_vision_radius = rules.vision_radius_by_vp[vp]


def activate_pending_vision(state: GameState, rules: RulesConfig | None = None) -> None:
    """Make pending vision purchases active for one round, then reset pending."""
    rules = RulesConfig() if rules is None else rules
    for player in state.players.values():
        player.active_vision_radius = player.next_vision_radius
        player.next_vision_radius = rules.default_vision_radius


def determine_winner(state: GameState, p90_latency_ns: dict[int, int] | None = None) -> GameResult:
    scores = {player_id: player.net_gold for player_id, player in state.players.items()}
    if scores[1] > scores[2]:
        return GameResult(winner_id=1, loser_id=2, reason="net_gold", net_gold=scores)
    if scores[2] > scores[1]:
        return GameResult(winner_id=2, loser_id=1, reason="net_gold", net_gold=scores)

    if p90_latency_ns is None or 1 not in p90_latency_ns or 2 not in p90_latency_ns:
        raise SimulatorRuleError("net gold tie requires both players' P90 latency")
    if p90_latency_ns[1] < p90_latency_ns[2]:
        return GameResult(winner_id=1, loser_id=2, reason="p90_latency", net_gold=scores)
    if p90_latency_ns[2] < p90_latency_ns[1]:
        return GameResult(winner_id=2, loser_id=1, reason="p90_latency", net_gold=scores)

    raise SimulatorRuleError("net gold and P90 latency are both tied; official tie-break is unspecified")


__all__ = [
    "GameResult",
    "activate_pending_vision",
    "apply_vision_purchase",
    "determine_winner",
    "gross_gold",
    "net_gold",
]
