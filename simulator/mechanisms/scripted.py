from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Action, GoldGenerationEvent


@dataclass(frozen=True)
class ScriptedRound:
    gold_generated: tuple[GoldGenerationEvent, ...] = ()
    npc_actions: dict[int, tuple[int | Action, ...]] = field(default_factory=dict)


@dataclass
class ScriptedMechanisms:
    rounds: dict[int, ScriptedRound] = field(default_factory=dict)

    def gold_generated(self, round_index: int) -> tuple[GoldGenerationEvent, ...]:
        return self.rounds.get(round_index, ScriptedRound()).gold_generated

    def npc_actions(self, round_index: int, npc_id: int) -> tuple[Action, ...]:
        actions = self.rounds.get(round_index, ScriptedRound()).npc_actions.get(npc_id, ())
        return tuple(Action(action) for action in actions)


__all__ = ["ScriptedMechanisms", "ScriptedRound"]
