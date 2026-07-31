from __future__ import annotations

import unittest
from dataclasses import dataclass

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.duel import DuelConfig, DuelMechanisms, DuelOrderMode, run_duel
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from simulator.types import Action, GameOutput, GoldGenerationEvent, Position


class DuelEnvTests(unittest.TestCase):
    def test_default_mechanism_short_duel_is_reproducible_with_replay(self) -> None:
        config = DuelConfig(
            episode=EpisodeConfig(rules=RulesConfig(round_count=5, snapshot_period=2), seed=20260731, map_id=1),
            order_mode=DuelOrderMode.RANDOM_ORDER,
        )

        result1 = run_duel(_stay_policy, _stay_policy, config=config, record_replay=True, p90_latency_ns={1: 1, 2: 2})
        result2 = run_duel(_stay_policy, _stay_policy, config=config, record_replay=True, p90_latency_ns={1: 1, 2: 2})

        self.assertEqual(result1.state.round_index, 5)
        self.assertEqual(result1.game_result.winner_id, 1)
        self.assertEqual(result1.replay, result2.replay)
        self.assertEqual(len(result1.rounds), 5)
        self.assertEqual(result1.replay["rounds"][-1]["end"]["round_index"], 5)

    def test_order_modes_place_agent_before_or_after_opponent(self) -> None:
        before = run_duel(
            _stay_policy,
            _stay_policy,
            config=DuelConfig(episode=_one_round_episode(), order_mode=DuelOrderMode.AGENT_BEFORE_OPPONENT, agent_player_id=1),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            p90_latency_ns={1: 1, 2: 2},
        )
        after = run_duel(
            _stay_policy,
            _stay_policy,
            config=DuelConfig(episode=_one_round_episode(), order_mode=DuelOrderMode.AGENT_AFTER_OPPONENT, agent_player_id=1),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            p90_latency_ns={1: 1, 2: 2},
        )

        self.assertEqual(before.rounds[0].first_player_id, 1)
        self.assertEqual(before.rounds[0].transition_result.dispatch_order, (1, 2))
        self.assertEqual(after.rounds[0].first_player_id, 2)
        self.assertEqual(after.rounds[0].transition_result.dispatch_order, (2, 1))

    def test_policy_sequence_output_and_move_decision_object_are_supported(self) -> None:
        result = run_duel(
            _SequencePolicy(),
            _MoveDecisionPolicy(),
            config=DuelConfig(episode=_one_round_episode(), order_mode=DuelOrderMode.AGENT_BEFORE_OPPONENT),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            p90_latency_ns={1: 1, 2: 2},
        )

        self.assertEqual(result.rounds[0].player_outputs[1].actions[0], int(Action.RIGHT))
        self.assertEqual(result.rounds[0].player_outputs[2], _stay_policy(None))
        self.assertEqual(result.state.player_unit(1, 0).position, Position(0, 1))

    def test_pre_round_gold_generation_is_visible_to_policy_and_replay_start(self) -> None:
        seen: list[int] = []

        def p1(game_input):
            seen.append(game_input.grid[0][1])
            return _stay_policy(game_input)

        mechanisms = _quiet_mechanisms()
        mechanisms.center_gold = _FixedGoldGenerator((GoldGenerationEvent(Position(0, 1), 5),))

        result = run_duel(
            p1,
            _stay_policy,
            config=DuelConfig(episode=_one_round_episode(), order_mode=DuelOrderMode.AGENT_BEFORE_OPPONENT),
            mechanisms=mechanisms,
            spawn=SpawnConfig(npc_ids=()),
            record_replay=True,
            p90_latency_ns={1: 1, 2: 2},
        )

        self.assertEqual(seen, [5])
        self.assertEqual(result.replay["rounds"][0]["start"]["grid"][0][1], 5)
        self.assertEqual(result.rounds[0].gold_generated, (GoldGenerationEvent(Position(0, 1), 5),))

    def test_minimal_duel_can_run_500_rounds(self) -> None:
        result = run_duel(
            _stay_policy,
            _stay_policy,
            config=DuelConfig(
                episode=EpisodeConfig(rules=RulesConfig(round_count=500), seed=1, map_id=1),
                order_mode=DuelOrderMode.AGENT_BEFORE_OPPONENT,
            ),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            p90_latency_ns={1: 1, 2: 2},
        )

        self.assertEqual(result.state.round_index, 500)
        self.assertEqual(len(result.rounds), 500)
        self.assertEqual(result.game_result.winner_id, 1)


def _stay_policy(_game_input) -> GameOutput:
    return GameOutput(actions=(Action.STAY,) * 6, k=3, order=0, vp=0)


def _one_round_episode() -> EpisodeConfig:
    return EpisodeConfig(rules=RulesConfig(round_count=1, snapshot_period=1), seed=7, map_id=1)


def _quiet_mechanisms() -> DuelMechanisms:
    return DuelMechanisms(
        center_gold=CenterGoldGenerator(CenterGoldConfig(center_a=0.0)),
        outer_gold=OuterGoldGenerator(OuterGoldConfig(first_round_offset_weights=((999, 1),))),
        bomb_refresher=BernoulliBombRefresher(BombConfig(0.0)),
    )


class _SequencePolicy:
    def __call__(self, _game_input) -> tuple[int, ...]:
        return (int(Action.RIGHT), int(Action.STAY), int(Action.STAY), int(Action.STAY), int(Action.STAY), int(Action.STAY), 1, 0, 0)


class _MoveDecisionPolicy:
    def move_decision(self, game_input) -> GameOutput:
        return _stay_policy(game_input)


@dataclass
class _FixedGoldGenerator:
    events: tuple[GoldGenerationEvent, ...]

    def generate(self, *_args) -> tuple[GoldGenerationEvent, ...]:
        return self.events


if __name__ == "__main__":
    unittest.main()
