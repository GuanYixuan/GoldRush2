from __future__ import annotations

import unittest
from dataclasses import dataclass

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepConfig, RoundStepEnv, RoundStepMechanisms
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from simulator.mechanisms.npc import M4aWeights, NpcEpisodeProfile
from simulator.state import GameState
from simulator.types import Action, GameOutput, GoldGenerationEvent, Position


class RoundStepEnvTests(unittest.TestCase):
    def test_reset_returns_action_time_observations_after_resource_generation(self) -> None:
        mechanisms = _quiet_mechanisms()
        mechanisms.center_gold = _FixedGoldGenerator(
            (
                GoldGenerationEvent(Position(0, 1), 5),
                GoldGenerationEvent(Position(0, 15), 6),
            )
        )
        env = RoundStepEnv(
            config=RoundStepConfig(_one_round_episode()),
            mechanisms=mechanisms,
            spawn=SpawnConfig(npc_ids=()),
        )

        observations = env.reset()

        self.assertEqual(observations[1].round, 0)
        self.assertEqual(observations[1].grid[0][1], 5)
        self.assertEqual(observations[2].grid[0][15], 6)

    def test_reset_reschedules_current_static2_batch_before_action_time(self) -> None:
        mechanisms = _quiet_mechanisms()
        mechanisms.outer_gold = OuterGoldGenerator(
            OuterGoldConfig(
                first_round_offset_weights=((0, 1),),
                gap_weights=((8, 1),),
                region_weights=((2, 1),),
                static2_total_weights=((88, 1),),
                outer_static0_count_weights=((0, 1),),
            )
        )
        env = RoundStepEnv(
            config=RoundStepConfig(_one_round_episode()),
            mechanisms=mechanisms,
            spawn=SpawnConfig(npc_ids=()),
        )

        env.reset()

        assert env.state is not None
        assert env.outer_state is not None
        self.assertEqual(env.state.round_index, 0)
        self.assertEqual(env.outer_state.next_static2_round, 8)
        self.assertEqual(env.outer_state.next_static2_region, 2)

    def test_step_uses_explicit_first_player_id_and_player_outputs(self) -> None:
        env = RoundStepEnv(
            config=RoundStepConfig(_one_round_episode()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            p90_latency_ns={1: 1, 2: 2},
        )
        env.reset()

        result = env.step(
            {
                1: _sequence_output(Action.RIGHT),
                2: _stay_output(),
            },
            first_player_id=2,
        )

        self.assertTrue(result.terminated)
        self.assertEqual(result.trace.first_player_id, 2)
        self.assertEqual(result.trace.transition_result.dispatch_order, (2, 1))
        self.assertEqual(result.state.player_unit(1, 0).position, Position(0, 1))
        self.assertEqual(result.observations, {})
        self.assertEqual(result.game_result.winner_id, 1)

    def test_npc_plans_from_action_time_state_before_first_player_executes(self) -> None:
        npc_policy = _RecordingNpcPolicy()
        mechanisms = _quiet_mechanisms()
        mechanisms.center_gold = _FixedGoldGenerator((GoldGenerationEvent(Position(0, 1), 5),))
        mechanisms.npc_policy = npc_policy
        env = RoundStepEnv(
            config=RoundStepConfig(_one_round_episode()),
            mechanisms=mechanisms,
            spawn=SpawnConfig(npc_ids=(-1,), npc_position=Position(8, 8)),
            p90_latency_ns={1: 1, 2: 2},
        )
        env.reset()

        env.step({1: _sequence_output(Action.RIGHT), 2: _stay_output()}, first_player_id=1)

        assert npc_policy.decision_state is not None
        self.assertEqual(npc_policy.decision_state.gold[Position(0, 1)], 5)
        self.assertEqual(npc_policy.decision_state.player_unit(1, 0).position, Position(0, 0))

    def test_step_returns_next_round_observations_when_not_terminated(self) -> None:
        env = RoundStepEnv(
            config=RoundStepConfig(EpisodeConfig(rules=RulesConfig(round_count=2, snapshot_period=1), seed=7, map_id=1)),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            p90_latency_ns={1: 1, 2: 2},
        )
        env.reset()

        result = env.step({1: _stay_output(), 2: _stay_output()}, first_player_id=1)

        self.assertFalse(result.terminated)
        self.assertEqual(result.observations[1].round, 1)
        self.assertTrue(result.observations[1].snapshot_valid)
        self.assertIsNone(result.game_result)

    def test_replay_records_rounds_without_policy_or_reward_fields(self) -> None:
        env = RoundStepEnv(
            config=RoundStepConfig(EpisodeConfig(rules=RulesConfig(round_count=2), seed=7, map_id=1)),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            record_replay=True,
            p90_latency_ns={1: 1, 2: 2},
        )
        env.reset()

        first = env.step({1: _stay_output(), 2: _stay_output()}, first_player_id=1)
        second = env.step({1: _stay_output(), 2: _stay_output()}, first_player_id=2)

        self.assertEqual(len(first.replay["rounds"]), 1)
        self.assertEqual(len(second.replay["rounds"]), 2)
        self.assertNotIn("reward", second.replay["rounds"][0])
        self.assertEqual(second.replay["rounds"][1]["events"]["dispatch_order"], [2, 1])

    def test_terminal_tie_without_latency_leaves_game_result_unset(self) -> None:
        env = RoundStepEnv(
            config=RoundStepConfig(_one_round_episode()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )
        env.reset()

        result = env.step({1: _stay_output(), 2: _stay_output()}, first_player_id=1)

        self.assertTrue(result.terminated)
        self.assertIsNone(result.game_result)

    def test_step_before_reset_fails_fast(self) -> None:
        env = RoundStepEnv(config=RoundStepConfig(_one_round_episode()), mechanisms=_quiet_mechanisms())

        with self.assertRaisesRegex(Exception, "not reset"):
            env.step({1: _stay_output(), 2: _stay_output()}, first_player_id=1)

    def test_minimal_round_step_can_run_500_rounds(self) -> None:
        env = RoundStepEnv(
            config=RoundStepConfig(EpisodeConfig(rules=RulesConfig(round_count=500), seed=1, map_id=1)),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            p90_latency_ns={1: 1, 2: 2},
        )
        observations = env.reset()

        last = None
        while observations:
            last = env.step({1: _stay_output(), 2: _stay_output()}, first_player_id=1)
            observations = last.observations

        assert last is not None
        self.assertEqual(last.state.round_index, 500)
        self.assertTrue(last.terminated)


def _one_round_episode() -> EpisodeConfig:
    return EpisodeConfig(rules=RulesConfig(round_count=1, snapshot_period=1), seed=7, map_id=1)


def _quiet_mechanisms() -> RoundStepMechanisms:
    return RoundStepMechanisms(
        center_gold=CenterGoldGenerator(CenterGoldConfig(center_a=0.0)),
        outer_gold=OuterGoldGenerator(OuterGoldConfig(first_round_offset_weights=((999, 1),))),
        bomb_refresher=BernoulliBombRefresher(BombConfig(0.0)),
    )


def _stay_output() -> GameOutput:
    return GameOutput(actions=(Action.STAY,) * 6, k=3, order=0, vp=0)


def _sequence_output(first_action: Action) -> tuple[int, ...]:
    return (int(first_action), int(Action.STAY), int(Action.STAY), int(Action.STAY), int(Action.STAY), int(Action.STAY), 1, 0, 0)


@dataclass
class _FixedGoldGenerator:
    events: tuple[GoldGenerationEvent, ...]

    def generate(self, *_args) -> tuple[GoldGenerationEvent, ...]:
        return self.events


class _RecordingNpcPolicy:
    def __init__(self) -> None:
        self.decision_state: GameState | None = None

    def sample_profile(self, _state: GameState, _rng) -> NpcEpisodeProfile:
        return NpcEpisodeProfile(weights=M4aWeights(), temperature=1.0, bomb_blind_p=0.0)

    def sample_order(self, state: GameState, _rng) -> tuple[int, ...]:
        return tuple(sorted(state.npcs))

    def decide_all(self, decision_state: GameState, _template, npc_order: tuple[int, ...], _rng, _profile) -> dict[int, tuple[Action, ...]]:
        self.decision_state = decision_state
        return {npc_id: (Action.STAY, Action.STAY, Action.STAY) for npc_id in npc_order}


if __name__ == "__main__":
    unittest.main()
