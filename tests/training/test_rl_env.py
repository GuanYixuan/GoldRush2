from __future__ import annotations

import unittest
from dataclasses import dataclass

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepMechanisms
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from simulator.types import Action, GameOutput, GoldGenerationEvent, Position
from training.opponents import OpponentSpec
from training.rl import (
    AGENT_DECISION_FAST,
    AGENT_DECISION_NEURAL,
    FAST_ORDER_LATENT_FIRST_RATE_MIXTURE,
    FastOrderConfig,
    SingleAgentEnvConfig,
    SingleAgentGoldRushEnv,
)
from training.rl.fast_order import make_fast_order_rng, make_latent_first_rate_rng, sample_latent_first_rate


class RLEnvTests(unittest.TestCase):
    def test_single_agent_env_runs_agent_after_opponent_and_returns_win_loss_reward(self) -> None:
        mechanisms = _quiet_mechanisms()
        mechanisms.center_gold = _FixedGoldGenerator((GoldGenerationEvent(Position(0, 1), 5),))
        env = SingleAgentGoldRushEnv(
            config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=mechanisms,
            spawn=SpawnConfig(npc_ids=()),
        )
        reset = env.reset(seed=7, map_id=1, agent_player_id=1)

        step = env.step(_sequence_output(Action.RIGHT))

        self.assertEqual(reset.observation.round, 0)
        self.assertTrue(step.terminated)
        self.assertFalse(step.truncated)
        self.assertEqual(step.reward, 1.0)
        self.assertEqual(step.info["first_player_id"], 2)
        self.assertEqual(step.info["agent_decision_mode"], AGENT_DECISION_NEURAL)
        self.assertFalse(step.info["fast_order_sampled"])
        self.assertFalse(step.info["agent_first"])
        self.assertEqual(step.info["game_result"].winner_id, 1)
        self.assertEqual(step.info["scores"]["net_gold"][1], 4)
        self.assertEqual(step.info["events"]["pickups"][1], 1)
        self.assertEqual(step.info["events"]["pickup_gold"][1], 4)
        self.assertEqual(step.info["events"]["bomb_triggers"][1], 0)
        self.assertEqual(step.info["events"]["bomb_lost_gold"][1], 0)
        self.assertEqual(step.info["events"]["tramples"][1], 0)
        self.assertEqual(step.info["events"]["trample_penalty"][1], 0)

    def test_non_terminal_step_reward_is_zero_and_returns_next_observation(self) -> None:
        env = SingleAgentGoldRushEnv(
            config=SingleAgentEnvConfig(
                episode=EpisodeConfig(rules=RulesConfig(round_count=2, snapshot_period=1), seed=7, map_id=1),
                opponent_spec=_stay_opponent_spec(),
            ),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )
        env.reset(agent_player_id=1)

        step = env.step(_stay_output())

        self.assertFalse(step.terminated)
        self.assertEqual(step.reward, 0.0)
        assert step.observation is not None
        self.assertEqual(step.observation.round, 1)

    def test_tie_break_treats_opponent_as_faster_for_both_agent_sides(self) -> None:
        rewards = []
        winners = []
        for agent_player_id in (1, 2):
            env = SingleAgentGoldRushEnv(
                config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
                mechanisms=_quiet_mechanisms(),
                spawn=SpawnConfig(npc_ids=()),
            )
            env.reset(seed=9, map_id=1, agent_player_id=agent_player_id)
            step = env.step(_stay_output())
            rewards.append(step.reward)
            winners.append(step.info["game_result"].winner_id)

        self.assertEqual(rewards, [-1.0, -1.0])
        self.assertEqual(winners, [2, 1])

    def test_step_before_reset_fails_fast(self) -> None:
        env = SingleAgentGoldRushEnv(config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()))

        with self.assertRaisesRegex(Exception, "not reset"):
            env.step(_stay_output())

    def test_fast_decision_mode_uses_latent_first_rate_for_order(self) -> None:
        env_first = SingleAgentGoldRushEnv(
            config=SingleAgentEnvConfig(
                episode=_one_round_episode(),
                opponent_spec=_stay_opponent_spec(),
                fast_order=FastOrderConfig(((1.0, 1.0, 1.0),)),
            ),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )
        env_first.reset(seed=11, map_id=1, agent_player_id=1)
        first_step = env_first.step(_stay_output(), agent_decision_mode=AGENT_DECISION_FAST)

        env_second = SingleAgentGoldRushEnv(
            config=SingleAgentEnvConfig(
                episode=_one_round_episode(),
                opponent_spec=_stay_opponent_spec(),
                fast_order=FastOrderConfig(((1.0, 0.0, 0.0),)),
            ),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )
        env_second.reset(seed=11, map_id=1, agent_player_id=1)
        second_step = env_second.step(_stay_output(), agent_decision_mode=AGENT_DECISION_FAST)

        self.assertEqual(first_step.info["first_player_id"], 1)
        self.assertTrue(first_step.info["fast_order_sampled"])
        self.assertTrue(first_step.info["agent_first"])
        self.assertEqual(first_step.info["latent_first_rate"], 1.0)
        self.assertEqual(second_step.info["first_player_id"], 2)
        self.assertTrue(second_step.info["fast_order_sampled"])
        self.assertFalse(second_step.info["agent_first"])
        self.assertEqual(second_step.info["latent_first_rate"], 0.0)

    def test_fast_order_rng_streams_are_reproducible_and_independent(self) -> None:
        config = FastOrderConfig(((1.0, 0.30, 0.70),))

        first_rate = sample_latent_first_rate(make_latent_first_rate_rng(123), config)
        second_rate = sample_latent_first_rate(make_latent_first_rate_rng(123), config)
        p1_draw = make_fast_order_rng(123, agent_player_id=1).random()
        p2_draw = make_fast_order_rng(123, agent_player_id=2).random()

        self.assertEqual(first_rate, second_rate)
        self.assertGreaterEqual(first_rate, 0.30)
        self.assertLessEqual(first_rate, 0.70)
        self.assertNotEqual(p1_draw, p2_draw)

    def test_default_fast_order_mixture_matches_conservative_regimes(self) -> None:
        self.assertEqual(
            FAST_ORDER_LATENT_FIRST_RATE_MIXTURE,
            (
                (0.40, 0.95, 1.00),
                (0.20, 0.70, 0.95),
                (0.20, 0.30, 0.70),
                (0.20, 0.02, 0.10),
            ),
        )


def _one_round_episode() -> EpisodeConfig:
    return EpisodeConfig(rules=RulesConfig(round_count=1, snapshot_period=1), seed=7, map_id=1)


def _quiet_mechanisms() -> RoundStepMechanisms:
    return RoundStepMechanisms(
        center_gold=CenterGoldGenerator(CenterGoldConfig(center_a=0.0)),
        outer_gold=OuterGoldGenerator(OuterGoldConfig(first_round_offset_weights=((999, 1),))),
        bomb_refresher=BernoulliBombRefresher(BombConfig(0.0)),
    )


def _stay_opponent_spec() -> OpponentSpec:
    return OpponentSpec(kind="python", name="stay")


def _stay_output() -> GameOutput:
    return GameOutput(actions=(int(Action.STAY),) * 6, k=3, order=0, vp=0)


def _sequence_output(first_action: Action) -> GameOutput:
    return GameOutput(actions=(int(first_action), int(Action.STAY), int(Action.STAY), int(Action.STAY), int(Action.STAY), int(Action.STAY)), k=1, order=0, vp=0)


@dataclass
class _FixedGoldGenerator:
    events: tuple[GoldGenerationEvent, ...]

    def generate(self, *_args) -> tuple[GoldGenerationEvent, ...]:
        return self.events


if __name__ == "__main__":
    unittest.main()
