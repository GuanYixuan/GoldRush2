from __future__ import annotations

import math
import unittest

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepMechanisms, RoundStepResult
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from simulator.rules.scoring import GameResult
from simulator.state import GameState, PlayerState, UnitState
from simulator.types import Action, GameOutput, Position
from training.opponents import OpponentSpec
from training.rl import BatchRolloutSampler, SingleAgentEnvConfig, SingleAgentGoldRushEnv, TerminalWinPlusMarginPotentialReward


class RewardTests(unittest.TestCase):
    def test_beta_zero_matches_terminal_win_loss_reward(self) -> None:
        reward = TerminalWinPlusMarginPotentialReward(beta=0.0)
        reward.reset(_state(p1_gold=0, p2_gold=0), agent_player_id=1)

        non_terminal = reward(_result(_state(p1_gold=500, p2_gold=0)), agent_player_id=1)
        terminal = reward(
            _result(_state(p1_gold=500, p2_gold=0), terminated=True, winner_id=1),
            agent_player_id=1,
        )

        self.assertEqual(non_terminal, 0.0)
        self.assertEqual(terminal, 1.0)

    def test_margin_increase_produces_positive_potential_reward(self) -> None:
        reward = TerminalWinPlusMarginPotentialReward(gamma=1.0, beta=1.0)
        reward.reset(_state(p1_gold=0, p2_gold=0), agent_player_id=1)

        value = reward(_result(_state(p1_gold=500, p2_gold=0)), agent_player_id=1)

        self.assertAlmostEqual(value, math.tanh(1.0))

    def test_margin_decrease_produces_negative_potential_reward(self) -> None:
        reward = TerminalWinPlusMarginPotentialReward(gamma=1.0, beta=1.0)
        reward.reset(_state(p1_gold=500, p2_gold=0), agent_player_id=1)

        value = reward(_result(_state(p1_gold=0, p2_gold=0)), agent_player_id=1)

        self.assertAlmostEqual(value, -math.tanh(1.0))

    def test_terminal_state_sets_next_potential_to_zero(self) -> None:
        reward = TerminalWinPlusMarginPotentialReward(gamma=1.0, beta=1.0, win_reward=0.0, loss_reward=0.0)
        reward.reset(_state(p1_gold=500, p2_gold=0), agent_player_id=1)

        value = reward(
            _result(_state(p1_gold=1000, p2_gold=0), terminated=True, winner_id=1),
            agent_player_id=1,
        )

        self.assertAlmostEqual(value, -math.tanh(1.0))

    def test_vision_spending_is_included_in_net_margin(self) -> None:
        reward = TerminalWinPlusMarginPotentialReward(gamma=1.0, beta=1.0)
        reward.reset(_state(p1_gold=0, p2_gold=0), agent_player_id=1)

        value = reward(_result(_state(p1_gold=100, p1_vision=200, p2_gold=0)), agent_player_id=1)

        self.assertLess(value, 0.0)
        self.assertAlmostEqual(value, math.tanh(-100.0 / 500.0))

    def test_stateful_reward_requires_reset(self) -> None:
        reward = TerminalWinPlusMarginPotentialReward()

        with self.assertRaisesRegex(Exception, "requires reset"):
            reward(_result(_state(p1_gold=0, p2_gold=0)), agent_player_id=1)

    def test_env_resets_stateful_reward_before_step(self) -> None:
        env = SingleAgentGoldRushEnv(
            config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            reward_fn=TerminalWinPlusMarginPotentialReward(beta=0.0),
        )

        env.reset(seed=7, map_id=1, agent_player_id=1)
        step = env.step(_stay_output())

        self.assertTrue(step.terminated)
        self.assertIn(step.reward, (-1.0, 1.0))

    def test_paired_rollout_resets_shared_stateful_reward(self) -> None:
        reward = TerminalWinPlusMarginPotentialReward(beta=0.0)
        sampler = BatchRolloutSampler(
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            reward_fn=reward,
        )

        batch = sampler.collect(_stay_policy, pair_count=1, seed=11, map_ids=(1,))

        self.assertEqual(batch.episode_count, 2)
        self.assertEqual([len(trajectory.transitions) for trajectory in batch.trajectories], [1, 1])
        self.assertTrue(all(trajectory.total_reward in (-1.0, 1.0) for trajectory in batch.trajectories))


def _state(
    *,
    p1_gold: int,
    p2_gold: int,
    p1_vision: int = 0,
    p2_vision: int = 0,
) -> GameState:
    return GameState(
        round_index=0,
        players={
            1: PlayerState(
                id=1,
                units=[UnitState(id=0, position=Position(0, 0), gold=p1_gold), UnitState(id=1, position=Position(0, 1), gold=0)],
                vision_spent=p1_vision,
            ),
            2: PlayerState(
                id=2,
                units=[UnitState(id=0, position=Position(16, 16), gold=p2_gold), UnitState(id=1, position=Position(16, 15), gold=0)],
                vision_spent=p2_vision,
            ),
        },
    )


def _result(state: GameState, *, terminated: bool = False, winner_id: int | None = None) -> RoundStepResult:
    game_result = None
    if terminated:
        if winner_id is None:
            raise ValueError("winner_id is required for terminal result")
        loser_id = 1 if winner_id == 2 else 2
        game_result = GameResult(
            winner_id=winner_id,
            loser_id=loser_id,
            reason="net_gold",
            net_gold={player_id: state.players[player_id].net_gold for player_id in (1, 2)},
        )
    return RoundStepResult(
        observations={},
        terminated=terminated,
        trace=None,  # type: ignore[arg-type]
        state=state,
        game_result=game_result,
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


def _stay_policy(_observation) -> GameOutput:
    return _stay_output()


if __name__ == "__main__":
    unittest.main()
