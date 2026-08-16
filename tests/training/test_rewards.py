from __future__ import annotations

import math
import unittest

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepMechanisms, RoundStepResult, RoundStepTrace
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from simulator.observation.sdk import GameInput
from simulator.rules.scoring import GameResult
from simulator.state import GameState, PlayerState, UnitState
from simulator.types import Action, GameOutput, GoldGenerationEvent, Position
from training.opponents import OpponentSpec
from training.rl import BatchRolloutSampler, SingleAgentEnvConfig, SingleAgentGoldRushEnv, TerminalWinMarginGoldGainReward


class RewardTests(unittest.TestCase):
    def test_default_dense_terms_disabled_matches_terminal_win_loss_reward(self) -> None:
        reward = TerminalWinMarginGoldGainReward(beta_margin=0.0, beta_gold_gain=0.0, beta_net_gold_gain=0.0)
        reward.reset(_state(p1_gold=0, p2_gold=0), agent_player_id=1)

        non_terminal = reward(_result(_state(p1_gold=500, p2_gold=0)), agent_player_id=1)
        terminal = reward(
            _result(_state(p1_gold=500, p2_gold=0), terminated=True, winner_id=1),
            agent_player_id=1,
        )

        self.assertEqual(non_terminal, 0.0)
        self.assertEqual(terminal, 1.0)

    def test_margin_increase_produces_positive_potential_reward(self) -> None:
        reward = TerminalWinMarginGoldGainReward(gamma=1.0, beta_win=0.0, beta_margin=1.0, beta_net_gold_gain=0.0)
        reward.reset(_state(p1_gold=0, p2_gold=0), agent_player_id=1)

        value = reward(_result(_state(p1_gold=500, p2_gold=0)), agent_player_id=1)

        self.assertAlmostEqual(value, math.tanh(1.0))

    def test_margin_decrease_produces_negative_potential_reward(self) -> None:
        reward = TerminalWinMarginGoldGainReward(gamma=1.0, beta_win=0.0, beta_margin=1.0, beta_net_gold_gain=0.0)
        reward.reset(_state(p1_gold=500, p2_gold=0), agent_player_id=1)

        value = reward(_result(_state(p1_gold=0, p2_gold=0)), agent_player_id=1)

        self.assertAlmostEqual(value, -math.tanh(1.0))

    def test_terminal_state_sets_next_potential_to_zero(self) -> None:
        reward = TerminalWinMarginGoldGainReward(gamma=1.0, beta_win=0.0, beta_margin=1.0, beta_net_gold_gain=0.0)
        reward.reset(_state(p1_gold=500, p2_gold=0), agent_player_id=1)

        value = reward(
            _result(_state(p1_gold=1000, p2_gold=0), terminated=True, winner_id=1),
            agent_player_id=1,
        )

        self.assertAlmostEqual(value, -math.tanh(1.0))

    def test_vision_spending_is_included_in_net_margin(self) -> None:
        reward = TerminalWinMarginGoldGainReward(gamma=1.0, beta_win=0.0, beta_margin=1.0, beta_net_gold_gain=0.0)
        reward.reset(_state(p1_gold=0, p2_gold=0), agent_player_id=1)

        value = reward(_result(_state(p1_gold=100, p1_vision=200, p2_gold=0)), agent_player_id=1)

        self.assertLess(value, 0.0)
        self.assertAlmostEqual(value, math.tanh(-100.0 / 500.0))

    def test_gold_gain_reward_uses_agent_gross_gold_delta(self) -> None:
        reward = TerminalWinMarginGoldGainReward(
            gamma=1.0,
            beta_win=0.0,
            beta_margin=0.0,
            beta_gold_gain=2.0,
            beta_net_gold_gain=0.0,
            gold_gain_scale=100.0,
        )
        reward.reset(_state(p1_gold=0, p2_gold=0), agent_player_id=1)

        value = reward(_result(_state(p1_gold=50, p1_vision=200, p2_gold=0)), agent_player_id=1)

        self.assertEqual(value, 1.0)
        self.assertIsNotNone(reward.last_components)
        assert reward.last_components is not None
        self.assertEqual(reward.last_components["gold_gain"], 50.0)
        self.assertEqual(reward.last_components["gold_gain_reward"], 1.0)

    def test_gold_gain_reward_is_clipped(self) -> None:
        reward = TerminalWinMarginGoldGainReward(
            gamma=1.0,
            beta_win=0.0,
            beta_margin=0.0,
            beta_gold_gain=2.0,
            beta_net_gold_gain=0.0,
            gold_gain_scale=100.0,
        )
        reward.reset(_state(p1_gold=0, p2_gold=0), agent_player_id=1)

        value = reward(_result(_state(p1_gold=300, p2_gold=0)), agent_player_id=1)

        self.assertEqual(value, 2.0)

    def test_stateful_reward_requires_reset(self) -> None:
        reward = TerminalWinMarginGoldGainReward()

        with self.assertRaisesRegex(Exception, "requires reset"):
            reward(_result(_state(p1_gold=0, p2_gold=0)), agent_player_id=1)

    def test_env_resets_stateful_reward_before_step(self) -> None:
        env = SingleAgentGoldRushEnv(
            config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            reward_fn=TerminalWinMarginGoldGainReward(beta_margin=0.0, beta_gold_gain=0.0, beta_net_gold_gain=0.0),
        )

        env.reset(seed=7, map_id=1, agent_player_id=1)
        step = env.step(_stay_output())

        self.assertTrue(step.terminated)
        self.assertIn(step.reward, (-1.0, 1.0))

    def test_paired_rollout_resets_shared_stateful_reward(self) -> None:
        reward = TerminalWinMarginGoldGainReward(beta_margin=0.0, beta_gold_gain=0.0, beta_net_gold_gain=0.0)
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

    def test_net_gold_gain_reward_uses_agent_net_gold_delta(self) -> None:
        reward = TerminalWinMarginGoldGainReward(
            gamma=1.0,
            beta_win=0.0,
            beta_margin=0.0,
            beta_gold_gain=0.0,
            beta_net_gold_gain=0.1,
            net_gold_gain_scale=50.0,
        )
        reward.reset(_state(p1_gold=0, p2_gold=0), agent_player_id=1)

        value = reward(_result(_state(p1_gold=25, p2_gold=0)), agent_player_id=1)

        self.assertEqual(value, 0.05)
        self.assertIsNotNone(reward.last_components)
        assert reward.last_components is not None
        self.assertEqual(reward.last_components["net_gold_gain"], 25.0)
        self.assertEqual(reward.last_components["clipped_net_gold_gain"], 0.5)
        self.assertEqual(reward.last_components["net_gold_gain_reward"], 0.05)

    def test_net_gold_gain_reward_charges_vision_spending(self) -> None:
        reward = TerminalWinMarginGoldGainReward(
            gamma=1.0,
            beta_win=0.0,
            beta_margin=0.0,
            beta_gold_gain=0.0,
            beta_net_gold_gain=0.1,
            net_gold_gain_scale=50.0,
        )
        reward.reset(_state(p1_gold=100, p1_vision=0, p2_gold=0), agent_player_id=1)

        value = reward(_result(_state(p1_gold=100, p1_vision=3, p2_gold=0)), agent_player_id=1)

        self.assertAlmostEqual(value, -0.006)
        assert reward.last_components is not None
        self.assertEqual(reward.last_components["net_gold_gain"], -3.0)

    def test_net_gold_gain_reward_is_clipped_to_signed_range(self) -> None:
        reward = TerminalWinMarginGoldGainReward(
            gamma=1.0,
            beta_win=0.0,
            beta_margin=0.0,
            beta_gold_gain=0.0,
            beta_net_gold_gain=0.1,
            net_gold_gain_scale=50.0,
        )
        reward.reset(_state(p1_gold=0, p1_vision=0, p2_gold=0), agent_player_id=1)

        positive = reward(_result(_state(p1_gold=200, p1_vision=0, p2_gold=0)), agent_player_id=1)
        negative = reward(_result(_state(p1_gold=200, p1_vision=300, p2_gold=0)), agent_player_id=1)

        self.assertEqual(positive, 0.1)
        self.assertEqual(negative, -0.1)

    def test_default_reward_uses_net_gold_gain_not_gross_gold_gain(self) -> None:
        reward = TerminalWinMarginGoldGainReward()

        self.assertEqual(reward.beta_gold_gain, 0.0)
        self.assertEqual(reward.beta_net_gold_gain, 0.1)
        self.assertEqual(reward.net_gold_gain_scale, 50.0)

    def test_vision_info_reward_ignores_no_purchase(self) -> None:
        reward = TerminalWinMarginGoldGainReward(
            beta_win=0.0,
            beta_margin=0.0,
            beta_gold_gain=0.0,
            beta_net_gold_gain=0.0,
            beta_vision_info=1.0,
        )
        reward.reset(_vision_state(round_index=0), agent_player_id=1)

        value = reward(
            _vision_result(round_index=1, vp=0, gold={(8, 11): 7}, next_gold_generated=()),
            agent_player_id=1,
        )

        self.assertEqual(value, 0.0)
        assert reward.last_components is not None
        self.assertEqual(reward.last_components["vision_info_gold"], 0.0)

    def test_vision_info_reward_ignores_base_radius_gold(self) -> None:
        reward = TerminalWinMarginGoldGainReward(
            beta_win=0.0,
            beta_margin=0.0,
            beta_gold_gain=0.0,
            beta_net_gold_gain=0.0,
            beta_vision_info=1.0,
        )
        reward.reset(_vision_state(round_index=0), agent_player_id=1)

        value = reward(
            _vision_result(round_index=1, vp=1, gold={(8, 10): 9}, next_gold_generated=()),
            agent_player_id=1,
        )

        self.assertEqual(value, 0.0)
        assert reward.last_components is not None
        self.assertEqual(reward.last_components["vision_info_gold"], 0.0)

    def test_vision_info_reward_counts_fresh_extra_ring_gold(self) -> None:
        reward = TerminalWinMarginGoldGainReward(
            beta_win=0.0,
            beta_margin=0.0,
            beta_gold_gain=0.0,
            beta_net_gold_gain=0.0,
            beta_vision_info=1.0,
            vision_info_scale=100.0,
            vision_info_reward_cap=1.0,
        )
        reward.reset(_vision_state(round_index=0), agent_player_id=1)

        value = reward(
            _vision_result(round_index=1, vp=1, gold={(8, 11): 7}, next_gold_generated=()),
            agent_player_id=1,
        )

        self.assertEqual(value, 0.07)
        assert reward.last_components is not None
        self.assertEqual(reward.last_components["vision_info_gold"], 7.0)
        self.assertEqual(reward.last_components["vision_info_cells"], 1.0)
        self.assertEqual(reward.last_components["vision_info_reward"], 0.07)

    def test_vision_info_reward_requires_fresh_or_increased_gold(self) -> None:
        reward = TerminalWinMarginGoldGainReward(
            beta_win=0.0,
            beta_margin=0.0,
            beta_gold_gain=0.0,
            beta_net_gold_gain=0.0,
            beta_vision_info=1.0,
            vision_info_scale=100.0,
            vision_info_reward_cap=1.0,
            vision_info_recent_window=5,
        )
        reward.reset(_vision_state(round_index=0, active_vision_radius=3), agent_player_id=1)

        stale_value = reward(
            _vision_result(round_index=1, vp=1, gold={(8, 11): 7}, next_gold_generated=()),
            agent_player_id=1,
        )
        increased_value = reward(
            _vision_result(
                round_index=2,
                vp=1,
                gold={(8, 11): 8},
                next_gold_generated=(GoldGenerationEvent(Position(8, 11), 1),),
            ),
            agent_player_id=1,
        )

        self.assertEqual(stale_value, 0.0)
        self.assertEqual(increased_value, 0.08)
        assert reward.last_components is not None
        self.assertEqual(reward.last_components["vision_info_gold"], 8.0)

    def test_vision_info_reward_is_capped(self) -> None:
        reward = TerminalWinMarginGoldGainReward(
            beta_win=0.0,
            beta_margin=0.0,
            beta_gold_gain=0.0,
            beta_net_gold_gain=0.0,
            beta_vision_info=1.0,
            vision_info_scale=100.0,
            vision_info_reward_cap=0.03,
        )
        reward.reset(_vision_state(round_index=0), agent_player_id=1)

        value = reward(
            _vision_result(round_index=1, vp=1, gold={(8, 11): 70}, next_gold_generated=()),
            agent_player_id=1,
        )

        self.assertEqual(value, 0.03)


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


def _vision_state(
    *,
    round_index: int,
    active_vision_radius: int = 2,
    gold: dict[tuple[int, int], int] | None = None,
) -> GameState:
    return GameState(
        round_index=round_index,
        players={
            1: PlayerState(
                id=1,
                units=[UnitState(id=0, position=Position(8, 8)), UnitState(id=1, position=Position(0, 0))],
                active_vision_radius=active_vision_radius,
            ),
            2: PlayerState(
                id=2,
                units=[UnitState(id=0, position=Position(16, 16)), UnitState(id=1, position=Position(16, 15))],
            ),
        },
        gold={Position(row, col): amount for (row, col), amount in (gold or {}).items()},
    )


def _vision_result(
    *,
    round_index: int,
    vp: int,
    gold: dict[tuple[int, int], int],
    next_gold_generated: tuple[GoldGenerationEvent, ...],
) -> RoundStepResult:
    state = _vision_state(round_index=round_index, active_vision_radius=3 if vp == 1 else 4 if vp == 2 else 2, gold=gold)
    return RoundStepResult(
        observations={1: _vision_observation(round_index=round_index, gold=gold)},
        terminated=False,
        trace=RoundStepTrace(
            round_index=round_index - 1,
            first_player_id=2,
            player_outputs={1: GameOutput(actions=(int(Action.STAY),) * 6, k=3, order=0, vp=vp), 2: _stay_output()},
            npc_order=(),
            npc_actions={},
            gold_generated=(),
            bomb_refresh_event=None,  # type: ignore[arg-type]
            transition_result=None,  # type: ignore[arg-type]
        ),
        state=state,
        next_round_gold_generated=next_gold_generated,
    )


def _vision_observation(*, round_index: int, gold: dict[tuple[int, int], int]) -> GameInput:
    grid = [[0 for _col in range(17)] for _row in range(17)]
    for (row, col), amount in gold.items():
        grid[row][col] = amount
    return GameInput(
        round=round_index,
        grid=grid,
        my_units=[(8, 8), (0, 0)],
        my_units_gold=(0, 0),
        gold_opp=0,
        visible_enemies=[(-1, -1), (-1, -1)],
        visible_npcs=[],
        snapshot_valid=False,
        snapshot=None,
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
