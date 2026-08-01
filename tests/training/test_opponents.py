from __future__ import annotations

import random
import unittest

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.duel import DuelConfig, DuelMechanisms, DuelOrderMode, run_duel
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from simulator.observation.sdk import GameInput
from simulator.types import Action, GameOutput
from training.opponents import EpisodeContext, LeagueEntry, OpponentLeague, OpponentSpec, build_runner
from training.opponents.params import ParamSpace, categorical, choice, int_uniform, uniform
from training.opponents.scripted import FastProbeV3LikeOpponent, GreedyVisibleGoldOpponent, RandomOpponent


class OpponentTests(unittest.TestCase):
    def test_param_space_sampling_is_reproducible(self) -> None:
        space = ParamSpace({
            "mode": choice("nearest", "highest"),
            "risk": uniform(0.0, 2.0),
            "period": int_uniform(3, 7),
            "vp": categorical({"never": 0.75, "periodic_9x9": 0.25}),
        })

        self.assertEqual(space.sample(random.Random(123)), space.sample(random.Random(123)))

    def test_param_space_rejects_bad_specs(self) -> None:
        with self.assertRaises(ValueError):
            ParamSpace({"bad": {"kind": "choice", "choices": []}})
        with self.assertRaises(ValueError):
            ParamSpace({"bad": {"kind": "categorical", "choices": ["a"], "weights": []}})
        with self.assertRaises(ValueError):
            ParamSpace({"bad": {"kind": "uniform", "low": 2.0, "high": 1.0}})
        with self.assertRaises(ValueError):
            ParamSpace({"bad": {"kind": "choice", "choices": ["a"], "typo": True}})

    def test_league_samples_specs_with_entry_params(self) -> None:
        league = OpponentLeague(
            train=(
                LeagueEntry(
                    id="greedy",
                    weight=1.0,
                    spec_template=OpponentSpec(kind="python", name="greedy_visible_gold", params={"vp_policy": "never"}),
                    param_space=ParamSpace({"target_score": choice("nearest", "highest")}),
                ),
            )
        )

        spec = league.sample_spec(random.Random(5), "train")

        self.assertEqual(spec.kind, "python")
        self.assertEqual(spec.name, "greedy_visible_gold")
        self.assertEqual(spec.params["vp_policy"], "never")
        self.assertIn(spec.params["target_score"], ("nearest", "highest"))

    def test_cpp_runners_are_reserved(self) -> None:
        with self.assertRaises(NotImplementedError):
            build_runner(OpponentSpec(kind="cpp_subprocess", name="probe"))

    def test_stay_runner_can_run_in_duel(self) -> None:
        runner = build_runner(OpponentSpec(kind="python", name="stay"))
        runner.reset(11, EpisodeContext(player_id=2, opponent_id=1, map_id=1, seed=11))

        result = run_duel(
            _stay_policy,
            runner,
            config=DuelConfig(
                episode=EpisodeConfig(rules=RulesConfig(round_count=3), seed=11, map_id=1),
                order_mode=DuelOrderMode.AGENT_BEFORE_OPPONENT,
            ),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
            p90_latency_ns={1: 1, 2: 2},
        )

        self.assertEqual(result.state.round_index, 3)
        self.assertEqual(result.rounds[0].player_outputs[2].actions, (int(Action.STAY),) * 6)

    def test_random_opponent_is_reproducible_after_reset(self) -> None:
        opponent = RandomOpponent(stay_prob=0.25)
        ctx = EpisodeContext(player_id=2, opponent_id=1)
        game_input = _basic_input()

        opponent.reset(99, ctx)
        first = opponent.act(game_input)
        opponent.reset(99, ctx)
        second = opponent.act(game_input)

        self.assertEqual(first, second)

    def test_greedy_visible_gold_moves_toward_best_visible_gold(self) -> None:
        opponent = GreedyVisibleGoldOpponent(target_score="highest", avoid_bombs=True)
        opponent.reset(1, EpisodeContext())
        game_input = _basic_input()
        game_input.grid[0][2] = 10

        output = opponent.act(game_input)

        self.assertEqual(output.order, 0)
        self.assertEqual(output.k, 2)
        self.assertEqual(output.actions[:2], (int(Action.RIGHT), int(Action.RIGHT)))

    def test_fast_probe_v3_like_moves_toward_visible_gold(self) -> None:
        opponent = FastProbeV3LikeOpponent(enable_vision=False)
        opponent.reset(1, EpisodeContext())
        game_input = _basic_input()
        game_input.grid[0][2] = 10

        output = opponent.act(game_input)

        self.assertEqual(output.order, 0)
        self.assertEqual(output.k, 6)
        self.assertEqual(output.actions[:2], (int(Action.RIGHT), int(Action.RIGHT)))

    def test_fast_probe_v3_like_buys_vision_after_bad_rounds(self) -> None:
        opponent = FastProbeV3LikeOpponent()
        opponent.reset(1, EpisodeContext())

        first = opponent.act(_basic_input())
        second_input = _basic_input()
        second_input.round = 1
        second = opponent.act(second_input)

        self.assertEqual(first.vp, 0)
        self.assertEqual(second.vp, 2)

    def test_build_runner_rejects_unknown_python_params(self) -> None:
        with self.assertRaises(ValueError):
            build_runner(OpponentSpec(kind="python", name="stay", params={"unexpected": 1}))


def _basic_input() -> GameInput:
    grid = [[0 for _ in range(17)] for _ in range(17)]
    return GameInput(
        round=0,
        grid=grid,
        my_units=[(0, 0), (16, 16)],
        my_units_gold=(0, 0),
        gold_opp=0,
        visible_enemies=[(-1, -1), (-1, -1)],
        visible_npcs=[],
        snapshot_valid=False,
        snapshot=None,
    )


def _stay_policy(_game_input) -> GameOutput:
    return GameOutput(actions=(int(Action.STAY),) * 6, k=3, order=0, vp=0)


def _quiet_mechanisms() -> DuelMechanisms:
    return DuelMechanisms(
        center_gold=CenterGoldGenerator(CenterGoldConfig(center_a=0.0)),
        outer_gold=OuterGoldGenerator(OuterGoldConfig(first_round_offset_weights=((999, 1),))),
        bomb_refresher=BernoulliBombRefresher(BombConfig(0.0)),
    )


if __name__ == "__main__":
    unittest.main()
