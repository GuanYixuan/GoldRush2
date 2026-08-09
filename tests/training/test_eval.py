from __future__ import annotations

import unittest
from dataclasses import dataclass

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepMechanisms
from simulator.errors import SimulatorRuleError
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from simulator.types import Action, GameOutput, GoldGenerationEvent, Position
from training.opponents import OpponentSpec
from training.rl import EvaluationCase, EvaluationConfig, SingleAgentEnvConfig, evaluate_policy
from training.rl.eval import _event_count
from training.rl.rollout import Trajectory, Transition


class EvaluationTests(unittest.TestCase):
    def test_evaluate_policy_returns_batch_and_metrics(self) -> None:
        config = EvaluationConfig(
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            cases=(EvaluationCase(seed=10, map_id=1), EvaluationCase(seed=11, map_id=1)),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )

        result = evaluate_policy(_stay_policy, config)

        self.assertEqual(result.batch.episode_count, 4)
        self.assertEqual(result.metrics["case_count"], 2)
        self.assertEqual(result.metrics["pair_count"], 2)
        self.assertEqual(result.metrics["win_rate"], 0.0)
        self.assertEqual(result.metrics["mean_pair_score"], -1.0)

    def test_eval_metrics_include_net_gold_margin(self) -> None:
        mechanisms = _quiet_mechanisms()
        mechanisms.center_gold = _FixedGoldGenerator((GoldGenerationEvent(Position(0, 1), 5),))
        config = EvaluationConfig(
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            cases=(EvaluationCase(seed=10, map_id=1),),
            mechanisms=mechanisms,
            spawn=SpawnConfig(npc_ids=()),
        )

        result = evaluate_policy(_right_policy, config)

        self.assertEqual(result.metrics["net_gold_margin_mean"], 2)
        self.assertEqual(result.metrics["net_gold_margin_min"], 0)
        self.assertEqual(result.metrics["net_gold_margin_max"], 4)
        self.assertIn("mean_agent_vision_spent", result.metrics)
        self.assertIn("mean_agent_pickups", result.metrics)

    def test_event_count_aggregates_all_trajectory_transitions(self) -> None:
        trajectory = Trajectory(
            episode_id="episode",
            pair_id="pair",
            pair_role="first",
            seed=1,
            map_id=1,
            agent_player_id=1,
            opponent_spec=_stay_opponent_spec(),
            transitions=(
                Transition(
                    observation=None,
                    action=_stay_output(),
                    reward=0.0,
                    next_observation=None,
                    terminated=False,
                    truncated=False,
                    info={"events": {"pickups": {1: 1, 2: 0}, "bomb_triggers": {1: 0, 2: 1}}},
                ),
                Transition(
                    observation=None,
                    action=_stay_output(),
                    reward=0.0,
                    next_observation=None,
                    terminated=True,
                    truncated=False,
                    info={"events": {"pickups": {1: 2, 2: 1}, "bomb_triggers": {1: 1, 2: 0}}},
                ),
            ),
        )

        self.assertEqual(_event_count(trajectory, "pickups", 1), 3)
        self.assertEqual(_event_count(trajectory, "pickups", 2), 1)
        self.assertEqual(_event_count(trajectory, "bomb_triggers", 1), 1)
        self.assertEqual(_event_count(trajectory, "bomb_triggers", 2), 1)

    def test_eval_is_reproducible(self) -> None:
        config = EvaluationConfig(
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            cases=(EvaluationCase(seed=20, map_id=1), EvaluationCase(seed=21, map_id=2)),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )

        first = evaluate_policy(_stay_policy, config)
        second = evaluate_policy(_stay_policy, config)

        self.assertEqual(first.metrics, second.metrics)
        self.assertEqual(_signature(first.batch), _signature(second.batch))

    def test_eval_rejects_empty_cases_and_bad_factories(self) -> None:
        with self.assertRaisesRegex(SimulatorRuleError, "at least one case"):
            EvaluationConfig(env_config=SingleAgentEnvConfig(episode=_one_round_episode()), cases=())

        with self.assertRaisesRegex(SimulatorRuleError, "seeds_per_slice"):
            EvaluationConfig.matrix_by_slice(
                env_config=SingleAgentEnvConfig(episode=_one_round_episode()),
                map_ids=(1,),
                opponent_specs=(_stay_opponent_spec(),),
                base_seed=1,
                seeds_per_slice=0,
            )

    def test_anchor_grid_uses_common_seeds_across_maps_and_opponents(self) -> None:
        config = EvaluationConfig.anchor_grid(
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=None),
            seeds=(1, 2),
            map_ids=(1, 2),
            opponent_specs=(_stay_opponent_spec(), _random_stay_opponent_spec()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )

        result = evaluate_policy(_stay_policy, config)

        self.assertEqual(len(config.cases), 8)
        self.assertEqual(result.metrics["case_count"], 8)
        self.assertEqual(result.metrics["map_ids"], (1, 2))
        self.assertEqual(result.metrics["opponent_count"], 2)

    def test_matrix_by_slice_uses_distinct_seed_ranges_per_map_opponent_slice(self) -> None:
        config = EvaluationConfig.matrix_by_slice(
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=None),
            map_ids=(1, 2),
            opponent_specs=(_stay_opponent_spec(), _random_stay_opponent_spec()),
            base_seed=100,
            seeds_per_slice=2,
            slice_seed_stride=10,
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )

        self.assertEqual([case.seed for case in config.cases], [100, 101, 110, 111, 120, 121, 130, 131])
        result = evaluate_policy(_stay_policy, config)
        self.assertEqual(result.batch.episode_count, 16)
        self.assertEqual(result.metrics["opponent_count"], 2)


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


def _random_stay_opponent_spec() -> OpponentSpec:
    return OpponentSpec(kind="python", name="random", params={"stay_prob": 1.0, "vp_policy": "never"})


def _stay_output() -> GameOutput:
    return GameOutput(actions=(int(Action.STAY),) * 6, k=3, order=0, vp=0)


def _right_output() -> GameOutput:
    return GameOutput(actions=(int(Action.RIGHT), int(Action.STAY), int(Action.STAY), int(Action.STAY), int(Action.STAY), int(Action.STAY)), k=1, order=0, vp=0)


def _stay_policy(_observation) -> GameOutput:
    return _stay_output()


def _right_policy(_observation) -> GameOutput:
    return _right_output()


def _signature(batch) -> tuple[tuple[str, int, int, float, int], ...]:
    return tuple(
        (
            trajectory.pair_id,
            trajectory.agent_player_id,
            trajectory.map_id,
            trajectory.total_reward,
            trajectory.terminal_info["scores"]["net_gold"][trajectory.agent_player_id],
        )
        for trajectory in batch.trajectories
    )


@dataclass
class _FixedGoldGenerator:
    events: tuple[GoldGenerationEvent, ...]

    def generate(self, *_args) -> tuple[GoldGenerationEvent, ...]:
        return self.events


if __name__ == "__main__":
    unittest.main()
