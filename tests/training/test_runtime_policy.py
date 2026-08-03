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
from training.rl import BatchRolloutSampler, RuntimePolicyWrapper, SingleAgentEnvConfig, evaluate_policy
from training.rl.runtime_policy import RuntimeAwarePolicy


class RuntimePolicyTests(unittest.TestCase):
    def test_wrapper_observes_features_and_commits_actions(self) -> None:
        seen = []
        policy = RuntimePolicyWrapper(lambda features: _record_and_stay(features, seen))
        game_input = _basic_rollout_input(round_index=0)

        action0 = policy(game_input)
        game_input.round = 1
        action1 = policy(game_input)

        self.assertEqual(action0, _stay_output())
        self.assertEqual(action1, _stay_output())
        self.assertEqual(seen[0]["scalars"][4], 0.0)
        self.assertEqual(seen[1]["scalars"][4], 1.0)

    def test_sampler_resets_runtime_policy_for_each_swapped_episode(self) -> None:
        call_counts = []
        policy = _RecordingRuntimePolicy()
        sampler = BatchRolloutSampler(
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=_quiet_mechanisms(),
            spawn=SpawnConfig(npc_ids=()),
        )

        batch = sampler.collect(policy, pair_count=1, seed=5, map_ids=(1,))
        call_counts.extend(policy.start_episode_calls)

        self.assertEqual(batch.episode_count, 2)
        self.assertEqual(call_counts, [1, 2])
        self.assertEqual(policy.first_scalar_values, [0.0, 0.0])

    def test_runtime_policy_can_run_through_evaluation(self) -> None:
        config = _eval_config()
        policy = RuntimePolicyWrapper(lambda _features: _stay_output())

        result = evaluate_policy(policy, config)

        self.assertEqual(result.batch.episode_count, 2)
        self.assertEqual(result.metrics["case_count"], 1)


def _record_and_stay(features, seen) -> GameOutput:
    seen.append(features)
    return _stay_output()


class _RecordingRuntimePolicy:
    def __init__(self) -> None:
        self.wrapper = RuntimePolicyWrapper(self._act_from_features)
        self.start_episode_calls: list[int] = []
        self.first_scalar_values: list[float] = []

    def start_episode(self, *, player_id: int) -> None:
        self.start_episode_calls.append(player_id)
        self.wrapper.start_episode(player_id=player_id)

    def __call__(self, game_input):
        return self.wrapper(game_input)

    def _act_from_features(self, features) -> GameOutput:
        self.first_scalar_values.append(float(features["scalars"][4]))
        return _stay_output()


def _eval_config():
    from training.rl import EvaluationCase, EvaluationConfig

    return EvaluationConfig(
        env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
        cases=(EvaluationCase(seed=7, map_id=1),),
        mechanisms=_quiet_mechanisms(),
        spawn=SpawnConfig(npc_ids=()),
    )


def _basic_rollout_input(round_index: int):
    from simulator.observation.sdk import GameInput

    return GameInput(
        round=round_index,
        grid=[[0 for _ in range(17)] for _ in range(17)],
        my_units=[(0, 0), (16, 16)],
        my_units_gold=(0, 0),
        gold_opp=0,
        visible_enemies=[(-1, -1), (-1, -1)],
        visible_npcs=[],
        snapshot_valid=False,
        snapshot=None,
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


@dataclass
class _FixedGoldGenerator:
    events: tuple[GoldGenerationEvent, ...]

    def generate(self, *_args) -> tuple[GoldGenerationEvent, ...]:
        return self.events


if __name__ == "__main__":
    unittest.main()
