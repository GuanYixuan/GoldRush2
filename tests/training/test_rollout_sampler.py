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
from training.rl import BatchRolloutSampler, SingleAgentEnvConfig


class RolloutSamplerTests(unittest.TestCase):
    def test_collect_returns_two_trajectories_per_pair(self) -> None:
        sampler = _sampler(_one_round_episode(), _stay_opponent_spec())

        batch = sampler.collect(_stay_policy, pair_count=3, seed=100, map_ids=(1,))

        self.assertEqual(batch.episode_count, 6)
        self.assertEqual(batch.transition_count, 6)
        self.assertEqual(len({trajectory.pair_id for trajectory in batch.trajectories}), 3)

    def test_transitions_preserve_observation_action_reward_next_observation(self) -> None:
        sampler = _sampler(_one_round_episode(), _stay_opponent_spec())

        batch = sampler.collect(_stay_policy, pair_count=1, seed=100, map_ids=(1,))
        transition = batch.trajectories[0].transitions[0]

        self.assertEqual(transition.observation.round, 0)
        self.assertEqual(transition.action, _stay_output())
        self.assertEqual(transition.reward, -1.0)
        self.assertTrue(transition.terminated)
        self.assertIsNone(transition.next_observation)

    def test_pair_metadata_is_stable(self) -> None:
        specs = (_stay_opponent_spec(),)
        sampler = _sampler(_one_round_episode(), _stay_opponent_spec())

        batch = sampler.collect(_stay_policy, pair_count=1, seed=123, map_ids=(1,), opponent_specs=specs)
        first, second = batch.trajectories

        self.assertEqual(first.pair_id, second.pair_id)
        self.assertEqual(first.seed, second.seed)
        self.assertEqual(first.map_id, second.map_id)
        self.assertEqual(first.opponent_spec, second.opponent_spec)
        self.assertEqual((first.agent_player_id, second.agent_player_id), (1, 2))
        self.assertEqual((first.pair_role, second.pair_role), ("first", "second"))

    def test_metrics_include_win_rate_and_pair_score(self) -> None:
        sampler = _sampler(_one_round_episode(), _stay_opponent_spec())

        batch = sampler.collect(_stay_policy, pair_count=2, seed=50, map_ids=(1,))
        metrics = batch.metrics()

        self.assertEqual(metrics["episode_count"], 4)
        self.assertEqual(metrics["transition_count"], 4)
        self.assertEqual(metrics["win_rate"], 0.0)
        self.assertEqual(metrics["loss_rate"], 1.0)
        self.assertEqual(metrics["mean_total_reward"], -1.0)
        self.assertEqual(metrics["mean_pair_score"], -1.0)
        self.assertEqual(metrics["mean_episode_steps"], 1)

    def test_sampler_is_reproducible(self) -> None:
        mechanisms = _quiet_mechanisms()
        mechanisms.center_gold = _FixedGoldGenerator((GoldGenerationEvent(Position(0, 1), 5),))
        sampler = BatchRolloutSampler(
            env_config=SingleAgentEnvConfig(episode=_one_round_episode(), opponent_spec=_stay_opponent_spec()),
            mechanisms=mechanisms,
            spawn=SpawnConfig(npc_ids=()),
        )

        first = sampler.collect(_right_policy, pair_count=2, seed=77, map_ids=(1,))
        second = sampler.collect(_right_policy, pair_count=2, seed=77, map_ids=(1,))

        self.assertEqual(_signature(first), _signature(second))


def _sampler(episode: EpisodeConfig, opponent_spec: OpponentSpec | None) -> BatchRolloutSampler:
    return BatchRolloutSampler(
        env_config=SingleAgentEnvConfig(episode=episode, opponent_spec=opponent_spec),
        mechanisms=_quiet_mechanisms(),
        spawn=SpawnConfig(npc_ids=()),
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
