from __future__ import annotations

import unittest

from simulator.config import EpisodeConfig, RulesConfig
from simulator.envs.round_step import RoundStepMechanisms
from simulator.mechanisms.bombs import BernoulliBombRefresher, BombConfig
from simulator.mechanisms.gold import CenterGoldConfig, CenterGoldGenerator, OuterGoldConfig, OuterGoldGenerator
from simulator.mechanisms.maps import SpawnConfig
from simulator.types import Action, GameOutput
from training.opponents import OpponentSpec
from training.rl import BatchRolloutSampler, RuntimePolicyWrapper, SingleAgentEnvConfig


class RuntimeTrajectoryTests(unittest.TestCase):
    def test_temporal_features_follow_previous_observation(self) -> None:
        policy = _RecordingRuntimePolicy((_right_output(), _down_output(), _stay_output()))
        sampler = _sampler(round_count=3)

        batch = sampler.collect(policy, pair_count=1, seed=101, map_ids=(1,))
        first_episode = policy.episodes[0]

        self.assertEqual(len(batch.trajectories[0].transitions), 3)
        self.assertEqual(first_episode.features[0]["visible_t1_count"], 0)
        self.assertEqual(first_episode.features[1]["visible_t1_count"], first_episode.features[0]["visible_t0_count"])
        self.assertEqual(first_episode.features[2]["visible_t1_count"], first_episode.features[1]["visible_t0_count"])

    def test_paired_episodes_reset_runtime_state(self) -> None:
        policy = _RecordingRuntimePolicy((_right_output(), _stay_output()))
        sampler = _sampler(round_count=2)

        sampler.collect(policy, pair_count=1, seed=102, map_ids=(1,))

        self.assertEqual([episode.player_id for episode in policy.episodes], [1, 2])
        self.assertEqual(policy.episodes[0].features[0]["visible_t1_count"], 0)
        self.assertEqual(policy.episodes[1].features[0]["visible_t1_count"], 0)
        self.assertEqual(policy.episodes[0].features[0]["schema"], "goldrush2_feature_v2")
        self.assertEqual(policy.episodes[1].features[0]["schema"], "goldrush2_feature_v2")

    def test_feature_schema_is_stable_within_episode(self) -> None:
        policy = _RecordingRuntimePolicy((_stay_output(),) * 5)
        sampler = _sampler(round_count=5)

        sampler.collect(policy, pair_count=1, seed=103, map_ids=(1,))

        for episode in policy.episodes:
            self.assertTrue(all(summary["schema"] == "goldrush2_feature_v2" for summary in episode.features))
            self.assertTrue(all(summary["plane_shape"] == (43, 17, 17) for summary in episode.features))
            self.assertTrue(all(summary["scalar_shape"] == (10,) for summary in episode.features))

    def test_runtime_rollout_runs_multi_round_without_round_regression(self) -> None:
        policy = _RecordingRuntimePolicy((_stay_output(),) * 20)
        sampler = _sampler(round_count=20)

        batch = sampler.collect(policy, pair_count=1, seed=104, map_ids=(1,))

        self.assertEqual([len(trajectory.transitions) for trajectory in batch.trajectories], [20, 20])
        self.assertEqual([len(episode.features) for episode in policy.episodes], [20, 20])
        self.assertTrue(all(episode.features[0]["visible_t1_count"] == 0 for episode in policy.episodes))


class _EpisodeRecord:
    def __init__(self, player_id: int) -> None:
        self.player_id = player_id
        self.features: list[dict[str, object]] = []
        self.actions: list[GameOutput] = []


class _RecordingRuntimePolicy:
    def __init__(self, actions: tuple[GameOutput, ...]) -> None:
        self.actions = actions
        self.episodes: list[_EpisodeRecord] = []
        self.current: _EpisodeRecord | None = None
        self.action_index = 0
        self.wrapper = RuntimePolicyWrapper(self._act_from_features)

    def start_episode(self, *, player_id: int) -> None:
        self.current = _EpisodeRecord(player_id)
        self.episodes.append(self.current)
        self.action_index = 0
        self.wrapper.start_episode(player_id=player_id)

    def __call__(self, game_input) -> GameOutput:
        return self.wrapper(game_input)

    def _act_from_features(self, features) -> GameOutput:
        assert self.current is not None
        action = self.actions[min(self.action_index, len(self.actions) - 1)]
        self.current.features.append(_feature_summary(features))
        self.current.actions.append(action)
        self.action_index += 1
        return action


def _feature_summary(features) -> dict[str, object]:
    channel_lookup = {name: idx for idx, name in enumerate(features["channel_names"])}
    visible_t0 = channel_lookup["visible_mask_t0"]
    visible_t1 = channel_lookup["visible_mask_t1"]
    return {
        "schema": features["feature_schema"],
        "plane_shape": tuple(features["planes"].shape),
        "scalar_shape": tuple(features["scalars"].shape),
        "visible_t0_count": int(features["planes"][visible_t0].sum()),
        "visible_t1_count": int(features["planes"][visible_t1].sum()),
    }


def _sampler(*, round_count: int) -> BatchRolloutSampler:
    return BatchRolloutSampler(
        env_config=SingleAgentEnvConfig(
            episode=EpisodeConfig(rules=RulesConfig(round_count=round_count, snapshot_period=1), seed=7, map_id=1),
            opponent_spec=_stay_opponent_spec(),
        ),
        mechanisms=_quiet_mechanisms(),
        spawn=SpawnConfig(npc_ids=()),
    )


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


def _down_output() -> GameOutput:
    return GameOutput(actions=(int(Action.DOWN), int(Action.STAY), int(Action.STAY), int(Action.STAY), int(Action.STAY), int(Action.STAY)), k=1, order=0, vp=0)


if __name__ == "__main__":
    unittest.main()
