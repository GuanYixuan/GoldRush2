from __future__ import annotations

import unittest

import torch

from training.rl import PpoBatch, PpoTransition


class PpoBufferTests(unittest.TestCase):
    def test_batch_stacks_transition_tensors_and_metadata(self) -> None:
        batch = PpoBatch.from_transitions([_transition(reward=1.0, done=True, episode_id="ep-0")])

        self.assertEqual(batch.transition_count, 1)
        self.assertEqual(tuple(batch.spatial_planes.shape), (1, 43, 17, 17))
        self.assertEqual(tuple(batch.scalars.shape), (1, 10))
        self.assertEqual(tuple(batch.fast_scalars.shape), (1, 2))
        self.assertEqual(tuple(batch.critic_planes.shape), (1, 26, 17, 17))
        self.assertEqual(tuple(batch.critic_scalars.shape), (1, 17))
        self.assertEqual(tuple(batch.actions.shape), (1, 6))
        self.assertEqual(batch.episode_ids, ("ep-0",))
        self.assertEqual(batch.map_ids, (1,))

    def test_batch_from_arrays_builds_tensors_and_metadata(self) -> None:
        batch = PpoBatch.from_arrays(
            spatial_planes=torch.zeros(2, 43, 17, 17),
            scalars=torch.zeros(2, 10),
            fast_scalars=torch.zeros(2, 2),
            critic_planes=torch.zeros(2, 26, 17, 17),
            critic_scalars=torch.zeros(2, 17),
            actions=torch.tensor([[4, 4, 4, 4, 4, 4], [0, 1, 2, 3, 4, 0]]),
            k=torch.tensor([3, 2]),
            order=torch.tensor([0, 1]),
            vp=torch.tensor([0, 2]),
            threshold_raw=torch.tensor([-1.0, 0.0]),
            threshold_int=torch.tensor([12, 17]),
            old_logprob=torch.tensor([-1.0, -2.0]),
            values=torch.tensor([0.0, 0.5]),
            rewards=torch.tensor([0.0, 1.0]),
            reward_sums=torch.tensor([0.0, 1.0]),
            taus=torch.tensor([1, 1]),
            fast_success=torch.tensor([False, False]),
            fast_status=torch.tensor([0, 0]),
            dones=torch.tensor([False, True]),
            episode_ids=("ep", "ep"),
            round_indices=torch.tensor([0, 1]),
            map_ids=(1, 1),
            agent_player_ids=torch.tensor([1, 1]),
            infos=({}, {"terminal": True}),
        )

        self.assertEqual(batch.transition_count, 2)
        self.assertEqual(batch.actions.dtype, torch.long)
        self.assertEqual(batch.rewards.dtype, torch.float32)
        self.assertEqual(batch.dones.dtype, torch.bool)
        self.assertEqual(batch.episode_ids, ("ep", "ep"))
        self.assertEqual(batch.infos[1], {"terminal": True})

    def test_batch_from_arrays_supports_gae_boundaries(self) -> None:
        batch = PpoBatch.from_arrays(
            spatial_planes=torch.zeros(3, 43, 17, 17),
            scalars=torch.zeros(3, 10),
            fast_scalars=torch.zeros(3, 2),
            critic_planes=torch.zeros(3, 26, 17, 17),
            critic_scalars=torch.zeros(3, 17),
            actions=torch.zeros(3, 6),
            k=torch.zeros(3),
            order=torch.zeros(3),
            vp=torch.zeros(3),
            threshold_raw=torch.zeros(3),
            threshold_int=torch.full((3,), 12),
            old_logprob=torch.zeros(3),
            values=torch.zeros(3),
            rewards=torch.tensor([1.0, 0.0, 2.0]),
            reward_sums=torch.tensor([1.0, 0.0, 2.0]),
            taus=torch.tensor([1, 1, 1]),
            fast_success=torch.tensor([False, False, False]),
            fast_status=torch.tensor([0, 0, 0]),
            dones=torch.tensor([True, False, True]),
            episode_ids=("a", "b", "b"),
            round_indices=torch.tensor([0, 0, 1]),
            map_ids=(1, 1, 1),
            agent_player_ids=torch.tensor([1, 2, 2]),
            infos=({}, {}, {}),
        )

        batch = batch.compute_gae(gamma=1.0, gae_lambda=1.0, normalize_advantage=False)

        self.assertTrue(torch.allclose(batch.returns, torch.tensor([1.0, 2.0, 2.0])))

    def test_gae_for_terminal_reward_is_hand_computable(self) -> None:
        batch = PpoBatch.from_transitions(
            [
                _transition(reward=0.0, done=False, value=0.0),
                _transition(reward=1.0, done=True, value=0.0),
            ]
        )

        batch = batch.compute_gae(gamma=1.0, gae_lambda=1.0, normalize_advantage=False)

        self.assertTrue(torch.allclose(batch.advantages, torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.allclose(batch.returns, torch.tensor([1.0, 1.0])))

    def test_gae_uses_tau_for_fast_steps(self) -> None:
        batch = PpoBatch.from_arrays(
            spatial_planes=torch.zeros(2, 43, 17, 17),
            scalars=torch.zeros(2, 10),
            fast_scalars=torch.zeros(2, 2),
            critic_planes=torch.zeros(2, 26, 17, 17),
            critic_scalars=torch.zeros(2, 17),
            actions=torch.zeros(2, 6),
            k=torch.zeros(2),
            order=torch.zeros(2),
            vp=torch.zeros(2),
            threshold_raw=torch.zeros(2),
            threshold_int=torch.full((2,), 12),
            old_logprob=torch.zeros(2),
            values=torch.zeros(2),
            rewards=torch.tensor([1.0, 2.0]),
            reward_sums=torch.tensor([3.0, 2.0]),
            taus=torch.tensor([2, 1]),
            fast_success=torch.tensor([True, False]),
            fast_status=torch.tensor([1, 0]),
            dones=torch.tensor([False, True]),
            episode_ids=("a", "a"),
            round_indices=torch.tensor([0, 1]),
            map_ids=(1, 1),
            agent_player_ids=torch.tensor([1, 1]),
            infos=({}, {}),
        )

        batch = batch.compute_gae(gamma=0.5, gae_lambda=1.0, normalize_advantage=False)

        self.assertTrue(torch.allclose(batch.returns, torch.tensor([3.5, 2.0])))

    def test_gae_resets_at_episode_boundaries(self) -> None:
        batch = PpoBatch.from_transitions(
            [
                _transition(reward=1.0, done=True, value=0.0, episode_id="a"),
                _transition(reward=0.0, done=False, value=0.0, episode_id="b"),
                _transition(reward=2.0, done=True, value=0.0, episode_id="b"),
            ]
        )

        batch = batch.compute_gae(gamma=1.0, gae_lambda=1.0, normalize_advantage=False)

        self.assertTrue(torch.allclose(batch.returns, torch.tensor([1.0, 2.0, 2.0])))

    def test_advantage_normalization_keeps_returns_raw(self) -> None:
        batch = PpoBatch.from_transitions(
            [
                _transition(reward=1.0, done=True, value=0.0),
                _transition(reward=3.0, done=True, value=0.0),
            ]
        )

        batch = batch.compute_gae(gamma=1.0, gae_lambda=1.0, normalize_advantage=True)

        assert batch.advantages is not None
        assert batch.returns is not None
        self.assertAlmostEqual(float(batch.advantages.mean().item()), 0.0, places=6)
        self.assertAlmostEqual(float(batch.advantages.std(unbiased=False).item()), 1.0, places=6)
        self.assertTrue(torch.allclose(batch.returns, torch.tensor([1.0, 3.0])))

    def test_minibatches_cover_batch(self) -> None:
        batch = PpoBatch.from_transitions([_transition(reward=float(idx), done=idx == 4) for idx in range(5)])
        batch = batch.compute_gae(gamma=1.0, gae_lambda=1.0)

        minibatches = list(batch.iter_minibatches(minibatch_size=2, shuffle=False))

        self.assertEqual([minibatch.transition_count for minibatch in minibatches], [2, 2, 1])
        self.assertEqual(tuple(minibatches[0].spatial_planes.shape), (2, 43, 17, 17))
        self.assertEqual(tuple(minibatches[0].critic_planes.shape), (2, 26, 17, 17))


def _transition(
    *,
    reward: float,
    done: bool,
    value: float = 0.0,
    episode_id: str = "episode",
) -> PpoTransition:
    return PpoTransition(
        spatial_planes=torch.zeros(43, 17, 17),
        scalars=torch.zeros(10),
        fast_scalars=torch.tensor([0.8, 0.25]),
        critic_planes=torch.zeros(26, 17, 17),
        critic_scalars=torch.zeros(17),
        actions=torch.tensor([4, 4, 4, 4, 4, 4]),
        k=torch.tensor(3),
        order=torch.tensor(0),
        vp=torch.tensor(0),
        threshold_raw=torch.tensor(-1.0),
        threshold_int=torch.tensor(12),
        old_logprob=torch.tensor(-1.0),
        value=torch.tensor(value),
        reward=reward,
        reward_sum=reward,
        tau=1,
        fast_success=False,
        fast_status=0,
        done=done,
        episode_id=episode_id,
        round_index=0,
        map_id=1,
        agent_player_id=1,
        info={},
    )


if __name__ == "__main__":
    unittest.main()
