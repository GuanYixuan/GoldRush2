from __future__ import annotations

import unittest

import torch

from training.rl import PpoBatch, PpoTransition


class PpoBufferTests(unittest.TestCase):
    def test_batch_stacks_transition_tensors_and_metadata(self) -> None:
        batch = PpoBatch.from_transitions([_transition(reward=1.0, done=True, episode_id="ep-0")])

        self.assertEqual(batch.transition_count, 1)
        self.assertEqual(tuple(batch.spatial_planes.shape), (1, 38, 17, 17))
        self.assertEqual(tuple(batch.scalars.shape), (1, 10))
        self.assertEqual(tuple(batch.actions.shape), (1, 6))
        self.assertEqual(batch.episode_ids, ("ep-0",))
        self.assertEqual(batch.map_ids, (1,))

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
        self.assertEqual(tuple(minibatches[0].spatial_planes.shape), (2, 38, 17, 17))


def _transition(
    *,
    reward: float,
    done: bool,
    value: float = 0.0,
    episode_id: str = "episode",
) -> PpoTransition:
    return PpoTransition(
        spatial_planes=torch.zeros(38, 17, 17),
        scalars=torch.zeros(10),
        actions=torch.tensor([4, 4, 4, 4, 4, 4]),
        k=torch.tensor(3),
        order=torch.tensor(0),
        vp=torch.tensor(0),
        old_logprob=torch.tensor(-1.0),
        value=torch.tensor(value),
        reward=reward,
        done=done,
        episode_id=episode_id,
        round_index=0,
        map_id=1,
        agent_player_id=1,
        info={},
    )


if __name__ == "__main__":
    unittest.main()
