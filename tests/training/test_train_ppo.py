from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from simulator.config import EpisodeConfig, RulesConfig
from training.models import PolicyNetworkConfig
from training.rl import EvaluationCase, PpoConfig
from training.rl import MultiprocessRolloutConfig
from training.scripts.train_ppo import (
    TrainPpoConfig,
    _same_role_reverse_fraction,
    load_checkpoint,
    opponent_spec_from_name,
    run_training,
)


class TrainPpoTests(unittest.TestCase):
    def test_same_role_reverse_fraction_excludes_role_boundary(self) -> None:
        actions = torch.tensor([[0, 1, 3, 2, 3, 4]])
        k = torch.tensor([3])

        result = _same_role_reverse_fraction(actions, k)

        self.assertEqual(result, 0.5)

    def test_smoke_training_writes_metrics_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _smoke_config(output_dir=Path(tmpdir))

            result = run_training(config)

            self.assertEqual(result.final_update, 1)
            self.assertTrue((Path(tmpdir) / "config.json").exists())
            self.assertTrue((Path(tmpdir) / "metrics.jsonl").exists())
            self.assertTrue(result.latest_checkpoint.exists())
            self.assertEqual(len(result.train_metrics), 1)
            records = _read_jsonl(Path(tmpdir) / "metrics.jsonl")
            self.assertEqual(records[0]["kind"], "train")
            self.assertEqual(records[0]["update"], 1)
            for key in _diagnostic_keys():
                self.assertIn(key, records[0])
                self.assertTrue(math.isfinite(float(records[0][key])))
            self.assertEqual(records[0]["optimizer_phase"], "full")
            self.assertEqual(records[0]["phase"], "ppo")
            self.assertEqual(records[0]["rollout_seed"], config.seed)
            self.assertFalse(records[0]["fixed_rollout_seeds"])
            self.assertAlmostEqual(float(records[0]["critic_lr"]), config.critic_learning_rate)
            self.assertAlmostEqual(float(records[0]["actor_lr"]), config.actor_learning_rate / config.actor_lr_ramp_updates)

    def test_checkpoint_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            first = run_training(_smoke_config(output_dir=output_dir))
            second = run_training(
                _smoke_config(
                    output_dir=output_dir,
                    total_updates=2,
                    resume_checkpoint=first.latest_checkpoint,
                )
            )

            self.assertEqual(second.final_update, 2)
            _model, checkpoint = load_checkpoint(second.latest_checkpoint, model_config=_small_model_config())
            self.assertEqual(checkpoint["update_index"], 2)
            records = _read_jsonl(output_dir / "metrics.jsonl")
            self.assertEqual([record["update"] for record in records if record["kind"] == "train"], [1, 2])

    def test_critic_warmup_and_actor_ramp_are_ppo_phase_relative(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _smoke_config(
                output_dir=Path(tmpdir),
                total_updates=3,
                critic_warmup_updates=1,
                actor_lr_ramp_updates=2,
            )

            result = run_training(config)

            self.assertEqual(result.final_update, 3)
            records = _read_jsonl(Path(tmpdir) / "metrics.jsonl")
            train_records = [record for record in records if record["kind"] == "train"]
            self.assertEqual([record["optimizer_phase"] for record in train_records], ["critic", "full", "full"])
            self.assertIsNone(train_records[0]["actor_lr"])
            self.assertAlmostEqual(float(train_records[0]["critic_lr"]), config.critic_learning_rate)
            self.assertAlmostEqual(float(train_records[1]["actor_lr"]), config.actor_learning_rate * 0.5)
            self.assertAlmostEqual(float(train_records[2]["actor_lr"]), config.actor_learning_rate)

    def test_fork_ppo_checkpoint_loads_model_without_resuming_update(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmp, tempfile.TemporaryDirectory() as fork_tmp:
            first = run_training(_smoke_config(output_dir=Path(first_tmp), total_updates=2))

            forked = run_training(
                _smoke_config(
                    output_dir=Path(fork_tmp),
                    total_updates=1,
                    fork_ppo_checkpoint=first.latest_checkpoint,
                )
            )

            self.assertEqual(forked.final_update, 1)
            records = _read_jsonl(Path(fork_tmp) / "metrics.jsonl")
            self.assertEqual(records[0]["init_source"], "ppo_fork")
            self.assertEqual(records[0]["fork_checkpoint_update"], 2)

    def test_rollout_seed_base_can_be_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _smoke_config(
                output_dir=Path(tmpdir),
                total_updates=2,
                rollout_seed_base=500,
                advance_rollout_seed=False,
            )

            run_training(config)

            records = _read_jsonl(Path(tmpdir) / "metrics.jsonl")
            train_records = [record for record in records if record["kind"] == "train"]
            self.assertEqual([record["rollout_seed"] for record in train_records], [500, 500])
            self.assertTrue(train_records[0]["fixed_rollout_seeds"])

    def test_eval_interval_runs_tiny_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _smoke_config(
                output_dir=Path(tmpdir),
                eval_interval=1,
                eval_cases=(
                    EvaluationCase(
                        seed=123,
                        map_id=1,
                        opponent_spec=opponent_spec_from_name("stay"),
                        tag="tiny_eval",
                    ),
                ),
            )

            result = run_training(config)

            self.assertEqual(len(result.eval_metrics), 1)
            self.assertEqual(result.eval_metrics[0]["kind"], "eval")
            self.assertIn("win_rate", result.eval_metrics[0]["metrics"])
            records = _read_jsonl(Path(tmpdir) / "metrics.jsonl")
            self.assertEqual([record["kind"] for record in records], ["train", "eval"])

    def test_multiprocess_rollout_training_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _smoke_config(
                output_dir=Path(tmpdir),
                total_updates=2,
                rollout_mode="multiprocess",
                multiprocess_rollout=MultiprocessRolloutConfig(
                    num_workers=2,
                    max_inference_batch_size=4,
                    inference_timeout_ms=1.0,
                ),
            )

            result = run_training(config)

            self.assertEqual(result.final_update, 2)
            self.assertEqual(result.train_metrics[0]["rollout_mode"], "multiprocess")
            self.assertEqual(result.train_metrics[0]["first_episodes"], 1)
            self.assertEqual(result.train_metrics[0]["second_episodes"], 1)
            self.assertFalse(result.train_metrics[0]["worker_pool_reused"])
            self.assertTrue(result.train_metrics[1]["worker_pool_reused"])
            self.assertEqual(result.train_metrics[1]["worker_startup_ms"], 0.0)
            self.assertTrue(result.latest_checkpoint.exists())

    def test_cli_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                "-m",
                "training.scripts.train_ppo",
                "--total-updates",
                "1",
                "--pair-count",
                "1",
                "--map-ids",
                "1",
                "--opponents",
                "stay",
                "--round-count",
                "1",
                "--beta-margin",
                "0.0",
                "--output-dir",
                tmpdir,
                "--model-width",
                "16",
                "--model-blocks",
                "1",
                "--scalar-hidden",
                "16",
                "16",
                "--actor-hidden",
                "32",
                "--critic-hidden",
                "32",
                "16",
                "--decoder-hidden",
                "16",
                "--decoder-embedding",
                "4",
                "--ppo-minibatch-size",
                "2",
                "--ppo-update-epochs",
                "1",
                "--ppo-target-kl",
                "-1",
            ]

            completed = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")

            payload = json.loads(completed.stdout)
            self.assertEqual(payload["final_update"], 1)
            self.assertTrue(Path(payload["latest_checkpoint"]).exists())


def _smoke_config(
    *,
    output_dir: Path,
    total_updates: int = 1,
    resume_checkpoint: Path | None = None,
    eval_interval: int | None = None,
    eval_cases: tuple[EvaluationCase, ...] = (),
    rollout_mode: str = "serial",
    multiprocess_rollout: MultiprocessRolloutConfig | None = None,
    critic_warmup_updates: int = 0,
    actor_lr_ramp_updates: int = 100,
    fork_ppo_checkpoint: Path | None = None,
    rollout_seed_base: int | None = None,
    advance_rollout_seed: bool = True,
) -> TrainPpoConfig:
    return TrainPpoConfig(
        seed=77,
        rollout_seed_base=rollout_seed_base,
        advance_rollout_seed=advance_rollout_seed,
        total_updates=total_updates,
        pair_count=1,
        map_ids=(1,),
        opponent_specs=(opponent_spec_from_name("stay"),),
        output_dir=output_dir,
        save_interval=1,
        eval_interval=eval_interval,
        eval_cases=eval_cases,
        rollout_mode=rollout_mode,
        multiprocess_rollout=multiprocess_rollout or MultiprocessRolloutConfig(),
        beta_margin=0.0,
        critic_warmup_updates=critic_warmup_updates,
        actor_lr_ramp_updates=actor_lr_ramp_updates,
        model=_small_model_config(),
        ppo=PpoConfig(update_epochs=1, minibatch_size=2, target_joint_kl=None),
        episode=EpisodeConfig(rules=RulesConfig(round_count=1, snapshot_period=1), seed=7, map_id=1),
        resume_checkpoint=resume_checkpoint,
        fork_ppo_checkpoint=fork_ppo_checkpoint,
    )


def _small_model_config() -> PolicyNetworkConfig:
    return PolicyNetworkConfig(
        width=16,
        residual_blocks=1,
        se_reduction=4,
        scalar_hidden=(16, 16),
        actor_hidden=32,
        critic_hidden=(32, 16),
        decoder_hidden=16,
        decoder_embedding=4,
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _diagnostic_keys() -> tuple[str, ...]:
    return (
        "action_fraction_up",
        "action_fraction_down",
        "action_fraction_left",
        "action_fraction_right",
        "action_fraction_stay",
        "stay_action_fraction",
        "k_fraction_0",
        "k_fraction_6",
        "order_fraction_0",
        "order_fraction_1",
        "ko_fraction_0",
        "ko_fraction_13",
        "k_edge_fraction",
        "same_role_reverse_fraction",
        "vp_fraction_0",
        "vp_fraction_2",
        "vp_nonzero_fraction",
        "train_win_rate",
        "train_net_gold_margin_mean",
        "train_agent_net_gold_mean",
        "train_opponent_net_gold_mean",
        "train_agent_pickups_mean",
        "train_opponent_pickups_mean",
        "train_agent_pickups_per_episode",
        "train_opponent_pickups_per_episode",
        "train_agent_pickup_gold_per_episode",
        "train_opponent_pickup_gold_per_episode",
        "train_agent_bomb_triggers_per_episode",
        "train_opponent_bomb_triggers_per_episode",
        "train_agent_bomb_lost_gold_per_episode",
        "train_opponent_bomb_lost_gold_per_episode",
        "train_agent_tramples_per_episode",
        "train_opponent_tramples_per_episode",
        "train_agent_trample_penalty_per_episode",
        "train_opponent_trample_penalty_per_episode",
        "train_agent_vision_spent_mean",
        "actor_grad_norm",
        "critic_grad_norm",
        "reward_net_gold_gain_mean",
        "reward_clipped_net_gold_gain_mean",
        "reward_net_gold_gain_reward_mean",
        "reward_vision_info_gold_mean",
        "reward_vision_info_cells_mean",
        "reward_vision_info_reward_mean",
        "return_mean",
        "return_std",
        "value_mean",
        "value_std",
        "advantage_mean",
        "advantage_std",
    )


if __name__ == "__main__":
    unittest.main()
