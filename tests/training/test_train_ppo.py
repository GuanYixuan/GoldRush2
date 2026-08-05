from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from simulator.config import EpisodeConfig, RulesConfig
from training.models import PolicyNetworkConfig
from training.rl import EvaluationCase, PpoConfig
from training.rl import MultiprocessRolloutConfig
from training.scripts.train_ppo import (
    TrainPpoConfig,
    load_checkpoint,
    opponent_spec_from_name,
    run_training,
)


class TrainPpoTests(unittest.TestCase):
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
                "--reward-beta",
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
) -> TrainPpoConfig:
    return TrainPpoConfig(
        seed=77,
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
        reward_beta=0.0,
        model=_small_model_config(),
        ppo=PpoConfig(update_epochs=1, minibatch_size=2, target_joint_kl=None),
        episode=EpisodeConfig(rules=RulesConfig(round_count=1, snapshot_period=1), seed=7, map_id=1),
        resume_checkpoint=resume_checkpoint,
    )


def _small_model_config() -> PolicyNetworkConfig:
    return PolicyNetworkConfig(
        width=16,
        residual_blocks=1,
        se_reduction=4,
        scalar_hidden=(16, 16),
        actor_hidden=32,
        critic_hidden=(32, 16),
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
        "vp_fraction_0",
        "vp_fraction_2",
        "vp_nonzero_fraction",
        "train_win_rate",
        "train_net_gold_margin_mean",
        "train_agent_net_gold_mean",
        "train_opponent_net_gold_mean",
        "train_agent_pickups_mean",
        "train_opponent_pickups_mean",
        "train_agent_vision_spent_mean",
        "return_mean",
        "return_std",
        "value_mean",
        "value_std",
        "advantage_mean",
        "advantage_std",
    )


if __name__ == "__main__":
    unittest.main()
