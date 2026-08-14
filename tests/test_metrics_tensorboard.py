from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.metrics_tensorboard import (
    DEFAULT_EXPERIMENT_ROOT,
    is_default_export_metric,
    metric_tag,
    parse_args,
    read_latest_update,
)


class MetricsTensorBoardTests(unittest.TestCase):
    def test_parse_args_defaults_to_manual_lab_watch(self) -> None:
        args = parse_args([])

        self.assertEqual(args.experiments, [DEFAULT_EXPERIMENT_ROOT])
        self.assertEqual(args.watch_seconds, 10.0)

    def test_parse_args_no_watch_disables_default_watch(self) -> None:
        args = parse_args(["--no-watch"])

        self.assertEqual(args.experiments, [DEFAULT_EXPERIMENT_ROOT])
        self.assertIsNone(args.watch_seconds)

    def test_default_export_keeps_dynamic_fast_metrics_and_filters_config(self) -> None:
        self.assertTrue(is_default_export_metric("fast_success_per_episode"))
        self.assertTrue(is_default_export_metric("actual_fast_first_rate"))
        self.assertTrue(is_default_export_metric("macro_tau_mean"))
        self.assertTrue(is_default_export_metric("fast_threshold_lr"))

        self.assertFalse(is_default_export_metric("fast_threshold_learning_rate"))
        self.assertFalse(is_default_export_metric("beta_net_gold_gain"))
        self.assertFalse(is_default_export_metric("net_gold_gain_scale"))
        self.assertFalse(is_default_export_metric("update"))

    def test_metric_tag_classifies_fast_metrics_and_config_cleanly(self) -> None:
        self.assertEqual(metric_tag("fast_threshold_lr"), "optimization/fast_threshold_lr")
        self.assertEqual(metric_tag("fast_success_per_episode"), "fast_option/fast_success_per_episode")
        self.assertEqual(metric_tag("actual_fast_first_rate"), "fast_option/actual_fast_first_rate")
        self.assertEqual(metric_tag("fast_threshold_learning_rate"), "config/fast_threshold_learning_rate")
        self.assertEqual(metric_tag("net_gold_gain_scale"), "config/net_gold_gain_scale")

    def test_read_latest_update_ignores_incomplete_trailing_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            path.write_bytes(
                json.dumps({"update": 7}).encode("utf-8")
                + b"\n"
                + json.dumps({"update": 8}).encode("utf-8")
                + b"\n"
                + b'{"update": 9'
            )

            self.assertEqual(read_latest_update(path), 8)

    def test_read_latest_update_returns_none_for_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            path.touch()

            self.assertIsNone(read_latest_update(path))


if __name__ == "__main__":
    unittest.main()
