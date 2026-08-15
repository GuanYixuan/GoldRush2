from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.submission import build_cpp_policy


def test_build_cpp_policy_no_build_generates_fast_runtime_release_assets() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        onnx_path = root / "actor.onnx"
        _write_fake_onnx_with_metadata(onnx_path)
        output_dir = root / "cpp"

        subprocess.run(
            [
                sys.executable,
                str(Path(build_cpp_policy.__file__).resolve()),
                "--onnx",
                str(onnx_path),
                "--output-dir",
                str(output_dir),
                "--module-name",
                "PlayerSmoke",
                "--fast-runtime-mode",
                "release",
                "--no-build",
            ],
            check=True,
            cwd=Path(build_cpp_policy.ROOT),
        )

        makefile = (output_dir / "Makefile").read_text(encoding="utf-8")
        player_cpp = (output_dir / "player.cpp").read_text(encoding="utf-8")
        metadata = json.loads((output_dir / "assembly_metadata.json").read_text(encoding="utf-8"))

        assert "-DPOLICY_RUNTIME_FAST_DEBUG=0" in makefile
        assert "policy_runtime/src/fast_option.cpp" not in makefile
        assert "fast_option_one_tu.cpp" in makefile
        assert (output_dir / "fast_option_one_tu.cpp").exists()
        assert (output_dir / "policy_runtime" / "fast_option.h").exists()
        assert '#include "policy_runtime/fast_option.h"' in player_cpp
        assert '#include "fast_option_one_tu.cpp"' in player_cpp
        assert "policy_runtime::FastRuntimeState runtime_" in player_cpp
        assert "runtime_.try_fast_output(*input, &fast_output)" in player_cpp
        assert "runtime_.prepare_neural(*input)" in player_cpp
        assert "runtime_.commit_neural(decision.output)" in player_cpp
        assert "runtime_.set_next_threshold(decision.threshold_int)" in player_cpp
        assert '"fast_scalars"' in player_cpp
        assert '"threshold_mu_raw"' in player_cpp
        assert '"threshold_log_std"' in player_cpp
        assert metadata["schema"] == "goldrush2_cpp_policy_assembly_v1"
        assert metadata["fast_runtime_mode"] == "release"
        assert metadata["policy_runtime_fast_debug"] == 0
        assert metadata["fast_core_one_tu"] is True
        assert metadata["stochastic"] is True
        assert metadata["critic_exported"] is False


def test_build_cpp_policy_no_build_generates_debug_fast_macro_when_requested() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        onnx_path = root / "actor.onnx"
        _write_fake_onnx_with_metadata(onnx_path)
        output_dir = root / "cpp"

        subprocess.run(
            [
                sys.executable,
                str(Path(build_cpp_policy.__file__).resolve()),
                "--onnx",
                str(onnx_path),
                "--output-dir",
                str(output_dir),
                "--module-name",
                "PlayerSmoke",
                "--fast-runtime-mode",
                "debug",
                "--no-build",
            ],
            check=True,
            cwd=Path(build_cpp_policy.ROOT),
        )

        makefile = (output_dir / "Makefile").read_text(encoding="utf-8")
        player_cpp = (output_dir / "player.cpp").read_text(encoding="utf-8")
        metadata = json.loads((output_dir / "assembly_metadata.json").read_text(encoding="utf-8"))

        assert "-DPOLICY_RUNTIME_FAST_DEBUG=1" in makefile
        assert "policy_runtime/src/fast_option.cpp" in makefile
        assert "fast_option_one_tu.cpp" not in makefile
        assert '#include "fast_option_one_tu.cpp"' not in player_cpp
        assert not (output_dir / "fast_option_one_tu.cpp").exists()
        assert metadata["fast_runtime_mode"] == "debug"
        assert metadata["policy_runtime_fast_debug"] == 1
        assert metadata["fast_core_one_tu"] is False


def test_build_cpp_policy_requires_explicit_fast_runtime_mode() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        onnx_path = root / "actor.onnx"
        _write_fake_onnx_with_metadata(onnx_path)

        result = subprocess.run(
            [
                sys.executable,
                str(Path(build_cpp_policy.__file__).resolve()),
                "--onnx",
                str(onnx_path),
                "--output-dir",
                str(root / "cpp"),
                "--module-name",
                "PlayerSmoke",
                "--no-build",
            ],
            cwd=Path(build_cpp_policy.ROOT),
            stderr=subprocess.PIPE,
            text=True,
        )

        assert result.returncode != 0
        assert "--fast-runtime-mode" in result.stderr


def test_build_cpp_policy_builds_release_so_with_embedded_model_bytes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        onnx_path = root / "actor.onnx"
        _write_fake_onnx_with_metadata(onnx_path)
        output_dir = root / "cpp"

        subprocess.run(
            [
                sys.executable,
                str(Path(build_cpp_policy.__file__).resolve()),
                "--onnx",
                str(onnx_path),
                "--output-dir",
                str(output_dir),
                "--module-name",
                "PlayerSmoke",
                "--fast-runtime-mode",
                "release",
            ],
            check=True,
            cwd=Path(build_cpp_policy.ROOT),
        )

        so_path = output_dir / "PlayerSmoke.so"
        metadata = json.loads((output_dir / "assembly_metadata.json").read_text(encoding="utf-8"))

        assert so_path.exists()
        assert so_path.stat().st_size < build_cpp_policy.SAFE_SO_BYTES
        assert metadata["so"] == str(so_path)
        assert metadata["so_bytes"] == so_path.stat().st_size


def test_build_cpp_policy_rejects_onnx_without_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        onnx_path = root / "actor.onnx"
        onnx_path.write_bytes(b"fake-onnx")

        result = subprocess.run(
            [
                sys.executable,
                str(Path(build_cpp_policy.__file__).resolve()),
                "--onnx",
                str(onnx_path),
                "--output-dir",
                str(root / "cpp"),
                "--module-name",
                "PlayerSmoke",
                "--fast-runtime-mode",
                "release",
                "--no-build",
            ],
            cwd=Path(build_cpp_policy.ROOT),
            stderr=subprocess.PIPE,
            text=True,
        )

        assert result.returncode != 0
        assert "missing actor ONNX metadata" in result.stderr


def test_build_cpp_policy_rejects_old_onnx_schema() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        onnx_path = root / "actor.onnx"
        _write_fake_onnx_with_metadata(onnx_path, schema="old_actor_export_v1")

        result = subprocess.run(
            [
                sys.executable,
                str(Path(build_cpp_policy.__file__).resolve()),
                "--onnx",
                str(onnx_path),
                "--output-dir",
                str(root / "cpp"),
                "--module-name",
                "PlayerSmoke",
                "--fast-runtime-mode",
                "release",
                "--no-build",
            ],
            cwd=Path(build_cpp_policy.ROOT),
            stderr=subprocess.PIPE,
            text=True,
        )

        assert result.returncode != 0
        assert "ONNX metadata schema" in result.stderr


def _write_fake_onnx_with_metadata(
    onnx_path: Path,
    *,
    schema: str = build_cpp_policy.EXPECTED_ONNX_SCHEMA,
) -> None:
    onnx_path.write_bytes(b"fake-onnx")
    metadata = {
        "schema": schema,
        "stochastic": True,
        "critic_exported": False,
        "checkpoint_update": 3,
        "action_head_schema": "candidate_cell_residual_v1_fast_threshold_calibrated_base_v1",
        "inputs": build_cpp_policy.EXPECTED_INPUTS,
        "outputs": build_cpp_policy.EXPECTED_OUTPUTS,
    }
    onnx_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
