# 策略导出与平台提交流程

本文记录主线神经策略的正式导出链路：

```text
PPO checkpoint -> actor-only stochastic FP32 ONNX -> C++ .so -> 本地 smoke -> 平台 self-play / 供挑战版本
```

当前提交路线只导出 actor，critic 完全不进入部署资产。actor 使用 `policy_runtime/` 中的 `goldrush2_feature_v2` C++ feature extractor；ONNX 额外接收 C++ 侧生成的随机张量，通过 Gumbel argmax 实现 stochastic 采样。ONNX 图中不应出现 `Random*` 或 `Multinomial` 算子。

## 分步工具

### 1. 导出 actor ONNX

```bash
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python -m training.export.export_actor_onnx \
  --checkpoint temp/ppo_manual_lab/runs/your_run/checkpoints/latest.pt \
  --output temp/submission_builds/your_build/actor.onnx
```

导出工具会执行：

- 检查 checkpoint schema 为 `ppo_train_v1`。
- 检查 actor feature shape 与当前 `goldrush2_feature_v2` 一致。
- 检查 action head schema 与当前 `candidate_cell_residual_v1` 一致。
- 校验 PyTorch 输出与 ONNXRuntime 输出完全一致。
- 检查 ONNX 图中没有平台不稳定随机算子。
- 写出 `actor.metadata.json`，包含 checkpoint update、模型配置、输入输出 shape 和算子类型。

### 2. 组装 C++ `.so`

```bash
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python tools/submission/build_cpp_policy.py \
  --onnx temp/submission_builds/your_build/actor.onnx \
  --output-dir temp/submission_builds/your_build/cpp \
  --module-name Player0811L
```

该工具会生成：

- `model_bytes.h`：内嵌 ONNX bytes。
- `player.cpp`：官方 `moveDecision` 入口、feature extractor、ONNX Runtime C API 调用与 stochastic 随机输入。
- `Makefile`。
- `<module-name>.so`。
- `assembly_metadata.json`。

`.so` 大小超过 `15_500_000` bytes 会 fail-fast。ONNX Runtime C API headers 来自 `policy_runtime/onnxruntime_c_api/`，不再依赖 `temp/`。

### 3. 本地 smoke

本地 smoke 需要能加载 conda 环境中的 `libonnxruntime.so`：

```bash
ORT_CAPI_DIR=$(conda run --no-capture-output -n goldrush python -c \
  'import onnxruntime, pathlib; print(pathlib.Path(onnxruntime.__file__).resolve().parent / "capi")')

LD_LIBRARY_PATH="$ORT_CAPI_DIR:$LD_LIBRARY_PATH" \
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python tools/submission/smoke_cpp_policy.py \
  --so temp/submission_builds/your_build/cpp/Player0811L.so \
  --map-id 2 \
  --round-count 50 \
  --agent-player-id 1
```

建议 P1/P2 各跑一次。通过标准是：

- `policyRuntimeReady()` 为 true。
- simulator 能完整跑完指定回合。
- `agent_vision_spent` 没有非预期增长。
- 无 C++ runtime 异常或 fallback 大量触发迹象。

### 4. 平台 self-play

该步骤会提交策略并发起一局公测对局，只有在明确需要平台验证时执行：

```bash
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python tools/submission/platform_submit_selfplay.py \
  --so temp/submission_builds/your_build/cpp/Player0811L.so \
  --map-id 2 \
  --model-a A0811L1A \
  --model-b A0811L1B
```

脚本会读取 `.env` 中的 `LOGIN_KEY`，不会打印 key 或 token。输出默认写入：

```text
temp/submission_runs/platform_selfplay/
```

完成后再用只读工具统计：

```bash
conda run --no-capture-output -n goldrush \
  python tools/platform_games.py stats <game_id> \
  --cache-dir temp/submission_runs/platform_selfplay/<run_dir> \
  --refresh
```

重点检查：

- `forfeit=0`。
- `vision_spent=0` 或符合预期。
- P99 / Max 决策耗时低于平台限制并留有余量。
- replay 已成功下载。

### 5. 上传供挑战版本

该步骤会修改平台上的供他人挑战版本，只能在用户明确要求时执行：

```bash
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python tools/submission/platform_publish_challenge.py \
  --so temp/submission_builds/your_build/cpp/Player0811L.so
```

输出默认写入：

```text
temp/submission_runs/platform_challenge/
```

## Agent 全流程指引

当用户要求“导出 latest 并做一次 self-play”时，推荐按以下顺序执行：

1. 确认 checkpoint：优先使用用户指定 run 的 `checkpoints/latest.pt`，并读取导出 metadata 中的 `checkpoint_update`。
2. 导出 ONNX：使用 `training.export.export_actor_onnx`，确认一致性检查和随机算子检查通过。
3. 组装 `.so`：使用 `tools/submission/build_cpp_policy.py`，确认 `.so` 大小低于阈值。
4. 本地 smoke：P1/P2 各至少一次，地图优先与计划平台验证地图一致。
5. 平台 self-play：用户已明确授权时执行；模型名使用短字母数字名，避免下划线。
6. 下载 replay 并统计：报告 `game_id`、replay 路径、forfeit、净金币、vision、P99/Max cost。
7. 供挑战版本：只有用户单独明确要求“设为供他人挑战版本”时执行。

不要把平台 self-play 和供挑战上传作为导出工具的默认行为。平台操作是有副作用的，应保持为独立显式命令。
