# GoldRush2 项目入口

本仓库围绕 GoldRush2.0 编程掘金争夺赛展开：整理规则与平台接口，分析 replay 与隐藏机制，构建本地 simulator、训练神经策略，并准备 C++/ONNX 提交路径。

规则事实来源统一见 `gamerules/gamerules.md`。平台接口、提交限制和 replay 格式分别见 `gamerules/platform.md`、`gamerules/submission.md`、`gamerules/replay.md`。

## 当前主线

当前神经策略主线是慢速后手 PPO：

- actor 使用可部署的 `goldrush2_feature_v1`，由 `policy_runtime/` 的 C++ runtime 提供。
- critic 使用训练期 privileged full-state feature，具体 schema 见 `docs/features/privileged_critic_feature_v1.md`。
- 网络结构见 `docs/policy_network_design.md`。
- PPO、reward 和训练栈入口见 `training/README.md`、`docs/rl_architecture.md`、`docs/reward_design.md`。

主线 PPO 入口：

```bash
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python -m training.scripts.train_ppo \
  --output-dir temp/ppo_runs/example \
  --rollout-mode multiprocess \
  --rollout-workers 64 \
  --pair-count 32 \
  --opponents fast_probe_v3_like
```

`temp/ppo_manual_lab/run_one.py` 只是兼容 wrapper；新的 PPO loop、optimizer、checkpoint、fork 和常用训练参数应维护在 `training.scripts.train_ppo`。

## 目录导航

- `gamerules/`：规则、平台 API、提交限制和官方 replay 口径。
- `docs/`：正式架构、schema、feature、reward、网络、推理和研究设计文档；入口见 `docs/README.md`。
- `training/`：BC、PPO、opponent league、评估、rollout 和模型训练入口；入口见 `training/README.md`。
- `policy_runtime/`：actor 侧 C++ feature extractor runtime，服务训练与最终 C++ 提交。
- `simulator/`：本地游戏环境、机制近似、round-step/duel runner 和 simulator full replay。
- `mechanism/`：隐藏机制的数据采集、双视角合并、离线特征抽取和机理分析。
- `fast_probing/`：快速 C++ 专家策略实验，`v3`/`fast_probe_v3_like` 也是当前 BC 与 opponent 先验的重要来源。
- `visualizer/`：本地 replay 可视化器，支持 official NDJSON、merged JSON 和 simulator full JSON。
- `tools/`：通用辅助脚本，如平台只读查询和 metrics 到 TensorBoard 导出。
- `official_sdk/`：官方 SDK 与样例数据。
- `temp/`：gitignore 的实验资产、临时 runner、长跑结果和手工分析输出。

## 常用命令

运行训练核心测试：

```bash
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python -m pytest \
  tests/training/test_rewards.py \
  tests/training/test_ppo.py \
  tests/training/test_train_ppo.py \
  tests/training/test_bc.py
```

运行 simulator 测试：

```bash
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python -m pytest tests/simulator
```

构建 actor feature runtime：

```bash
conda run --no-capture-output -n goldrush \
  python setup.py build_ext --inplace
```

打开 replay 可视化器：

```bash
conda run --no-capture-output -n goldrush \
  python -m visualizer.main <replay-path>
```

通用工具说明见 `tools/README.md`。涉及平台提交、发起对局或下载日志时，必须先阅读 `AGENTS.md` 和对应工具文档。
