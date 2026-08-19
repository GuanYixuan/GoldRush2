# Training 训练栈入口

本目录承载 GoldRush2 神经策略训练代码。游戏规则以 `gamerules/gamerules.md` 为准；高层训练架构见 `docs/rl_architecture.md`；网络、feature、reward 细节分别见 `docs/policy_network_design.md`、`docs/features/` 和 `docs/reward_design.md`。

## 当前主线

当前主线训练的是慢速神经网络后手策略。训练环境显式采用：

```text
fast opponent action -> NPC action -> neural agent action
```

模型使用 actor/critic 双输入：

- actor 使用可提交的 `goldrush2_feature_v2`，由 `policy_runtime.FeatureExtractor` 从官方 `GameInput` 提取。
- critic 使用训练期 `goldrush2_privileged_critic_feature_v2`，由 simulator full state 与同 transition 的 actor 信息态共同构造。
- PPO policy loss 只更新 actor 路径；value loss 只更新 privileged critic 路径。
- BC 只初始化/训练 actor 路径，不依赖 privileged critic。

主线 PPO 入口是：

```bash
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python -m training.scripts.train_ppo \
  --output-dir temp/ppo_runs/example \
  --rollout-mode multiprocess \
  --rollout-workers 64 \
  --pair-count 32 \
  --opponents fast_probe_v3_like
```

`temp/ppo_manual_lab/run_one.py` 只是兼容 wrapper，保留旧 manual lab 参数习惯；不要在其中重新实现 PPO loop、optimizer、checkpoint 或 fork 逻辑。

## 目录职责

- `training/models/`：神经网络结构与动作概率语义实现。当前核心是 `GoldRushPolicyNetwork`，包含自回归 actor 和 privileged critic。
- `training/rl/`：单智能体环境适配、rollout、PPO batch、reward、评估和 PPO update。
- `training/bc/`：BC 数据采集、审计、加载、loss 和 actor-only 训练。
- `training/opponents/`：训练和评估用 opponent runner、scripted opponents 与 league 参数采样。
- `training/configs/`：稳定评估配置。
- `training/export/`：可部署 actor 导出工具，当前支持 actor-only stochastic FP32 ONNX。
- `training/scripts/`：可执行入口，当前包括 PPO、BC 采集/审计/训练。

## 常用能力

`training.scripts.train_ppo` 支持：

- `--init-model-checkpoint`：从 BC checkpoint 初始化 actor。
- `--fork-ppo-checkpoint`：从 PPO checkpoint 继承完整模型权重并开始新 run，不继承旧 optimizer。
- `--resume-checkpoint`：从当前 run checkpoint 续训，恢复 optimizer。
- `--actor-learning-rate` / `--critic-learning-rate`：actor 和 critic 分离学习率。
- `--critic-warmup-updates`：前若干 update 只训练 critic。
- `--actor-lr-ramp-updates`：PPO phase 内 actor lr 线性 ramp。
- `--rollout-seed-base` / `--advance-rollout-seed`：显式控制 rollout seed。
- `--save-updates`：保存指定 update checkpoint。
- `--eval-interval`：按 update 周期运行训练内评估；默认关闭。打开后使用 `ParallelEvalPool` 多进程评估，默认 `--eval-workers 64`、`--eval-max-inference-batch-size 64`、`--eval-inference-timeout-ms 2.0`、`--no-eval-deterministic`。若训练打开 fast runtime，评估同步使用 fast runtime。

## 最小测试

训练栈改动后优先跑：

```bash
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python -m pytest \
  tests/training/test_rewards.py \
  tests/training/test_ppo.py \
  tests/training/test_train_ppo.py \
  tests/training/test_bc.py
```

涉及 multiprocess rollout、feature schema、policy network 或 eval 时，补跑对应测试：

```bash
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python -m pytest \
  tests/training/test_multiprocess_rollout.py \
  tests/training/test_privileged_critic_features.py \
  tests/training/test_policy_network.py \
  tests/training/test_eval.py
```
