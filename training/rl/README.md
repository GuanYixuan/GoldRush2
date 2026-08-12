# RL 训练模块

`training/rl/` 提供 GoldRush2 PPO 训练所需的环境适配、reward、rollout、PPO batch、PPO update 和评估 API。训练入口位于 `training.scripts.train_ppo`；高层入口见 `training/README.md`。

## 边界

- 默认训练语义是 `agent_after_opponent`：opponent 先动，NPC 行动，agent 后动。
- 双方 observation 都来自同一个行动前状态，不把 opponent 行动后的状态泄漏给 agent。
- env 默认 reward 是 `WinLossReward`，用于基础环境与测试；PPO 主线显式传入 `TerminalWinMarginGoldGainReward`。
- PPO batch 保存 actor feature 与 privileged critic feature。policy/logprob/entropy 只读 actor feature；value/return/EV 只读 critic feature。
- PPO 仍使用完整 episode rollout，GAE terminal bootstrap 固定为 `0`；暂不支持 rollout chunk bootstrap。
- paired episodes 只用于采样组织和降噪统计，训练轨迹仍按单局独立计算 reward。

## 主要文件

- `env.py`：`SingleAgentGoldRushEnv`，把 `RoundStepEnv` 适配成单智能体 agent-vs-opponent 环境。
- `rewards.py`：`WinLossReward` 与 `TerminalWinMarginGoldGainReward`。
- `ppo_buffer.py`：`PpoTransition`、`PpoBatch`、GAE 和 minibatch iterator。
- `ppo.py`：serial PPO rollout、`ppo_update()`、`critic_only_update()`、EV 和梯度统计。
- `multiprocess_rollout.py` / `rollout_mp/`：多进程 rollout pool、worker、shared memory、scheduler 和 batch assembly。
- `privileged_critic_features.py`：训练期 critic full-state feature extractor。
- `eval.py`：`EvaluationCase`、`EvaluationConfig` 和 `evaluate_policy()`。
- `eval_mp/`：轻量多进程评估，主进程持有 model/GPU 并批量推理，worker 只返回 episode summary。
- `runtime_policy.py`：把 actor-only runtime policy 接入 sampler/eval 时序。
- `sampler.py` / `rollout.py`：通用 policy callable rollout 和 paired episode 数据结构。

## Reward

PPO 主线 reward 为 `TerminalWinMarginGoldGainReward`，schema 为 `terminal_win_margin_gold_gain_v1`。详细设计见 `docs/reward_design.md`。

## Rollout

serial rollout 使用 `collect_ppo_rollouts()`。多进程 rollout 使用 `MultiprocessRolloutPool`，主进程持有 model/GPU 并批量推理，worker 只运行 simulator、opponent 和 feature extractor。

多进程入口保持 on-policy 语义：每个 update 的 rollout batch 来自当前冻结模型。推荐较大训练配置使用：

```bash
--rollout-mode multiprocess \
--rollout-workers 64 \
--rollout-max-inference-batch-size 64 \
--rollout-inference-timeout-ms 2
```

`TrainPpoConfig.rollout_seed_base` 与 `advance_rollout_seed` 控制每个 update 的 rollout seed。默认 `rollout_seed_base=None` 时使用 `seed`；`advance_rollout_seed=True` 时第 `u` 个 update 使用：

```text
rollout_seed = base + (u - 1) * pair_count
```

## PPO Update

`ppo_update()` 使用 actor/critic 参数集合分离：

- actor loss = policy loss - entropy bonus，只更新 actor encoder、actor heads、embedding 和 decoder。
- critic loss = `value_coef * value_loss`，只更新 critic encoder 和 value head。
- 每个 minibatch 分别记录 `actor_grad_norm` 与 `critic_grad_norm`，`grad_norm` 保持为二者最大值的均值。

`critic_only_update()` 用于 critic warmup 阶段，只训练 critic 路径，metrics 中 `actor_grad_norm=None`。

主线训练入口支持：

- `--actor-learning-rate` / `--critic-learning-rate`
- `--critic-warmup-updates`
- `--actor-lr-ramp-updates`
- `--init-model-checkpoint`
- `--fork-ppo-checkpoint`
- `--resume-checkpoint`
- `--save-updates`

若同时设置 critic warmup 和 actor lr ramp，ramp 按 PPO phase 内 update 数计算。例如 warmup 100、ramp 100 时，第 101 个总 update 是第 1 个 PPO update，actor lr 为目标值的 `1/100`。

## Checkpoint

- `resume_checkpoint`：恢复模型和 optimizer，从 checkpoint 的 `update_index + 1` 继续。
- `fork_ppo_checkpoint`：只加载模型权重，重新创建 optimizer，从新 run 的 update 1 开始。
- `init_model_checkpoint`：从 BC checkpoint 初始化 actor；critic 随当前 PPO model 随机初始化。

三者互斥。checkpoint schema 为 `ppo_train_v1`，保存 `optimizer_phase`、`init_source`、`fork_metadata`、`train_config` 和 `last_metrics`。

## 评估

正式评估应使用冻结 case 列表。当前约定的评估口径：

- 终局胜负。
- 最终净金币差。
- 己方毛金币与累计 `vision_spent`。
- 分 map、opponent、机制参数切片统计。

`training.configs.eval.anchor_eval_config()` 提供固定 anchor eval 配置；更大矩阵评估应显式冻结 seed、map、opponent 和参数范围。

大规模或训练中评估优先使用 `ParallelEvalPool` / `evaluate_parallel()`。它复用 `rollout_mp` 的 central inference 结构，但不保存 PPO transition，只返回每局净金币、胜负、pickup、炸弹、踩踏和视野消耗 summary。`ParallelEvalConfig.deterministic=False` 是默认口径；需要 paired 可复现对比时显式打开 deterministic。
