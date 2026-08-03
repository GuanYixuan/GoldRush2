# RL Environment

`training/rl/` 目前只提供最小环境适配层，用于把 `simulator.envs.round_step` 包成 agent-vs-opponent 的训练 rollout。

## 当前边界

- agent action 仍直接使用官方形状 `GameOutput(actions, k, order, vp)`；暂不实现神经网络 `ActionCodec`。
- 默认训练语义是 `agent_after_opponent`：opponent 先动，NPC 行动，agent 后动。
- 双方 observation 都来自同一个行动前状态，不把 opponent 行动后的状态泄漏给 agent。
- reward 只实现终局 `WinLossReward`：非终局 `0`，agent 胜 `+1`，agent 负 `-1`。
- 同分判定按“agent 慢、opponent 快”的 P90 假设处理，避免交换 P1/P2 时引入固定玩家 ID 偏置。

## 主要入口

- `SingleAgentGoldRushEnv`：单局环境。
- `BatchRolloutSampler`：直接驱动 `SingleAgentGoldRushEnv` 批量采样 paired episodes，并输出 `EpisodeBatch`。
- `RuntimePolicyWrapper`：把 `policy_runtime.FeatureExtractor` 接入现有 policy callable 时序；每局由 sampler 调用 `start_episode(player_id=...)` 重置 runtime。
- `EvaluationCase` / `EvaluationConfig` / `evaluate_policy`：显式评估 case 集与评估汇总。正式评估应冻结 case 列表；`anchor_grid()` 用于公共 seed 对照，`matrix_by_slice()` 用于每个 `(map, opponent)` 切片独立 seed 集。
- `Transition` / `Trajectory` / `EpisodeBatch`：rollout 数据结构。episode 步数从 `len(trajectory.transitions)` 派生，不作为单独字段冻结。

训练时仍应把两条 episode trajectory 作为独立样本；pair 只用于采样组织、评估聚合和降噪统计。

第一版固定 anchor eval 配置见 `training.configs.eval.anchor_eval_config()`：

- 10 个公共 anchor seeds。
- 3 张公开地图。
- `greedy_visible_gold` 与 `fast_probe_v3_like` 两个对手。
- 共 60 个 paired cases，即 120 条单局 episodes。
