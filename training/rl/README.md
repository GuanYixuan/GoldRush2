# RL Environment

`training/rl/` 提供 GoldRush2.0 第一版 PPO 训练所需的环境适配、reward、rollout buffer 和单次 PPO update 核心。训练入口脚本、checkpoint 和 eval interval 仍未实现。

## 当前边界

- agent action 仍直接使用官方形状 `GameOutput(actions, k, order, vp)`；神经网络 action 采样工具在 `training.models.policy_network` 中实现。
- 默认训练语义是 `agent_after_opponent`：opponent 先动，NPC 行动，agent 后动。
- 双方 observation 都来自同一个行动前状态，不把 opponent 行动后的状态泄漏给 agent。
- env 默认 reward 是终局 `WinLossReward`：非终局 `0`，agent 胜 `+1`，agent 负 `-1`。PPO v1 主线应显式使用 `TerminalWinPlusMarginPotentialReward`，默认 `beta=0.2`；smoke/debug 可设 `beta=0` 退化为纯终局胜负 reward。
- 同分判定按“agent 慢、opponent 快”的 P90 假设处理，避免交换 P1/P2 时引入固定玩家 ID 偏置。
- PPO v1 只支持完整 episode rollout，GAE terminal bootstrap 固定为 `0`；暂不支持 rollout chunk bootstrap 或并行环境。

## 主要入口

- `SingleAgentGoldRushEnv`：单局环境。
- `BatchRolloutSampler`：直接驱动 `SingleAgentGoldRushEnv` 批量采样 paired episodes，并输出 `EpisodeBatch`。
- `RuntimePolicyWrapper`：把 `policy_runtime.FeatureExtractor` 接入现有 policy callable 时序；每局由 sampler 调用 `start_episode(player_id=...)` 重置 runtime。
- `PpoBatch` / `PpoTransition`：PPO 训练 batch 与 transition 数据结构，支持 GAE 和 minibatch iterator。
- `collect_ppo_rollouts()`：用 `GoldRushPolicyNetwork` 和 `BatchRolloutSampler` 配置采集 PPO batch。
- `ppo_update()`：对已计算 GAE 的 `PpoBatch` 执行一次 PPO update。
- `EvaluationCase` / `EvaluationConfig` / `evaluate_policy`：显式评估 case 集与评估汇总。正式评估应冻结 case 列表；`anchor_grid()` 用于公共 seed 对照，`matrix_by_slice()` 用于每个 `(map, opponent)` 切片独立 seed 集。
- `Transition` / `Trajectory` / `EpisodeBatch`：rollout 数据结构。episode 步数从 `len(trajectory.transitions)` 派生，不作为单独字段冻结。

## PPO v1 计划

第一版神经 PPO 的模型放在 `training/models/policy_network.py`，PPO 算法与 buffer 放在 `training/rl/ppo.py`、`training/rl/ppo_buffer.py`。下一步训练入口应放在 `training/scripts/train_ppo.py`。具体网络输出、forward 返回对象、rollout buffer 字段和 reward schema 以 `docs/policy_network_design.md` 与 `docs/reward_design.md` 为准。

训练时仍应把两条 episode trajectory 作为独立样本；pair 只用于采样组织、评估聚合和降噪统计。

第一版固定 anchor eval 配置见 `training.configs.eval.anchor_eval_config()`：

- 10 个公共 anchor seeds。
- 3 张公开地图。
- `greedy_visible_gold` 与 `fast_probe_v3_like` 两个对手。
- 共 60 个 paired cases，即 120 条单局 episodes。
