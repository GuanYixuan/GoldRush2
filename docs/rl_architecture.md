# RL 栈架构总览

本文是 GoldRush2.0 RL 栈的入口导航，描述模块边界、当前实现状态和近期推进顺序。游戏规则以 `gamerules/gamerules.md` 为准，平台提交与推理约束见 `gamerules/submission.md` 和 `docs/neural_inference.md`。

## 核心假设

本项目默认训练 **慢速神经网络后手策略**，主要通过快速 opponent league 进行 against-league 训练。

依据：

- GoldRush2.0 行动顺序由双方 `moveDecision` 响应耗时决定。
- 神经网络策略通常慢于手写规则策略，实战中大概率后手。
- 标准同构 self-play 会高估神经网络策略先手或同步行动时的表现。
- 主训练分布应近似：

```text
fast opponent action -> NPC action -> neural agent action
```

平台真实顺序为：

```text
faster player action -> NPC action -> slower player action
```

因此训练环境必须显式建模行动顺序，不应在算法层隐式假定双方同步。

## 模块边界

当前 RL 栈按根目录分层：

```text
simulator/
  # 游戏环境、官方规则、机制近似、观测生成、回放导出

policy_runtime/
  # 训练和部署共享的策略运行时；重点是有状态 feature extractor

training/
  opponents/
    # opponent league、快速规则策略、固定评测对手
  rl/
    # 单智能体环境适配、rollout、reward、evaluation
  configs/
    # 固定训练/评估配置
  scripts/
    # 后续训练、评估、导出入口
  experiments/
    # 后续实验记录和可复现 run 配置
```

职责约束：

- `simulator/` 描述真实世界如何演化，不保存 agent 的 belief 或策略记忆。
- `policy_runtime/` 描述 agent 如何从 observation 维护记忆、生成网络输入，并最终进入 C++ 提交策略。
- `training/opponents/` 只管理对手接口、参数化和 league 采样。
- `training/rl/` 只管理训练 rollout、评估和算法所需数据结构，不实现游戏规则。
- `training/configs/` 固化可复现配置，避免评估集散落在脚本常量中。

## 当前实现

`simulator/` 已具备本地训练所需底座：

- 官方规则层：移动、碰撞、金币拾取、炸弹、踩踏、视野购买、snapshot、胜负结算。
- 机制近似层：地图、金币生成、炸弹刷新、NPC 策略。
- `envs.round_step`：训练核心 step API，调用方显式传入双方 `GameOutput` 和先手方。
- `envs.duel`：本地固定策略对战和 replay 生成。
- `replay`：导出 simulator full replay。

`training/opponents/` 已实现 Python scripted opponents：

- `stay`
- `random`
- `greedy_visible_gold`
- `fast_probe_v3_like`

`training/rl/` 已实现第一版 rollout 与评估链路：

- `SingleAgentGoldRushEnv`：把 `RoundStepEnv` 包成 agent-vs-opponent 单智能体环境。
- `WinLossReward`：非终局 `0`，终局胜 `+1`、负 `-1`。
- `Transition` / `Trajectory` / `EpisodeBatch`：rollout 数据结构。
- `BatchRolloutSampler`：直接驱动 env，按 paired episodes 批量采样。
- `EvaluationCase` / `EvaluationConfig` / `evaluate_policy`：显式 case-based evaluation。

`training/configs/eval.py` 已实现第一版 `anchor_eval_config()`：

- 10 个公共 anchor seeds。
- 3 张公开地图。
- `greedy_visible_gold` 与 `fast_probe_v3_like` 两个对手。
- 共 60 个 paired cases，即 120 条单局 episodes。

## 数据流

训练主链路目标是：

```text
Python simulator
  -> official-shaped GameInput
  -> policy_runtime feature extractor
  -> neural policy
  -> official-shaped GameOutput
  -> Python simulator transition
```

当前 `training/rl` 仍直接使用 `GameInput -> policy callable -> GameOutput`，尚未接入 `policy_runtime` 和神经网络。下一阶段应先补齐 `policy_runtime` 的最小状态机，再设计网络输入输出。

## 关键原则

### 训练与部署一致

feature extractor 会同时用于 Python 训练和最终 C++ 提交策略。训练侧不应另写 Python 版正式特征逻辑，否则会引入训练/部署漂移。

### Observation 不泄漏后手信息

平台中双方策略拿到的是行动前同一份状态。因此即使 agent 被训练成后手，它的 observation 也必须来自行动前状态，不能看到 opponent 行动后的结果。

### Episode randomization 只作用于机制近似层

官方规则层保持固定；随机化用于金币生成、炸弹刷新、NPC 策略、地图/机制参数等未知或近似部分。

### BC 是辅助损失，不是 reward

行为克隆可用于 warm start 和迁移，但不进入 `env.step()` 返回的 reward。相关细节见 `docs/reward_design.md`。

### Fast option 是未来扩展

conditional fast option 用于缓解神经网络常态后手问题，但不属于当前 MVP。架构上只需避免把 action、policy state 和 rollout schema 设计死。相关细节见 `docs/fast_option_design.md`。

## 评估口径

训练可以使用 dense reward 或 auxiliary loss，但评估必须以官方目标为准：

- 终局胜负。
- 最终净金币差。
- 毛金币与累计 `vision_spent`。
- 分 map、opponent、机制参数的胜率和净金币差。

评估最小单位是显式 `EvaluationCase` 下的 paired episodes：

```text
EvaluationCase(seed, map_id, opponent_spec)
  -> agent=P1, opponent=P2
  -> agent=P2, opponent=P1
```

两局共享同一 seed、map、opponent 和机制配置。训练轨迹仍按单局独立计算 reward；pair 只用于评估聚合和降噪。

评估集分层：

- `anchor_eval`：少量公共 seeds × selected maps × selected opponents，用于 debug 和横向对照。
- `matrix_eval`：每个 `(map, opponent)` 切片使用不同但确定的 seed 集，用于主要模型选择。
- `holdout_eval`：独立 seed、opponent 参数或机制参数，只做阶段性评估。

同分判定按主训练假设处理：agent 慢、opponent 快，因此同分时 opponent 胜，避免 P1/P2 交换后引入固定玩家 ID 偏置。

## 下一阶段

下一步进入 `policy_runtime/`：

1. 建立最小 C++ feature extractor 状态机。
2. 固定 `reset -> observe -> commit_action` 时序。
3. 先输出基础可测 feature，不急于做复杂 belief。
4. 提供 Python 训练侧调用方式。
5. 用 rollout 中的 observation/action 序列验证重置、状态更新和训练/部署一致性。

暂不推进：

- 完整网络结构。
- PPO 或其它 RL 算法。
- ONNX 导出链路。
- fast option。

## 相关文档

- `simulator/README.md`：模拟器规则层、机制层、环境接口和 replay 导出。
- `training/opponents/README.md`：opponent league 和 scripted opponents。
- `training/rl/README.md`：当前 RL env、rollout、evaluation API。
- `docs/reward_design.md`：reward、critic auxiliary target 和 BC warm start 设计。
- `docs/fast_option_design.md`：conditional fast option 未来扩展设计。
- `docs/neural_inference.md`：模型推理、ONNX Runtime、量化和提交体积/速度实验结论。
