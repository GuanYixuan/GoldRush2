# RL 栈架构总览

本文是 GoldRush2 RL 栈的高层导航。游戏规则以 `gamerules/gamerules.md` 为准；平台提交与推理约束见 `gamerules/submission.md` 和 `docs/neural_inference.md`；训练代码入口见 `training/README.md`。

## 执行顺序与条件快速路径

比赛前期的基础策略是常态后手的神经网络策略，主要通过快速 opponent league 做 against-league 训练。比赛后期，主线加入 conditional fast option：局部条件满足时由 C++ fast controller 直接返回动作，否则才运行神经网络。最终训练主线同时启用了 fast runtime，并训练了控制下一回合触发阈值的 threshold head。

这一设计基于：

- GoldRush2 行动顺序由双方 `moveDecision` 响应耗时决定。
- 神经网络策略通常慢于手写规则策略，实战中大概率后手。
- 同构同步 self-play 会高估神经网络策略先手或同步行动时的表现。

普通神经路径仍近似为：

```text
fast opponent action -> NPC action -> neural agent action
```

fast 命中时则近似为：

```text
fast agent action -> NPC action -> opponent action
```

平台真实顺序为：

```text
faster player action -> NPC action -> slower player action
```

因此训练环境必须显式建模两条路径的行动顺序，不应在算法层隐式假定双方同步。通用训练入口保留显式开关，`--enable-fast-runtime-features` 默认关闭；这只是 CLI 默认值，不代表比赛后期主线未使用 fast runtime。完整实现与历史结论见 `docs/fast_option_design.md`。

## 模块边界

```text
simulator/
  游戏环境、官方规则、机制近似、观测生成、回放导出

policy_runtime/
  训练和部署共享的 actor feature extractor、fast controller 与跨回合 runtime state

training/
  models/
    自回归 actor、privileged critic、动作概率语义
  bc/
    BC 数据采集、审计、加载和 actor-only 训练
  opponents/
    opponent runner、scripted opponents、league 参数采样
  rl/
    单智能体环境适配、reward、rollout、PPO batch/update、eval
  scripts/
    PPO/BC 命令入口
```

职责约束：

- `simulator/` 描述真实世界如何演化，不保存 agent 的 belief 或策略记忆。
- `policy_runtime/` 描述 actor 如何从官方 observation 维护记忆、生成可提交 feature。
- `training/opponents/` 只管理对手接口、参数化和 league 采样。
- `training/rl/` 只管理训练 rollout、评估和算法所需数据结构，不实现游戏规则。
- `training/models/` 只消费已冻结的 actor/critic feature，不读取 simulator state。

## 当前训练数据流

```text
RoundStepEnv action-time state
  -> official-shaped GameInput
  -> policy_runtime FastRuntimeState
     -> fast 命中：直接生成 official-shaped GameOutput
     -> fast 未命中：backfill pending，提取 actor feature
        -> simulator full-state privileged critic feature extractor
        -> GoldRushPolicyNetwork actor/critic 双输入
        -> 普通动作 + 下一回合 threshold
  -> simulator transition/reward
  -> PpoBatch
  -> GAE
  -> PPO update
```

部署路径只使用 actor feature 和 actor-only 网络入口；privileged critic feature 只存在于训练期 value function。

## 关键原则

### 训练与部署一致

actor feature extractor 同时用于 Python 训练和最终 C++/ONNX 提交策略。训练侧不得另写可提交 actor feature 逻辑，否则会引入训练/部署漂移。

### Observation 不泄漏后手信息

平台中双方策略拿到的是行动前同一份状态。因此即使 agent 被训练成后手，它的 observation 也必须来自行动前状态，不能看到 opponent 行动后的结果。

### BC 是行为先验，不是 reward

BC 当前用于 actor-only warm start。它不进入 `env.step()` reward，也不替代官方评估口径。相关细节见 `docs/reward_design.md` 和 `training/bc/README.md`。

### Fast option 是已实现的可选路径

fast option、threshold head、pending/backfill 和训练/部署共享的 C++ runtime core 已完成实现，并在比赛后期进入主线。当前通用训练 CLI 不默认启用这条路径；需要训练或评估 fast 行为时必须显式打开 `--enable-fast-runtime-features`，需要独立控制 threshold head 学习率时再传入 `--fast-threshold-learning-rate`。

赛后冻结评估确认 fast controller 相对关闭 fast 有明显收益，但没有确认 learned threshold residual 优于 calibrated prior。这里应区分“fast path 整体有效”和“threshold 学习带来额外增益”两项结论，具体数据见 `docs/fast_option_design.md`。

## 评估口径

训练可以使用 dense reward、BC 初始化和 privileged critic，但评估必须以官方目标为准：

- 终局胜负。
- 最终净金币差。
- 己方毛金币与累计 `vision_spent`。
- 分 map、opponent、机制参数的切片表现。

主线评估最小单位是显式 `EvalTask`。训练内评估由 `training.scripts.train_ppo` 将冻结的 seed、map 和 opponent 切片展开成双方视角的 `EvalTask`：

```text
EvalTask(seed, map_id, opponent_spec, agent_player_id=1)
EvalTask(seed, map_id, opponent_spec, agent_player_id=2)
```

两局共享同一 seed、map、opponent 和机制配置。训练轨迹仍按单局独立计算 reward；双方视角只用于评估聚合和降噪。训练中和大规模本地评估应使用 `ParallelEvalPool` / `evaluate_parallel()`。

## 文档导航

- `training/README.md`：训练栈入口、主线 PPO 能力和常用测试。
- `training/AGENTS.md`：训练代码修改规则。
- `training/rl/README.md`：RL 模块 API、rollout、PPO update、checkpoint 语义。
- `training/bc/README.md`：BC 数据和 actor-only 训练。
- `training/opponents/README.md`：opponent league 和 scripted opponents。
- `docs/policy_network_design.md`：网络结构、动作概率、PPO 双输入语义。
- `docs/reward_design.md`：reward、BC warm start 和评估约束。
- `docs/features/README.md`：actor/critic feature schema 导航。
- `docs/neural_inference.md`：模型推理、ONNX Runtime、量化和提交体积/速度实验结论。
