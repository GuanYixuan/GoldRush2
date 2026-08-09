# Reward 与行为先验设计

本文记录 GoldRush2.0 RL 训练中 reward、critic 辅助目标和 BC warm start 的设计细节。架构入口见 `docs/rl_architecture.md`。

## 当前实现

当前 `training/rl` 已实现两个 reward：

- `WinLossReward`：env smoke/default 使用，非终局 reward 为 `0`，agent 胜为 `+1`，agent 负为 `-1`。
- `TerminalWinMarginGoldGainReward`：PPO 主线 reward，schema 为 `terminal_win_margin_gold_gain_v1`。

`SingleAgentGoldRushEnv` 默认仍使用 `WinLossReward`，用于保持基础环境、rollout、evaluation 和 paired sampling 的测试语义稳定。PPO 训练入口应显式传入 `TerminalWinMarginGoldGainReward`。

PPO smoke/debug 时将 `beta_margin=0` 且 `beta_gold_gain=0`，同一个 reward 类会退化为纯终局胜负 reward，用于验证 PPO 代码链路、GAE、logprob、KL、checkpoint 和 evaluation。

## 核心原则

- 最终优化目标是胜负，而不是己方绝对金币量。
- dense reward 用于缓解 500 回合信用分配问题，不替代官方终局目标。
- margin shaping 应基于双方相对净资产。
- 当 BC 先验在 PPO 中被侵蚀、agent 连基础吃金币行为都无法保持时，可以显式加入己方金币增量 dense reward，作为行为先验保护和信用分配辅助。
- 训练环境可以使用 simulator privileged state 计算 reward；actor 输入仍必须保持官方合法 observation 或由 `policy_runtime` 维护的 belief。

## 净资产与 Margin

定义玩家 `i` 在回合 `t` 的净资产：

```text
W_i(t) = G_i(t) - C_i^vp(t)
```

其中：

- `G_i(t)`：玩家两名角色当前总持币。
- `C_i^vp(t)`：截至当前累计购买视野费用。

定义 agent 相对 opponent 的净金币差：

```text
M_t = W_agent(t) - W_opponent(t)
```

一回合的竞争局面变化为：

```text
DeltaM_t = M_{t+1} - M_t
```

它自然包含：

- 己方拾取金币：正向。
- 对手拾取金币：负向。
- 己方踩炸弹或踩踏：负向。
- 对手踩炸弹或踩踏：正向。
- 己方购买视野：即时负成本。
- 对手购买视野：相对正收益。

## PPO Reward: Win / Margin / Gold Gain

当前 PPO reward 拆成三个独立系数：

```text
reward_schema = "terminal_win_margin_gold_gain_v1"

Phi_t = tanh(M_t / 500)
shaping_t = gamma * Phi_{t+1} - Phi_t
gold_gain_t = max(0, G_agent(t+1) - G_agent(t))
gold_gain_reward_t = clip(gold_gain_t / gold_gain_scale, 0, 1)

r_t =
    beta_win * r_win_t
    + beta_margin * shaping_t
    + beta_gold_gain * gold_gain_reward_t
```

其中：

- `M_t = W_agent(t) - W_opponent(t)`。
- `W_i(t) = G_i(t) - C_i^vp(t)`，即毛金币减累计视野花费。
- `G_agent(t)` 使用己方 gross gold，不扣视野费用；这样 dense gold 项只奖励“吃到金币”，不把“少买视野”误当成吃金币能力。
- `gamma` 使用 PPO 配置中的 `0.9999`。
- 默认 `beta_win=1.0`、`beta_margin=0.2`、`beta_gold_gain=0.0`，因此默认行为等价于旧的终局胜负加 margin potential shaping。
- 终止状态的 potential 置为 `0`，避免终局 margin 被重复计入。
- `margin_scale=500` 是人工固定尺度：`M=500` 时 `tanh(1)≈0.76`，`M=1000` 时 `tanh(2)≈0.96`，表示 500 金币已是很大差距，1000 金币基本饱和。
- `gold_gain_scale=100` 是第一版 dense gold 尺度，单回合吃到 100 金币即达到该项上限。
- 实现上该 reward 在 `env.reset()` 后用初始 `GameState` 初始化 `Phi_t`；如果未 reset 就调用，会 fail-fast。

设计理由：

- 最终优化目标仍是胜负，终局 `+1/-1` 保持主目标地位。
- potential shaping 给 500 回合长时任务提供更稳定的中间学习信号。
- `tanh` 有界，避免大额金币事件造成 reward 尺度失控。
- 固定 `s_M=500` 能保持不同训练 run 的 reward 语义一致，避免每次用 rollout 分位数估计导致尺度漂移。
- dense gold 项直接补充“发现金币、走过去、吃掉金币”的短期信号，并通过 clip 限制偶发大金币造成的 advantage 方差。

`beta_margin=0, beta_gold_gain=0` smoke 的验收目标不是胜率提升，而是 loss 有限、无 NaN、GAE/terminal 处理正确、KL/clip fraction/value 输出可解释。

## 终局胜负 Reward

终局主 reward 定义为：

```text
r_win_T =
    +1 if agent wins
    -1 if agent loses
```

在当前训练假设下，agent 慢、opponent 快；若金币完全相同，tie-break 应判 opponent 胜。若后续做不含真实耗时假设的反事实实验，可以单独改为同分 `0`。

## 后续 Reward 备选

若 `terminal_win_margin_gold_gain_v1` 过弱或过强，优先做小范围 ablation：

- `s_M = 300 / 500 / 800`。
- `beta_margin = 0.1 / 0.2 / 0.3`。
- `beta_gold_gain = 0.5 / 1.0 / 2.0`。
- `gold_gain_scale = 50 / 100 / 200`。
- 关闭 shaping，回到纯终局胜负。

不建议第一版直接使用 `DeltaM_t` dense reward。若后续需要，可考虑：

```text
r_margin_t = asinh(DeltaM_t / s_delta)
r_t = r_win_t + alpha * r_margin_t
```

其中 `s_delta` 应从 rollout 或 replay 的 `|DeltaM_t|` 分布中固定估计。该方案更直接，但噪声更大，也更容易让策略过度追逐短期金币流，因此不作为第一版主线。

注意：GoldRush 是部分可观测问题；如果 shaping 使用完整 simulator state，严格的 potential-based 最优策略不变性结论不能直接照搬，应把它视为训练启发式。

## Critic 多头

Value function 不建议只用一个 head 同时拟合胜率、最终金币差和短期金币流。推荐共享 backbone 后使用多头：

1. `win_value`
   预测最终胜负，范围 `[-1, 1]`，作为 PPO 主 value。

2. `margin_value`
   预测压缩后的最终净金币差，例如：

   ```text
   asinh(M_T / s)
   ```

   作为辅助监督，帮助网络理解经济优势。

3. `cashflow_value`
   预测未来短窗口净金币差变化：

   ```text
   sum_{k=1..H} DeltaM_{t+k}
   ```

   用于学习近期金币、NPC、路径和风险模式。

critic 可以在训练中使用全局状态、真实对手金币、真实视野费用等 privileged 信息；部署时只需要 actor。

## 评估约束

训练可以使用 dense margin，但评估必须以官方目标为准：

- 终局胜负。
- 最终净金币差。
- 己方毛金币与累计 `vision_spent`。
- 分 opponent / 分机制参数的胜率和净金币差。

dense reward 只能用于优化，不应替代官方评估。

## BC Warm Start 与行为先验

训练框架应支持可配置的行为克隆辅助损失，用于新网络结构初始化、策略迁移和早期训练稳定。

第一版 PPO 不使用 BC warm start，先跑通 `feature -> network -> PPO -> eval` 主链路。BC 作为后续扩展加入。

BC 不作为环境 reward，不进入 `env.step()` 返回的 reward。它是训练算法侧的 auxiliary loss：

```text
loss = rl_loss
     + lambda_bc * bc_loss
     + lambda_aux_value * aux_value_loss
```

`lambda_bc` 必须可配置，并支持按训练阶段退火。

### 数据来源

BC teacher action 可以来自：

- `fast_probing/` 中的启发式策略。
- `training/opponents/` 中的 opponent league 策略。
- 平台 replay 或官方样本转换出的轨迹。
- 历史 checkpoint 或旧网络结构的离线推理结果。

这些来源应统一转换为官方动作格式：

```text
k, order, actions[0:6], vp
```

如果未来策略输出包含候选计划、价值图或其它辅助头，BC 数据也可以扩展为对应 teacher target，但基础动作监督应保持稳定。

### Loss 形态

基础动作 BC loss 建议按动作头拆分：

- `k`：分类交叉熵。
- `order`：二分类交叉熵。
- `vp`：分类交叉熵。
- `actions[0:6]`：逐步分类交叉熵。

总 BC loss 可按头加权：

```text
bc_loss =
    w_k * CE(k)
  + w_order * CE(order)
  + w_vp * CE(vp)
  + w_actions * mean_t CE(action_t)
```

如果某些 teacher 只提供部分动作标签，loss 应支持 mask；不要为了补齐标签引入伪监督。

### 使用阶段

推荐用法：

1. 新网络结构上线时，先做纯 BC 或高 `lambda_bc` 预热，学习合法动作和基础经济行为。
2. 进入 against-league RL 后继续保留小权重 BC，减少策略早期漂移。
3. 后期逐步降低 `lambda_bc`，让终局胜负和 dense margin 主导优化。

BC loss 只用于优化和迁移，不作为评估指标。策略是否变强仍以终局胜负、净金币差和 opponent/机制 holdout 表现为准。
