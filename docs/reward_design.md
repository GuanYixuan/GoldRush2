# Reward 与行为先验设计

本文记录 GoldRush2.0 RL 训练中 reward、critic 辅助目标和 BC warm start 的设计细节。架构入口见 `docs/rl_architecture.md`。

## 当前实现

当前 `training/rl` 只实现最小 `WinLossReward`：

- 非终局 reward 为 `0`。
- agent 胜为 `+1`。
- agent 负为 `-1`。

这是为了先跑通 environment、rollout、evaluation 和 paired sampling。后续训练算法接入后，再引入 dense margin、potential shaping 和 auxiliary value heads。

## 核心原则

- 最终优化目标是胜负，而不是己方绝对金币量。
- dense reward 用于缓解 500 回合信用分配问题，不替代官方终局目标。
- reward 应基于双方相对净资产，而不是只奖励己方拾取金币。
- 不为拾取、炸弹、踩踏、视野购买等事件重复添加 reward；这些事件已经体现在净资产变化中。
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

## Dense Margin Reward

不建议直接使用 `DeltaM_t`，应做尺度压缩。首选：

```text
r_margin_t = asinh(DeltaM_t / s)
```

其中 `s` 应从初期 rollout、官方 replay 或 fast policy 对局估计，例如取 `|DeltaM_t|` 的 P90/P95，并在一次训练 run 内固定。

`asinh` 的优点：

- 小变化区域近似线性。
- 大额金币事件仍保持大小顺序。
- 极端样本不会支配梯度。

可选的更保守形式：

```text
r_margin_t = clip(DeltaM_t / s, -c, c)
```

但 hard clipping 会更早丢失大金币事件之间的差异。

## 终局胜负 Reward

终局主 reward 定义为：

```text
r_win_T =
    +1 if agent wins
    -1 if agent loses
```

在当前训练假设下，agent 慢、opponent 快；若金币完全相同，tie-break 应判 opponent 胜。若后续做不含真实耗时假设的反事实实验，可以单独改为同分 `0`。

## Curriculum

建议分阶段调整 dense margin 与终局胜负的权重：

```text
阶段一：r_t = r_margin_t
阶段二：r_t = alpha * r_margin_t + r_win_t
阶段三：r_t = r_win_t + beta * shaping_t
```

其中 `alpha` 逐步退火：

```text
1.0 -> 0.3 -> 0.1 -> 0.05 或 0
```

阶段目标：

- 阶段一：快速学会基本经济行为、路径收益、避险和视野成本。
- 阶段二：从最大化经济 margin 转向最大化实际胜率。
- 阶段三：以终局胜负为主，只保留小幅 shaping 或完全关闭 dense reward。

## Potential-Based Shaping

后期可将 dense margin 改成 potential-based shaping：

```text
Phi(s_t) = clip(M_t / s, -K, K)
shaping_t = gamma * Phi(s_{t+1}) - Phi(s_t)
r_t = r_win_t + beta * shaping_t
```

终止状态的 potential 应置为 `0`。这种形式比长期直接叠加事件 reward 更干净，主要用于告诉模型哪些动作改善了相对经济局面。

注意：GoldRush 是部分可观测问题；如果 `Phi` 使用完整 simulator state，严格的最优策略不变性结论不能直接照搬，应把它视为训练启发式。

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
