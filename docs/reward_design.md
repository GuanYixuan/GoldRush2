# Reward 与行为先验设计

本文记录 GoldRush2.0 RL 训练中 reward、critic 辅助目标和 BC warm start 的设计细节。架构入口见 `docs/rl_architecture.md`。

## 当前实现

当前 `training/rl` 已实现两个 reward：

- `WinLossReward`：env smoke/default 使用，非终局 reward 为 `0`，agent 胜为 `+1`，agent 负为 `-1`。
- `TerminalWinMarginGoldGainReward`：PPO 主线 reward，schema 为 `terminal_win_margin_gold_gain_v1`。

`SingleAgentGoldRushEnv` 默认仍使用 `WinLossReward`，用于保持基础环境、rollout、evaluation 和 paired sampling 的测试语义稳定。PPO 训练入口应显式传入 `TerminalWinMarginGoldGainReward`。

当前主线 PPO 默认使用 `beta_win=0.0, beta_margin=0.0, beta_gold_gain=0.0, beta_net_gold_gain=0.1`，先以己方净金币 dense reward 稳定“找金币、吃金币、控制风险和视野成本”的基础行为。需要做纯终局胜负 smoke/debug 时，可显式设 `beta_win=1, beta_margin=0, beta_gold_gain=0, beta_net_gold_gain=0`。

## 核心原则

- 最终优化目标是胜负，而不是己方绝对金币量。
- dense reward 用于缓解 500 回合信用分配问题，不替代官方终局目标。
- margin shaping 应基于双方相对净资产。
- 当 BC 先验在 PPO 中被侵蚀、agent 连基础吃金币行为都无法保持时，可以显式加入己方金币增量 dense reward，作为行为先验保护和信用分配辅助。默认使用己方净金币增量，避免 gross gold 奖励鼓励无节制购买视野。
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

## PPO Reward: Win / Margin / Gold Gain / Vision Info

当前 PPO reward 拆成五个独立系数：

```text
reward_schema = "terminal_win_margin_gold_gain_v1"

Phi_t = tanh(M_t / margin_scale)
shaping_t = gamma * Phi_{t+1} - Phi_t
gold_gain_t = max(0, G_agent(t+1) - G_agent(t))
gold_gain_reward_t = clip(gold_gain_t / gold_gain_scale, 0, 1)
net_gold_gain_t = W_agent(t+1) - W_agent(t)
net_gold_gain_reward_t = clip(net_gold_gain_t / net_gold_gain_scale, -1, 1)
vision_info_reward_t = min(
    vision_info_reward_cap,
    beta_vision_info * min(1, fresh_extra_gold_t / vision_info_scale)
)

r_t =
    beta_win * r_win_t
    + beta_margin * shaping_t
    + beta_gold_gain * gold_gain_reward_t
    + beta_net_gold_gain * net_gold_gain_reward_t
    + vision_info_reward_t
```

其中：

- `M_t = W_agent(t) - W_opponent(t)`。
- `W_i(t) = G_i(t) - C_i^vp(t)`，即毛金币减累计视野花费。
- `G_agent(t)` 使用己方 gross gold，不扣视野费用；该 ablation 项只奖励“吃到金币”，但不会惩罚视野花费。
- `net_gold_gain_t` 使用己方净金币增量，包含拾取金币、炸弹/踩踏损失和己方视野成本，不受对手当回合得失直接影响，噪声低于 margin delta。
- `gamma` 使用 PPO 配置中的默认 `0.97`。
- 默认 `beta_win=0.0`、`beta_margin=0.0`、`beta_gold_gain=0.0`、`beta_net_gold_gain=0.1`，因此默认行为是己方净金币 dense reward；终局胜负和 margin shaping 是显式可调项。
- 终止状态的 potential 置为 `0`，避免终局 margin 被重复计入。
- `margin_scale=200` 是当前默认尺度；该项只在 `beta_margin > 0` 时影响 reward。
- `gold_gain_scale=100` 是第一版 dense gold 尺度，单回合吃到 100 金币即达到该项上限。
- `net_gold_gain_scale=50` 是默认净金币 dense 尺度；单回合净赚 50 金币达到正向上限，单回合净亏 50 金币达到负向下限。
- `beta_vision_info=0.0` 默认关闭；开启后只奖励购买视野带来的“新鲜额外可见金币”信息，不改变 actor observation。
- 实现上该 reward 在 `env.reset()` 后用初始 `GameState` 初始化 `Phi_t`；如果未 reset 就调用，会 fail-fast。

设计理由：

- 最终优化目标仍是胜负；当前默认先关闭终局项，是为了在校准阶段降低长时信用分配难度，优先恢复可学习的基础经济行为。
- potential shaping 可以给长时任务提供中间学习信号，但当前默认关闭，避免在基础吃金币行为尚不稳定时引入对手相关噪声。
- `tanh` 有界，避免大额金币事件造成 reward 尺度失控。
- 固定 `margin_scale` 能保持不同训练 run 的 reward 语义一致，避免每次用 rollout 分位数估计导致尺度漂移。
- net dense gold 项直接补充“发现金币、走过去、吃掉金币、控制视野成本”的短期信号，并通过双向 clip 限制偶发大额变化造成的 advantage 方差。

`beta_margin=0, beta_gold_gain=0, beta_net_gold_gain=0` smoke 的验收目标不是胜率提升，而是 loss 有限、无 NaN、GAE/terminal 处理正确、KL/clip fraction/value 输出可解释。

## 购买视野信息 Reward

`vision_info_reward_t` 是用于唤醒视野购买探索的可选辅助项，默认关闭。它不直接补贴视野成本，而是奖励“购买视野后额外看到此前难以确认的新鲜金币信息”。

`TerminalWinMarginGoldGainReward` 维护两个 `17x17` 小表：

- `last_seen_round[row][col]`：agent 最近一次实际看见该格的回合。
- `last_gold_increase_round[row][col]`：该格最近一次由下一轮金币生成事件增加的回合。

每次 `RoundStepEnv.step()` 非终局时会先结算本轮行动，再为下一轮 observation 生成并应用金币；这些下一轮金币生成事件通过 `RoundStepResult.next_round_gold_generated` 暴露给 reward。若 agent 本轮 `vp > 0`，reward 在下一轮 observation 中计算：

```text
extra_cells = actual_visible_cells - base_radius_2_visible_cells
```

只统计 `extra_cells` 中当前 observation 可见且金币金额大于 `0` 的格子。一个金币格会被视为“新鲜信息”，当且仅当：

```text
round_index - last_seen_round[pos] > vision_info_recent_window
or last_seen_round[pos] < last_gold_increase_round[pos]
```

因此：

- 靠移动自然进入基础 `5x5` 视野看到的金币不奖励。
- 买视野看到额外环带金币才可能奖励。
- 最近看过且金币没有新增的格子不重复奖励。
- 最近看过空地，但之后该格刷出或增加金币，再通过购买视野看到，会奖励。

训练入口参数为 `--beta-vision-info`、`--vision-info-scale`、`--vision-info-reward-cap`、`--vision-info-recent-window`。开实验时应同时关注 `vp_fraction_1/2`、`vp_nonzero_fraction`、`reward_vision_info_*`、`train_agent_vision_spent_mean` 和固定 seed eval，避免策略学成无条件买视野。

## 终局胜负 Reward

终局主 reward 定义为：

```text
r_win_T =
    +1 if agent wins
    -1 if agent loses
```

在当前训练假设下，agent 慢、opponent 快；若金币完全相同，tie-break 应判 opponent 胜。若后续做不含真实耗时假设的反事实实验，可以单独改为同分 `0`。

## BC Warm Start 与行为先验

当前 BC 已进入主线训练工作流，但口径是简化的 actor-only warm start：

- `training/bc/` 负责采集 teacher 数据、审计数据集、训练 actor-only BC checkpoint。
- `training.scripts.train_ppo --init-model-checkpoint` 从 BC checkpoint 加载 actor 路径权重。
- critic encoder 和 value head 不从 BC checkpoint 初始化，因为 privileged critic feature 的 channel/scalar 语义与 actor feature 不同。
- BC loss 不进入 PPO update；PPO 阶段只优化 RL loss 和 value loss。

BC 不作为环境 reward，不进入 `env.step()` 返回的 reward。它的当前作用是给 actor 提供“往中心走、吃金币、基础动作合法性和经济行为”的初始化先验，减少 PPO 从随机策略探索时的冷启动噪声。

未来如果需要继续保护 BC 行为，可在 PPO 算法侧增加可退火的 BC auxiliary loss。但这应作为新训练配置显式引入，不能混入 reward：

```text
loss = rl_loss
     + lambda_bc * bc_loss
     + lambda_aux_value * aux_value_loss
```

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

### 当前使用阶段

当前推荐用法：

1. 用 `fast_probe_v3_like` self-play 的后手方采集 BC 数据。
2. 训练 actor-only BC checkpoint。
3. PPO 使用 `--init-model-checkpoint` 初始化 actor。
4. PPO 通过 `--critic-warmup-updates`、`--actor-lr-ramp-updates`、dense net gold reward 和分离 actor/critic lr 控制早期漂移。

BC loss 只用于 BC 训练和迁移，不作为 PPO reward 或最终评估指标。策略是否变强仍以终局胜负、净金币差和 opponent/机制 holdout 表现为准。
