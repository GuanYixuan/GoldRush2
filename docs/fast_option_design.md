# Conditional Fast Option 设计

本文记录 fast option 的当前推荐方向。它用于缓解“神经网络策略毫秒级、常态后手，很难击败微秒级高质量规则策略”的问题。

游戏规则决定双方拿到同一份行动前 observation，但实际执行顺序由 `moveDecision()` 响应耗时决定：

```text
更快玩家行动 -> NPC 行动 -> 更慢玩家行动
```

同一格金币、炸弹和碰撞由先行动者优先结算。因此 fast option 的价值不是降低提交 P90 本身，而是在局部资源竞争中把部分回合从“神经网络后手”切换成“fast path 先手”。

## 核心方案

网络在第 `t` 回合正常输出当前官方动作，同时输出一个第 `t+1` 回合使用的连续 fast threshold：

```text
normal_action_t = (actions[0:6], k, order, vp)
threshold_raw_t -> sigmoid -> threshold_t in [low, high]
```

第 `t+1` 回合收到新 `GameInput` 后，fast controller 不运行神经网络，只读取上一回合保存的 `threshold_t` 和当前可见地图：

```text
threshold_int = clamp(floor(threshold_t + 0.5), 4, 30)

if first_hit_5x5_gold(input_t1, threshold_int) and greedy_path_constructed:
    return FastGoldGrabOption(input_t1, threshold_int)  # ultra-fast action, likely first
else:
    run_neural_policy(input_t1)                       # slow action, likely second
```

这不是动作缓存。上一回合网络并不知道下一回合具体会 reveal 哪个金币格；fast controller 必须在新 observation 上动态扫描和构造动作。

第一版不学习显式 `armed` 二分 gate。触发由 deterministic controller 决定，网络只学习“下一回合 fast path 应该多激进”。较低 threshold 增加触发率，较高 threshold 近似禁用普通金币抢夺。

## Threshold Head

推荐第一版使用连续标量：

```text
threshold = low + (high - low) * sigmoid(threshold_raw)
```

初始建议：

```text
low = 4
high = 30
initial_threshold ~= 8
```

`threshold=8` 已在 fast option 预研中显示较高触发率和 episode 层正收益；把 head 初始化到 8 可以让接入初期接近已有强 baseline。可见金币 `>30` 的情况很少，`high=30` 已基本能表达“关闭普通抢金 fast path”，同时比 `64/80` 保留更宽的有效 sigmoid 梯度区间。

训练时把 `threshold_raw` 作为一个连续 stochastic action：

```text
mu_raw = threshold_head(state, sampled_normal_action, fast_belief)
threshold_raw ~ Normal(mu_raw, std_raw)
threshold = 4 + 26 * sigmoid(threshold_raw)
```

PPO 保存并 teacher-force rollout 中采样到的 `threshold_raw`，logprob 在 raw 空间按 Normal 计算；`threshold` 只是给 simulator 和 runtime 使用的确定性变换。这样 threshold head 能用标准 policy gradient 学习，而不是依赖不可导的 `gold >= threshold` 触发条件。

部署时也使用 stochastic action 口径。C++ runtime 应从模型输出的分布参数采样普通动作和 `threshold_raw`，再执行同一 sigmoid 与整数化逻辑；不得在文档或默认提交路径中把 argmax/均值路径作为默认策略。

fast path 只读取保存好的整数 threshold，不在 fast 回合运行神经网络。连续 threshold 转整数时统一使用：

```text
threshold_int = floor(threshold + 0.5)
threshold_int = clamp(threshold_int, 4, 30)
```

不得使用 Python `round()` 的 banker rounding。训练 simulator 和 C++ runtime 必须使用同一口径；trigger 条件为 `grid_gold >= threshold_int`。

## Fast Effective Belief

fast option 的真实收益依赖触发后是否达成预期抢金效果。官方 `GameInput` 不提供双方耗时、真实执行顺序或事件日志；为了避免训练/部署不匹配，第一版不使用“真实是否先手”作为 belief 更新口径，而使用可由部署侧同样计算的抢金效果 proxy。

推荐使用 Beta 后验：

```text
fast_effective_alpha = 4
fast_effective_beta = 1
p_fast_effective = alpha / (alpha + beta)  # 初始 80%
fast_effective_confidence = min((alpha + beta) / 20, 1)
```

每局开始 reset 为 `alpha=4, beta=1`，不跨局继承。初始 `p_fast_effective=0.8`，初始 `fast_effective_confidence=0.25`；约 15 次有效 fast 更新后达到满置信。

每次 fast path 触发后，下一次 neural observe 前用上回合保存的局面和动作做一次只模拟已知金币的轻量模拟：

```text
expected_gain = simulate_known_gold_pickups(saved_input_t, fast_output_t)
actual_delta = current_my_units_gold[inferred_fast_role] - saved_my_units_gold[inferred_fast_role]
score = clip(actual_delta / max(expected_gain, 1), 0, 1)
```

只在 `expected_gain > 0` 时更新 Beta：

```text
alpha += score
beta += 1 - score
```

如果实际收益为负，`score` 会被 clip 到 `0`，等价于一次明确失败样本。第一版不模拟炸弹损失、踩踏、NPC 或敌方行动；这些因素如果导致实际金币变化低于已知金币模拟收益，会自然表现为低 score。

`p_fast_effective` 和 `fast_effective_confidence` 应作为 threshold head 的额外 scalar 输入。它们不进入 actor 主干或 actor scalar FiLM，但 PPO 重算 logprob 时必须能复现同一输入；critic value 也应看到这两个 scalar，否则同一 observation 下未来 fast 有效性分布不同，value 会更难拟合。

第一版不额外保存逐次 belief/debug 字段；`score`、`expected_gain` 和 `actual_delta` 只用于更新 Beta belief，并按 update 聚合到默认诊断指标。

### 轻量金币模拟口径

`simulate_known_gold_pickups()` 只模拟己方 fast output 在上回合保存的可见局面中能解释的金币收益：

- 用 saved `GameInput.grid` 中 `>0` 的金币。
- 按官方规则 `ceil(0.65 * gold)` 计算拾取，并更新该格剩余金币；bounce 可能再次进入同一格。
- 只统计 fast 选择角色对应动作段带来的收益。
- 对越界、`grid < 0` 或可见敌方占位导致的非法移动，该步不执行，后续动作继续尝试，尽量贴近官方非法移动语义。
- 不模拟未知炸弹、已知炸弹损失、踩踏、NPC、敌方行动或赛后视野花费。

该 proxy 的语义是：

```text
fast_effective_score_sample =
  实际持币变化 / 按上回合可见金币和自身 fast 动作可解释的预期收益
```

它不是“真实先手概率”，但更适合训练和部署一致地调节 threshold：如果近期 fast trigger 经常拿不到预期金币，threshold 应自然变保守。

### 依赖的 Fast-Path 性质

上述 proxy 使用了当前 fast-path 的若干性质。实现前必须逐项审计 C++ fast-path，若实际实现不满足，应先修正实现或调整 proxy：

- fast-path 只在当前两个角色5x5范围内（不考虑视野）选目标和寻路；路径经过格在保存的 `GameInput.grid` 中应是已知格。
- fast-path 不选择或进入 `grid < 0` 的格子，因此按设计不会主动选择障碍、炸弹或迷雾格。
- fast-path 当前以 `grid_gold >= threshold_int` 的 first-hit 可见金币作为目标。
- fast-path 输出至多 6 步官方动作，且只让一个己方角色承担抢金动作；另一个角色保持不动或无关。
- fast-path 至多追加一次 bounce；bounce 也必须落在保存局面中可见且 `grid >= 0` 的格子。
- fast-path 可能在去目标或 bounce 过程中吃到路径上的其它已知金币，因此 expected gain 不能只按目标格金币计算。
- fast-path 保存 pending 时必须同时保存足以复现模拟的 `GameInput` 关键字段、实际 `GameOutput` 和触发前两角色持币；不保存 `fast_role`。

`inferred_fast_role` 在慢路径 backfill 时后推得到。实现应根据 pending 中保存的 `GameOutput`、`k/order` 和保存局面，把两个角色各自的官方动作段分别重放到可见金币模拟中，选择产生正 `expected_gain` 或唯一非停动作段的角色。按当前 fast-path 约束，成功 fast action 只应由一个角色承担抢金动作，因此该反推应唯一；训练/simulator 中若无法唯一反推应 fail-fast，部署侧可跳过本次 belief 更新以保持运行稳健。

## 训练语义

引入 fast option 后，策略不再是单一“永远后手”：

```text
neural path: opponent -> NPC -> agent
fast path:   agent -> NPC -> opponent
```

训练环境必须显式建模这个顺序差异，否则网络学不到 fast option 的真实价值。

### Latent 先手率建模

平台 sweep 显示，`mutual odd nonstay` 口径下我方 fast path 对大多数公开对手几乎必定先手，但存在一小批亚微秒级强规则对手会把我方先手率压到 `2%..40%`。训练不能直接照抄经验分布，否则 threshold head 容易学成过度激进；第一版采用更保守的对手速度 regime mixture。

每局开始采样一个隐藏变量：

```text
latent_first_rate = p_fast
```

该变量表示本局 fast trigger 与对手 fast/非停动作竞争时，我方先执行的概率。推荐训练分布：

```text
40%: dominant
  p_fast ~ Uniform(0.95, 1.00)

20%: strong
  p_fast ~ Uniform(0.70, 0.95)

20%: contested
  p_fast ~ Uniform(0.30, 0.70)

20%: crushed
  p_fast ~ Uniform(0.02, 0.10)
```

每次 agent 走 fast path 时，simulator 按 `Bernoulli(p_fast)` 抽取本次是否由我方先执行；同一局内 `p_fast` 保持不变，以模拟稳定的对手速度类型，并让 `Fast Effective Belief` 有机会通过同局历史逐渐校准。agent 走 neural fallback 时仍按慢速后手建模：

```text
if agent_decision_mode == fast:
    agent_first = fast_order_rng.bernoulli(p_fast)
else:
    agent_first = false
```

`fast_order_rng` 必须是独立随机流，不与地图、金币、炸弹、policy sampling 或 opponent sampling 共用 RNG。

`p_fast` 是环境隐藏变量，不直接进入 actor/critic feature。训练和部署都只暴露 `p_fast_effective`、`fast_effective_confidence` 等由抢金效果 proxy 更新出的 belief；这样可以避免训练时把真实先手率泄露给网络，而部署时无法复现。rollout metrics 应记录 `latent_first_rate`、实际 fast 先手率、fast trigger 数和 fast effective score，便于区分“速度 regime 不利”和“threshold 学坏”。

### 标准 PPO 归因

`threshold_t` 是第 `t` 回合采样的 action component，但它控制第 `t+1` 回合是否能走 fast path。第一版按标准 delayed-action MDP 处理，不拆独立 threshold loss：

```text
state_t = 当前 action-time observation + runtime pending threshold_{t-1}
action_t = normal_action_t + threshold_t
```

rollout 中 `threshold_raw_t` 和 `threshold_logprob_t` 记录在第 `t` 条 transition 上。第 `t+1` 回合如果 fast controller 触发，其收益进入后续 reward/return，再通过普通 GAE 回传到 `A_t`，并与第 `t` 条 transition 的联合 logprob 一起参与 PPO ratio。

因此第一版 PPO actor loss 只有一套标准联合目标：

```text
old_logprob_t =
  logprob(normal_action_t)
  + logprob(threshold_raw_t)

policy_loss_t =
  PPOClip(old_logprob_t, new_logprob_t, advantage_t)
```

不把 `threshold_logprob_t` 搬到第 `t+1` 条 transition，也不额外构造 `threshold_loss`、`A_threshold_t` 或 one-step fast delta 辅助目标。这样实现最接近标准 PPO，避免一开始引入额外权重和长期/短期收益冲突。

为降低分析盲区，rollout metrics 仍应把第 `t` 的 threshold 与第 `t+1` 的 fast 后果串起来聚合统计。这些只用于诊断，不进入第一版 loss。

### PPO 必需字段

PPO batch / shared memory 必须保存能重算 logprob/value 的字段：

```text
fast_scalars_t
threshold_raw_sample_t
threshold_logprob_t
threshold_int_t
old_logprob_t   # normal action + threshold_raw
```

PPO 更新时必须 teacher-force rollout 中采样到的 `threshold_raw_sample_t`，禁止用当前模型重新采样。critic 仍只输出普通 state value，不为 threshold 单独输出 value head。

### 默认诊断指标

第一版默认按 update 聚合以下 metrics，不保存逐次 debug 字段：

```text
threshold_raw_mean
threshold_raw_std
threshold_int_mean
threshold_int_p10
threshold_int_p50
threshold_int_p90
threshold_log_std
threshold_entropy
threshold_approx_kl

fast_success_per_episode
fast_miss_no_target_per_episode
fast_path_fail_per_episode
neural_fallback_per_episode
fast_nonstay_per_episode

latent_first_rate_mean
actual_fast_first_rate
fast_order_samples_per_episode

p_fast_effective_mean
p_fast_effective_p50
fast_effective_confidence_mean
fast_effective_updates_per_episode
fast_effective_score_mean
fast_expected_gain_mean
fast_actual_delta_mean

fast_pickup_gold_per_episode
fast_bomb_lost_gold_per_episode
fast_trample_penalty_per_episode
one_step_fast_delta_per_episode
```

命名上避免单独使用含义不清的 `fast_triggered`。`fast_success` 专指真正返回 ultra-fast action；`fast_miss_no_target` 表示 first-hit 没有找到目标；`fast_path_fail` 表示找到目标但 greedy path 构造失败；`neural_fallback` 表示本回合最终回退 neural path。

`one_step_fast_delta_per_episode` 在训练 rollout 中使用轻量 proxy：

```text
one_step_fast_delta = actual_delta - expected_gain
```

其中 `actual_delta` 是 `inferred_fast_role` 从 pending 回合到当前回合的持币变化，`expected_gain` 是 saved visible gold 下重放 pending fast output 得到的可解释金币收益。该指标不是 paired neural baseline regret；wrapper on/off 的真实差值只在 CRN eval 中报告。

第一版不加入 `last_fast_role`、`last_fast_target`、`last_fast_action_count`、逐次 threshold bucket 等 debug-only 字段。若后续需要人工排查 replay，可另开 debug schema 或临时日志，不进入默认 PPO batch。

## Runtime State

fast path 可能跳过完整 `FeatureExtractor.observe()`，以保持亚微秒级耗时。因此 runtime 需要 shadow state 延迟补账：

```text
t neural:
  observe(input_t)
  output normal_action_t + threshold_t
  commit_action(normal_action_t)
  save threshold_t

t+1 fast:
  if fast controller triggers:
      output fast_action_t1
      save pending(input_t1, fast_action_t1)
      return immediately

t+2 neural:
  if pending:
      observe(input_t1)
      commit_action(fast_action_t1)
      clear pending
  observe(input_t2)
  run neural
```

### Pending Backfill 结构

pending 只在 fast path 成功并直接返回 ultra-fast action 时创建。若 first-hit miss 或 greedy path 构造失败，本回合应回退 neural path，不创建 pending。

第一版 pending 只保存 backfill 必需信息；任何可延后到下一次 neural 回合的 belief/diagnostic 计算都不得进入 fast 热路径。特别地，fast path 不计算也不保存 `expected_known_gain`。

推荐结构：

```cpp
struct PendingSnapshotRegion {
    int id;
    int gold_generated;
    int gold_remaining;
    int occupants;
};

struct PendingFastBackfill {
    bool valid;

    int round;
    int8_t grid_padded21[21 * 21];
    Position my_units[2];
    int my_units_gold[2];
    int gold_opp;
    Position visible_enemies[2];
    int num_visible_npcs;
    NpcInfo visible_npcs[MAX_NPCS];

    int snapshot_valid;
    PendingSnapshotRegion snapshot_regions[REGION_COUNT];

    GameOutput output;
    int threshold_int;
};
```

`grid_padded21` 使用 padding=2 的 `21x21` 网格，padding 区填 `GRID_FOG`。真实 `17x17` 网格按以下规则压缩：

```text
-5/-3/-1/0 原样保存
gold > 127 clip 到 127
其它 gold 原样保存
```

`int8 clip 127` 不破坏 `threshold_int <= 30` 的 first-hit 触发，也尽量保留高金币的相对幅度。actor feature 当前对 `gold_count` 使用 `/20` 后 clip 到 `3`，因此 `>60` 的 actor plane 等价；若未来 feature 需要未裁剪金币值，pending grid schema 必须重新评估。

`PendingSnapshotRegion` 只保存 actor feature 当前实际使用的 snapshot 字段。`window_begin/window_end/enter/leave/gold_collected` 第一版不保存；若后续 actor feature 使用这些字段，pending schema 必须同步升级。

### Backfill 顺序

下一次 neural 回合收到 `input_now` 后，必须先处理 pending，再构造当前回合 feature：

```text
if pending.valid:
    restored_input = restore_game_input(pending)

    expected_gain = simulate_known_gold_pickups(restored_input, pending.output)
    update_fast_effective_belief(pending, input_now, expected_gain)

    observe(restored_input)
    commit_action(pending.output)
    clear pending

features_now = observe(input_now)
fast_scalars_now = current_belief_scalars()
run neural model
commit_action(neural_output)
save threshold for next round
```

belief 更新发生在当前 neural model 推理前，使 `fast_scalars_now` 能反映上一次 fast 的实际效果。`simulate_known_gold_pickups()` 在慢路径使用 restored pending input 和 pending output 现算；它可以受 `int8 clip 127` 影响，但不影响 fast 动作或 feature backfill。第一版接受该 proxy 近似以保护 fast 热路径。

已有预研显示 shadow backfill 可以保持 feature state 等价；主线实现前仍必须保留覆盖测试，尤其是：

- temporal planes shift。
- `bomb_belief_mask` 刷新和不可见保持。
- `static_2_mask` 在线发现。
- snapshot memory。
- `commit_action()` 相关状态。
- fast miss/path fail 不创建 pending，而是回退 neural path。
- `int8 clip 127` restore 后 actor feature 与原始输入在当前 schema 下等价。
- Fast Effective Belief 在当前 neural 推理前更新，且每局 reset。

## Fast Controller 边界

第一版 controller 固定为 `FastGoldGrabOption`：

- 读取当前 `GameInput.grid`。
- 使用预研当前主线的 `first-hit` 口径：按固定顺序扫描两个角色各自 5x5 窗口，遇到第一个 `grid_gold >= threshold_int` 的可见金币格立即选中。
- 命中 target 后仍计算两个己方角色到该格的 Manhattan distance，选择更近角色执行。
- 用 greedy path 构造至多 6 步动作；若缩短 Manhattan distance 的候选步均不可走，则不触发并回退 neural path。
- 路径格必须可见、非障碍、非炸弹、非可见敌人。
- 保留当前已验证的 backtrack-bounce 和 hand-fused output 版本。
- 使用 compact pending 保存 fast 回合 backfill 所需 metadata。

不允许网络输出任意程序。

## 实验路线

推荐分阶段推进：

1. 固定 threshold wrapper。
   - 使用 `threshold=8`。
   - 接入真实 fast/neural 执行顺序。
   - 先做 CRN eval，确认 wrapper on/off 的收益、炸弹损失和触发率。

2. 连续 threshold head。
   - 初始化到 `threshold ~= 8`。
   - 只训练 threshold head 或使用较小学习率。
   - 记录 threshold 分布、触发率、fast effective score 和 regret。

3. PPO 联合微调。
   - 从稳定神经网络 checkpoint fork。
   - 保持 fast controller 固定。
   - 让 actor 学会创造/避免 fast 机会。

4. 更复杂 option。
   - 只有当固定 controller 和 threshold head 稳定有效后，再考虑 region、template 或显式 enable。
   - 不优先恢复二分 armed gate；若需要，应改成 value-aware 预测，而不是硬 `should_arm` BCE。

## 评估指标

fast option 实验除官方净金币/胜率外，必须额外报告：

- `threshold_raw_*` 和 `threshold_int_*` 分布与饱和比例。
- fast success/miss/path-fail/fallback rate。
- fast nonstay rate。
- latent/actual fast first rate。
- fast effective score/confidence。
- fast pickup gold。
- fast bomb lost。
- fast trample penalty。
- one-step fast delta，即训练 rollout 中的 `actual_delta - expected_gain` proxy。
- wrapper on/off paired delta。

特别注意 500 round 中炸弹损失随持币增加放大。fast option 不能只看即时 pickup，也必须检查后续位置质量和 bomb loss。
