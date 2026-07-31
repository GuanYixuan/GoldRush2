# NPC 行为建模总览

本文汇总 `mechanism/results/npc/` 下各子实验的结论，并给出面向本地模拟器与 RL 训练的推荐 NPC 建模方案。目标不是完全复刻官方 NPC 策略，而是构造一个机制结算准确、策略行为可调、能通过 domain randomization 覆盖较广情况的近似模型。

## 数据与子实验

目前 NPC 分析主要基于以下 merged replay 和导出的特征表：

- 地图1：`symobs-a-map1-100-20260727-231446`
- 地图2：`symobs-a-map2-100-20260727-231704`
- 地图3：`symobs-m3a-map3-100-20260728-200953`
- 地图3：`symobs-m3b-map3-100-20260728-202647`

相关子文档：

- `npc_behavior_feature_observation.md`：NPC 基础动作、调度、拾金和局部金币响应。
- `npc_feature_tables_and_bomb_interaction.md`：特征表导出和 NPC 与炸弹交互。
- `npc_id_independence_analysis.md`：不同 `npc_id` 是否需要独立策略参数。
- `npc_step_path_and_gold_target_analysis.md`：step 独立性、path 结构和金币目标函数。
- `npc_repeated_pickup_analysis.md`：动态 `65%` 拾取规则与反复进入金币格。
- `npc_step_count_and_zero_move_analysis.md`：走步数分布、0 步行为和炸弹封路例外。

## 已确认的机制层结论

以下部分应在本地模拟器中尽量准确实现，因为它们直接影响环境状态转移。

### NPC 数量与行动顺序

公测规则下每局有 `7` 个 NPC。每轮调度顺序稳定表现为：

```text
[fast_player, 7 个 NPC 的随机排列, slow_player]
```

两图各 `50000` 轮都观察到全部 `7! = 5040` 种 NPC 内部排列。各 NPC 出现在内部调度位置的次数接近均匀，因此第一版模拟器可把 NPC 内部顺序建模为每轮均匀随机 permutation。

### NPC 移动执行

连续可见轨迹中：

- 每个 NPC 每轮 `actions` 长度均为 `3`。
- 用 `start_pos + actions` 重放后与 `end_pos` 完全一致。
- 未观察到越界或撞静态障碍导致的非法移动。
- `停` 动作存在但较少，三步非停约 `97%`。

因此模拟器中可以先枚举合法 3-step path，再从候选 path 中采样。合法性至少排除边界与静态障碍；NPC 与玩家、NPC 之间可重叠，不应作为阻挡。

### 金币拾取结算

NPC pickup 字段几乎可由以下规则精确解释：

```text
按 dispatch_order 依次行动；
NPC 每次移动进入一个仍有金币的格子：
    pickup = ceil(0.65 * 当前剩余金币)
    当前格金币 -= pickup
```

在保守样本中，按同轮 7 个 NPC 共享金币堆重放后：

```text
exact match: 99.994%
within 1:    99.998%
```

这说明拾金金额不是另一个需要拟合的随机过程。模拟器应直接按动态 `65%` 规则结算，并让更早行动的 NPC 改变后续 NPC 看到的剩余金币。

### 炸弹触发结算

NPC 会触发炸弹并使炸弹消失。可见数据中，NPC 目标格为仍存在可见炸弹时，基本都会记录为 `bomb_trigger`。炸弹本身的生成与刷新机制应引用 `mechanism/results/bomb/bomb_resampling_observation.md`，NPC 行为模型只需要在 path 打分时加入炸弹风险，并在执行阶段按真实状态触发炸弹。

## 策略层统计结论

以下部分描述 NPC 如何选择路径。它们不需要在模拟器中被写成硬规则，更适合作为随机效用模型的特征。

### NPC 行为可视为近似与 id 无关

当前数据不支持给不同 `npc_id` 设计不同策略：

- 各 id 动作分布相对总体的 TVD 只有千分量级。
- 按 `step_idx` 分开后差异仍很小。
- 邻接炸弹避让、邻接金币吸引在各 id 上高度一致。
- 无条件 pickup、bomb trigger、region 分布的 id 波动跨图不稳定。

建议共享一套 NPC 策略参数，保留 `npc_id` 作为诊断字段即可。

### Step 独立模型不足

NPC 的三步动作存在明显相关性。完整 3-step 序列经验分布相比 step 独立经验模型：

```text
LL gain ~= 0.255 nats / trajectory
```

高频序列包括三步直线，也包括明显折返结构。合并统计中：

- 三步直线且非停：`21.83%`
- 含相邻反向移动：`32.79%`
- `a0 == a2 != a1`：`14.35%`

因此第一版策略模型不建议只做逐 step 独立采样。更推荐枚举合法 3-step path 后做 path-level 采样。

### 金币是最强策略信号

NPC 明显响应金币：

- 第一步按最近金币方向的启发式命中率约 `54%`，显著高于随机合法动作约 `36%`。
- 3-step path 的实际静态金币收益显著高于随机 path。
- 但 NPC 不等于简单最大化静态金币金额和。

关键修正是使用动态 `65%` path reward，而不是简单把经过格金币金额相加：

```text
reward = 0
for pos in path:
    if 这一步移动进入 pos 且 local_gold[pos] > 0:
        gain = ceil(0.65 * local_gold[pos])
        reward += gain
        local_gold[pos] -= gain
```

动态 `65%` 收益下，实际 path 成为合法最优 path 的比例从静态收益的 `35.48%` 提升到：

```text
chosen is best: 53.74%
chosen avg:     2.763
random avg:     0.897
best avg:       5.107
```

这说明动态金币收益是当前解释力最强的单一策略特征，但仍不足以完整解释路径选择。

### 反复进入金币格是重要行为模式

由于每次只能拾取当前金币堆的 `65%`，离开后再进入同一格可以继续获得剩余金币。数据支持该行为模式：

- `revisit_after_gold_cell` 占全部轨迹 `10.02%`。
- 该组 pickup rate 为 `92.68%`。
- 该组动态收益均值 `6.046`，显著高于总体 chosen avg `2.763`。
- 约 `49.41%` 的该类轨迹在动态模拟中真的对同一金币格产生多次收益。

因此 path reward 必须支持同一金币格重复进入后的剩余金币扣减。是否额外加入 `revisit_gold_bonus` 可作为策略随机化参数，而不应替代动态收益结算。

### 炸弹是强负效用但不是硬约束

NPC 对邻接炸弹有强避让信号：

```text
最近可见炸弹距离为 1 时：
直接踩向炸弹: 2.80%
远离炸弹:     82.53%
```

但可见炸弹触发仍有大量样本，因此炸弹不应建模成绝对不可进入。更合理的是在 path score 中加入较强负效用，并保留小概率因随机性、金币收益或视野误差而踩炸弹。

## 推荐建模方案

面向本地模拟器和 RL 训练，推荐采用 path-level 随机效用模型：

```text
每轮：
    先确定 fast_player / slow_player
    对 7 个 NPC 采样一个随机 permutation 作为执行顺序

NPC 决策阶段：
    固定一个 NPC 决策快照，即先手玩家行动后、任何 NPC 执行前的状态
    对 7 个 NPC 分别枚举所有合法 3-step path
    基于同一个决策快照为每条 path 计算 score
    按 softmax(score / temperature) 为每个 NPC 采样 path

NPC 执行阶段：
    按随机 permutation 依次执行已采样 path
    每步按真实规则结算金币和炸弹
```

第一版模拟器推荐使用 M4a 作为默认 score：

```text
score(path) =
    w_gold * dynamic_reward_div10(path)
  + w_enter_bomb * enter_bomb_count(path)
  + w_stay * stay_count(path)
  + w_straight * straight3(path)
  + w_backtrack * backtrack(path)
  + w_bomb_trapped_stay * bomb_trapped_stay(path)
```

其中：

- `dynamic_reward_div10 = dynamic_pickup_reward / 10`。
- `dynamic_pickup_reward(path)`：按 NPC 决策快照中的金币堆和 `ceil(65%)` 规则模拟该 NPC path 的收益；只扣减该候选 path 内部的局部剩余金币，不扣减同轮其它 NPC 的候选或已采样 path。
- `enter_bomb_count(path)`：path 中移动进入当前可见炸弹格的次数；炸弹是强负效用，不是硬约束。
- `stay_count(path)`：三步动作中 `action=4` 的次数。
- `straight3(path)`：三步同向且非停。
- `backtrack(path)`：存在相邻两步反向移动。
- `bomb_trapped_stay(path)`：起点所有合法相邻非停格都是当前可见炸弹，且候选 path 为 `[4,4,4]`。

M5c 加入 `first_two_same_nonstay`、`last_two_same_nonstay` 和 `sandwich` 后，valid NLL 更低、形状统计更贴近 replay，但首动作和前两步 argmax 正确率低于 M4a。出于简洁、易实现和调参稳定性，第一版 simulator 默认不加入 M5 细形状项；M5c 保留为统计复现 profile 或后续 ablation 对照。

决策时序对照见 `npc_path_policy_fit.md`：在同一 train/valid split 下，`simultaneous_start` 口径的 M4a valid NLL 为 `3.267048`，优于 `ordered` 口径的 `3.285297`。因此第一版 simulator 默认采用“同时生成 7 个 NPC actions，再按随机顺序执行结算”，而不是“按执行顺序逐个决策并立即执行”。

该模型的定位是：

- 机制层准确：调度顺序、合法移动、金币扣减、炸弹触发应尽量贴近规则。
- 策略层可调：金币贪婪程度、避弹强度、路径形状偏好和随机性通过参数控制。
- 训练层鲁棒：不同 episode 随机化参数，让 RL agent 见到一族 NPC，而不是过拟合单一 NPC。

## 建议的 domain randomization 参数

第一版采用 episode-level randomization：每局开始时采样一套全局 M4a profile，并在整局内固定。这样能模拟“本局 NPC 整体更贪金币 / 更保守避弹 / 更偏直线”的风格，同时避免每次决策重抽权重造成过强噪声。

默认 M4a profile 可从以下范围开始，后续再根据训练稳定性和 replay 拟合残差收窄：

```text
temperature      ~ Uniform(0.7, 2.5)
w_gold           *= LogUniform(0.5, 2.0)
w_enter_bomb     *= LogUniform(0.5, 2.0)
w_stay           += Uniform(-0.8, 0.8)
w_straight       += Uniform(-0.8, 0.8)
w_backtrack      += Uniform(-0.5, 0.5)
w_bomb_trapped_stay += Uniform(-3.0, 3.0)
bomb_blind_p     ~ Uniform(0.0, 0.15)
```

`bomb_blind_p` 用于模拟 NPC 偶尔忽略炸弹风险。实现上可以在计算 `bomb_risk` 前按概率屏蔽该 NPC 本次决策中的炸弹特征，但执行阶段仍应真实触发炸弹。

per-NPC 差异当前没有 replay 证据支持，因此不要为 7 个 NPC 各自独立采样完整权重。若训练需要更宽覆盖，可在 episode profile 基础上加入小幅 per-NPC jitter，默认幅度为全局扰动的约 `20%`：

```text
for each npc at episode reset:
    npc_temperature *= LogUniform(0.9, 1.1)
    npc_weight += Normal(0, 0.1 * abs(episode_weight))
```

其中 `npc_weight` 指 M4a 各线性权重。jitter 在整局内固定，不在每次 NPC 决策时重采样。第一版实现可以先关闭 per-NPC jitter；若开启，建议作为配置项，默认 `npc_jitter_scale=0.2`。

如果希望更保守，可以先使用以下最小模型：

```text
score(path) =
    w_gold * dynamic_reward_div10(path)
  + w_enter_bomb * enter_bomb_count(path)
```

该模型已经覆盖最重要的两类行为：抢金币和避炸弹。随后再加入 path shape 与 region bias。

## 拟合结果摘要

已完成 M1/M2a/M3 path shape、M4a 炸弹封路停留例外和 M5 细 path shape 的第一版 path-level softmax 拟合，完整实验结果见 `npc_path_policy_fit.md`。本次拟合使用四组 run 的完整 game 覆盖、每 10 条 ordered 样本取 1 条，共 `86962` 条样本。

当前验证集结果：

| model | valid NLL | actual path prob |
| --- | ---: | ---: |
| M0 uniform | `4.370542` | `1.352%` |
| M1 dynamic reward | `4.122165` | `2.964%` |
| M2a dynamic + enter bomb | `4.052143` | `3.260%` |
| M3a + stay | `3.443826` | `5.304%` |
| M3b + straight3 | `3.317798` | `6.634%` |
| M3c + backtrack | `3.287138` | `6.838%` |
| M4a + trapped stay | `3.285297` | `6.852%` |
| M5a + first two same | `3.269292` | `6.973%` |
| M5b + last two same | `3.252415` | `7.124%` |
| M5c + sandwich | `3.250092` | `7.114%` |

上表是 ordered 特征口径下的逐项拟合结果。从复现统计分布看 M5c 的 NLL 最低；从简洁易行和 prefix argmax 正确率看，M4a 更适合作为第一版模拟器默认 NPC 策略中心。

后续补充的决策时序对照显示，在 M1/M2a/M3c/M4a 上，`simultaneous_start` 口径均优于 `ordered` 口径。第一版 simulator 因此使用 simultaneous-start M4a 权重：

| timing | model | valid NLL | top1 | actual path prob |
| --- | --- | ---: | ---: | ---: |
| ordered | M4a | `3.285297` | `17.52%` | `6.852%` |
| simultaneous_start | M4a | `3.267048` | `18.08%` | `7.097%` |

默认 M4a：

```text
score(path) =
    2.073989 * dynamic_reward_div10(path)
  - 2.795283 * enter_bomb_count(path)
  - 3.120374 * stay_count(path)
  + 1.183404 * straight3(path)
  - 0.562663 * backtrack(path)
  + 10.776788 * bomb_trapped_stay(path)
```

其中 `dynamic_reward_div10 = dynamic_pickup_reward / 10`。M2a 已足以校正直接踩炸弹频率；原 M2 中 `adjacent_bomb_count` 增益极小且权重为小正值，暂不建议进入默认策略。M3/M4a 显示 path shape 是强解释项，尤其 `stay_count` 和 `straight3` 能分别解释 NPC 几乎走满三步、三步直线显著高于 uniform 的现象。

另见 `npc_step_count_and_zero_move_analysis.md`：当 NPC 所有合法相邻非停方向都是可见炸弹时，`P(0步)` 约为 `97.95%`。M4a 引入的 `bomb_trapped_stay` 专门修正该稀有例外；在 ordered valid trapped 子集上，M3c 给 `[4,4,4]` 的平均概率约 `0.007%`，M4a 提升到 `59.47%`。M5 系列继承该修正。

M5c 可作为 ordered 特征口径下的统计复现 profile；若后续需要在 simulator 中启用 M5c，应按 `simultaneous_start` 口径重新拟合后再冻结默认权重：

```text
score(path) =
    2.021454 * dynamic_reward_div10(path)
  - 2.789653 * enter_bomb_count(path)
  - 2.815642 * stay_count(path)
  + 0.196427 * straight3(path)
  - 0.384252 * backtrack(path)
  + 10.516389 * bomb_trapped_stay(path)
  + 0.847191 * first_two_same_nonstay(path)
  + 0.707631 * last_two_same_nonstay(path)
  + 0.322904 * sandwich(path)
```

## 当前解释力估计

按现有实验，大致可以这样理解模型解释力：

- 移动执行规则：连续可见样本中接近 `100%`。
- NPC 内部调度：可用随机 permutation 很好解释。
- pickup 金额结算：动态 `65%` 规则 exact match `99.994%`。
- id 无关性：当前数据支持共享策略参数。
- path 选择：动态 `65%` 金币收益可使 `chosen is best` 达到 `53.74%`，显著高于随机，但仍留下大量策略残差。
- 炸弹避让：邻接炸弹场景中 `82.53%` 远离，说明负效用很强，但不是硬约束。

因此，当前最推荐的模拟器策略不是“精确拟合官方 NPC”，而是：

```text
准确实现状态转移；
用动态金币收益 + 炸弹风险 + path 形状的 softmax 策略采样 NPC path；
通过参数随机化覆盖从贪金币到保守避弹、从直线偏好到折返偏好的多种 NPC 行为。
```

## 后续验证方向

后续若需要提高保真度，优先做以下验证：

1. 拟合 path-level softmax 的 held-out log-likelihood，与 step softmax 对照。
2. 分图检查参数是否稳定，避免把地图布局差异误当成 NPC 固有偏好。
3. 在移动玩家数据上重新验证 ordered dynamic reward，因为当前数据收集方式下玩家基本不移动。
4. 检查是否需要显式加入不可见金币/炸弹的记忆或噪声项。

在 RL 训练目标下，除非复现实测 replay 成为硬需求，否则不建议过早加入 per-cell 或 per-id 的细粒度参数。
