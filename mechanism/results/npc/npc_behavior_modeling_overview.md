# NPC 行为建模总览

本文汇总 NPC 行为机制分析结论，并给出面向 simulator/RL 训练的推荐建模方案。本文是当前 NPC 推荐实现的唯一权威入口；其它 NPC 子文档只记录实验结果或历史诊断。目标不是完全复刻官方 NPC，而是构造一个机制结算准确、行为分布接近、参数可随机化的近似模型。

## 数据与子实验

主要数据：

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
- `npc_path_policy_fit.md`：canonical path-level softmax 拟合结果。
- `../simulator_alignment/map1_a_default_alignment.md`：simulator rollout 与真实 replay 分布对齐。
- `../simulator_alignment/npc_repeat_consumption_ablation.md`：pickup bonus / 残堆 cleanup rollout ablation。

## Canonical 口径

后续 NPC 建模统一采用：

```text
回合开始：
    刷新炸弹
    生成金币并落到状态

Plan 阶段：
    玩家与 NPC 都基于同一个 round-start state 产出动作
    7 个 NPC actions 同时产出

执行阶段：
    first_player 执行并结算
    7 个 NPC 按随机 permutation 执行并结算
    second_player 执行并结算
```

即 NPC 不看先手玩家执行后的状态，也不看同轮前序 NPC 已执行后的状态。当前 simulator 已按该口径实现。

## 机制层结论

- 公测规则下每局有 `7` 个 NPC。
- NPC 内部执行顺序近似每轮均匀随机 permutation；两图各 `50000` 轮中观察到全部 `7! = 5040` 种排列。
- 每个 NPC 每轮 `actions` 长度为 `3`，连续可见轨迹可用 `start_pos + actions` 重放到 `end_pos`。
- 未观察到 NPC 越界或撞静态障碍导致的非法移动；模拟器中 NPC path 合法性应排除边界与静态障碍。
- NPC 与玩家、NPC 之间可重叠，不应作为移动阻挡。
- NPC pickup 金额由动态 `ceil(65%)` 规则解释；执行阶段应按真实 dispatch order 扣减地面金币。
- NPC 会触发炸弹并移除炸弹；行为策略中炸弹应是强负效用，而不是硬约束。

## 策略层结论

- 当前数据不支持不同 `npc_id` 使用不同策略；推荐共享一套参数。
- 逐 step 独立动作模型不足，3-step path 有明显结构偏好；推荐枚举合法 3-step path 后做 path-level softmax。
- 动态 `65%` path reward 是稳定的一阶金币信号，但单独使用会低估实际 pickup 次数。
- `pickup_count` 是强特征，表示 path 中发生金币 pickup 的步数；它解释了 simulator 早期“小金币堆残留偏多”的主要偏差。
- `static_pickup_count` 比动态 `pickup_count` 的 path-level NLL 更好；它表示“移动进入 round-start 时有金币的格子”的固定效用，不模拟 path 内金币量扣减。
- `center_delta_chebyshev` 是稳定中心吸引项；加入 pickup 后仍保持正权重。
- `stay_count` 强负，NPC 几乎总是走满三步；但炸弹封路时 `[4,4,4]` 是明确例外。
- M5 细 path shape 项能进一步降低 NLL，但第一版默认可先保留 M4 结构以便调参。

## 推荐默认模型

第一版 simulator 默认建议使用 canonical M4e：

```text
score(path) =
    0.907331 * dynamic_reward_div10(path)
  + 0.912740 * static_pickup_count(path)
  - 2.716949 * enter_bomb_count(path)
  - 3.065158 * stay_count(path)
  + 1.209899 * straight3(path)
  - 0.709615 * backtrack(path)
  + 10.916492 * bomb_trapped_stay(path)
  + 0.167728 * center_delta_chebyshev(path)
```

特征含义：

- `dynamic_reward_div10 = dynamic_pickup_reward / 10`。
- `dynamic_pickup_reward(path)`：按决策快照中的金币堆和 `ceil(65%)` 规则模拟该 NPC path 内部收益；只扣减候选 path 自己经过的局部金币，不扣减其它 NPC path。
- `pickup_count(path)`：path 中实际移动进入有金币格并发生 pickup 的步数。
- `static_pickup_count(path)`：path 中每次移动进入 round-start 时有金币的格子计 `1`；该特征不模拟 path 内金币扣减，amount 为 `1` 的金币格重复进出也可多次计数。
- `enter_bomb_count(path)`：path 中移动进入当前可见炸弹格的次数。
- `stay_count(path)`：三步动作中 `action=4` 的次数。
- `straight3(path)`：三步同向且非停。
- `backtrack(path)`：存在相邻两步反向移动。
- `bomb_trapped_stay(path)`：起点所有合法相邻非停格都是当前可见炸弹，且 path 为 `[4,4,4]`。
- `center_delta_chebyshev(path) = start_chebyshev - end_chebyshev`，其中 `chebyshev(pos)=max(abs(row-8), abs(col-8))`。

M4e 的 valid NLL 为 `3.177952`，相比不含 pickup/center 的 M4a `3.267048` 明显改善；相比 M4d 动态 `pickup_count` 的 `3.192912` 也有进一步改善。

## 统计复现 Profile

如果目标更偏 replay 分布复现，可使用 canonical M5g：

```text
score(path) =
    0.875531 * dynamic_reward_div10(path)
  + 0.905037 * static_pickup_count(path)
  - 2.710227 * enter_bomb_count(path)
  - 2.758740 * stay_count(path)
  + 0.219614 * straight3(path)
  - 0.526579 * backtrack(path)
  + 10.652026 * bomb_trapped_stay(path)
  + 0.826686 * first_two_same_nonstay(path)
  + 0.716532 * last_two_same_nonstay(path)
  + 0.274567 * sandwich(path)
  + 0.167938 * center_delta_chebyshev(path)
```

M5g valid NLL 为 `3.142093`，是当前拟合序列中最低。但参数更多，且收益主要是细 path shape 校准；面向第一版 simulator/RL 训练，默认仍推荐 M4e。

## Domain Randomization

第一版采用 episode-level randomization：每局开始采样一套全局 NPC profile，整局内固定。

建议初始范围：

```text
temperature          ~ Uniform(0.7, 2.5)
w_dynamic_reward     *= LogUniform(0.5, 2.0)
w_static_pickup      += Uniform(-0.35, 0.35)
w_enter_bomb         *= LogUniform(0.5, 2.0)
w_stay               += Uniform(-0.8, 0.8)
w_straight           += Uniform(-0.8, 0.8)
w_backtrack          += Uniform(-0.5, 0.5)
w_bomb_trapped_stay  += Uniform(-3.0, 3.0)
w_center             += Uniform(-0.12, 0.12)
bomb_blind_p         ~ Uniform(0.0, 0.15)
```

`bomb_blind_p` 只影响决策阶段是否看见炸弹；执行阶段仍按真实状态触发炸弹。

不同 `npc_id` 当前不建议独立采样完整权重。若训练需要更宽覆盖，可在 episode profile 基础上加入小幅 per-NPC jitter，默认关闭或设为较小值：

```text
for each npc at episode reset:
    npc_temperature *= LogUniform(0.9, 1.1)
    npc_weight += Normal(0, 0.1 * abs(episode_weight))
```

## 当前解释力

- 移动执行规则：连续可见样本中接近 `100%`。
- NPC 内部调度：随机 permutation 足够解释。
- pickup 金额结算：动态 `65%` 规则 exact match `99.994%`。
- id 无关性：当前数据支持共享策略参数。
- path 选择：canonical M4e valid NLL `3.177952`，actual path probability `8.043%`；M5g valid NLL `3.142093`，actual path probability `8.354%`。
- rollout 对齐：基础 `pickup_count` bonus 能显著修正 simulator 小金币堆残留和地面总量偏高的问题。

因此当前最推荐的模拟器策略是：

```text
准确实现状态转移；
NPC 同轮基于同一个 round-start state 规划；
使用动态金币收益 + static pickup 基础效用 + 炸弹风险 + path 形状 + 中心吸引的 softmax 采样；
通过 episode-level 参数随机化覆盖行为差异。
```

## 后续验证

后续若继续提高保真度，优先：

1. 在 map2/map3 rollout 上复查 M4e 是否同样改善小金币堆残留。
2. 检查 pickup/center 权重在分图上的稳定性。
3. 在有移动玩家的数据上复核 canonical plan 口径。
4. 若 replay 分布复现成为硬需求，再考虑启用 M5g 或引入局部区域驻留项。
