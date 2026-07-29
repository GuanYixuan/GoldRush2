# NPC Step/Path 与金币目标函数分析

本文基于已导出的 NPC 特征表和 merged replay，分析两个问题：

1. NPC 三步动作是否可先按 step 独立动作近似，还是需要考虑 3-step path。
2. NPC 对金币的响应更接近哪类简单目标函数。

本文新增使用 map3 的两组数据。

## 数据范围

| 标记 | run_id | map_id | continuous trajectories |
| --- | --- | ---: | ---: |
| map1 | `symobs-a-map1-100-20260727-231446` | 1 | `341207` |
| map2 | `symobs-a-map2-100-20260727-231704` | 2 | `342464` |
| map3a | `symobs-m3a-map3-100-20260728-200953` | 3 | `297528` |
| map3b | `symobs-m3b-map3-100-20260728-202647` | 3 | `299418` |

map3 已导出 NPC 特征表：

```text
mechanism/data/processed/npc_features/symobs-m3a-map3-100-20260728-200953/
mechanism/data/processed/npc_features/symobs-m3b-map3-100-20260728-202647/
```

四组数据合计：

```text
continuous trajectories = 1280617
step rows = 3841851
```

## Step 独立性检查

先用最简单的 step 独立模型作为对照：

```text
P(a0, a1, a2) = P0(a0) * P1(a1) * P2(a2)
```

其中 `Pi` 是对应 `step_idx=i` 的经验动作分布。再与完整 3-step 序列经验分布比较。

| 数据 | unique sequences | H(sequence) | sum H(step) | LL gain / trajectory | MI(step0,step1) | MI(step1,step2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| map1 | `123` | `4.0404` | `4.2261` | `0.1857` | `0.0841` | `0.0564` |
| map2 | `125` | `3.9992` | `4.3563` | `0.3571` | `0.1932` | `0.1284` |
| map3a | `109` | `3.9921` | `4.2576` | `0.2655` | `0.1298` | `0.0973` |
| map3b | `109` | `3.9825` | `4.2538` | `0.2712` | `0.1320` | `0.1001` |
| 合并 | `125` | `4.0322` | `4.2869` | `0.2548` | `0.1269` | `0.0904` |

解释：

- `H(sequence)` 明显小于 `sum H(step)`，说明三步动作之间有实质相关性。
- 相比 step 独立模型，完整序列经验分布每条轨迹提升约 `0.255 nats` log-likelihood。
- `step0 -> step1` 的互信息高于 `step1 -> step2`，说明前两步关联更强。

这不严格证明 NPC 一定先规划完整 path；也可能是逐步决策但状态和目标有持续性。不过它足以说明：只用 step 独立动作分布作为 NPC 模型会丢掉重要结构。

## 序列形态

合并数据最高频序列：

| actions | count |
| --- | ---: |
| `[0,0,0]` | `70271` |
| `[1,1,1]` | `70244` |
| `[3,3,3]` | `70019` |
| `[2,2,2]` | `69077` |
| `[1,1,3]` | `35276` |
| `[0,0,2]` | `35161` |
| `[1,1,2]` | `35028` |
| `[0,0,3]` | `34105` |
| `[0,2,2]` | `32544` |
| `[0,3,3]` | `32275` |
| `[1,3,3]` | `31084` |
| `[1,2,2]` | `30701` |

合并形态统计：

| 指标 | count | freq |
| --- | ---: | ---: |
| 三步直线且非停 | `279611` | `21.83%` |
| 含相邻两步反向移动 | `419869` | `32.79%` |
| `a0 == a2 != a1` | `183786` | `14.35%` |
| 三步非停 | `1248254` | `97.47%` |
| displacement = 3 | `777831` | `60.74%` |
| displacement = 1 | `474184` | `37.03%` |

结论：

- NPC 基本每轮都走满 3 步。
- 三步直线非常常见。
- 反向/折返也不少，说明不是单纯“选一个方向走三格”。
- map3 的高频序列更偏左右方向，说明 step/path 统计会受地图和局部目标分布影响。

## Step 分布跨图变化

合并 step 动作分布：

| step_idx | 上 | 下 | 左 | 右 | 停 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `27.34%` | `27.22%` | `22.12%` | `22.10%` | `1.23%` |
| 1 | `23.43%` | `23.60%` | `25.86%` | `25.94%` | `1.16%` |
| 2 | `21.99%` | `21.85%` | `27.56%` | `27.52%` | `1.08%` |

但分图看，map1/map2 的 step0 更偏上下，而 map3 的 step1/step2 更偏左右。该差异提示：step bias 不是纯粹的内置常数，更可能和地图布局、金币分布、站位观测窗口共同相关。

## 金币目标函数：第一步

对每个连续可见 NPC round，读取 `start.grid` 的可见金币，比较 NPC 第一步是否选择几个启发式下的最优合法动作。合法动作只排除边界和静态障碍。

启发式：

- `nearest`：行动后到最近可见金币距离最小。
- `nearest_then_amount`：先最近距离，再比较该距离上的金币金额。
- `value_over_dist`：最大化 `gold_amount / (distance + 1)`。
- `target_gold`：只看第一步目标格是否有金币；当没有邻接金币时大量动作并列，解释力有限。

合并结果：

| heuristic | cases | chosen is best | random best expectation | gain |
| --- | ---: | ---: | ---: | ---: |
| nearest | `1279914` | `54.24%` | `35.81%` | `18.43%` |
| nearest_then_amount | `1279914` | `47.90%` | `30.15%` | `17.74%` |
| value_over_dist | `1279914` | `52.62%` | `33.59%` | `19.03%` |
| target_gold | `1279914` | `74.02%` | `68.13%` | `5.89%` |

`target_gold` 的总体命中率虚高，因为当最近金币距离大于 1 时，所有合法动作通常都无法第一步踩到金币，动作大量并列。它只适合解释邻接金币场景。

按起点最近可见金币距离分桶看，`value_over_dist` 在大多数桶中略优于单纯最近距离：

| nearest gold dist | cases | nearest hit | value/dist hit | random expectation for value/dist |
| ---: | ---: | ---: | ---: | ---: |
| 0 | `167165` | `25.50%` | `46.94%` | `33.29%` |
| 1 | `376625` | `53.16%` | `51.98%` | `29.39%` |
| 2 | `270794` | `59.76%` | `53.26%` | `34.07%` |
| 3 | `156836` | `66.89%` | `55.86%` | `35.93%` |
| 4 | `94859` | `65.40%` | `55.77%` | `36.21%` |
| 5 | `70891` | `62.25%` | `54.67%` | `36.25%` |
| `>=6` | `142744` | `55.01%` | `53.04%` | `38.46%` |

结论：

- NPC 第一步明确响应金币。
- 单纯最近距离已经有较强解释力。
- `value / (distance + 1)` 的总体增益略高，说明金额也有作用，但不是简单“最近距离相同再看金额”。
- 当 NPC 已经站在金币上方附近或可见金币很密集时，简单启发式解释会变差，需要 path 和拾取规则一起分析。

## 金币目标函数：3-step 路径收益

进一步枚举每个 NPC 起点的所有合法 3-step 序列，用 `start.grid` 上可见金币金额计算路径经过格的简单收益和：

```text
path_reward = sum(max(0, start_grid[pos_step]))
```

该指标未模拟先前 NPC/玩家拾取，也未按 `ceil(65%)` 扣减地面金币，只用于比较路径是否朝高金币收益方向移动。

| 数据 | chosen avg reward | best legal avg reward | random legal avg reward | chosen is best |
| --- | ---: | ---: | ---: | ---: |
| map1 | `4.499` | `12.799` | `1.779` | `26.58%` |
| map2 | `4.783` | `14.862` | `2.038` | `18.45%` |
| map3a | `3.461` | `7.914` | `1.174` | `50.40%` |
| map3b | `3.516` | `7.975` | `1.191` | `50.30%` |
| 合并 | `4.104` | `11.090` | `1.570` | `35.48%` |

结论：

- NPC 选择的 3-step 路径收益显著高于随机合法路径。
- 但它远低于当前可见金币下的最优合法路径，说明 NPC 不是简单最大化当前可见金币收益。
- map3 的 `chosen is best` 明显更高，可能是金币/障碍/站位使可选高收益路径更少或更明确；这说明路径收益模型需要分地图验证。

## 建模建议

当前证据支持把 NPC 模型从 step baseline 往 path-aware 方向推进，但不要直接写死复杂目标函数。

建议下一步模型顺序：

1. **Step baseline**

```text
P(action | step_idx, legal_actions)
```

2. **Step softmax + 局部特征**

```text
score(action) =
    step_bias[step_idx, action]
  + w_nearest_gold * delta_nearest_gold_dist(action)
  + w_value_dist * best_gold_value_over_dist_after(action)
  + w_target_gold * target_gold(action)
  + bomb_terms(action)
  + geometry_terms(action)
```

3. **3-step path model**

枚举所有合法长度 3 path，用 path 级特征打分：

```text
score(path) =
    path_shape_terms(path)
  + visible_gold_reward_terms(path)
  + bomb_risk_terms(path)
  + endpoint_geometry_terms(path)
```

优先比较 step softmax 与 path model 的 held-out log-likelihood。若 path model 增益仍接近本文序列经验分布相对 step 独立的增益，说明 NPC 决策确实更适合 path-level 表达。

## 当前结论

- 三步动作之间存在明显相关性，step 独立模型不足。
- 但该相关性不等价于已证明 NPC 预规划完整路径；仍需与带状态特征的 step softmax 对照。
- NPC 对金币有清晰响应，`value / distance` 类特征值得进入模型。
- NPC 不会简单最大化当前可见金币收益；炸弹、地图几何、随机性、隐藏状态或更复杂目标都可能参与。
- map3 提供了有价值的交叉验证：总体结论一致，但 step 方向偏置和 path reward 命中率与 map1/map2 有明显差异。
