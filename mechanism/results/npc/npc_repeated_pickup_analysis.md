# NPC 反复拾取金币堆分析

本文验证一个关键启发：由于进入金币格只能拾取当前金币堆的 `65%`，离开后再进入同一格可以继续拾取剩余金币，因此 NPC 的折返/来回移动可能是在反复消耗同一金币堆。

## 数据范围

使用四组已导出的 NPC 特征和 merged replay：

- `symobs-a-map1-100-20260727-231446`
- `symobs-a-map2-100-20260727-231704`
- `symobs-m3a-map3-100-20260728-200953`
- `symobs-m3b-map3-100-20260728-202647`

合计连续可见 NPC 轨迹：

```text
1230752
```

## 动态 65% 收益定义

对一条 3-step path，使用 `start.grid` 中可见金币初始化局部金币堆，并按规则模拟：

```text
reward = 0
for pos in path:
    if this step moves into pos and local_gold[pos] > 0:
        gain = ceil(0.65 * local_gold[pos])
        reward += gain
        local_gold[pos] -= gain
```

该定义和此前简单的：

```text
path_reward = sum(start_grid[pos_step])
```

不同。动态 65% 收益会正确处理同一格多次进入后的金币剩余量，因此更贴近真实拾取规则。

## Pickup 字段校验

先做规则校验：在同一轮中，按 `dispatch_order` 模拟 7 个 NPC 共享消耗 `start.grid` 中的可见金币堆，并与 replay 的 `end.npcs[].pickup` 比较。

为避免不可见金币污染，只使用以下保守样本：

- 同轮 7 个 NPC 都连续可见。
- 每个 NPC 的三步路径涉及的格子在 `start.grid` 中均可见。
- 按 NPC 内部调度顺序依次扣减同一个局部金币表。

结果：

| 数据 | full observable rounds | NPC samples | exact match | within 1 | observed avg | simulated avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| map1 | `41798` | `292586` | `99.986%` | `99.995%` | `2.7828` | `2.7830` |
| map2 | `42804` | `299628` | `100.000%` | `100.000%` | `2.9650` | `2.9650` |
| map3a | `10862` | `76034` | `100.000%` | `100.000%` | `2.0775` | `2.0775` |
| map3b | `11531` | `80717` | `100.000%` | `100.000%` | `2.1261` | `2.1261` |
| 合并 | `106995` | `748965` | `99.994%` | `99.998%` | `2.7133` | `2.7134` |

误差分布几乎全为 `0`：

```text
observed_pickup - simulated_pickup:
0: 748923
-1: 26
-2: 11
-3: 2
-4: 1
-5: 1
<-5: 1
```

结论：

- NPC pickup 字段基本可由“按 NPC 调度顺序共享金币堆 + 每次进入拾取 ceil(65%)”精确解释。
- 这确认了后续 NPC 目标函数应使用动态 65% 收益，而不是静态金币金额和。

## 动态收益路径对照

枚举每个 NPC 起点的所有合法 3-step path，比较 NPC 实际 path 的动态 65% 收益、合法最优 path、随机合法 path。

| 数据 | cases | chosen avg | best avg | random avg | chosen best | chosen positive | best positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| map1 | `339514` | `3.0167` | `5.7696` | `0.9913` | `45.87%` | `53.23%` | `83.15%` |
| map2 | `340854` | `3.2354` | `6.7711` | `1.1698` | `36.23%` | `56.93%` | `90.89%` |
| map3a | `273998` | `2.2923` | `3.6480` | `0.6657` | `69.46%` | `38.76%` | `57.58%` |
| map3b | `276386` | `2.3372` | `3.6875` | `0.6754` | `69.41%` | `38.96%` | `57.93%` |
| 合并 | `1230752` | `2.7634` | `5.1071` | `0.8973` | `53.74%` | `47.83%` | `73.94%` |

相比此前静态 path reward，动态 65% 模型更贴近真实规则，并显著提高了“chosen path 是最优 path”的比例：

- 静态金币金额和模型：合并 chosen best `35.48%`
- 动态 65% 收益模型：合并 chosen best `53.74%`

结论：

- NPC path 明显优于随机合法 path。
- 动态 65% 收益能解释相当一部分 path 选择。
- 但仍不是完全贪婪最优，特别是 map1/map2 仍有较大 best gap，说明炸弹、几何、随机性或隐藏目标仍需建模。

## 折返与重复拾取

按 path 形态分组：

| 组别 | cases | freq | pickup rate | pickup amount/event | dynamic reward avg | chosen best | repeated gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| immediate_backtrack | `404617` | `32.88%` | `45.40%` | `5.446` | `2.765` | `60.03%` | `15.06%` |
| sandwich (`a0 == a2 != a1`) | `179750` | `14.60%` | `45.31%` | `4.810` | `2.422` | `58.31%` | `12.01%` |
| straight3 | `270280` | `21.96%` | `40.90%` | `5.679` | `2.603` | `49.77%` | `0.00%` |
| revisit_after_cell | `431676` | `35.07%` | `43.30%` | `5.408` | `2.617` | `59.08%` | `14.12%` |
| revisit_after_gold_cell | `123337` | `10.02%` | `92.68%` | `5.789` | `6.046` | `61.80%` | `49.41%` |
| other | `439376` | `35.70%` | `49.05%` | `5.541` | `3.004` | `51.30%` | `0.00%` |

这里：

- `immediate_backtrack`：三步中存在相邻两步互为反向。
- `sandwich`：`a0 == a2 != a1`，典型来回结构。
- `revisit_after_cell`：path 中重复进入某个已访问格。
- `revisit_after_gold_cell`：重复进入的格子在 `start.grid` 中有可见金币。
- `repeated gain`：动态模拟中同一金币格产生至少两次 pickup gain。

关键结论：

- `revisit_after_gold_cell` 只占全部轨迹 `10.02%`，但 pickup rate 高达 `92.68%`。
- 该组动态收益均值 `6.046`，显著高于总体 chosen avg `2.763`。
- 该组中约 `49.41%` 真的在动态模拟里对同一金币格产生多次收益。
- 因此“反复来回消耗同一金币堆”不是所有折返的唯一原因，但确实是一类强信号、高收益、常见到不可忽略的 NPC 行为模式。

## 对建模的影响

此前我们认为 NPC 不是简单最大化静态可见金币和；本实验进一步说明原因之一是静态 reward 定义错了。真实可用的 path reward 应考虑：

```text
1. 只有移动进入金币格才 pickup。
2. 每次 pickup 为 ceil(65% * remaining_gold)。
3. 同一格离开后再进入可以继续 pickup。
4. 同一轮更早行动的 NPC 会改变该格 remaining_gold。
```

建议后续 NPC path model 使用动态收益特征：

```text
score(path) =
    path_shape_terms(path)
  + dynamic_pickup_reward(path)
  + repeated_gold_reentry_bonus(path)
  + bomb_risk_terms(path)
  + geometry_terms(path)
```

其中 `dynamic_pickup_reward(path)` 至少应有两种版本：

- `solo_dynamic_reward`：只考虑该 NPC 自己在当前 `start.grid` 上拾取。
- `ordered_dynamic_reward`：按 `dispatch_order` 模拟前序 NPC 后的剩余金币。

当前玩家不动，所以暂不需要模拟快玩家对金币堆的改变；若未来使用移动玩家数据，这一项需要补上。

## 结论

当前数据强支持该启发：

```text
NPC 的折返/来回移动中，有相当一部分是在利用 65% pickup 规则反复消耗同一金币堆。
```

更精确地说：

- 动态 65% 规则几乎精确解释 replay 的 NPC `pickup` 字段。
- 重复进入可见金币格的轨迹占 `10.02%`，且几乎必然伴随 pickup。
- 动态 65% path reward 比静态金币金额和更能解释 NPC path 选择。
- 但 NPC 仍不是纯粹动态收益最大化器；应继续加入炸弹风险、path 形状和几何项。
