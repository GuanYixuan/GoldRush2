# NPC 中心吸引项检验

本文检验 NPC path policy 是否需要加入“向中心靠拢”的几何项。动机是 simulator rollout 中偶尔出现中心区域没有 NPC，而 replay 中 NPC 更常聚集在中心附近。

本文是中心项的隔离实验记录，不冻结当前默认实现。最终 simulator 默认权重以 `mechanism/results/npc/npc_behavior_modeling_overview.md` 为准。

## 口径

使用与 NPC path policy 主拟合一致的数据和抽样口径：

- `symobs-a-map1-100-20260727-231446`
- `symobs-a-map2-100-20260727-231704`
- `symobs-m3a-map3-100-20260728-200953`
- `symobs-m3b-map3-100-20260728-202647`

样本：

| split | samples |
| --- | ---: |
| train | `68849` |
| valid | `18113` |

输出文件：

```text
temp/npc_center_bias/model_metrics.csv
temp/npc_center_bias/center_feature_summary.csv
temp/npc_center_bias/weights.json
```

本次使用 simultaneous-start 决策口径：同一轮 7 个 NPC 都基于同一个 NPC 决策快照计算候选 path 特征，不扣减同轮其它 NPC 的已采样 path。

## 中心特征

令中心坐标为 `(8, 8)`：

```text
d2(pos) = (row - 8)^2 + (col - 8)^2
manhattan(pos) = abs(row - 8) + abs(col - 8)
chebyshev(pos) = max(abs(row - 8), abs(col - 8))
```

测试以下候选项：

- `center_delta_d2_div10 = (start_d2 - end_d2) / 10`，越大表示越靠近中心。
- `neg_end_center_d2_div10 = -end_d2 / 10`。
- `end_region1 = I(end in center_9x9)`。
- `path_mean_center_delta_d2_div10 = mean(start_d2 - step_pos_d2) / 10`。
- `center_delta_manhattan = start_manhattan - end_manhattan`。
- `center_delta_chebyshev = start_chebyshev - end_chebyshev`。

`center_delta_d2_div10` 与 `neg_end_center_d2_div10` 对同一个样本的 softmax 等价，因为 `start_d2` 对所有候选 path 都是常数。

## 拟合结果

在 simultaneous-start M4a 基础上单独加入一个中心项：

| model | valid NLL | top1 | actual path prob |
| --- | ---: | ---: | ---: |
| M4a simultaneous | `3.267048` | `18.08%` | `7.097%` |
| M4a + center delta d2 | `3.248345` | `17.93%` | `7.363%` |
| M4a + neg end d2 | `3.248345` | `17.93%` | `7.363%` |
| M4a + end region1 | `3.249897` | `18.23%` | `7.339%` |
| M4a + path mean center delta | `3.253572` | `17.07%` | `7.297%` |
| M4a + center delta manhattan | `3.251573` | `17.98%` | `7.340%` |
| M4a + center delta chebyshev | `3.246296` | `17.87%` | `7.402%` |

中心项带来稳定 NLL 改善：

```text
M4a + center_delta_chebyshev:  valid NLL 改善 0.02075 nats / trajectory
M4a + center_delta_d2:         valid NLL 改善 0.01870 nats / trajectory
M4a + end_region1:            valid NLL 改善 0.01715 nats / trajectory
M4a + center_delta_manhattan:  valid NLL 改善 0.01547 nats / trajectory
```

## 中心统计残差

valid split 中，实际 path 与 M4a 期望的中心特征：

| feature | actual | M4a expected | uniform expected |
| --- | ---: | ---: | ---: |
| `center_delta_d2_div10` | `-0.000905` | `-0.237133` | `-0.104485` |
| `neg_end_center_d2_div10` | `-2.952614` | `-3.188842` | `-3.056194` |
| `end_region1` | `0.541379` | `0.491015` | `0.513080` |
| `path_mean_center_delta_d2_div10` | `0.007891` | `-0.131101` | `-0.063907` |
| `center_delta_manhattan` | `-0.005852` | `-0.287075` | `-0.133588` |
| `center_delta_chebyshev` | `-0.012422` | `-0.243084` | `-0.118141` |

解释：

- 实际 `center_delta_d2_div10` 约为 `0`，即整体没有明显远离中心。
- M4a 期望为 `-0.237`，明显预测 NPC 会远离中心。
- 实际 `end_region1` 为 `54.14%`，M4a 期望只有 `49.10%`，低估中心 9x9 终点比例约 `5.0pp`。
- uniform 也低估中心，但 M4a 低估更严重，说明这不是纯粹候选集合几何造成的。

加入 `center_delta_d2` 后：

| feature | actual | model expected | uniform expected |
| --- | ---: | ---: | ---: |
| `center_delta_d2_div10` | `-0.000905` | `0.000631` | `-0.104485` |
| `neg_end_center_d2_div10` | `-2.952614` | `-2.951078` | `-3.056194` |
| `end_region1` | `0.541379` | `0.519711` | `0.513080` |
| `path_mean_center_delta_d2_div10` | `0.007891` | `0.021938` | `-0.063907` |

该项几乎完全修正了平均中心距离漂移，但仍低估 `end_region1`。

加入 `end_region1` 后：

| feature | actual | model expected | uniform expected |
| --- | ---: | ---: | ---: |
| `center_delta_d2_div10` | `-0.000905` | `-0.113111` | `-0.104485` |
| `neg_end_center_d2_div10` | `-2.952614` | `-3.064820` | `-3.056194` |
| `end_region1` | `0.541379` | `0.541678` | `0.513080` |

该项几乎完全修正中心 9x9 终点比例，但对中心距离漂移修正较弱。

## 权重

三种距离项对照：

| feature | weight | valid NLL | 备注 |
| --- | ---: | ---: | --- |
| `center_delta_chebyshev` | `0.180587` | `3.246296` | 当前 NLL 最好，形状上接近中心方形吸引。 |
| `center_delta_d2_div10` | `0.158346` | `3.248345` | 平滑、连续惩罚远离中心。 |
| `center_delta_manhattan` | `0.108561` | `3.251573` | 有改善，但弱于前两者。 |
| `end_region1` | `0.687821` | `3.249897` | 能直接修正中心 9x9 occupancy，但阈值较硬。 |

推荐优先使用切比雪夫距离项：

```text
score(path) += 0.180587 * center_delta_chebyshev(path)
```

等价写法：

```text
score(path) += 0.180587 * (-end_chebyshev(path))
```

完整 M4a + center delta chebyshev 权重：

```text
score(path) =
    2.106687 * dynamic_reward_div10(path)
  - 2.796592 * enter_bomb_count(path)
  - 3.126171 * stay_count(path)
  + 1.212827 * straight3(path)
  - 0.580930 * backtrack(path)
  + 10.882944 * bomb_trapped_stay(path)
  + 0.180587 * center_delta_chebyshev(path)
```

若希望保留更平滑的欧氏风格，也可以使用平方欧氏 profile：

```text
score(path) += 0.158346 * center_delta_d2_div10(path)
```

若目标直接修正中心 9x9 occupancy，可考虑 `end_region1` profile：

```text
score(path) += 0.687821 * end_region1(path)
```

但该项是区域阈值特征，比距离项更硬，跨地图和后续 domain randomization 的鲁棒性可能弱一些。

## 结论

当前数据支持 NPC 存在独立于金币、炸弹和 path shape 的中心吸引项。

第一版 simulator 可在 M4a 上加入中心距离项：

```text
score(path) += w_center * (start_chebyshev - end_chebyshev)
```

推荐中心权重中心值：

```text
w_center = 0.180587
```

用于 RL domain randomization 时，可先使用：

```text
w_center += Uniform(-0.14, 0.14)
```

后续还应做 rollout 分布对齐，重点检查：

- `P(center_9x9_npc_count == 0)`
- `P(center_9x9_npc_count >= 4)`
- 每轮 NPC 平均 `d2` 曲线
- region1 occupancy
