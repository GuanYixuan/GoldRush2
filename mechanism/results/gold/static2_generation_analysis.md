# 外围 static_map=2 高额 Batch 分析

本文分析外围 `region != 1 && static_map=2` 格的高额金币 batch 生成。结论基于地图1、地图2、地图3 M3-A/M3-B 的双视角合并 replay。

## 数据与产物

专题特征由 `mechanism/scripts/analysis/analyze_static2_generation.py` 从金币特征表抽取：

```bash
conda run -n goldrush python mechanism/scripts/analysis/analyze_static2_generation.py
```

输入：

```text
mechanism/data/processed/gold_features/<run_id>/cell_transitions.csv
```

输出：

```text
mechanism/data/processed/static2_analysis/maps123_static2_generation/
```

主要文件：

- `static2_region_rounds.csv`：每个 run / game / round / region 一行。
- `static2_batches.csv`：每个 observed high batch 一行。
- `static2_batch_cells.csv`：observed high batch 中每个正增量 `static_map=2` 格一行。
- `summary.json`：分析摘要。

覆盖批次：

| 地图 | run_id | 对局数 |
| --- | --- | ---: |
| 地图1 | `symobs-a-map1-100-20260727-231446` | `100` |
| 地图2 | `symobs-a-map2-100-20260727-231704` | `100` |
| 地图3 M3-A | `symobs-m3a-map3-100-20260728-200953` | `100` |
| 地图3 M3-B | `symobs-m3b-map3-100-20260728-202647` | `100` |

总计：

| 指标 | 数值 |
| --- | ---: |
| game-rounds | `197600` |
| region-rounds | `790400` |
| observed high batches | `16360` |
| full high batches | `15326` |
| observed multi-high game-rounds | `4` |
| full multi-high game-rounds | `0` |

本文使用两个观测口径：

```text
is_high_batch_observed =
    static2_delta_sum >= 50
    or static2_max_delta >= 20

is_full_high_batch_observed =
    static2_delta_sum >= 50
```

`observed` 口径用于触发时序、region 选择和外围普通格耦合分析；`full` 口径沿用原 `sum>=50`，主要用于 batch total、positive cell count 等金额分布估计，避免把地图3漏观察 partial batch 当成真实金额下降。

## 触发频率

| 数据 | game-rounds | observed high | observed rate | full high | partial-only | observed multi-high rounds | full total mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 地图1 | `49400` | `4067` | `0.08233` | `4037` | `30` | `0` | `88.1719` |
| 地图2 | `49400` | `4127` | `0.08354` | `4122` | `5` | `0` | `88.3503` |
| 地图3 pooled | `98800` | `8166` | `0.08265` | `7167` | `999` | `4` | `80.8122` |
| 全部 | `197600` | `16360` | `0.08279` | `15326` | `1034` | `4` | `84.7782` |

改用 `observed` 口径后，地图3 high rate 从 `0.07254` 回到 `0.08265`，与地图1/2高度接近，说明此前地图3偏低主要是 `static_map=2` 漏观察造成。`observed` 口径只引入 `4` 个同轮多 high region，集中在地图3，数量很小；`full` 口径仍为 `0`。

地图3的 full batch total mean 仍低于地图1/2，主要可能来自地图3静态结构和单局 `static_map=2` 可见覆盖差异；触发机制结构本身没有变化。

## 触发间隔

high batch 不是每回合独立 Bernoulli。按同一 game 内相邻 high batch 的 round gap 统计：

| 数据 | observed gaps | min | mean | `8..16` 占比 | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| 地图1 | `3967` | `8` | `12.134` | `100.00%` | 干净落在 `8..16` |
| 地图2 | `4027` | `8` | `11.959` | `99.95%` | 仅 `2` 个 `17..30` |
| 地图3 pooled | `7966` | `0` | `12.101` | `98.64%` | `4` 个 gap `<8` 来自同轮多 observed region |

新口径下三张图的 observed gap 主体都集中在 `8..16`。地图3原先较多 `17..30` 长间隔被 partial high batch 补回，进一步支持 static2 high batch 是带最小间隔的状态触发，而不是每回合独立 Bernoulli。

三张图 observed gap 的 `8..16` 计数：

| gap | count |
| ---: | ---: |
| 8 | `1781` |
| 9 | `1771` |
| 10 | `1759` |
| 11 | `1514` |
| 12 | `1980` |
| 13 | `1747` |
| 14 | `1970` |
| 15 | `1605` |
| 16 | `1723` |

建模建议：static2 high batch 触发器不应写成每回合 `Bernoulli(p)`；更合适的是在一次 high batch 后采样下一次触发间隔，第一版可用经验 categorical：

```text
next_gap ~ EmpiricalCategorical({8..16 dominant, rare longer gaps})
```

若只做粗仿真，可用 `UniformInteger(8, 16)` 作为第一近似。

## 相位关系

按 `round_mod5` 和 `round_mod20` 统计，high batch 没有像炸弹那样锁定固定相位。

全图 `round_mod5`：

| round_mod5 | count |
| ---: | ---: |
| 0 | `3138` |
| 1 | `3166` |
| 2 | `3363` |
| 3 | `3559` |
| 4 | `3134` |

全图 `round_mod20` 也分散在 `0..19`，没有固定刷新相位。因此 static2 high batch 与 snapshot 5轮窗口、炸弹 20轮刷新不是同一个相位机制。

## Region 选择

high region 计数：

| 数据 | region 2 | region 3 | region 4 | region 5 |
| --- | ---: | ---: | ---: | ---: |
| 地图1 | `907` | `854` | `1100` | `1206` |
| 地图2 | `965` | `844` | `1005` | `1313` |
| 地图3 pooled | `2120` | `1985` | `2010` | `2051` |
| 全部 | `3992` | `3683` | `4115` | `4570` |

相邻 high batch 的 region transition 中，同 region 连续触发是允许的：

| 数据 | transitions | same-region transitions |
| --- | ---: | ---: |
| 地图1 | `3967` | `1302` |
| 地图2 | `4027` | `1215` |
| 地图3 pooled | `7966` | `2265` |
| 全部 | `15960` | `4782` |

region 计数和 batch total mean 都受可见覆盖影响，不宜直接解释为真实 region 权重。第一版生成器可先用近似均匀 region，或按地图/布局使用经验权重。

## Batch Total 分布

本节使用 `is_full_high_batch_observed=1` 的 full high batch 统计，避免把 partial high 的可见片段误当成真实 batch total。

| 数据 | mean | median | p10 | p90 | min | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 地图1 | `88.172` | `89` | `66` | `107` | `50` | `112` |
| 地图2 | `88.350` | `89` | `69` | `107` | `53` | `112` |
| 地图3 pooled | `80.812` | `84` | `54` | `106` | `50` | `112` |
| 全部 | `84.778` | `87` | `59` | `106` | `50` | `112` |

全图最高频 batch total：

| total | count |
| ---: | ---: |
| 80 | `653` |
| 96 | `576` |
| 92 | `526` |
| 84 | `507` |
| 104 | `491` |
| 100 | `471` |
| 81 | `454` |
| 88 | `424` |
| 66 | `392` |
| 108 | `380` |

batch total 呈现明显离散峰值，建议优先用经验 categorical 建模，而不是简单正态/均匀分布。

## Positive Cell Count

本节同样使用 full high batch 统计。

high batch 中正增量 `static_map=2` 格数量：

| positive cells | 地图1 | 地图2 | 地图3 pooled | 全部 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `13` | `1` | `20` | `34` |
| 2 | `353` | `70` | `893` | `1316` |
| 3 | `1066` | `554` | `2622` | `4242` |
| 4 | `1031` | `1750` | `1812` | `4593` |
| 5 | `1574` | `1747` | `1820` | `5141` |

batch total 与 positive cell count 强相关：

| positive cells | batches | mean total | range |
| ---: | ---: | ---: | --- |
| 1 | `34` | `63.06` | `50..107` |
| 2 | `1316` | `62.06` | `50..112` |
| 3 | `4242` | `71.88` | `50..112` |
| 4 | `4593` | `91.55` | `64..112` |
| 5 | `5141` | `95.33` | `80..112` |

地图3的 `2/3` positive cell batch 更多，是其 batch total mean 较低的重要原因；这可能反映地图3 static2 几何/可见覆盖，而不是单格金额机制改变。

## 候选格选择与金额分配

地图1/2 的漏观察较少，可用 `visible_static2_cells=5` 的 full high batch 分析 high region 内 5 个 `static_map=2` 候选格的选择过程。

在该干净口径下，positive cell count 分布为：

| 数据 | batches | k=3 | k=4 | k=5 |
| --- | ---: | ---: | ---: | ---: |
| 地图1 | `2158` | `38` | `546` | `1574` |
| 地图2 | `2202` | `23` | `432` | `1747` |
| 地图1/2 combined | `4360` | `1.4%` | `22.4%` | `76.2%` |

这说明 high batch 不是“5 个 static2 格各自独立触发”。更像是一次 batch 先决定覆盖格数 `k`，其中多数覆盖全部 5 格，少数覆盖 4 格，极少覆盖 3 格。

`k=4` 时等价于“5 个候选格中漏掉 1 个”。实测遗漏格不是完全均匀：

| 数据 | region | 最高遗漏格 | 遗漏次数 | 最低遗漏格 | 遗漏次数 |
| --- | ---: | --- | ---: | --- | ---: |
| 地图1 | 2 | `(1,6)` | `70` | `(0,12)` | `9` |
| 地图1 | 4 | `(15,6)` | `94` | `(15,10)` | `25` |
| 地图2 | 2 | `(0,8)` | `45` | `(1,12)` | `26` |
| 地图2 | 3 | `(9,1)` | `37` | `(5,0)` | `12` |

遗漏率与距离存在弱到中等关系，但不足以完全解释：

| 特征 | 与遗漏率相关 |
| --- | ---: |
| 到中心平方欧氏距离 | `-0.459` |
| 到中心曼哈顿距离 | `-0.529` |
| 到中心切比雪夫距离 | `-0.328` |
| 到边缘距离 | `+0.328` |

方向上，越靠中心的 static2 格越容易被漏掉，越贴边的格越不容易被漏掉。但同距离格仍有明显差异，例如地图1 region 2 的 `(1,6)` 与 `(1,10)` 距离特征相同，遗漏率分别为 `0.479` 和 `0.137`。因此不建议只用中心距离或边缘距离替代格子级经验权重。

金额分配过程更清晰：同一个 full high batch 内，各正增量格子的金额差几乎总是 `0` 或 `1`。因此它更像是先采样 batch total `T`，再把 `T` 均分到选中的 `k` 个格：

```text
base = T // k
rem = T % k

for cell in selected_cells:
    add base
for cell in sample_without_replacement(selected_cells, rem):
    add 1
```

完整 5 候选格样本中，`T` 主要在 `80..112`，且 `80..95` 与 `96..112` 两个区间的 `k` 分布几乎一致：

| T 区间 | batches | k=3 | k=4 | k=5 |
| --- | ---: | ---: | ---: | ---: |
| `80..95` | `2176` | `1.5%` | `22.5%` | `76.0%` |
| `96..112` | `2184` | `1.3%` | `22.4%` | `76.3%` |

第一版建议把 `T` 与 `k` 近似独立采样，但格子子集选择保留按 region / cell 的经验权重；若要进一步简化，可先用 `sample_without_replacement(candidates, k)`，但这会丢掉已观测到的格子级偏置。

## 单格金额

本节同样使用 full high batch 内的正增量单格统计。

high batch 内 `static_map=2` 正增量单格金额主体落在 `16..28`，但存在少量更高 outlier。全图单格金额均值 `21.85`。

全图主要金额频率：

| amount | freq |
| ---: | ---: |
| 16 | `6.02%` |
| 17 | `8.19%` |
| 18 | `8.14%` |
| 19 | `9.81%` |
| 20 | `12.38%` |
| 21 | `12.20%` |
| 22 | `10.59%` |
| 23 | `4.89%` |
| 24 | `4.93%` |
| 25 | `4.71%` |
| 26 | `4.67%` |
| 27 | `4.54%` |
| 28 | `2.86%` |
| `29..37` | `6.35%` |
| `>=38` | `0.66%` |

不建议把 high batch 简单建成“独立采样若干单格金额再相加”。更稳定的结构是先采样 batch total 或 positive cell count，再分配到 region 内 `static_map=2` 格。

## Non-high static2 正事件

按 `observed` 口径排除 high 后，仍有正增量的 region-round：

| 数据 | region-rounds | events | delta_sum | delta >= 40 |
| --- | ---: | ---: | ---: | ---: |
| 地图1 | `1737` | `1829` | `9712` | `0` |
| 地图2 | `1676` | `1757` | `9351` | `1` |
| 地图3 pooled | `3482` | `3880` | `22926` | `100` |

`max_static2_delta >= 20` 捕获了地图3大量 partial high batch，使地图3 non-high 残差从 `4481` 个 region-round 降到 `3482`，`delta_sum` 从 `65762` 降到 `22926`。剩余残差仍不应建成强独立生成机制。

## 与外围普通格的耦合

high batch 与外围普通格 `static_map=0` 的关系在地图1/2/3一致：

- high region 内普通格事件近似为 `0`；地图3 observed 口径下有 `1` 个事件级例外。
- high region 外其它外围 region 会出现普通格小额事件。
- high region 外普通格事件数主要集中在 `3..5`。

详见 `mechanism/results/outer_static0_generation_analysis.md`。

## 建模建议

第一版 static2 生成器建议：

```text
state.next_static2_round initialized from empirical first-trigger distribution

for each round t:
    if t != state.next_static2_round:
        continue

    high_region = sample_outer_region()
    total = sample_empirical_static2_batch_total(map_id)
    k = sample_empirical_static2_positive_cell_count(map_id)
    cells = sample_static2_cells_by_region_weight(high_region, k)

    base = total // k
    rem = total % k
    for cell in cells:
        add base
    for cell in sample_without_replacement(cells, rem):
        add 1

    state.next_static2_round = t + sample_empirical_static2_gap(map_id)
```

简化版：

```text
gap ~ UniformInteger(8, 16)
region ~ near-uniform over outer regions
total ~ empirical_static2_batch_total
k ~ EmpiricalCategorical({3: 0.014, 4: 0.224, 5: 0.762})
selected_cells ~ sample_without_replacement(static2_cells[region], k)
split total nearly evenly over selected_cells
```

关键点：

1. 不要把 high batch 触发写成每回合独立 Bernoulli；相邻 high batch 存在 `8` 回合最小间隔，主体为 `8..16`。
2. 同一回合同一时间只触发一个外围 region。
3. high region 内 static2 候选格多数全覆盖，少数覆盖 4 格，极少覆盖 3 格；子集选择存在格子级偏置。
4. static2 金额不是单格独立采样，而是 batch total 在选中格上近似均分。
5. high batch 所在 region 的普通格近似不生成，小额普通格生成发生在其它外围 region。
6. 地图3 batch total 较低和 non-high 中等片段较多，应优先解释为地图结构/可见覆盖差异，后续可按地图单独校准参数。
