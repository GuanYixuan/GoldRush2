# 外围普通格金币生成分析

本文单独分析中心区外 `static_map=0` 普通格的小额金币生成。目标是判断其触发条件、位置分布和金额分布，并给出第一版生成模型建议。

## 数据与口径

数据来自两组已合并 replay 的 `cell_transitions.csv`：

- 地图1：`symobs-a-map1-100-20260727-231446`
- 地图2：`symobs-a-map2-100-20260727-231704`

筛选条件：

- `region != 1`
- `static_map = 0`
- 排除 `window_begin = 0` 的初始窗口
- 只使用已进入 `cell_transitions.csv` 的干净可见 cell transition

建模单位仍是：

```text
end[t-1] -> start[t]
```

本文把 `region != 1 && static_map=2` 的单轮单区域 `delta_sum >= 50` 称为 `static2 high batch`。

## 总体频率

| 数据 | region-rounds | exposure | events | event rate/exposure | amount/event |
| --- | ---: | ---: | ---: | ---: | ---: |
| 地图1 | `197600` | `7238001` | `15480` | `0.002139` | `5.5037` |
| 地图2 | `197600` | `7730448` | `16157` | `0.002090` | `5.3121` |
| 合并 | `395200` | `14968449` | `31637` | `0.002114` | `5.4058` |

如果只看总体 event rate，外围普通格像是低概率背景生成；但按 `static2 high batch` 分层后，触发条件非常不一样。

## 触发条件

合并数据中共有 `98800` 个 game-round，其中：

- `static2 high batch` game-round：`8159`
- 无 high batch game-round：`90641`
- 同一 game-round 多个外围 region 同时 high batch：`0`

high batch 触发区域计数：

| high region | count |
| ---: | ---: |
| 2 | `1871` |
| 3 | `1698` |
| 4 | `2105` |
| 5 | `2485` |

按同一 region-round 条件分层：

| 条件 | region-rounds | positive region-rounds | events | exposure event rate | amount/event |
| --- | ---: | ---: | ---: | ---: | ---: |
| same region high batch | `8159` | `0` | `0` | `0.000000` |  |
| same region any static2 event | `11607` | `2485` | `3894` | `0.008889` | `5.0056` |
| no same region static2 event | `383593` | `16435` | `27743` | `0.001909` | `5.4620` |
| not high batch | `387041` | `18920` | `31637` | `0.002158` | `5.4058` |

这里 “same region high batch” 的普通格事件数为 `0`，说明外围普通格小额生成不是 high batch 区域内部的伴随生成。

按 game-round 看，关系更清楚：

| 条件 | region-rounds | positive region-rounds | events | exposure event rate | amount/event |
| --- | ---: | ---: | ---: | ---: | ---: |
| high batch 所在 region | `8159` | `0` | `0` | `0.000000` |  |
| high batch 之外的其它 region | `24477` | `18830` | `31486` | `0.033856` | `5.4067` |
| 无 high batch 的 region | `362564` | `90` | `151` | `0.000011` | `5.2185` |

结论：外围普通格生成几乎完全由 `static2 high batch` 触发，但发生在 high batch 区域之外的其它三个外围 region。无 high batch 时的背景生成非常弱，当前可先忽略或作为极低概率残差项处理。

## High Batch 后的事件数分布

在 `8159` 个 high batch game-round 中，外围普通格总事件数分布如下。这里总事件数只统计 high batch 区域之外的普通格，因为 high batch 所在 region 的普通格未观察到事件。

| events | count | freq |
| ---: | ---: | ---: |
| 0 | `7` | `0.09%` |
| 1 | `119` | `1.46%` |
| 2 | `851` | `10.43%` |
| 3 | `2365` | `28.99%` |
| 4 | `2584` | `31.67%` |
| 5 | `1456` | `17.85%` |
| 6 | `537` | `6.58%` |
| 7 | `197` | `2.41%` |
| 8 | `35` | `0.43%` |
| 9 | `7` | `0.09%` |
| 10 | `1` | `0.01%` |

统计量：

| metric | value |
| --- | ---: |
| mean events/high round | `3.8591` |
| mean delta/high round | `20.8648` |
| median delta/high round | `22` |
| p10 delta/high round | `14` |
| p90 delta/high round | `26` |
| delta range | `0..28` |

这说明外围普通格不是每个 cell 独立低概率背景生成。若在 high batch 之外的其它 region-round 上用整体 cell rate `0.033856` 做独立 Bernoulli，预测分布会明显更分散；观测分布集中在每个 high round 总共 `3..5` 个事件。第一版更稳妥的做法是先采样 high round 的普通格事件总数，再分配位置。

## 区域关系

下表按 high batch 区域和目标普通格区域统计。对角线为 high batch 所在 region，普通格事件均为 `0`。

| high region | target region | region-rounds | positive region-rounds | exposure | events | event rate | amount/event |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | `1871` | `0` | `74636` | `0` | `0.000000` |  |
| 2 | 3 | `1871` | `1366` | `72859` | `2221` | `0.030484` | `5.3597` |
| 2 | 4 | `1871` | `1509` | `74704` | `2549` | `0.034121` | `5.5002` |
| 2 | 5 | `1871` | `1418` | `61719` | `2240` | `0.036294` | `5.4438` |
| 3 | 2 | `1698` | `1356` | `67622` | `2321` | `0.034323` | `5.4601` |
| 3 | 3 | `1698` | `0` | `66087` | `0` | `0.000000` |  |
| 3 | 4 | `1698` | `1287` | `67745` | `2190` | `0.032327` | `5.4648` |
| 3 | 5 | `1698` | `1196` | `55780` | `1863` | `0.033399` | `5.5566` |
| 4 | 2 | `2105` | `1721` | `83722` | `2964` | `0.035403` | `5.3637` |
| 4 | 3 | `2105` | `1584` | `81841` | `2622` | `0.032038` | `5.3555` |
| 4 | 4 | `2105` | `0` | `83922` | `0` | `0.000000` |  |
| 4 | 5 | `2105` | `1543` | `68956` | `2486` | `0.036052` | `5.4035` |
| 5 | 2 | `2485` | `2028` | `98950` | `3471` | `0.035078` | `5.5483` |
| 5 | 3 | `2485` | `1841` | `97044` | `3093` | `0.031872` | `5.2334` |
| 5 | 4 | `2485` | `1981` | `99051` | `3466` | `0.034992` | `5.2819` |
| 5 | 5 | `2485` | `0` | `81879` | `0` | `0.000000` |  |

非对角区域的 event rate 大致在 `0.030..0.036`，差异不大。考虑到 region 5 仍受布局 A 漏格影响，第一版可以近似把普通格位置均匀分配到 high batch 区域之外的可生成普通格；后续再用更多覆盖数据校准区域权重或 cell 权重。

## 金额分布

外围普通格单格事件金额主要落在 `1..11`，但不是均匀分布，而是偏向小金额。合并数据：

| amount | freq |
| ---: | ---: |
| 1 | `12.39%` |
| 2 | `11.39%` |
| 3 | `10.41%` |
| 4 | `9.51%` |
| 5 | `10.28%` |
| 6 | `8.55%` |
| 7 | `9.14%` |
| 8 | `7.67%` |
| 9 | `7.41%` |
| 10 | `6.08%` |
| 11 | `6.60%` |
| `12..18` 合计 | `0.57%` |

建议第一版直接使用 `mechanism/data/processed/gold_features/amount_distributions/cell_event_amount_distribution.csv` 中 `combined/outer_static0_cell` 的经验 categorical 分布。若需要更简单的近似，可截断为偏小的 `1..11` 离散分布；不建议用均匀 `1..11`。

## 建模建议

外围普通格第一版模型建议依附于 `static2 high batch`：

```text
for each round t:
    high_region = sampled_static2_high_batch_region_or_none()

    if high_region is not None:
        k = sample_empirical_outer_static0_event_count_given_high()
        candidate_cells = outer_static0_cells excluding high_region
        selected_cells = sample_without_replacement(candidate_cells, k)
        for cell in selected_cells:
            add sample_empirical_outer_static0_amount()

    # optional residual background
    for cell in outer_static0_cells:
        if Bernoulli(0.000011):
            add sample_empirical_outer_static0_amount()
```

其中：

- `sample_empirical_outer_static0_event_count_given_high()` 使用本文件 “High Batch 后的事件数分布” 表。
- `sample_empirical_outer_static0_amount()` 使用 `combined/outer_static0_cell` 金额经验分布。
- 残差背景项极弱，且当前只观察到 `151` 个事件；第一版生成器可以先不实现，或保留为可关闭项。

这个模型比“所有外围普通格每回合独立低概率触发”更符合当前数据，因为后者无法解释 high batch 回合中普通格事件数集中在 `3..5`，也无法解释 high batch 所在 region 的普通格事件为 `0`。
