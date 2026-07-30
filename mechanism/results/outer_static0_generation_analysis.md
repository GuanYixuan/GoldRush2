# 外围普通格金币生成分析

本文分析中心区外 `static_map=0` 普通格的小额金币生成。结论基于地图1、地图2、地图3 M3-A/M3-B 的双视角合并 replay。

## 数据与口径

输入特征表：

```text
mechanism/data/processed/gold_features/<run_id>/cell_transitions.csv
```

筛选条件：

- `region != 1`
- `static_map = 0`
- 排除 `window_begin = 0`
- 使用干净可见 cell transition

建模单位：

```text
end[t-1] -> start[t]
```

本文把 `region != 1 && static_map=2` 的单轮单区域

```text
static2_delta_sum >= 50 or static2_max_delta >= 20
```

称为 observed `static2 high batch`。其中 `static2_delta_sum >= 50` 另称为 full high batch，主要用于金额分布估计。

## 总体观察

只看总体 event rate，外围普通格像是低概率小额生成；但按 `static2 high batch` 分层后，地图1/2/3共同显示它不是独立背景主导，而是 high batch 的互补区域伴随项。

## 触发条件

| 数据 | game-rounds | observed high | observed high rate | full high | partial-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 地图1 | `49400` | `4067` | `0.08233` | `4037` | `30` |
| 地图2 | `49400` | `4127` | `0.08354` | `4122` | `5` |
| 地图3 pooled | `98800` | `8166` | `0.08265` | `7167` | `999` |

新 observed 口径下地图3 high rate 与地图1/2接近，说明原 `sum>=50` 口径低估了地图3触发次数。

按普通格所在区域相对 high batch 的关系分层：

| 数据 | 条件 | region-rounds | positive region-rounds | events | exposure event rate | amount/event |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 地图1 | high 所在 region | `4067` | `0` | `0` | `0.000000` |  |
| 地图1 | high 之外 region | `12201` | `9287` | `15480` | `0.034576` | `5.5037` |
| 地图1 | 未观测到 high | `181332` | `0` | `0` | `0.000000` |  |
| 地图2 | high 所在 region | `4127` | `0` | `0` | `0.000000` |  |
| 地图2 | high 之外 region | `12381` | `9629` | `16151` | `0.033212` | `5.3118` |
| 地图2 | 未观测到 high | `181092` | `4` | `6` | `0.000001` | `6.0000` |
| 地图3 pooled | high 所在 region | `8166` | `1` | `1` | `0.000003` | `1.0000` |
| 地图3 pooled | high 之外 region | `24482` | `18129` | `30331` | `0.035130` | `5.4116` |
| 地图3 pooled | 未观测到 high | `362552` | `232` | `363` | `0.000028` | `5.3967` |

结论：

- high batch 所在 region 的普通格事件在地图1/2为 `0`，地图3 pooled 仅 `1` 个事件，可视为近似为 `0`。
- high batch 之外的其它三个外围 region 出现小额普通格事件，event rate 约 `0.033..0.035`。
- 地图1/2 的 `no_high` 几乎为 0；地图3 `no_high` 在新口径下也降到很弱，仍不应作为强背景率建模。

## High Batch 后的事件数分布

统计 high batch game-round 中，high region 外普通格总事件数：

| events | 地图1 freq | 地图2 freq | 地图3 pooled freq |
| ---: | ---: | ---: | ---: |
| 0 | `0.12%` | `0.05%` | `0.25%` |
| 1 | `1.70%` | `1.21%` | `2.65%` |
| 2 | `10.82%` | `9.98%` | `13.46%` |
| 3 | `29.58%` | `28.32%` | `29.79%` |
| 4 | `32.38%` | `31.04%` | `28.75%` |
| 5 | `16.57%` | `19.24%` | `15.54%` |
| 6 | `6.22%` | `6.91%` | `7.44%` |
| 7 | `2.14%` | `2.66%` | `1.85%` |
| `>=8` | `0.47%` | `0.58%` | `0.28%` |

三张图都集中在每个 high round 约 `3..5` 个事件，不符合“所有外围普通格每回合独立低概率触发”的宽分散形态。

## 金额分布

外围普通格触发后金额主要落在 `1..11`，但偏向小金额，不建议用均匀 `1..11`。

地图1/2 combined 金额均值 `5.4058`；地图3 pooled 金额均值 `5.4113`。地图3 pooled 频率：

| amount | freq |
| ---: | ---: |
| 1 | `12.35%` |
| 2 | `11.38%` |
| 3 | `11.41%` |
| 4 | `9.67%` |
| 5 | `9.14%` |
| 6 | `9.02%` |
| 7 | `7.40%` |
| 8 | `8.64%` |
| 9 | `7.07%` |
| 10 | `6.60%` |
| 11 | `6.67%` |
| `12..19` | `0.65%` |

## 建模建议

外围普通格第一版模型应依附于 `static2 high batch`：

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
    # 地图1/2 可近似忽略；地图3单布局残差不应用来直接校准强背景率。
```

其中：

- `sample_empirical_outer_static0_event_count_given_high()` 使用跨图 high 后事件数经验分布，主体为 `3..5`。
- `sample_empirical_outer_static0_amount()` 使用外围普通格金额经验 categorical。
- 不应把外围普通格建成由中心距离主导的独立背景生成。

该结论由地图1/2/3共同支持。
