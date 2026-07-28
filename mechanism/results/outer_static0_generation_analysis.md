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

本文把 `region != 1 && static_map=2` 的单轮单区域 `delta_sum >= 50` 称为 `static2 high batch`。

## 总体观察

只看总体 event rate，外围普通格像是低概率小额生成；但按 `static2 high batch` 分层后，地图1/2/3共同显示它不是独立背景主导，而是 high batch 的互补区域伴随项。

## 触发条件

| 数据 | game-rounds | high batch | high rate | multi-high rounds | high batch mean total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 地图1 | `49400` | `4037` | `0.08172` | `0` | `88.1719` |
| 地图2 | `49400` | `4122` | `0.08344` | `0` | `88.3503` |
| 地图3 pooled | `98800` | `7167` | `0.07254` | `0` | `80.8122` |

地图3 high batch 可见率和总量低于地图1/2，但 M3-A/M3-B 结果接近，说明该差异不只是单布局偶然噪声；仍可能受地图3静态结构和单局 `static_map=2` 可见覆盖影响。

按普通格所在区域相对 high batch 的关系分层：

| 数据 | 条件 | region-rounds | positive region-rounds | events | exposure event rate | amount/event |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 地图1 | high 所在 region | `4037` | `0` | `0` | `0.000000` |  |
| 地图1 | high 之外 region | `12111` | `9214` | `15359` | `0.034571` | `5.5049` |
| 地图1 | 未观测到 high | `181452` | `73` | `121` | `0.000018` | `5.3471` |
| 地图2 | high 所在 region | `4122` | `0` | `0` | `0.000000` |  |
| 地图2 | high 之外 region | `12366` | `9616` | `16127` | `0.033203` | `5.3132` |
| 地图2 | 未观测到 high | `181112` | `17` | `30` | `0.000004` | `4.7000` |
| 地图3 pooled | high 所在 region | `7167` | `0` | `0` | `0.000000` |  |
| 地图3 pooled | high 之外 region | `21501` | `15978` | `26838` | `0.035345` | `5.4036` |
| 地图3 pooled | 未观测到 high | `366532` | `2384` | `3857` | `0.000299` | `5.4651` |

结论：

- high batch 所在 region 的普通格事件在三张图上均为 `0`。
- high batch 之外的其它三个外围 region 出现小额普通格事件，event rate 约 `0.033..0.035`。
- 地图1/2 的 `no_high` 几乎为 0；地图3 `no_high` 稍高，但仍比 `other_high` 弱两个数量级，且更可能混入未完整观测到的 high batch。

## High Batch 后的事件数分布

统计 high batch game-round 中，high region 外普通格总事件数：

| events | 地图1 freq | 地图2 freq | 地图3 pooled freq |
| ---: | ---: | ---: | ---: |
| 0 | `0.12%` | `0.05%` | `0.25%` |
| 1 | `1.71%` | `1.21%` | `2.58%` |
| 2 | `10.87%` | `9.99%` | `13.02%` |
| 3 | `29.63%` | `28.36%` | `29.22%` |
| 4 | `32.33%` | `31.03%` | `29.08%` |
| 5 | `16.45%` | `19.21%` | `15.96%` |
| 6 | `6.27%` | `6.89%` | `7.67%` |
| 7 | `2.16%` | `2.67%` | `1.90%` |
| `>=8` | `0.47%` | `0.58%` | `0.32%` |

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
