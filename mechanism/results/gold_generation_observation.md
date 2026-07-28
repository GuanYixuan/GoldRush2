# 金币生成规律观察

本文记录基于双视角合并 replay 的金币生成规律。结论由地图1、地图2、地图3 M3-A/M3-B 共同推出，并统一用于后续生成建模。

## 数据范围

已抽取金币特征的批次：

| 地图 | run_id | 对局数 | clean cell transitions | positive events |
| --- | --- | ---: | ---: | ---: |
| 地图1 | `symobs-a-map1-100-20260727-231446` | `100` | `11243402` | `118761` |
| 地图2 | `symobs-a-map2-100-20260727-231704` | `100` | `11941033` | `130079` |
| 地图3 M3-A | `symobs-m3a-map3-100-20260728-200953` | `100` | `8989792` | `76358` |
| 地图3 M3-B | `symobs-m3b-map3-100-20260728-202647` | `100` | `8985776` | `77530` |

特征表位置：

```text
mechanism/data/processed/gold_features/<run_id>/
```

主要文件：

- `cell_transitions.csv`
- `snapshot_windows.csv`
- `window_observed.csv`
- `cell_summary.csv`
- `summary.json`

专题分析：

- 中心区距离模型与独立触发：`mechanism/results/center_distance_model_comparison.md`
- 外围普通格生成：`mechanism/results/outer_static0_generation_analysis.md`

## Snapshot 对齐

金币生成发生在回合行动前，cell 级观测统一使用：

```text
end[t-1] -> start[t]
```

实测对齐：

```text
snapshot.window=[5k, 5k+4]
t = 5k+1 .. 5k+5
window_begin = floor((t - 1) / 5) * 5
```

地图1/2 中该口径与官方 snapshot 在充分覆盖的区域基本一致。地图3 M3-A/M3-B 中相位仍一致，但单局只使用一种布局，region 3/5 和中心 `(8,8)` 存在漏观测，不能直接用单布局 `window_observed.delta_sum` 校准全区域总量。

## 中心区小额生成

中心区 `region=1 && static_map=0` 的生成规律：

1. 每个中心非障碍格每回合按自身触发概率近似独立触发。
2. 触发概率由到中心 `(8,8)` 的平方欧氏距离主导。
3. 触发后金额近似 `UniformInteger(1, 11)`。

推荐模型：

```text
center_rate(row, col) = A * exp(-B * ((row - 8)^2 + (col - 8)^2))

if Bernoulli(center_rate(row, col)):
    add UniformInteger(1, 11)
```

跨图参数：

| 数据 | A | B | amount/event | 独立触发 TVD |
| --- | ---: | ---: | ---: | ---: |
| 地图1 | `0.05962` | `0.06745` | `5.984` | `0.00190` |
| 地图2 | `0.05956` | `0.06726` | `6.016` | `0.00505` |
| 地图3 pooled | `0.05641` | `0.06260` | `6.0020` | `0.002..0.004` |

第一版可固定 `B ~= 0.065`，按地图或更多数据校准 `A`。若需要沿用地图1/2参数，`B=0.06735` 仍是可接受近似。

注意：

- 地图1/2 中 `(4,11)` 是站位/观测干扰格，拟合时应排除。
- 地图3中心普通格少，且 `(8,8)` 未直接覆盖，事件均值低于地图1/2不是机制反例。

## 外围 `static_map=2` 高额 Batch

外围 `region != 1 && static_map=2` 存在高额 batch 生成。以单轮单区域 `static2 delta_sum >= 50` 作为保守定义：

| 数据 | game-rounds | high batch | high rate | 同轮多 high region | batch total mean | range |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 地图1 | `49400` | `4037` | `0.08172` | `0` | `88.1719` | `50..112` |
| 地图2 | `49400` | `4122` | `0.08344` | `0` | `88.3503` | `53..112` |
| 地图3 pooled | `98800` | `7167` | `0.07254` | `0` | `80.8122` | `50..112` |

结论：

- 三张图均支持“同一 game-round 至多一个外围 region 触发 high batch”。
- 地图3可见 high rate 和 batch mean 低于地图1/2，可能由地图结构和单局 `static_map=2` 可见覆盖共同造成；核心结构没有变化。
- 第一版生成器可把 high batch 作为外围生成主事件，再采样区域与 batch total。

## 外围普通格小额生成

外围 `region != 1 && static_map=0` 普通格的小额生成不是由中心距离主导的独立背景，而是 `static2 high batch` 的互补区域伴随项。

跨图条件分层：

| 数据 | 条件 | events | exposure event rate | amount/event |
| --- | --- | ---: | ---: | ---: |
| 地图1 | high 所在 region | `0` | `0.000000` |  |
| 地图1 | high 之外 region | `15359` | `0.034571` | `5.5049` |
| 地图1 | 未观测到 high | `121` | `0.000018` | `5.3471` |
| 地图2 | high 所在 region | `0` | `0.000000` |  |
| 地图2 | high 之外 region | `16127` | `0.033203` | `5.3132` |
| 地图2 | 未观测到 high | `30` | `0.000004` | `4.7000` |
| 地图3 pooled | high 所在 region | `0` | `0.000000` |  |
| 地图3 pooled | high 之外 region | `26838` | `0.035345` | `5.4036` |
| 地图3 pooled | 未观测到 high | `3857` | `0.000299` | `5.4651` |

触发后金额主要落在 `1..11`，但偏向小金额，不建议用均匀 `1..11`。第一版应使用外围普通格经验 categorical 分布。

推荐结构：

```text
if static2_high_batch_triggered:
    high_region = sampled_high_region
    k = sample_outer_static0_event_count_given_high()
    candidate_cells = outer_static0_cells excluding high_region
    selected_cells = sample_without_replacement(candidate_cells, k)
    for cell in selected_cells:
        add sample_outer_static0_amount()
```

地图1/2 的 `no_high` 背景几乎为 0；地图3 `no_high` 稍高，但仍比 `other_high` 弱两个数量级，且可能混入未完整观测到的 high batch。第一版不建议把它建成强背景率。

## 建模建议

第一版金币生成器可按以下结构实现：

```text
for each round t:
    # center small gold
    for cell in center_9x9_non_obstacle:
        p = A * exp(-B * squared_distance(cell, (8, 8)))
        if Bernoulli(p):
            add UniformInteger(1, 11)

    # outer high batch
    if Bernoulli(p_static2_high_batch):
        high_region = sample_outer_region()
        total = sample_static2_batch_total()
        allocate total over static_map=2 cells in high_region

        # coupled outer ordinary gold
        k = sample_outer_static0_event_count_given_high()
        cells = sample_without_replacement(
            outer_static0_cells excluding high_region,
            k,
        )
        for cell in cells:
            add sample_outer_static0_amount()

    # optional residual background
    # 可先关闭；若开启，应保持远弱于 high batch 伴随项。
```

参数建议：

- `B`：中心径向斜率第一版取 `0.065` 或沿用地图1/2 的 `0.06735`。
- `A`：按地图或目标校准；地图1/2 约 `0.0596`，地图3 pooled 约 `0.0564`。
- `p_static2_high_batch`：地图1/2 约 `0.0826`，地图3可见约 `0.0725`。
- `static2_batch_total`：优先使用经验 categorical；地图1/2 mean 约 `88.26`，地图3可见 mean 约 `80.81`。
- `outer_static0_amount`：使用经验 categorical，均值约 `5.4`。

后续如果要进一步提升精度，应优先补更完整的同局外围覆盖，而不是把当前单布局漏观测解释成机制变化。
