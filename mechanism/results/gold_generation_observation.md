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
- 外围 `static_map=2` 高额 batch：`mechanism/results/static2_generation_analysis.md`

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

外围 `region != 1 && static_map=2` 存在高额 batch 生成。当前使用 observed / full 两个观测口径：

| 数据 | game-rounds | observed high | observed rate | full high | partial-only | full batch total mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 地图1 | `49400` | `4067` | `0.08233` | `4037` | `30` | `88.1719` |
| 地图2 | `49400` | `4127` | `0.08354` | `4122` | `5` | `88.3503` |
| 地图3 pooled | `98800` | `8166` | `0.08265` | `7167` | `999` | `80.8122` |

结论：

- `observed high` 判据为 `static2_delta_sum >= 50 or static2_max_delta >= 20`，用于触发时序和耦合；`full high` 判据为 `static2_delta_sum >= 50`，用于金额分布估计。
- 三张图均支持“同一 game-round 至多一个外围 region 触发 high batch”；地图3 observed 口径有 `4` 个同轮多 region 例外，数量很小。
- high batch 不是每回合独立 Bernoulli；相邻 high batch 存在 `8` 回合最小间隔，主体间隔为 `8..16`。
- 地图3 observed high rate 与地图1/2接近；原 full 口径偏低主要由 `static_map=2` 漏观察造成。
- 第一版生成器应把 high batch 作为带间隔状态的外围主事件，再采样区域、batch total 与金额分配。

## 外围普通格小额生成

外围 `region != 1 && static_map=0` 普通格的小额生成不是由中心距离主导的独立背景，而是 `static2 high batch` 的互补区域伴随项。

跨图条件分层：

| 数据 | 条件 | events | exposure event rate | amount/event |
| --- | --- | ---: | ---: | ---: |
| 地图1 | high 所在 region | `0` | `0.000000` |  |
| 地图1 | high 之外 region | `15480` | `0.034576` | `5.5037` |
| 地图1 | 未观测到 high | `0` | `0.000000` |  |
| 地图2 | high 所在 region | `0` | `0.000000` |  |
| 地图2 | high 之外 region | `16151` | `0.033212` | `5.3118` |
| 地图2 | 未观测到 high | `6` | `0.000001` | `6.0000` |
| 地图3 pooled | high 所在 region | `1` | `0.000003` | `1.0000` |
| 地图3 pooled | high 之外 region | `30331` | `0.035130` | `5.4116` |
| 地图3 pooled | 未观测到 high | `363` | `0.000028` | `5.3967` |

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

地图1/2 的 `no_high` 背景几乎为 0；地图3 `no_high` 在新 observed 口径下也降到很弱。第一版不建议把它建成强背景率。

## 建模建议

第一版金币生成器可按以下结构实现：

```text
for each round t:
    # center small gold
    for cell in center_9x9_non_obstacle:
        p = A * exp(-B * squared_distance(cell, (8, 8)))
        if Bernoulli(p):
            add UniformInteger(1, 11)

    # outer high batch, stateful gap trigger
    if t == state.next_static2_round:
        high_region = sample_outer_region()
        total = sample_static2_batch_total()
        k_static2 = sample_static2_positive_cell_count()
        static2_cells = sample_static2_cells_by_region_weight(
            high_region,
            k_static2,
        )
        base = total // k_static2
        rem = total % k_static2
        for cell in static2_cells:
            add base
        for cell in sample_without_replacement(static2_cells, rem):
            add 1

        # coupled outer ordinary gold
        k = sample_outer_static0_event_count_given_high()
        cells = sample_without_replacement(
            outer_static0_cells excluding high_region,
            k,
        )
        for cell in cells:
            add sample_outer_static0_amount()

        state.next_static2_round = t + sample_static2_gap()

    # optional residual background
    # 可先关闭；若开启，应保持远弱于 high batch 伴随项。
```

参数建议：

- `B`：中心径向斜率第一版取 `0.065` 或沿用地图1/2 的 `0.06735`。
- `A`：按地图或目标校准；地图1/2 约 `0.0596`，地图3 pooled 约 `0.0564`。
- `static2_gap`：优先使用 observed high 的经验 categorical；三图主体为 `8..16`。粗仿真可先用 `UniformInteger(8, 16)`。
- `static2_batch_total`：优先使用 full high 的经验 categorical；地图1/2 mean 约 `88.26`，地图3 full 口径 mean 约 `80.81`。
- `static2_positive_cell_count`：完整可见地图1/2中约 `k=3: 1.4%`、`k=4: 22.4%`、`k=5: 76.2%`；选中子集存在格子级偏置，不能完全由中心/边缘距离解释。
- `outer_static0_amount`：使用经验 categorical，均值约 `5.4`。

后续如果要进一步提升精度，应优先补更完整的同局外围覆盖，而不是把当前单布局漏观测解释成机制变化。
