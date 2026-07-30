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

- 初始金币生成：`mechanism/results/gold/initial_gold_analysis.md`
- 中心区距离模型与独立触发：`mechanism/results/gold/center_distance_model_comparison.md`
- 外围普通格生成：`mechanism/results/gold/outer_static0_generation_analysis.md`
- 外围 `static_map=2` 高额 batch：`mechanism/results/gold/static2_generation_analysis.md`

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

注意：首个 snapshot 窗口 `window=[0,4]` 的 `gold_generated` 包含 `round=0.start` 已存在的初始金币，不能直接用于后续回合间生成率拟合。

## 初始金币生成

官方 3 局上帝视角 replay 显示，`round=0.start.grid` 初始金币只出现在中心区 `region=1 && static_map=0`：

| replay | center positive cells | center total | outer static0 total | outer static2 total |
| --- | ---: | ---: | ---: | ---: |
| `official_sdk/data/full/g0.txt` | `50` | `59` | `0` | `0` |
| `official_sdk/data/full/g1.txt` | `50` | `53` | `0` | `0` |
| `official_sdk/data/full/g2.txt` | `50` | `70` | `0` | `0` |

三局的 50 个初始正金币坐标集合完全一致，且每格至少 `+1`。首个 snapshot 窗口验证：

```text
snapshot[0,4].center.gold_generated
    = initial_center_gold
    + round1..5 center_generation
```

用 400 局双视角 snapshot 对比后，推荐初始模型为：

```text
initialize_gold():
    for cell in initial_center_base_cells_50:
        add 1

    run_center_small_gold_once()
```

其中 `run_center_small_gold_once()` 使用后续中心小额生成同一套 `center_A / center_B / UniformInteger(1,11)`。详见 `mechanism/results/gold/initial_gold_analysis.md`。

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

金币生成属于 `simulator/` 的机制近似层，不应写死在官方规则层中。第一版建议使用 `GoldGenerationConfig` 驱动：对齐 replay 时使用 `fit` 配置，RL 主训练时每局由 `EpisodeConfig` 采样一组参数，固定评估时使用若干 `eval_holdout` 配置。

推荐生成器结构：

```text
initialize_gold():
    for cell in initial_center_base_cells_50:
        add 1
    run_center_small_gold_once(config.center_A, config.center_B)

for each round t:
    # center small gold
    for cell in center_9x9_non_obstacle:
        p = config.center_A * exp(
            -config.center_B * squared_distance(cell, (8, 8))
        )
        if Bernoulli(p):
            add UniformInteger(1, 11)

    # outer high batch, stateful gap trigger
    if t == state.next_static2_round:
        high_region = sample(config.static2_region_dist)
        total = sample(config.static2_total_dist)
        k_static2 = sample(config.static2_positive_cell_count_dist)
        static2_cells = sample_static2_cells(
            high_region,
            k_static2,
            config.static2_cell_subset_mode,
            config.static2_cell_subset_weights,
        )
        base = total // k_static2
        rem = total % k_static2
        for cell in static2_cells:
            add base
        # remainder is uniformly assigned within selected static2 cells
        for cell in sample_without_replacement(static2_cells, rem):
            add 1

        # coupled outer ordinary gold
        k = sample(config.outer_static0_event_count_dist_given_high)
        cells = sample_without_replacement(
            outer_static0_cells excluding high_region,
            k,
        )
        for cell in cells:
            add sample(config.outer_static0_amount_dist)

        state.next_static2_round = t + sample(config.static2_gap_dist)
```

### 参数列表

| 参数 | 含义 | fit 默认建议 | RL 随机化建议 |
| --- | --- | --- | --- |
| `center_A` | 中心区总体触发强度 | 地图1/2 约 `0.0596`，地图3 pooled 约 `0.0564` | 每局轻微扰动，例如围绕 fit 值做 `0.9..1.1` scale |
| `center_B` | 中心区到中心平方距离衰减 | `0.065` 或地图1/2 `0.06735` | 小幅扰动，保持径向结构不变 |
| `static2_first_round_dist` | 每局第一次 high batch 的回合 | 用 observed high 的首触发经验分布 | 随机初始相位，避免策略记死固定开局节奏 |
| `static2_gap_dist` | 相邻 high batch 间隔 | observed high 经验 categorical；主体 `8..16` | 在经验分布、`UniformInteger(8,16)`、轻微长尾版本间采样 |
| `static2_region_dist` | high batch 触发的外围 region | 近似均匀，或按地图/布局经验权重 | 对 region 权重做 Dirichlet 扰动 |
| `static2_total_dist` | full high batch total | full high 经验 categorical；地图1/2 mean 约 `88.26`，地图3 full 约 `80.81` | 经验分布加轻微 scale/shift，但保持离散峰值 |
| `static2_positive_cell_count_dist` | high region 内被加钱的 static2 格数 | 完整可见地图1/2：`k=3:1.4%`、`k=4:22.4%`、`k=5:76.2%` | 在经验分布附近扰动；不建议改成逐格独立 Bernoulli |
| `static2_cell_subset_mode` | 给定 region 和 k 后如何选 static2 子集 | `empirical_cell_weight`；简化版可用 `uniform_without_replacement` | 训练时混合经验权重与均匀抽样，提升鲁棒性 |
| `static2_cell_subset_weights` | region / cell 级子集或遗漏权重 | 使用地图1/2完整可见样本估计；距离只作弱先验 | 每局对权重加平滑/扰动，避免过拟合单格偏置 |
| `outer_static0_event_count_dist_given_high` | high batch 后其它外围 region 普通格事件数 | 经验分布，主体 `3..5` | 对事件数分布小幅扰动 |
| `outer_static0_amount_dist` | 外围普通格触发后金额 | 经验 categorical，均值约 `5.4` | 在经验分布附近扰动；不建议用均匀 `1..11` 替代 |
| `outer_static0_residual_rate` | 未观测 high 时的弱背景生成率 | 默认为 `0` 或极弱；地图3 no-high 残差也很小 | 可采样极弱背景率，但应远弱于 high batch 伴随项 |

### 固定与随机化边界

建议固定的结构：

- 中心区按格近似独立触发。
- 中心区触发率由到 `(8,8)` 的平方欧氏距离主导。
- 中心区触发后金额固定为 `UniformInteger(1, 11)`。
- 初始金币先在固定 50 个中心格上各放 `1`，再额外执行一次中心小额生成 tick。
- static2 high batch 使用状态化 gap 触发，不使用逐回合 Bernoulli。
- static2 金额按 batch total 在选中格上近似均分。
- static2 余数分配固定为在选中格内均匀随机分配。
- 外围 static0 小额生成依附 static2 high batch。

建议随机化的数值：

- 中心区 `A/B`。
- static2 首触发、gap、region 权重、batch total、positive cell count。
- static2 子集选择权重。
- outer static0 事件数、金额分布和极弱 residual 背景。

RL 训练中 actor 不应看到 `GoldGenerationConfig` 的真实参数；这些属于 simulator privileged state。固定评估应包含 `fit` 配置和若干 holdout 随机化配置，分别衡量机制拟合表现和机制误差鲁棒性。
