# 金币生成规律初步观察

本文记录基于双视角合并 replay 的金币生成规律初步验证。数据均来自本地离线产物，未访问评测平台。

## 数据范围

已抽取特征的批次：

| 地图 | run_id | 对局数 | clean cell transitions | positive events |
| --- | --- | --- | ---: | ---: |
| 地图1 | `symobs-a-map1-100-20260727-231446` | 100 | `11243402` | `118761` |
| 地图2 | `symobs-a-map2-100-20260727-231704` | 100 | `11941033` | `130079` |

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

金额分布频率表：

```text
mechanism/data/processed/gold_features/amount_distributions/
```

包含：

- `cell_event_amount_distribution.csv`
- `static2_batch_total_distribution.csv`
- `static2_batch_event_count_distribution.csv`
- `static2_batch_cell_amount_distribution.csv`

## Snapshot 对齐

金币生成发生在回合行动前，cell 级观测使用：

```text
end[t-1] -> start[t]
```

实测表明，`snapshot.window=[5k, 5k+4]` 与如下 transition 聚合最吻合：

```text
t = 5k+1 .. 5k+5
window_begin = floor((t - 1) / 5) * 5
```

在该对齐下，排除 `window_begin=0` 后，地图1与地图2中 region 2/3/4 的 `window_observed.delta_sum` 与 `snapshot.gold_generated` 几乎完全一致；region 5 差异主要来自布局 A 的右侧漏格。

## 中心区域

region 1 的生成呈现中心加权空间分布，而不是均匀分布。

金额分布：

- 地图1：`amount/event = 5.984`
- 地图2：`amount/event = 6.016`

两图中中心区单次新增金额都近似均匀落在 `1..11`。

两图合并后的中心区单格事件金额频率：

| amount | freq |
| ---: | ---: |
| 1 | `9.10%` |
| 2 | `9.19%` |
| 3 | `8.97%` |
| 4 | `9.14%` |
| 5 | `9.03%` |
| 6 | `8.94%` |
| 7 | `9.14%` |
| 8 | `9.22%` |
| 9 | `9.15%` |
| 10 | `9.12%` |
| 11 | `9.00%` |

高频格集中在中心附近。

地图1 top cells：

```text
(8,9), (8,7), (9,8), (7,8), (8,8)
```

地图2 top cells：

```text
(8,8), (8,7), (8,9), (7,8), (9,8)
```

`snapshot.window=[0,4]` 的中心区生成量明显偏高，可能混入初始地图金币，应作为初始状态单独处理，不直接纳入稳态生成模型。

## 外围普通格

外围 `static_map=0` 普通格存在低概率小额生成。

| 地图 | event rate | amount/event |
| --- | ---: | ---: |
| 地图1 | `0.00212` | `5.504` |
| 地图2 | `0.00207` | `5.312` |

金额分布同样主要落在 `1..11`。

两图合并后的外围普通格单格事件金额频率：

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
| `>=12` | `0.57%` |

### 距离效应检查

已对外围 `region != 1 && static_map=0` 普通格按到中心 `(8,8)` 的 Chebyshev / Manhattan 距离分桶，并用 cell 聚合的 Poisson offset 模型检查 `events ~ exposure * exp(a + b * distance)`。

结论：当前不支持“外围普通格生成主要由到中心距离决定”。距离效应即使存在，也明显弱于 `static_map=2` 机制。

Chebyshev 距离分桶 event rate：

| 地图 | d=5 | d=6 | d=7 | d=8 | slope rate ratio / distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| 地图1 | `0.00191` | `0.00213` | `0.00216` | `0.00223` | `1.0457` |
| 地图2 | `0.00194` | `0.00219` | `0.00209` | `0.00207` | `1.0130` |
| 合并 | `0.00193` | `0.00216` | `0.00212` | `0.00214` | `1.0278` |

Manhattan 距离检查同样不稳定：地图1 斜率对应每距离单位 rate ratio `1.0165`，地图2 为 `1.0012`，合并后为 `1.0084`。

因此，建模优先级应为：

1. 先区分 `static_map=0` 与 `static_map=2`。
2. 对 `static_map=0` 外围普通格先使用区域/地图级背景率。
3. 只有在更多布局补齐可见性后，再考虑加入弱距离项。

## 外围静态 2 格

外围 `static_map=2` 格呈现高额 batch 生成特征。

| 地图 | event rate | amount/event |
| --- | ---: | ---: |
| 地图1 | `0.01981` | `20.651` |
| 地图2 | `0.02146` | `19.363` |

两图合并后的外围 `static_map=2` 单格事件金额分布是混合分布：少量小额事件落在 `1..11`，主体落在 `16..28`。

| amount range | freq |
| --- | ---: |
| `1..11` 合计 | `8.61%` |
| 16 | `6.99%` |
| 17 | `7.27%` |
| 18 | `7.37%` |
| 19 | `9.37%` |
| 20 | `12.53%` |
| 21 | `10.64%` |
| 22 | `8.86%` |
| 23 | `4.76%` |
| 24 | `4.48%` |
| 25 | `4.18%` |
| 26 | `3.29%` |
| 27 | `3.27%` |
| 28 | `2.28%` |

以 `static2` 单轮单区域 `delta_sum >= 50` 作为高额 batch 的保守阈值：

| 地图 | batch 总数 | 每局均值 | 同轮多区域 batch |
| --- | ---: | ---: | ---: |
| 地图1 | `4037` | `40.37` | `0` |
| 地图2 | `4122` | `41.22` | `0` |

高额 batch 在每个 game-round 中只观察到一个外围区域触发。

两图合并后，高额 batch 总量共 `8159` 次，均值 `88.262`，范围 `50..112`。最高频 batch total：

| total | count |
| ---: | ---: |
| 80 | `427` |
| 92 | `386` |
| 81 | `338` |
| 96 | `321` |
| 100 | `297` |
| 84 | `284` |
| 104 | `220` |
| 88 | `217` |
| 98 | `210` |
| 83 | `205` |

batch 中可见 `static_map=2` 格数量分布：

| cells | count |
| ---: | ---: |
| 1 | `14` |
| 2 | `423` |
| 3 | `1620` |
| 4 | `2781` |
| 5 | `3321` |

区域计数：

| 地图 | region 2 | region 3 | region 4 | region 5 |
| --- | ---: | ---: | ---: | ---: |
| 地图1 | `907` | `854` | `1100` | `1176` |
| 地图2 | `964` | `844` | `1005` | `1309` |

region 5 受布局 A 漏格影响，不宜直接和其它区域比较真实生成强度。

## 建模建议

第一版金币生成模型建议按“何时触发”和“触发后分布”拆成三层。建模单位统一使用：

```text
end[t-1] -> start[t]
```

建模时应先过滤：

- `snapshot.window=[0,4]`
- `window_observed.transition_rounds < 5`
- 任何含不可见、障碍、炸弹或负标记的 cell transition

### 中心区小额生成

触发时机：

- 每回合尝试生成。
- 当前数据基本支持“中心区各格按各自 rate 近似独立触发”，而不是某些回合集中触发很多格、长期平均后伪装成逐格 rate。
- 排除 `(4,11)` 和初始窗口后，两图合并每个 game-round 的中心区事件数均值为 `1.8094`。
- 观测到的每回合事件数分布与独立 Bernoulli 预测高度一致；详细对照见 `mechanism/results/center_distance_model_comparison.md`。
- 两图合并后的每回合事件数经验分布：

| events | freq |
| ---: | ---: |
| 0 | `16.02%` |
| 1 | `29.56%` |
| 2 | `27.30%` |
| 3 | `16.35%` |
| 4 | `7.30%` |
| 5 | `2.57%` |
| `>=6` | `0.91%` |

位置分布：

- 不建议用中心 `9x9` 均匀分布，也不必优先使用逐格经验权重。
- 第一版建议使用平方欧氏距离径向触发概率：

```text
center_rate(row, col) = A * exp(-0.06735 * ((row - 8)^2 + (col - 8)^2))
```

- `center_rate(row, col)` 表示该中心区非障碍格在一次 `end[t-1] -> start[t]` 回合转换中触发一次小额金币生成事件的概率。
- 当前两组数据的中心点基准率 `A` 约为 `0.0596`；后续可用更多布局继续校准。
- `(4,11)` 已确认为观测/站位干扰，拟合和参数校准时先排除该格，不让其污染径向规律。

触发后金额：

- 两图合并均值为 `6.0008`。
- 单格金额几乎均匀落在 `1..11`。
- 第一版可建模为：

```text
amount ~ UniformInteger(1, 11)
```

### 外围普通格小额生成

触发时机：

- 只考虑 `region != 1 && static_map=0`。
- 低概率，不由到中心距离主导。
- 两图合并后，每个 game-round 的外围普通格事件数均值为 `0.3170`。
- 每回合事件数经验分布：

| events | freq |
| ---: | ---: |
| 0 | `91.79%` |
| 1 | `0.12%` |
| 2 | `0.85%` |
| 3 | `2.38%` |
| 4 | `2.60%` |
| 5 | `1.47%` |
| `>=6` | `0.79%` |

事件数 `3..5` 比 `1` 更常见，说明外围普通格小额生成可能和“外围区域刷新”同步，而不是完全独立背景噪声。第一版可以先简化为：

```text
outer static0 background, per-cell low probability
```

更好的后续结构是：

```text
若某外围 region 本回合触发刷新，则该 region 的若干 static_map=0 格也生成小额金币。
```

触发后金额：

- 两图合并均值为 `5.4058`。
- 金额主要落在 `1..11`，且偏向小金额。
- 第一版建议直接使用 `cell_event_amount_distribution.csv` 中 `combined/outer_static0_cell` 的经验 categorical 分布；若需要简化，可用偏小的 `1..11` 离散分布。

### 外围静态 2 高额 batch

触发时机：

- 只考虑 `region != 1 && static_map=2`。
- 以单轮单区域 `static2 delta_sum >= 50` 作为高额 batch 的保守定义。
- 两图合并后共有 `8159` 次高额 batch。
- 总 game-round 数为 `99800`，触发率约为：

```text
p_static2_batch = 8159 / 99800 = 0.0818
```

- 两图中未观察到同一 game-round 多个外围区域同时触发高额 batch。

区域选择：

```text
observed counts:
region 2: 1871
region 3: 1698
region 4: 2105
region 5: 2485
```

region 5 受布局 A 漏格影响，不宜直接解释为真实生成强度更高。第一版生成器中可先使用：

```text
region ~ Categorical(region 2..5), 暂设近似均匀
```

或只用 region 2/3/4 估计区域权重，等 `C(A)` 补齐后再定 region 5。

触发后总量：

- batch total 均值为 `88.262`。
- 范围为 `50..112`。
- 第一版建议直接使用 `static2_batch_total_distribution.csv` 中 `combined` 的经验 categorical 分布：

```text
batch_total ~ EmpiricalStatic2BatchTotal
```

触发后位置与分配：

- 当前观测到的 batch 中可见 `static_map=2` 格数量分布受可见性和 region 5 漏格影响，不能直接当作真实参与格数量。
- 更稳的第一版生成方式：

```text
1. 以 p_static2_batch ~= 0.0818 判定本回合是否触发高额 batch。
2. 若触发，选择一个外围 region。
3. 从 empirical batch_total 分布采样总量。
4. 将总量分配到该 region 的 static_map=2 点。
```

若需要临时做单格金额近似，可使用 `static2_batch_cell_amount_distribution.csv` 中 `combined` 的 batch 内单格金额分布；其主体落在 `16..28`，均值为 `21.5292`。但不建议直接独立采样多个单格金额后相加，因为 batch total 的离散形态更稳定。

### 结构化生成器草案

```text
for each round t:
    # center small gold
    for cell in center_9x9_non_obstacle:
        p = A * exp(-0.06735 * squared_distance(cell, (8, 8)))
        if Bernoulli(p):
            add UniformInteger(1, 11)

    # outer high batch
    if Bernoulli(0.0818):
        region = sample_outer_region()
        total = sample_empirical_static2_batch_total()
        allocate total over static_map=2 cells in region

        # likely coupled small gold
        generate small static_map=0 events in same region
        amount ~ empirical_outer_static0_amount

    # optional weak background
    for outer static_map=0 cells not handled above:
        trigger with very low background probability
```

后续需要用 `C(A)` 布局或其它覆盖补足 region 5 漏格，再判断外围区域之间是否真正同分布。
