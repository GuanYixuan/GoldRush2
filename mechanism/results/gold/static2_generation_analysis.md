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

## 首次触发时间

本文使用 `observed high` 口径统计每局第一次 static2 high batch。现有 `cell_transitions.csv` 不包含 `window=[0,4]` 的回合间 transition，但 400 局 snapshot 的首个窗口中外围 region 的 `gold_generated` 均为 `0`，官方 full replay 的 `window=[0,4]` 外围生成也为 `0`；因此当前数据支持首次 high batch 不早于 round `8`。

按地图统计：

| 数据 | games | min | max | mean | first round 分布 |
| --- | ---: | ---: | ---: | ---: | --- |
| 地图1 | `100` | `8` | `13` | `11.90` | `{8:10, 12:60, 13:30}` |
| 地图2 | `100` | `8` | `14` | `11.60` | `{8:30, 11:10, 13:30, 14:30}` |
| 地图3 pooled | `200` | `8` | `14` | `10.75` | `{8:44, 9:34, 10:50, 13:16, 14:56}` |
| 全部 | `400` | `8` | `14` | `11.25` | `{8:84, 9:34, 10:50, 11:10, 12:60, 13:76, 14:86}` |

地图3两批采集的首次触发分布差异明显：

| run_id | games | first round 分布 |
| --- | ---: | --- |
| `symobs-m3a-map3-100-20260728-200953` | `100` | `{9:13, 10:15, 13:16, 14:56}` |
| `symobs-m3b-map3-100-20260728-202647` | `100` | `{8:44, 9:21, 10:35}` |

实现建议：

- 不建议继续用 `0..16` 均匀 offset；该分布会产生大量当前未观测到的 `0..7` 首触发。
- 对齐 replay 的 `fit` 配置应使用按地图/布局的经验 categorical。
- 若暂时不区分地图/布局，粗略训练配置可用全局经验分布，或至少使用 `UniformInteger(8,14)` 而不是 `0..16`。
- 首次触发分布与后续 gap 分布应分开建模：首次触发支持目前为 `8..14`，后续 gap 主体为 `8..16`。

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

相邻 high batch 的 region transition 中，同 region 连续触发是允许的，且不是纯粹由 region 总体不均匀导致。下表的“独立基线”按实际 prev / next region 边际分布计算，即如果相邻 region 独立时预期的 repeat rate：

| 数据 | transitions | same-region transitions | repeat rate | 独立基线 | 差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 地图1 | `3967` | `1302` | `0.3282` | `0.2554` | `+0.0728` |
| 地图2 | `4027` | `1215` | `0.3017` | `0.2564` | `+0.0453` |
| 地图3 pooled | `7966` | `2265` | `0.2843` | `0.2502` | `+0.0341` |
| 地图1/2 combined | `7994` | `2517` | `0.3149` | `0.2556` | `+0.0593` |
| 全部 | `15960` | `4782` | `0.2996` | `0.2515` | `+0.0481` |

因此当前数据支持 region 选择存在一阶 persistence：上一轮 high batch 的 region 比独立模型下更容易再次出现。该效应在地图1/2更明显，地图3较弱但同向。

按 prev region 分层的转移率：

| 数据 | prev region | next 2 | next 3 | next 4 | next 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 地图1 | 2 | `0.257` | `0.218` | `0.225` | `0.300` |
| 地图1 | 3 | `0.259` | `0.255` | `0.261` | `0.225` |
| 地图1 | 4 | `0.207` | `0.130` | `0.397` | `0.267` |
| 地图1 | 5 | `0.181` | `0.237` | `0.211` | `0.370` |
| 地图2 | 2 | `0.297` | `0.214` | `0.230` | `0.259` |
| 地图2 | 3 | `0.269` | `0.274` | `0.179` | `0.278` |
| 地图2 | 4 | `0.175` | `0.159` | `0.296` | `0.369` |
| 地图2 | 5 | `0.219` | `0.203` | `0.250` | `0.327` |
| 地图3 pooled | 2 | `0.293` | `0.213` | `0.297` | `0.197` |
| 地图3 pooled | 3 | `0.272` | `0.291` | `0.204` | `0.234` |
| 地图3 pooled | 4 | `0.285` | `0.167` | `0.269` | `0.279` |
| 地图3 pooled | 5 | `0.198` | `0.285` | `0.234` | `0.284` |

地图1/2中 region `4/5` 的自重复更强；地图3转移更均匀。gap 分层看，`8..16` 各 gap 下 repeat rate 均在独立基线之上附近波动，没有只由某个固定 gap 贡献的迹象。

region 计数和 batch total mean 都受可见覆盖影响，不宜直接解释为真实 region 权重。第一版生成器可先用近似均匀 region，或按地图/布局使用经验权重；更贴近数据的版本应把 region 选择建成一阶 Markov：

```text
if previous_high_region exists:
    with p_repeat:
        high_region = previous_high_region
    otherwise:
        high_region = sample_other_or_empirical_region()
else:
    high_region = sample_first_region_dist()
```

粗略参数可先令 `p_repeat` 落在 `0.28..0.33`，地图1/2略高、地图3略低；如果追求回放拟合，则直接使用按地图的经验转移矩阵。

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

低 positive cell count 样本的 total mean 明显偏低，主要来自候选格未完整暴露或被动态对象/炸弹遮挡的观测口径；不能直接解释为官方先随机采样了较小的 `k`。

## 候选格占用与金额分配

占用回查由 `mechanism/scripts/analysis/analyze_static2_occupancy.py` 完成。该脚本把每个 observed high batch 回到 merged replay 中，统计 high region 的全部 `static_map=2` 候选格、当前回合 start 的玩家/NPC占用格、炸弹格、以及正增量格。

默认输出：

```text
mechanism/data/processed/static2_occupancy_analysis/maps123_static2_occupancy/
```

全量 1/2/3 observed high batch 中，正增量从未落在被占用候选格上：

| 口径 | batches | 有 blocked 候选 | positive on blocked | positive cells > legal candidates |
| --- | ---: | ---: | ---: | ---: |
| all observed high | `16360` | `8108` | `0` | `0` |
| all full high | `15326` | `7086` | `0` | `0` |

在全部候选格干净可见、且没有炸弹遮挡的 full high 样本中，positive cell count 精确等于未被玩家/NPC占用的合法候选数：

| 口径 | blocked candidates | batches | positive cell count | mean total |
| --- | ---: | ---: | --- | ---: |
| 地图1/2 full high，5候选全可见 | 0 | `3321` | `{5:3321}` | `95.25` |
| 地图1/2 full high，5候选全可见 | 1 | `978` | `{4:978}` | `95.04` |
| 地图1/2 full high，5候选全可见 | 2 | `61` | `{3:61}` | `93.66` |
| 1/2/3 full high，候选全可见 | 0 | `5141` | `{5:5141}` | `95.33` |
| 1/2/3 full high，候选全可见 | 1 | `2071` | `{4:2071}` | `95.18` |
| 1/2/3 full high，候选全可见 | 2 | `116` | `{3:116}` | `94.31` |
| 1/2/3 full high，候选全可见 | 3 | `3` | `{2:3}` | `95.33` |

因此当前数据更支持：

- high batch 选中一个外围 region 后，尝试作用于该 region 内全部 `static_map=2` 候选格。
- 被玩家/NPC占用的候选格不会获得本次正增量；炸弹格在 clean-visible 口径下不可直接验证，但全量统计同样没有观测到正增量落在炸弹格。
- 官方没有延后整个 high batch：存在 blocked candidate 时同回合仍在其它合法候选格生成。
- 地图1/2/3样本没有出现所有候选格都被占用的极端情况；map4 受控堵点实验补充显示，当 high region 的 `legal_candidates=0` 时，本次 high 不会整次跳过，也不会重采样到有 static2 的 region，而是退化为 high region 外外围普通格上的 combined fallback batch；被堵 high region 的 snapshot `gold_generated` 为 `0`，其它外区合计仍为 high-sized total。

金额分配过程更清晰：同一个 full high batch 内，各正增量格子的金额差几乎总是 `0` 或 `1`。因此它更像是先采样 batch total `T`，再把 `T` 均分到合法候选格：

```text
k = len(legal_static2_cells)
base = T // k
rem = T % k

for cell in legal_static2_cells:
    add base
for cell in sample_without_replacement(legal_static2_cells, rem):
    add 1
```

完整可见 full high 样本中，`T` 主要在 `80..112`，且 blocked candidates 为 `0/1/2` 时 mean total 分别约 `95.33/95.18/94.31`，说明 `T` 与动态占用基本无关；占用主要改变分母 `k`，从而提高剩余合法格的单格金额。

## map4 零合法 static2 fallback

map4 的外围 `static_map=2` 候选格明显少于地图1/2/3，其中 region 3 和 region 5 各只有一个候选格。2026-08-15 对这两个单候选外区做了 `40` 局受控堵点实验：

| 批次 | run_id | 局数 | 堵点 | target fallback |
| --- | --- | ---: | --- | ---: |
| region5 单堵 | `map4-static2-zero-region5-20260815-20` | `20` | `(2,13)` | `242` |
| region3 单堵 | `map4-static2-zero-region3-20260815-20` | `20` | `(16,3)` | `237` |

原始 replay：

```text
mechanism/data/raw/observer_runs/map4-static2-zero-region5-20260815-20/
mechanism/data/raw/observer_runs/map4-static2-zero-region3-20260815-20/
```

分析脚本和详细临时记录：

```text
temp/map4_static2_zero_candidate_experiment/analyze_fallback_events.py
temp/map4_static2_zero_candidate_experiment/README.md
```

识别口径：

- 用双方视角抽取 `end[t-1] -> start[t]` 的 clean visible positive cell delta。
- 正常 high 由 unblocked region 的 `static_map=2` 高额增量识别。
- fallback 由“无 normal static2 high、恰有一个外区普通格生成量为 `0`、其它外区普通格合计 `>=50` 且正增量格数 `>=5`”识别；该零生成外区视为被选中的 high region。
- snapshot full total 需按 `end[t-1] -> start[t]` 归入包含 `t-1` 的 snapshot window 对齐。

精细结论：

| 指标 | pooled target fallback |
| --- | ---: |
| events | `479` |
| 有 snapshot full total | `475` |
| full total mean | `119.11` |
| full total range | `100..140` |
| visible static0 count mean | `17.19` |
| visible static0 sum mean | `112.50` |
| 估计真实 `k` mean | 约 `18.2` |

这组数据推翻了“三局粗扫后提出的 high total 均分到少数 fallback cells”的假设。target fallback 的单格金额主体是 `1..11`，并有少量 `12..29` 尾部；正常 high 的伴随 static0 对照仅约 `4.2` 个可见格、总量约 `22`。因此更合理的模型是一个 combined fallback static0 batch，而不是“fallback high component + 独立伴随 static0”两个可分事件。

simulator 当前落地为：

- `DEFAULT_STATIC2_ZERO_FALLBACK_TOTAL_WEIGHTS`：pooled target snapshot full total，范围 `100..140`。
- `DEFAULT_STATIC2_ZERO_FALLBACK_COUNT_WEIGHTS`：基于 visible count 和覆盖补偿后的 pooled target `k` 分布，范围 `12..26`。
- `DEFAULT_STATIC2_ZERO_FALLBACK_AMOUNT_WEIGHTS`：pooled target visible static0 单格金额经验分布，范围 `1..29`。

生成时先采样 fallback total 和 `k`，从 high region 外外围 `static_map=0` 普通格无放回采样 `k` 个落点，再按 fallback 单格金额经验分布采样并调整到 total；零候选 fallback 分支不再额外生成正常伴随 static0。

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

不建议把 high batch 简单建成“独立采样若干单格金额再相加”。更稳定的结构是先采样 batch total，再分配到 high region 内未被动态对象占用的 `static_map=2` 合法格。

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

    high_region = sample_outer_region_conditioned_on_previous()
    total = sample_empirical_static2_batch_total(map_id)
    cells = [
        cell for cell in static2_cells[high_region]
        if not occupied_by_player_or_npc(cell) and not blocked_by_bomb(cell)
    ]
    if not cells:
        total = sample_empirical_zero_static2_fallback_total()
        k = sample_empirical_zero_static2_fallback_count()
        fallback_cells = sample_without_replacement(static0_cells_outside(high_region), k)
        amounts = sample_empirical_zero_static2_fallback_amounts_conditioned_on_sum(total, k)
        for cell, amount in zip(fallback_cells, amounts):
            add amount
        state.next_static2_round = t + sample_empirical_static2_gap(map_id)
        continue

    k = len(cells)
    base = total // k
    rem = total % k
    for cell in cells:
        add base
    for cell in sample_without_replacement(cells, rem):
        add 1

    generate companion outer_static0 outside high_region
    state.next_static2_round = t + sample_empirical_static2_gap(map_id)
```

简化版：

```text
gap ~ UniformInteger(8, 16)
region ~ near-uniform over outer regions, or first-order Markov
total ~ empirical_static2_batch_total
cells = unoccupied static2 cells in high region
split total nearly evenly over cells
```

关键点：

1. 不要把 high batch 触发写成每回合独立 Bernoulli；相邻 high batch 存在 `8` 回合最小间隔，主体为 `8..16`。
2. 同一回合同一时间只触发一个外围 region。
3. 相邻 high batch 的 region 存在一阶 persistence；简化版可近似均匀，拟合版建议使用按地图经验转移矩阵。
4. high region 内 static2 候选格在未被玩家/NPC占用时全覆盖；占用格不获得本次正增量。
5. 若 high region 的合法 static2 候选格为 `0`，不跳过、不重采样，而是生成 high region 外外围普通格 combined fallback batch；40 局 map4 单堵点实验支持 full total 范围 `100..140`、均值约 `119`，`k` 均值约 `18`，单格金额主体 `1..11` 且少量 `12..29` 尾部。当前 simulator 采样 fallback total 和 count，再按单格金额经验分布采样并调整到 total。
6. 正常 static2 金额不是单格独立采样，而是 batch total 在合法候选格上近似均分；零候选 fallback 不支持“把 high total 均分到少数普通格”，更像普通格批量小额生成。
7. high batch 所在 region 的普通格近似不生成，小额普通格生成发生在其它外围 region。
8. 地图3 batch total 较低和 non-high 中等片段较多，应优先解释为地图结构/可见覆盖差异，后续可按地图单独校准参数。
