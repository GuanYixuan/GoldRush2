# 初始金币生成分析

本文分析 `round=0.start.grid` 中开局已经存在的金币。该机制不同于后续 `end[t-1] -> start[t]` 的回合间生成，应在金币生成器中作为初始化阶段单独处理。

## 数据与产物

分析脚本：

```bash
conda run -n goldrush python mechanism/scripts/analysis/analyze_initial_gold.py
```

输入：

```text
official_sdk/data/full/g0.txt
official_sdk/data/full/g1.txt
official_sdk/data/full/g2.txt
mechanism/data/processed/gold_features/<run_id>/snapshot_windows.csv
```

输出：

```text
mechanism/data/processed/initial_gold_analysis/full_and_snapshots/
```

主要文件：

- `official_full_initial_gold.csv`：官方 3 局上帝视角 replay 的 `round=0.start` 初始金币统计。
- `initial_center_base_cells.csv`：3 局共同出现初始金币的 50 个中心格坐标。
- `snapshot_window0_model_comparison.csv`：用 400 局双视角数据的首个 snapshot 窗口验证初始模型。
- `summary.json`：摘要。

## 上帝视角观察

官方 3 局 full replay 的 `round=0.start.grid` 是全图可见，可以直接观察初始金币：

| replay | center positive cells | center total | amount counts | outer static0 total | outer static2 total |
| --- | ---: | ---: | --- | ---: | ---: |
| `g0.txt` | `50` | `59` | `{1:49, 10:1}` | `0` | `0` |
| `g1.txt` | `50` | `53` | `{1:49, 4:1}` | `0` | `0` |
| `g2.txt` | `50` | `70` | `{1:47, 4:1, 8:1, 11:1}` | `0` | `0` |

结论：

- 初始金币只出现在中心区 `region=1 && static_map=0`。
- 外围 `static_map=0` 和外围 `static_map=2` 初始金币均为 `0`。
- 三局都有且仅有 `50` 个中心格为正金币。
- 这 `50` 个格子的坐标集合在三局中完全一致。
- 这些格子都有基础 `+1`，少数格有额外金币。

固定初始中心基础格：

```text
(4,5) (4,6) (4,7) (4,8) (4,9) (4,10) (4,11)
(5,4) (5,6) (5,7) (5,8) (5,9) (5,10) (5,12)
(6,4) (6,5) (6,6) (6,7) (6,8) (6,9) (6,10) (6,11) (6,12)
(7,4) (7,5) (7,6) (7,8) (7,10) (7,11) (7,12)
(8,5) (8,7) (8,9) (8,11)
(9,4) (9,5) (9,6) (9,8) (9,10) (9,11) (9,12)
(10,4) (10,5) (10,6) (10,7) (10,8) (10,9) (10,10) (10,11) (10,12)
```

## Snapshot 口径

官方 snapshot 的首个窗口为 `window=[0,4]`，发布于 `round=5`。在 3 局 full replay 中：

```text
snapshot[0,4].center.gold_generated
    = round0.start 初始中心金币
    + round1..5 中心回合间生成
```

验证结果：

| replay | 初始中心 | round1..5 中心生成 | snapshot[0,4] center generated |
| --- | ---: | ---: | ---: |
| `g0.txt` | `59` | `90` | `149` |
| `g1.txt` | `53` | `55` | `108` |
| `g2.txt` | `70` | `23` | `93` |

因此首个 snapshot 窗口不能直接用于后续中心生成率拟合；它混入了初始金币。

## 模型对比

候选模型：

```text
base50:
    for cell in initial_center_base_cells:
        add 1

base50_plus_extra_tick:
    for cell in initial_center_base_cells:
        add 1
    run_center_small_gold_once()
```

其中 `run_center_small_gold_once()` 使用后续中心生成同一套 `center_A / center_B / UniformInteger(1,11)`。

用 400 局双视角 replay 的 `snapshot_windows.csv` 对比 `window=[0,4]` 的中心区 `gold_generated` 均值：

| 数据 | games | observed window0 center mean | center tick expectation | base50 + 5 ticks | base50 + extra tick + 5 ticks |
| --- | ---: | ---: | ---: | ---: | ---: |
| 地图1 | `100` | `115.55` | `10.9423` | `104.71` | `115.65` |
| 地图2 | `100` | `119.34` | `12.1562` | `110.78` | `122.94` |
| 地图3 pooled | `200` | `84.91` | `6.1487` | `80.74` | `86.89` |

`base50_plus_extra_tick` 明显更接近 `window0` 观测。该结果支持：初始金币不是只有固定 50 个 `+1`，还很可能额外执行了一次中心小额生成 tick。

## 初始基础格的矩形假设

此前文档把 3 局官方 full replay 中共同出现的 `50` 个格作为固定 mask。进一步检查发现，在官方样本地图上，这 `50` 个格并不像任意坐标集合，而是可以由一个更简单的矩形规则解释：

```text
base_rect = rows 4..10, cols 4..12
initial_base_cells = base_rect 内 static_map=0 的格
                     excluding round0.start 被动态对象占用的格
```

在官方 `g0/g1/g2` 对应的地图1静态布局中：

- `rows=4..10, cols=4..12` 是一个 `7x9` 矩形。
- 该矩形内有 `51` 个 `static_map=0` 格。
- `(8,8)` 也是 `static_map=0`，但 `round0.start` 被 NPC 占用，grid 显示为 `-4`。
- NPC 在 `round0.end` 离开 `(8,8)` 后，该格显示为 `0`，不是 `1`；因此当前 full replay 不支持 `(8,8)` 底下也有初始基础金币。
- 排除 `(8,8)` 后，矩形内剩余 `50` 个普通格与 `initial_center_base_cells.csv` 完全一致。

因此，对地图1而言，`50` 更像是“矩形内可见合法普通格”的结果，而不是直接手写的固定数量。

但跨地图推广时，数据不支持“同一矩形内所有 `static_map=0` 合法格”这一通用解释：

| 口径 | 地图1 | 地图2 | 地图3 |
| --- | ---: | ---: | ---: |
| `rows=4..10, cols=4..12` 内 `static_map=0`，排除 `(8,8)` | `50` | `53` | `24` |
| 整个中心 9x9 内 `static_map=0` | `65` | `69` | `27` |
| 首窗口反推 base 数量：`window0_center_mean - 6 * center_tick_expectation` | `49.90` | `46.40` | `48.01` |

地图3是主要冲突点：它在该 `7x9` 矩形内只有 `24` 个 `static_map=0`，整个中心 9x9 也只有 `27` 个 `static_map=0`，但首窗口统计反推的初始化基础量仍接近 `48`。这说明当前不能把初始化基础金币简单建成“地图相关矩形内所有普通合法格”。

目前更稳妥的解释是：

- 地图1 full replay 强支持存在一个 `rows=4..10, cols=4..12` 相关的初始 mask / 尝试生成区域。
- 地图2/3没有 full replay 直接观测，首窗口只能做聚合反推，且它与“合法普通格矩形”模型不一致。
- 可能的真实机制包括固定坐标 mask、矩形尝试生成但 snapshot 统计口径与障碍/占用有差异，或首窗口反推仍混有其它系统偏差。
- 在获得地图2/3上帝视角前，不建议把模拟器初始化改成按地图静态布局重新枚举矩形合法格。

## 建模建议

第一版初始化模型：

```text
initialize_gold():
    for cell in initial_center_base_cells:
        add 1

    run_center_small_gold_once()
```

其中：

- `initial_center_base_cells` 暂时固定为官方 full replay 观察到的 50 个中心格。
- `run_center_small_gold_once()` 使用与后续中心小额生成相同的 `center_A / center_B / UniformInteger(1,11)`。
- 外围 `static_map=0` 和 `static_map=2` 初始为 `0`。

谨慎点：

- 官方 full replay 只有 3 局，且都来自官方样本地图；地图1上 `50` 可由 `rows=4..10, cols=4..12` 矩形排除 `(8,8)` 解释，但地图2/3聚合统计不支持简单推广为“矩形内所有 `static_map=0`”。
- 对齐当前数据时，仍建议先使用官方 full replay 观察到的固定 50 格 mask；不要按地图2/3静态布局重新枚举矩形合法格。
- 若后续获得地图2/3上帝视角，应优先验证初始基础金币坐标是否随地图 static layout 改变，以及 `(8,8)` 等开局占用格底下是否存在被遮挡的初始金币。
