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

- 官方 full replay 只有 3 局，且都来自官方样本地图；对地图2/3的初始 50 格是否完全相同，目前主要由首个 snapshot 间接支持。
- 若后续获得地图2/3上帝视角，应优先验证初始 50 格坐标是否随地图 static layout 改变。
