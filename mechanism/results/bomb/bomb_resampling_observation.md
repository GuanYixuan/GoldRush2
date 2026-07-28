# 炸弹重采样规律观察

本文记录基于双视角合并 replay 的炸弹刷新规律初步验证。数据均来自本地离线产物，未访问评测平台。

## 数据范围

| 地图 | run_id | 对局数 |
| --- | --- | ---: |
| 地图1 | `symobs-a-map1-100-20260727-231446` | `100` |
| 地图2 | `symobs-a-map2-100-20260727-231704` | `100` |

输入文件：

```text
mechanism/data/processed/merged_replays/<run_id>/<game_id>.json
```

## 规则约束

已确认规则：

- 回合流程第 1 步为“生成新的金币或炸弹”，发生在玩家策略调用前。
- 炸弹每 `20` 轮刷新一次。
- `grid = -3` 表示炸弹。
- 角色移动到炸弹格时扣除当前持币 `10%`，向上取整；炸弹触发后消失。
- NPC 也会触发炸弹并使炸弹消失。
- 炸弹不会刷新在玩家、NPC、金币或已有炸弹所在格。
- 同一个位置不会同时有炸弹和金币。

本文只讨论当前公测数据中的可观测实现规律，不保证正式赛沿用。

## 实验口径

炸弹刷新发生在行动前，因此刷新变化使用：

```text
end[t-1] -> start[t]
```

行动触发导致的炸弹消失使用：

```text
start[t] -> end[t]
```

所有 cell transition 均只统计前后两帧该格都可见的样本：

```text
prev_grid[row][col] != -5
curr_grid[row][col] != -5
```

因此本文结论是“联合视野下可观察样本”的结论，不把 merged replay 当作上帝视角。玩家位置在 merged replay 中基本完整；NPC 只统计已可见条目，因此候选格里对 NPC 占用的排除是保守近似。

## 刷新相位

`end[t-1] -> start[t]` 的炸弹出现和清除只发生在 `round % 20 == 0`。

地图1：

| `round % 20` | appeared | cleared | same bomb |
| ---: | ---: | ---: | ---: |
| 0 | `36963` | `32593` | `2617` |
| 1..19 | `0` | `0` | 非刷新轮原炸弹保持 |

地图2：

| `round % 20` | appeared | cleared | same bomb |
| ---: | ---: | ---: | ---: |
| 0 | `39263` | `34599` | `3034` |
| 1..19 | `0` | `0` | 非刷新轮原炸弹保持 |

结论：

- 刷新相位严格对齐 `round % 20 == 0`。
- 非刷新相位没有观察到行动前的炸弹无故出现或消失。
- `start[t] -> end[t]` 中只观察到炸弹消失，没有观察到行动过程中炸弹出现，符合“炸弹只在回合开始前刷新”的规则。

行动阶段触发消失：

| 数据 | start bomb observations | disappeared during action | disappear rate |
| --- | ---: | ---: | ---: |
| 地图1 | `770520` | `4506` | `0.5848%` |
| 地图2 | `826818` | `4646` | `0.5619%` |
| 合并 | `1597338` | `9152` | `0.5730%` |

## 刷新机制

刷新轮上，观测到的旧炸弹大多会消失，但也有少量同一坐标在刷新后仍为炸弹：

| 数据 | prev bombs | cleared | same coord bomb | same rate |
| --- | ---: | ---: | ---: | ---: |
| 地图1 | `35210` | `32593` | `2617` | `7.431%` |
| 地图2 | `37633` | `34599` | `3034` | `8.062%` |
| 合并 | `72843` | `67192` | `5651` | `7.758%` |

这些 `same coord bomb` 不必解释为旧炸弹真实保留。若模型为“先清空旧炸弹，再在候选格上重新采样”，同一坐标被再次采样的期望数量为：

| 数据 | observed same coord bomb | expected if clear and resample |
| --- | ---: | ---: |
| 地图1 | `2617` | `2788.5` |
| 地图2 | `3034` | `2998.9` |

因此当前数据支持“整套重采样”：刷新轮清空旧炸弹后，再对当前候选格重新采样。刷新后同一坐标仍为炸弹，主要可由重采样抽回同一格解释。

## 刷新后数量分布

候选格定义为刷新前可见、非障碍、非金币、非玩家占用、且没有已知 NPC 的格。旧炸弹格在“清空后重采样”模型下视为候选格。

刷新轮统计：

| 数据 | refresh transitions | candidate mean | current bombs mean | current bombs var | sampled probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| 地图1 | `2400` | `207.969` | `16.492` | `15.383` | `0.07930` |
| 地图2 | `2400` | `221.158` | `17.624` | `17.364` | `0.07969` |
| 合并 | `4800` | `214.563` | `17.058` | 未汇总 | `0.07950` |

若每个候选格独立以同一概率采样炸弹，地图1的预测方差约为 `15.410`，地图2约为 `16.446`，与观测方差接近。相比“每次刷新固定采样 K 个炸弹”，当前数据更支持：

```text
for cell in eligible_cells:
    if Bernoulli(p_bomb):
        place bomb
```

其中：

```text
p_bomb ~= 0.0795
```

## 空间分布

刷新轮上，按区域统计的候选格采样率：

| region | sampled | candidate | rate |
| ---: | ---: | ---: | ---: |
| 1 | `20030` | `249962` | `0.08013` |
| 2 | `16488` | `208535` | `0.07907` |
| 3 | `15907` | `199544` | `0.07972` |
| 4 | `16100` | `204166` | `0.07886` |
| 5 | `13352` | `167696` | `0.07962` |

按静态地图取值统计：

| static_map | sampled | candidate | rate |
| ---: | ---: | ---: | ---: |
| 0 | `77389` | `973812` | `0.07947` |
| 2 | `4488` | `56091` | `0.08001` |

结论：

- 当前不需要为炸弹采样引入明显的 region 权重。
- `static_map=2` 与普通 `static_map=0` 的炸弹采样率接近，不应排除 `static_map=2`。
- 各 region 的采样率差异很小，第一版不建议加入额外空间权重。

## 建模建议

第一版炸弹重采样模型：

```text
for each round t:
    if t % 20 == 0:
        clear all existing bombs

        for cell in all_cells:
            if cell is obstacle:
                continue
            if cell has gold:
                continue
            if cell has player:
                continue
            if cell has NPC:
                continue

            if Bernoulli(0.0795):
                place bomb at cell
```

注意事项：

- 刷新发生在本回合行动前，建模输入状态应为 `end[t-1]`，输出状态为 `start[t]`。
- 旧炸弹格应先清空；若清空后该格没有金币、玩家或 NPC，则可以被重新采样为炸弹。
- 行动阶段只处理炸弹触发消失，不生成新炸弹。
- 官方规则写有“炸弹不会刷新在已有炸弹所在格”，但当前观测到的同格炸弹数量与“清空后重采样抽回同一格”的期望接近。第一版实现应按当前数据建模，后续若有 full replay 或更多布局数据再复核该语义。

## 后续验证

建议后续补一张炸弹 transition 特征表，避免每次重读完整 merged replay：

```text
mechanism/data/processed/bomb_features/<run_id>/
```

优先字段：

- `bomb_cell_transitions.csv`：`end[t-1] -> start[t]` 的 appeared / cleared / same。
- `bomb_refresh_rounds.csv`：每个 `game_id, round` 的候选格数量、刷新后炸弹数量、采样率。
- `bomb_candidate_summary.csv`：按 cell / region / static_map 聚合候选次数和采样次数。

若后续能获得更完整的 NPC 全图位置，应重新检查候选格定义中 NPC 排除条件对 `p_bomb` 的影响。
