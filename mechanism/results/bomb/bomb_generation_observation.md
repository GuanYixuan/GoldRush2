# 炸弹生成规律观察

本文记录地图1、地图2、地图3 M3-A/M3-B 上的炸弹生成与重采样规律。数据来自本地双视角合并 replay 的离线分析，未访问评测平台。

## 数据范围

炸弹 transition 特征由 `mechanism/scripts/analysis/extract_bomb_features.py` 抽取：

```bash
conda run -n goldrush python mechanism/scripts/analysis/extract_bomb_features.py --run-id <run_id>
```

输出目录：

```text
mechanism/data/processed/bomb_features/<run_id>/
```

已抽取批次：

| 地图 | run_id | 对局数 | bomb transitions | appeared | disappeared |
| --- | --- | ---: | ---: | ---: | ---: |
| 地图1 | `symobs-a-map1-100-20260727-231446` | `100` | `12046100` | `36963` | `32593` |
| 地图2 | `symobs-a-map2-100-20260727-231704` | `100` | `12801700` | `39263` | `34599` |
| 地图3 M3-A | `symobs-m3a-map3-100-20260728-200953` | `100` | `9663800` | `30908` | `26982` |
| 地图3 M3-B | `symobs-m3b-map3-100-20260728-202647` | `100` | `9663800` | `31203` | `27260` |

特征口径：

```text
end[t-1] -> start[t]
prev_grid != -3 && curr_grid == -3  => appeared
prev_grid == -3 && curr_grid != -3  => disappeared
```

只统计连续可见且非静态障碍的格子，因此不可见格上的刷新不能直接观测。

## 规则约束

已确认规则与数据推断：

- 回合流程第 1 步为“生成新的金币或炸弹”，发生在玩家策略调用前。
- 炸弹每 `20` 轮刷新一次。
- `grid = -3` 表示炸弹。
- 角色移动到炸弹格时扣除当前持币 `10%`，向上取整；炸弹触发后消失。
- NPC 也会触发炸弹并使炸弹消失。
- 炸弹不会刷新在玩家、NPC、金币或已有炸弹所在格。
- 同一个位置不会同时有炸弹和金币。

本文只讨论当前可观测实现规律，不保证后续平台实现永远不变。

## 刷新相位

三张图中，可见炸弹新增和刷新消失都只发生在 `round % 20 == 0`。

| 数据 | appeared at mod 0 | appeared at mod 1..19 | disappeared at mod 0 | disappeared at mod 1..19 |
| --- | ---: | ---: | ---: | ---: |
| 地图1 | `36963` | `0` | `32593` | `0` |
| 地图2 | `39263` | `0` | `34599` | `0` |
| 地图3 pooled | `62111` | `0` | `54242` | `0` |

按 game-round：

| 数据 | refresh rounds | appeared mean | appeared range | disappeared mean | disappeared range | visible curr bombs mean |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| 地图1 | `2400` | `15.4013` | `4..30` | `13.5804` | `3..29` | `16.4917` |
| 地图2 | `2400` | `16.3596` | `5..32` | `14.4163` | `4..27` | `17.6238` |
| 地图3 pooled | `4800` | `12.9398` | `2..28` | `11.3004` | `1..27` | `13.9277` |

非刷新相位：

| 数据 | non-refresh rounds | appeared mean | disappeared mean |
| --- | ---: | ---: | ---: |
| 地图1 | `47500` | `0.0000` | `0.0000` |
| 地图2 | `47500` | `0.0000` | `0.0000` |
| 地图3 pooled | `95000` | `0.0000` | `0.0000` |

结论：炸弹生成可建模为每 `20` 轮在 `round % 20 == 0` 进行一次整相位刷新/重采样；其它回合不新增炸弹。

行动阶段 `start[t] -> end[t]` 中只观察到炸弹消失，没有观察到行动过程中炸弹出现，符合“炸弹只在回合开始前刷新”的规则。地图1/2行动阶段可见炸弹触发消失：

| 数据 | start bomb observations | disappeared during action | disappear rate |
| --- | ---: | ---: | ---: |
| 地图1 | `770520` | `4506` | `0.5848%` |
| 地图2 | `826818` | `4646` | `0.5619%` |
| 合并 | `1597338` | `9152` | `0.5730%` |

## 重采样机制

刷新轮上，观测到的旧炸弹大多会消失，但有少量同一坐标在刷新后仍为炸弹：

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

因此地图1/2 的细粒度候选格分析支持“整套重采样”：刷新轮先清空旧炸弹，再对当前候选格重新采样。刷新后同一坐标仍为炸弹，主要可由重采样抽回同一格解释。

## 刷新后数量分布

候选格定义为刷新前可见、非障碍、非金币、非玩家占用、且没有已知 NPC 的格。旧炸弹格在“清空后重采样”模型下视为候选格。

地图1/2 刷新轮统计：

| 数据 | refresh transitions | candidate mean | current bombs mean | current bombs var | sampled probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| 地图1 | `2400` | `207.969` | `16.492` | `15.383` | `0.07930` |
| 地图2 | `2400` | `221.158` | `17.624` | `17.364` | `0.07969` |
| 合并 | `4800` | `214.563` | `17.058` | 未汇总 | `0.07950` |

若每个候选格独立以同一概率采样炸弹，地图1的预测方差约 `15.410`，地图2约 `16.446`，与观测方差接近。相比“每次刷新固定采样 K 个炸弹”，当前数据更支持：

```text
for cell in eligible_cells:
    if Bernoulli(p_bomb):
        place bomb
```

其中：

```text
p_bomb ~= 0.0795
```

地图3当前特征表验证了刷新相位、约束和空间分布；候选格 Bernoulli 重采样率仍以地图1/2 的细粒度候选分析为主要依据。

## 生成约束

可见新增炸弹中没有违反以下约束的样本：

| 数据 | appeared on previous visible gold | appeared on previous visible player/NPC occupied cell | appeared on current visible player/NPC occupied cell |
| --- | ---: | ---: | ---: |
| 地图1 | `0` | `0` | `0` |
| 地图2 | `0` | `0` | `0` |
| 地图3 pooled | `0` | `0` | `0` |

这支持规则中“炸弹不会刷新在玩家、NPC、金币或已有炸弹所在格”的可见部分。不可见格仍不能由本口径直接证明。

## 空间分布

按 region 的 appeared/exposure：

| 数据 | region 1 | region 2 | region 3 | region 4 | region 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 地图1 | `0.002839` | `0.003238` | `0.003097` | `0.003110` | `0.003166` |
| 地图2 | `0.002807` | `0.003186` | `0.003189` | `0.003135` | `0.003136` |
| 地图3 pooled | `0.002847` | `0.003252` | `0.003262` | `0.003250` | `0.003317` |

按 `static_map` 的 appeared/exposure：

| 数据 | static_map=0 | static_map=2 |
| --- | ---: | ---: |
| 地图1 | `0.003140` | `0.002226` |
| 地图2 | `0.003139` | `0.002170` |
| 地图3 pooled | `0.003236` | `0.002961` |

地图1/2 候选格采样率的细粒度统计也显示 region 与 `static_map` 差异很小：

| group | sampled | candidate | rate |
| --- | ---: | ---: | ---: |
| region 1 | `20030` | `249962` | `0.08013` |
| region 2 | `16488` | `208535` | `0.07907` |
| region 3 | `15907` | `199544` | `0.07972` |
| region 4 | `16100` | `204166` | `0.07886` |
| region 5 | `13352` | `167696` | `0.07962` |
| static_map 0 | `77389` | `973812` | `0.07947` |
| static_map 2 | `4488` | `56091` | `0.08001` |

结论：

- 当前不需要为炸弹采样引入明显的 region 权重。
- `static_map=2` 与普通 `static_map=0` 都可能生成炸弹，不应排除 `static_map=2`。
- `appeared/exposure` 中 `static_map=2` 略低，可能受 high batch 金币、占用和可见性影响；按候选格定义校正后采样率接近。

## 建模建议

第一版炸弹生成器：

```text
for each round t:
    if t % 20 != 0:
        do not generate new bombs
        return

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
- 官方规则写有“炸弹不会刷新在已有炸弹所在格”，但当前观测到的同格炸弹数量与“清空后重采样抽回同一格”的期望接近。第一版实现应按当前数据建模，后续若有 full replay 或更完整同局覆盖数据再复核该语义。

NPC 与炸弹触发/消耗行为另见 `mechanism/results/npc/npc_feature_tables_and_bomb_interaction.md`；该文档分析的是触发和消耗，不应与本生成相位文档混为一谈。
