# map1 layout A simulator 对齐实验

本文是历史 rollout 对齐实验，记录当时 simulator default 下发现的金币堆残留偏差。当前 NPC 默认实现已更新为 `npc_behavior_modeling_overview.md` 中的 M4e；本文结果不代表当前 M4e default 的最终对齐表现。

## 实验目的

验证实验当时的 simulator 默认机制在 `map1 + symmetry observer layout A` 场景下，是否会复现真实 replay 中的地面金币堆分布。重点检查此前肉眼观察到的现象：simulator rollout 是否留下更多小金币堆，尤其是 `amount=1/2/3`。

本实验不预设偏差来源来自金币生成、NPC policy 或炸弹机制，只先做分布对齐和初步诊断。

## 数据与脚本

- 真实数据：`mechanism/data/processed/merged_replays/symobs-a-map1-100-20260727-231446/`，共 `100` 局。
- 仿真数据：固定 map1，双方接入 `mechanism/scripts/probes/symmetry_observer/layout_a.py`，使用实验当时的 simulator 默认机制 rollout `100` 局，每局 `500` 回合。
- 临时实验目录：`temp/simulator_alignment/map1_a_default/`
- 实验脚本：`temp/simulator_alignment/map1_a_default/run_alignment_experiment.py`
- 运行命令：

```bash
conda run --no-capture-output -n goldrush python temp/simulator_alignment/map1_a_default/run_alignment_experiment.py --games 100 --rounds 500 --sample-replays 3
```

产物：

- `temp/simulator_alignment/map1_a_default/stats/real_visible/`
- `temp/simulator_alignment/map1_a_default/stats/sim_visible/`
- `temp/simulator_alignment/map1_a_default/stats/sim_full/`
- `temp/simulator_alignment/map1_a_default/comparison/summary.md`
- `temp/simulator_alignment/map1_a_default/sim_replays/` 保留前 3 局 simulator full replay 样本。

## 统计口径

真实合并 replay 是双视角联合可见 replay，不是上帝视角。因此主比较口径使用“双方当前视野联合可见区域”：

- `real_visible`：真实 merged replay 的联合可见区域。
- `sim_visible`：simulator full state 按双方当前视野 mask 后的同口径统计。
- `sim_full`：simulator 上帝视角全图统计，只做辅助诊断，不直接和真实数据下结论。

每局每回合统计 `start` 和 `end` 两帧；下表以 `end` 帧为主。

## 主要结果

| 指标，end frame | real_visible | sim_visible | sim_full |
| --- | ---: | ---: | ---: |
| 平均可见格数 | `280.3340` | `280.5100` | `289.0000` |
| 平均金币堆数 | `24.1315` | `36.1630` | `37.6643` |
| 平均地面金币总量 | `150.8855` | `180.4635` | `190.1063` |
| 平均每堆金额 | `6.1712` | `4.9954` | `5.0635` |
| `amount=1` 占比 | `21.53%` | `29.24%` | `28.90%` |
| `amount<=2` 占比 | `36.82%` | `45.68%` | `45.36%` |
| `amount<=3` 占比 | `48.97%` | `59.82%` | `59.44%` |
| 平均中心 NPC 数 | `4.0065` | `3.6321` | `3.6505` |
| 平均炸弹数 | `15.3427` | `14.6288` | `15.0917` |

`start` 帧同样显示偏差：

| 指标，start frame | real_visible | sim_visible | sim_full |
| --- | ---: | ---: | ---: |
| 平均金币堆数 | `26.0393` | `37.7014` | `39.2239` |
| 平均地面金币总量 | `169.8994` | `194.3281` | `204.3342` |
| 平均每堆金额 | `6.4470` | `5.1571` | `5.2327` |
| `amount<=3` 占比 | `46.43%` | `58.33%` | `58.00%` |

结论：当前 simulator 确实留下了更多小金币堆；并且不只是 `amount=1` 偏多，而是整体地面金币堆数量偏高、平均每堆金额偏低。

## 按区域分解

end frame 金币堆占比：

| 区域 | real_visible | sim_visible | sim_full |
| --- | ---: | ---: | ---: |
| center9 | `42.32%` | `40.28%` | `38.68%` |
| static2 | `24.06%` | `23.38%` | `23.86%` |
| other | `33.62%` | `36.34%` | `37.46%` |

end frame 各区域 `amount<=3` 占比：

| 区域 | real_visible | sim_visible | sim_full |
| --- | ---: | ---: | ---: |
| center9 | `58.86%` | `69.27%` | `69.27%` |
| static2 | `35.18%` | `50.63%` | `50.60%` |
| other | `46.40%` | `55.26%` | `54.93%` |

区域分解说明：偏差不是中心金币单独造成的；`static2` 高额格和其它普通格也都有明显“小残堆”偏高。尤其 `static2` 的 `amount<=3` 从真实 `35.18%` 升到仿真 `50.63%`，提示 NPC 对高额 batch 的消耗深度或覆盖率可能不足。

## 时间分桶

按 `end` frame、每 100 回合分桶：

| 回合段 | real piles | sim_visible piles | real total | sim_visible total | real <=3/frame | sim <=3/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-99 | `19.80` | `26.26` | `120.69` | `147.38` | `9.82` | `14.46` |
| 100-199 | `24.69` | `36.91` | `156.82` | `184.52` | `12.06` | `21.98` |
| 200-299 | `25.76` | `38.13` | `161.49` | `186.61` | `12.53` | `23.08` |
| 300-399 | `25.39` | `39.41` | `156.87` | `190.57` | `12.52` | `24.02` |
| 400-499 | `25.02` | `40.11` | `158.56` | `193.23` | `12.15` | `24.62` |

偏差在 100 回合后持续扩大，表现为 simulator 库存逐步累积，而不是只在开局相位出现短暂差异。

## 拾取与减少事件

真实 merged replay 没有逐步 pickup 事件；本实验用同一格 `start/end` 都可见且金额下降作为 `visible reduction` 近似，会低估真实拾取。

- real visible reductions：`204689`
- sim visible reductions：`161557`
- sim explicit pickup events：`177721`
- sim pickup actor：`{'npc': 177627, 'player_unit': 94}`
- sim bomb triggers：`{'player_unit': 375, 'npc': 3755}`

可见减少事件少于真实，同时地面金币堆更多，支持“仿真 NPC 对金币堆的实际清理不足”这一方向。由于真实缺少上帝视角事件流，这里只能作为诊断线索，不能单独证明 NPC policy 是唯一原因。

## 解释与后续建议

当前结果支持以下判断：

1. simulator 小金币堆残留偏多是真实存在的分布偏差。
2. 偏差在同等可见口径下仍然明显，因此不是由真实 replay 非全图导致的观测错觉。
3. 偏差同时体现在 `amount<=3` 占比、平均金币堆数、平均每堆金额、时间累积趋势上。
4. 偏差不局限于中心区域；`static2` 和普通外围格也明显偏小。
5. 更优先怀疑 NPC 对金币堆的覆盖率/重复消耗深度不足，其次再检查金币生成 batch 强度与相位。

推荐下一步实验：

- 固定当前金币/炸弹机制，只调整 NPC 策略，增加“继续消耗当前/近邻残堆”的局部惯性或重复采食项，观察 `amount<=3`、平均金币堆数和 visible reductions 是否向真实靠拢。
- 对 static2 high batch 单独做生成量/消耗量平衡表，确认 simulator 是生成过多、NPC 消耗不足，还是两者都有。
- 在同一脚本下加入 map2/map3 交叉验证，避免为 map1 layout A 过拟合。

后续已完成 NPC 重复消耗与基础 pickup ablation，见 `mechanism/results/simulator_alignment/npc_repeat_consumption_ablation.md`。再后续的 path-level 拟合已把该方向吸收到当前推荐 M4e 中；当前默认实现以 `mechanism/results/npc/npc_behavior_modeling_overview.md` 为准。
