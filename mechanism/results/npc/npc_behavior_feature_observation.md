# NPC 行为统计特征初步观察

本文记录基于双视角合并 replay 的 NPC 行为统计特征。目标是先建立可靠的观测事实和后续特征表设计，而不是直接假设 NPC 的完整决策机制。

## 数据范围

| 地图 | run_id | 对局数 | rounds |
| --- | --- | ---: | ---: |
| 地图1 | `symobs-a-map1-100-20260727-231446` | `100` | `50000` |
| 地图2 | `symobs-a-map2-100-20260727-231704` | `100` | `50000` |

输入文件：

```text
mechanism/data/processed/merged_replays/<run_id>/<game_id>.json
```

## 规则与 replay 口径

已确认规则：

- 公测 NPC 总数为 `7`。
- 所有 NPC 开局出生在地图中心。
- 每个 NPC 每轮最多移动 `3` 步。
- NPC 行动位于两个玩家行动之间。
- NPC 可与玩家、NPC 重叠，不阻挡玩家移动。
- NPC 会拾取金币，也会触发炸弹并使炸弹消失。
- NPC 移动策略、内部顺序不公开。

replay 口径：

- NPC 按 `id` 对齐，当前样本 id 为 `-7..-1`。
- `round t` 的 `end.npcs[].actions` 是本轮 NPC 动作。
- `round t` 的 `start.npcs[]` 常常是上一轮 `end.npcs[]` 状态副本，因此连续轨迹应使用同轮 `start.npcs[id].position -> end.npcs[id].position` 验证。
- 双视角合并 replay 仍不是上帝视角；未出现在 `npcs[]` 的 NPC 不能当作不存在。

本文把同一轮 `start` 和 `end` 中同 id NPC 均可见，且 `end.npcs[].actions` 存在的样本称为“连续可见轨迹”。

## 可见性与数据质量

NPC 可见覆盖率较高，但仍是条件于当前双视角布局的观测结果。

| 数据 | possible end NPC entries | observed end NPC entries | coverage |
| --- | ---: | ---: | ---: |
| 地图1 | `350000` | `345044` | `98.58%` |
| 地图2 | `350000` | `345582` | `98.74%` |

每轮 `end.npcs` 数量分布：

| 数据 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 地图1 | `100` | `0` | `2` | `4` | `21` | `221` | `3725` | `45927` |
| 地图2 | `100` | `1` | `4` | `9` | `21` | `158` | `3277` | `46430` |

连续可见轨迹：

| 数据 | continuous trajectories | replay match | mismatch | invalid moves |
| --- | ---: | ---: | ---: | ---: |
| 地图1 | `341207` | `341207` | `0` | `0` |
| 地图2 | `342464` | `342464` | `0` | `0` |

结论：

- 连续可见样本中，按动作 `0/1/2/3/4 = 上/下/左/右/停` 重放，`start_pos + actions` 与 `end_pos` 完全一致。
- 当前未观察到 NPC 尝试越界或撞静态障碍导致原地失败。
- `actions` 长度均为 `3`。因此 step 级特征可以可靠抽取。

## 调度顺序

`end.dispatch_order` 在两图全部 `100000` 个 round 中形态完整：

```text
[fast_player, 7 个 NPC 的排列, slow_player]
```

| 数据 | complete dispatch rounds | NPC block between players | P1 first | P2 first | unique NPC permutations |
| --- | ---: | ---: | ---: | ---: | ---: |
| 地图1 | `50000` | `50000` | `24950` | `25050` | `5040` |
| 地图2 | `50000` | `50000` | `25099` | `24901` | `5040` |

`5040 = 7!`，两图都观察到了全部 NPC 内部排列。各 NPC 成为第一个 NPC 行动者的次数也接近均匀：

| npc_id | 地图1 first count | 地图2 first count |
| ---: | ---: | ---: |
| -7 | `7146` | `7178` |
| -6 | `7133` | `7223` |
| -5 | `7250` | `7150` |
| -4 | `7165` | `7128` |
| -3 | `7153` | `7008` |
| -2 | `7184` | `7092` |
| -1 | `6969` | `7221` |

结论：NPC 内部行动顺序高度像每轮对 7 个 NPC 做均匀随机排列。第一版建模可以把 NPC 内部顺序视为随机 permutation。

## 动作分布

连续可见轨迹中，所有 NPC 每轮都有 3 个动作，总动作数为 `2051013`。

合并动作频率：

| action | 含义 | count | freq |
| ---: | --- | ---: | ---: |
| 0 | 上 | `523765` | `25.54%` |
| 1 | 下 | `525343` | `25.61%` |
| 2 | 左 | `486323` | `23.71%` |
| 3 | 右 | `486661` | `23.73%` |
| 4 | 停 | `28921` | `1.41%` |

每轮非停动作数：

| non-stay actions | count | freq |
| ---: | ---: | ---: |
| 0 | `3711` | `0.54%` |
| 1 | `3761` | `0.55%` |
| 2 | `10266` | `1.50%` |
| 3 | `665933` | `97.41%` |

终点相对起点的 Manhattan 位移：

| displacement | count | freq |
| ---: | ---: | ---: |
| 0 | `5981` | `0.87%` |
| 1 | `225723` | `33.02%` |
| 2 | `7996` | `1.17%` |
| 3 | `443971` | `64.94%` |

分步动作存在明显结构：

- 第 1 步偏上下方向。
- 第 2 步四个方向接近均匀。
- 第 3 步偏左右方向。
- `停` 很少，但在地图2明显多于地图1。

这提示后续不宜直接把 3 个 step 当作完全同分布动作；更稳的特征表应保留 `step_idx`。

## 空间分布

连续轨迹的起点区域分布：

| region | count | freq |
| ---: | ---: | ---: |
| 1 | `402089` | `58.82%` |
| 2 | `75616` | `11.06%` |
| 3 | `68357` | `10.00%` |
| 4 | `74987` | `10.97%` |
| 5 | `62622` | `9.16%` |

结论：

- NPC 大部分时间仍在中心 region 1。
- 外围四区占比接近，但受地图布局和双视角站位影响，不能直接解释为全局稳态区域分布。
- 区域迁移主要发生在中心与相邻外围区域之间；外围区之间的直接迁移较少，符合区域几何结构。

## 拾金统计

| 数据 | NPC entries | pickup events | pickup rate | pickup sum | amount / event |
| --- | ---: | ---: | ---: | ---: | ---: |
| 地图1 | `345044` | `175551` | `50.88%` | `959360` | `5.465` |
| 地图2 | `345582` | `188752` | `54.62%` | `1027022` | `5.442` |
| 合并 | `690626` | `364303` | `52.75%` | `1986382` | `5.453` |

合并 pickup 事件主要集中在中心区：

| region | pickup events | pickup sum | amount / event |
| ---: | ---: | ---: | ---: |
| 1 | `237729` | `1086502` | `4.570` |
| 2 | `32194` | `224639` | `6.978` |
| 3 | `29986` | `202803` | `6.763` |
| 4 | `33350` | `244834` | `7.342` |
| 5 | `31044` | `227604` | `7.332` |

结论：

- NPC 拾金频繁，约半数可见 NPC round 有 pickup。
- 中心区拾金次数最多，但外围单次 pickup 金额更高，符合外围高额金币批次的存在。
- 后续建模不能只做运动模型，必须把金币拾取作为行为反馈特征。

## 局部金币响应

为避免直接推测 NPC 策略，本文只做一个弱检验：在连续可见轨迹中，用 `start.grid` 的当前可见金币计算 NPC 起点到最近可见金币的距离，比较 NPC 第一步是否比随机合法动作更倾向于靠近金币。

合并结果：

| 指标 | count / value | freq |
| --- | ---: | ---: |
| cases with visible gold | `683657` |  |
| chosen first action is best among legal actions | `346129` | `50.63%` |
| random legal action best expectation | `244719.5` | `35.79%` |
| first action closer to nearest visible gold | `316597` | `46.31%` |
| first action same distance | `129364` | `18.93%` |
| first action farther | `237696` | `34.77%` |

若排除 `nearest_gold_distance = 0` 的情况，趋近金币信号更明显：第一步靠近最近可见金币约 `55.4%`，选择最优合法动作约 `56.3%`。

结论：

- NPC 对局部可见金币存在显著响应信号。
- 该统计只说明“金币相关特征值得进入模型”，还不能说明 NPC 只根据最近金币行动。
- 因为 NPC 位于玩家/NPC 行动顺序中间，`start.grid` 之后可能已被先手玩家改变，所以后续需要把行动顺序和玩家先手路径纳入更精细的特征表。

## 建模前特征表建议

建议先落以下特征表，再开始拟合模型：

```text
mechanism/data/processed/npc_features/<run_id>/
```

### `npc_round_observations.csv`

每个可连续观察的 `game_id, round, npc_id` 一行：

- `map_id`
- `start_row, start_col, end_row, end_col`
- `start_region, end_region`
- `actions`
- `dispatch_index`
- `visible_by_start, visible_by_end`
- `pickup`
- `start_nearest_gold_dist`
- `start_nearest_bomb_dist`
- `start_nearest_player_dist`
- `gold_count_r3, gold_sum_r3, max_gold_r3`

### `npc_step_events.csv`

每个 NPC step 一行：

- `game_id, round, npc_id, step_idx`
- `action`
- `before_row, before_col, after_row, after_col`
- `target_region`
- `target_static_map`
- `target_grid_at_round_start`
- `moved`
- `distance_to_nearest_visible_gold_before`
- `distance_to_nearest_visible_gold_after`
- `distance_to_nearest_visible_bomb_before`
- `distance_to_nearest_visible_bomb_after`

### `npc_dispatch_orders.csv`

每轮一行：

- `game_id, round`
- `first_player`
- `npc_order`
- 每个 `npc_id` 的 `npc_dispatch_index`

### `npc_interactions.csv`

只记录事件：

- pickup 事件
- 走到 start 可见炸弹格的潜在触发事件
- 与玩家或其它 NPC 重叠事件
- 区域 enter / leave 事件

## 后续建模顺序

建议按如下顺序递进：

1. 先用 `npc_dispatch_orders.csv` 验证内部顺序是否可固定为随机 permutation。
2. 用 `npc_step_events.csv` 建立 step-index-aware baseline，即按 `step_idx`、region、NPC id 的动作经验分布。
3. 加入局部金币、炸弹、边界、障碍、中心距离等特征，拟合五动作 softmax。
4. 若 step 级模型解释不足，再改为 3-step path-level 模型，对完整长度为 3 的路径打分。

第一版不建议直接设计复杂状态机。当前最确定的信号是：NPC 每轮 3 步动作可完整重放，内部顺序近似随机，局部金币响应明显。
