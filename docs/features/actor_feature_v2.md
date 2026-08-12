# Actor Feature v2 设计

本文冻结 `goldrush2_feature_v2` 的增量设计。v2 继承 `goldrush2_feature_v1` 的全部 channel、scalar、状态机和输入边界，只在 spatial planes 末尾新增少量可部署机制先验和 belief planes。v1 完整语义见 `docs/features/actor_feature_v1.md`。

## 定位

v2 的目标是补上三类低参数高价值信息，并按正式规则澄清修正障碍推断：

- snapshot 中最近窗口各区域 `gold_generated`。
- 中心小额金币生成概率先验。
- 炸弹当前存在概率 belief。
- 在线高额外围金币推断出的 static2 候选 belief，以及到候选格的静态导航距离图。
- 障碍物只假设满足上下对称或左右对称至少一条，不再使用 180 度中心对称推断。

正式比赛不会采用当前三张公开地图，因此 actor feature 不再内置公开地图库、公开地图 obstacle/static2 坐标或闭集地图识别逻辑。

actor 输入仍只能来自官方 `GameInput`、历史 observation、规则允许的地图对称性/机理数据推断和 feature extractor 自身状态。禁止使用 simulator full state、训练期 `map_id` 真值、真实 `MapTemplate.special_cells`、公开地图坐标或未来金币生成信息。

## 输出形态

v2 输出：

```text
feature_schema: "goldrush2_feature_v2"
spatial_planes: 43 x 17 x 17
scalars: 10
tokens: none
```

`scalars` 与 v1 完全一致。前 38 个 spatial channel 与 v1 完全一致，新增 channel 放在最后：

```text
...
last_snapshot_gold_remaining_map
prev_snapshot_gold_remaining_map
last_snapshot_occupants_map
last_snapshot_gold_generated_map
center_gold_spawn_prob
bomb_belief_mask
static_2_mask
to_static2_distance
```

其中 `static_2_mask` 是 binary plane，取值为 `0.0` 或 `1.0`。其它新增 plane 为固定缩放的 float plane。

## Snapshot Generated Map

`last_snapshot_gold_generated_map` 使用最近一次有效 snapshot 中对应区域的 `gold_generated`，广播到该区域：

```text
last_snapshot_gold_generated_map[cell] =
    clip(last_snapshot.regions[region(cell)].gold_generated / 100, 0, 2)
```

无历史有效 snapshot 时，该 plane 全 0。

该 plane 与 v1 的 `last_snapshot_gold_remaining_map` / `prev_snapshot_gold_remaining_map` 使用同样的 region broadcast 和 `/100` clipping 口径。它表达最近 snapshot 窗口内区域生成量，而不是当前地面金币真值。

## Center Gold Spawn Prior

`center_gold_spawn_prob` 按 privileged critic 中同名 plane 的概率核构造，但必须受 actor 当前已知障碍过滤：

```text
if cell in center 9x9 and not known_obstacle(cell):
    center_gold_spawn_prob[cell] = exp(-B * ((row - 8)^2 + (col - 8)^2))
else:
    center_gold_spawn_prob[cell] = 0
```

其中：

```text
B = 0.06735
known_obstacle(cell) = obstacle_known_mask[cell] == 1 and obstacle_mask[cell] == 1
```

该 plane 不包含总体强度 `A`，范围约为 `0..1`。未知障碍格不按障碍处理，保持中心概率先验；只有 actor 已经直接或按 v1 障碍记忆推断为障碍的格子才置 0。

## 障碍轴对称推断

v2 覆盖 v1 的障碍推断规则：不再使用 180 度中心对称 `(r, c) -> (16-r, 16-c)`。正式地图只保证静态障碍至少满足以下一条：

```text
up_down symmetry:    (r, c) -> (16-r, c)
left_right symmetry: (r, c) -> (r, 16-c)
```

feature extractor 每局维护：

```text
possible_symmetry_axes = {up_down, left_right}
direct_obstacle_status[17][17] in {unknown, free, obstacle}
inferred_obstacle_status[17][17] in {unknown, free, obstacle}
```

每次直接观察到非迷雾格：

```text
if visible cell is obstacle:
    direct_obstacle_status[cell] = obstacle
else:
    direct_obstacle_status[cell] = free
```

直接观测永远优先。随后根据直接观测排除不再可能的轴：

```text
if any directly observed pair conflicts under up_down:
    remove up_down from possible_symmetry_axes

if any directly observed pair conflicts under left_right:
    remove left_right from possible_symmetry_axes
```

最后重算推断缓存：

```text
if direct_obstacle_status[cell] is known:
    effective_status[cell] = direct_obstacle_status[cell]
else if only one axis remains possible:
    effective_status[cell] = direct_obstacle_status[mirror(cell, axis)]
else if both axes remain possible:
    up_down_status = direct_obstacle_status[mirror(cell, up_down)]
    left_right_status = direct_obstacle_status[mirror(cell, left_right)]
    if both are known and equal:
        effective_status[cell] = up_down_status
    else:
        effective_status[cell] = unknown
else:
    effective_status[cell] = unknown
```

若只剩一条轴可能，但镜像格尚未直接观察，则该格仍为 unknown。若两条轴都可能，必须两条轴都能推出同一个状态，才输出 inferred known；否则保持 unknown。

`obstacle_known_mask` / `obstacle_mask`、`unit*_known_obstacle_distance`、`center_gold_spawn_prob` 的障碍过滤、`bomb_belief_mask` 的障碍清零，以及 `to_static2_distance` 的阻挡规则都使用该 `effective_status`。

## Bomb Belief Mask

`bomb_belief_mask` 表达 actor 对当前该格存在炸弹的 belief，取值范围 `[0, 1]`。第一版只使用硬规则和机制固定概率：

```text
BOMB_SPAWN_PROBABILITY = 0.0795
BOMB_REFRESH_PERIOD = 20
```

状态初始化：

```text
bomb_belief_mask[cell] = 0.0795 for all cells
known_obstacle(cell) -> 0
```

每次 `observe(game_input)` 时，如果当前回合是炸弹刷新回合：

```text
if round % 20 == 0:
    reset bomb_belief_mask[cell] = 0.0795 for all cells
    known_obstacle(cell) -> 0
```

随后用当前 observation 覆盖可见格：

```text
visible bomb cell -> 1.0
visible non-bomb cell -> 0.0
visible obstacle cell -> 0.0
```

非刷新回合不重置不可见格：

```text
if round % 20 != 0:
    invisible cells keep previous belief
    visible cells are overwritten by current observation as above
```

特别地，如果某格此前 belief 为 `1.0`，非刷新回合中本轮不可见，则保持 `1.0`。炸弹只会在刷新回合或被玩家/NPC触发时消失；不可见触发无法由 actor 确认，因此保守保留该风险。

设计边界：

- actor 不尝试重建真实炸弹候选集合，因为不可见金币、NPC、敌方位置不可知。
- 当前可见金币、己方、可见敌人、可见 NPC 所在格都按 visible non-bomb 覆盖为 `0.0`。
- 已知障碍格始终为 `0.0`。
- `bomb_mask_t0..t4` 继续保留原始可见炸弹历史；`bomb_belief_mask` 是派生 belief，不替代 temporal 观测。

## Static2 推断状态

feature extractor 每局只维护在线 static2 推断状态：

```text
observed_high_outer_gold_mask: 17 x 17 bool
```

初始化：

```text
observed_high_outer_gold_mask = all false
```

`reset(player_id)` 必须清空上述状态。round 回退触发 reset 或 fail-fast 时，也必须清空。

## 在线高金币发现

v2 使用一个固定阈值：

```text
STATIC2_GOLD_THRESHOLD = 16
```

每次 `observe(game_input)` 时，对当前可见格更新：

```text
if region_id(cell) != 1 and visible_gold(cell) >= 16:
    observed_high_outer_gold_mask[cell] = true
```

规则约束：

- 不要求见证金币生成 delta；只要当前可见外围金币数达到阈值即可。
- `observed_high_outer_gold_mask` 在同一局内持久保留，即使后续该格金币被拾取或不可见，也不清除。
- 不做 weak/strong 两级证据。
- 不因重复看到同一格高金币而额外累计任何分数。
- 不把一个高金币格扩展到同 region 其它 static2 候选格；region 扩展留给后续 schema。
- 阈值 `16` 来自 static2 high batch 单格金额主体区间。它比 `20` 更激进，会提高发现率，同时接受少量外围普通金币累积误报。

## `static_2_mask` 输出规则

输出：

```text
static_2_mask[cell] =
    1.0 if observed_high_outer_gold_mask[cell]
    0.0 otherwise
```

该 plane 是硬 binary feature，不含概率含义。它只表达“该格已被在线高额外围金币观测标记为可疑 static2 格”。开局没有任何 static2 坐标先验。

## Static2 Distance

`to_static2_distance` 是到当前 `static_2_mask` 目标集合的多源已知障碍 BFS 距离图：

```text
targets = {cell | static_2_mask[cell] == 1 and not known_obstacle(cell)}

if targets is not empty:
    d(cell) = shortest path distance from cell to nearest target
    to_static2_distance[cell] = clip(d(cell) / 32, 0, 2)
else:
    to_static2_distance[cell] = 2
```

阻挡规则沿用 v1 的已知障碍 BFS 距离图：

```text
blocked(cell) = obstacle_known_mask[cell] == 1 and obstacle_mask[cell] == 1
```

未知格按可通行处理。玩家、敌人、NPC、金币和炸弹都不作为该距离图的阻挡，因为该 plane 表达的是静态地图导航势场，不是本回合精确 legal path。

若没有任何在线 static2 target belief，则全图取最大值 `2`。

## 设计边界

`static_2_mask` 有意避免引入更多参数和状态：

- 不维护公开地图坐标或地图 posterior/log weight。
- 不维护证据强弱分数。
- 不维护 region confidence。
- 不根据 high gold 自动推断同 region 其它未知候选格。
- 不用 snapshot 直接改写 `static_2_mask`。

snapshot 的区域统计由 v1 的 `last_snapshot_gold_remaining_map`、`prev_snapshot_gold_remaining_map`、`last_snapshot_occupants_map` 和 v2 的 `last_snapshot_gold_generated_map` 表达。若后续需要 `static2_region_priority_map`、`unit*_to_static2_distance` 或 `static2_high_imminence`，必须使用新的 schema 或在 v2 文档中明确扩展后再实现。

## 版本与兼容

v2 schema 固定为：

```text
feature_schema = "goldrush2_feature_v2"
```

任何使用 v1 checkpoint 的模型不能直接消费 v2 feature。训练 checkpoint、ONNX 导出、C++ 部署和 runtime binding 必须记录并校验 feature schema、channel 数量和 channel 顺序。

## 测试要求

v2 在 v1 测试基础上新增：

- `feature_schema == "goldrush2_feature_v2"`。
- `spatial_planes.shape == (43, 17, 17)`、`scalars.shape == (10,)`。
- 前 38 个 channel 与 v1 同输入输出一致。
- `last_snapshot_gold_generated_map` 使用最近有效 snapshot 的 `gold_generated / 100` 区域 broadcast；无 snapshot 时全 0。
- `center_gold_spawn_prob` 与 critic 概率核一致，并对 actor 已知障碍格置 0。
- `bomb_belief_mask` 初始化和刷新回合将未知/非障碍格置为 `0.0795`，已知障碍格置 0。
- `bomb_belief_mask` 可见炸弹格为 1，可见非炸弹格为 0。
- 非刷新回合不可见格保持上一轮 bomb belief；此前为 1 的不可见格保持 1。
- 障碍推断不使用 180 度中心对称；直接观测优先，轴对称候选按上下/左右冲突排除。
- 两条轴都可能时，只有上下/左右镜像直接观测推出同一状态，才输出 inferred known。
- 只剩一条轴可能时，按该轴镜像直接观测推断；镜像未观测则保持 unknown。
- `static_2_mask` 初始全 0。
- 当前可见外围金币 `>=16` 的格会写入并持久保留 online mask。
- 中心区金币 `>=16` 不写入 online mask。
- `to_static2_distance` 对 `static_2_mask` 中非已知障碍 target 取 0，并按已知障碍 BFS 距离 `clip(distance / 32, 0, 2)` 填充。
- 无 static2 target belief 时，`to_static2_distance` 全图为 2。
- reset 清空 symmetry axis 状态和 online mask。
