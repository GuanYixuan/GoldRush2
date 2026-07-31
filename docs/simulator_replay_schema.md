# Simulator 上帝视角 Replay Schema

本文定义 `simulator/` 本地环境导出的上帝视角 replay 格式。该格式服务于 simulator 调试、机制观测和可视化器开发，不是评测平台官方 NDJSON，也不是双视角合并 replay。

官方 replay 格式仍以 `gamerules/replay.md` 为准；双视角合并派生格式见 `docs/merged_replay_schema.md`。可视化器必须通过顶层 `format` 区分三类输入，不应把 simulator full replay 伪装成 official 或 merged replay。

## 设计目标

- 保存 simulator 运行时的完整上帝视角状态，便于本地复盘。
- 明确区分静态地图、资源格、玩家实体和 NPC 实体。
- 记录双方 `GameOutput`、NPC actions、规则事件、机制事件和 snapshot，便于定位机制或规则实现问题。
- 不导出不可观测且不计分的 NPC 内部持币账本。
- 避免沿用官方 full replay 中的派生 grid 标记；visualizer 如需玩家/NPC 叠层，应自行从实体字段派生。

## 顶层格式

Simulator replay 使用单个 JSON 文档：

```json
{
  "format": "goldrush2_simulator_full_replay",
  "format_version": 1,
  "source": {
    "seed": 123,
    "map_id": 1,
    "map_name": "official_map_1",
    "mechanisms": {}
  },
  "players": {
    "player1": "policy_a",
    "player2": "policy_b"
  },
  "maps": [],
  "rounds": [],
  "forfeit": null
}
```

字段说明：

| 字段 | 必填 | 含义 |
| --- | --- | --- |
| `format` | 是 | 固定为 `goldrush2_simulator_full_replay`。 |
| `format_version` | 是 | 当前为 `1`。 |
| `source.seed` | 建议 | 本局 simulator seed，可为 `null`。 |
| `source.map_id` | 是 | simulator 地图模板 ID。 |
| `source.map_name` | 建议 | simulator 地图模板名。 |
| `source.mechanisms` | 是 | 机制模型元信息；字段内容由 runner 填充。 |
| `players` | 是 | 玩家名映射，key 固定为 `player1` / `player2`。 |
| `maps` | 是 | `17x17` 静态地图，取值见下文。 |
| `rounds` | 是 | 回合数组。 |
| `forfeit` | 否 | 预留判负信息；正常本地对战通常省略。 |

### `maps`

`maps` 是 simulator 静态地图层，不是局中资源层：

| 值 | 含义 |
| --- | --- |
| `0` | 普通可通行格。 |
| `1` | 静态障碍，不可进入。 |
| `2` | 特殊非阻挡格，可通行；当前用于外围金币生成候选层。 |

可视化器应把 `maps=1` 渲染为障碍。`maps=2` 不是障碍，不应阻挡 NPC 或玩家路径显示。

## 轮次格式

每个回合对象结构如下：

```json
{
  "round": 0,
  "start": {},
  "end": {},
  "actions": {},
  "events": {},
  "snapshot": null
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `round` | 回合编号，从 `0` 开始。 |
| `start` | 本回合行动前完整上帝视角状态。对 `envs.duel` 导出的 replay，该状态已经包含本回合开始时生成/刷新后的资源。 |
| `end` | 本回合行动后完整上帝视角状态，`round_index` 通常为 `round + 1`。 |
| `actions` | 双方玩家输出、NPC 动作和可选 NPC 执行顺序。 |
| `events` | 本轮机制事件和规则事件。 |
| `snapshot` | 本轮发布的 simulator `Snapshot`；无新 snapshot 时为 `null`。 |

重要语义：

- `events.gold_generated` 和 `events.bomb_refresh` 是回合开始前的机制事件。对 `envs.duel` 输出，它们已经反映在 `start` 状态中，可视化器不应把这些事件再次应用到 `start`。
- `start -> end` 的状态差异主要由玩家移动、NPC 移动、拾金、炸弹触发、踩踏和视野购买产生。
- `rounds` 应按 `round` 连续递增；当前 recorder 对顺序不匹配 fail-fast。

## 状态帧格式

`start` 和 `end` 使用同一结构：

```json
{
  "round_index": 0,
  "grid": [],
  "players": [],
  "npcs": [],
  "gold": [],
  "bombs": []
}
```

### `grid`

`grid` 是 `17x17` 资源格，只表达地形和资源：

| 值 | 含义 |
| --- | --- |
| `-3` | 炸弹。 |
| `-1` | 静态障碍。 |
| `0` | 空地。 |
| `>=1` | 该格地面金币数量。 |

Canonical simulator replay 不使用这些官方/派生标记：

| 值 | 禁止原因 |
| --- | --- |
| `-5` | simulator replay 是上帝视角，没有迷雾。 |
| `-4` | 玩家/NPC 叠层不写入 `grid`，应从实体字段派生。 |
| `-2` | 玩家/NPC 叠层不写入 `grid`，应从实体字段派生。 |

`grid` 与 `gold` / `bombs` 是冗余表达，便于不同消费者读取。可视化器可优先使用 `grid` 渲染资源层，并用 `gold` / `bombs` 做索引或校验。

资源不变量：

- 金币数量必须为正。
- 金币不能和障碍或炸弹重合。
- 炸弹不能和障碍或金币重合。
- 玩家和 NPC 不写入 `grid`，它们可以站在金币或炸弹以外的同一格；NPC 之间、NPC 与玩家之间允许重叠。

### `players`

玩家数组按玩家 ID 排序：

```json
{
  "id": 1,
  "gold": 0,
  "vision_spent": 0,
  "active_vision_radius": 2,
  "next_vision_radius": 2,
  "units": [
    {
      "id": 0,
      "position": [0, 0],
      "gold": 0
    }
  ]
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `id` | 玩家 ID，当前为 `1` 或 `2`。 |
| `gold` | 该玩家两个单位持币总和，未扣除视野花费。 |
| `vision_spent` | 累计视野花费，用于赛后净金币结算。 |
| `active_vision_radius` | 本回合对该玩家生效的视野半径。 |
| `next_vision_radius` | 已购买、将在下一回合生效的视野半径；行动结算后通常被重置为默认值。 |
| `units[].id` | 单位 ID，当前为 `0` 或 `1`。 |
| `units[].position` | `[row, col]`。 |
| `units[].gold` | 单位当前持币。 |

### `npcs`

NPC 数组按 NPC ID 排序：

```json
{
  "id": -1,
  "position": [8, 8]
}
```

Simulator 不维护 `npc.gold`，replay 也不得伪造该字段。NPC 拾取金币只通过 `events.interactions[].pickups` 记录，NPC 持币不参与玩家胜负结算。

### `gold` 与 `bombs`

`gold` 是地面金币列表：

```json
{
  "position": [8, 8],
  "amount": 10
}
```

`bombs` 是炸弹位置列表：

```json
[[4, 5], [6, 7]]
```

两者均按位置排序导出。

## Actions

`actions` 记录策略和机制层给出的动作：

```json
{
  "players": {
    "1": {
      "actions": [4, 4, 4, 4, 4, 4],
      "k": 3,
      "order": 0,
      "vp": 0
    },
    "2": {
      "actions": [4, 4, 4, 4, 4, 4],
      "k": 3,
      "order": 0,
      "vp": 0
    }
  },
  "npcs": {
    "-1": [0, 3, 3]
  },
  "npc_order": [-1, -2, -3]
}
```

动作编码与官方 SDK 一致：

| 值 | 含义 |
| --- | --- |
| `0` | 上。 |
| `1` | 下。 |
| `2` | 左。 |
| `3` | 右。 |
| `4` | 停。 |

`players` 和 `npcs` 的 key 使用字符串形式。`npc_order` 表示本轮 NPC 执行顺序；没有 NPC 时为空数组或省略。

## Events

`events` 记录机制事件和规则事件：

```json
{
  "gold_generated": [],
  "movement": [],
  "interactions": [],
  "dispatch_order": [1, -1, -2, 2],
  "bomb_refresh": {
    "cleared": [],
    "spawned": []
  },
  "mechanisms": {}
}
```

### `gold_generated`

回合开始前生成的金币事件：

```json
{
  "position": [8, 8],
  "amount": 10
}
```

对 `envs.duel` 输出，这些金币已包含在 `start.grid` / `start.gold` 中。

### `bomb_refresh`

炸弹刷新事件：

```json
{
  "cleared": [[4, 5]],
  "spawned": [[6, 7]]
}
```

对 `envs.duel` 输出，刷新后的炸弹集合已包含在 `start.grid` / `start.bombs` 中。非刷新回合通常为两个空数组。

### `movement`

移动事件来自规则层：

```json
{
  "player_id": 1,
  "unit_id": 0,
  "action": 3,
  "from_pos": [0, 0],
  "to_pos": [0, 1],
  "status": "moved",
  "blocked_by": null
}
```

NPC 移动事件也复用该结构，但当前规则层导出为 `player_id=0`、`unit_id=<npc_id>`。更稳妥的轨迹渲染方式是读取 `actions.npcs` 并从 `start.npcs` 重放；`movement` 主要用于调试规则结算。

`status` 当前可能包含：

| 值 | 含义 |
| --- | --- |
| `moved` | 成功移动。 |
| `stayed` | 停留。 |
| `blocked_out_of_bounds` | 玩家尝试越界被阻挡。 |
| `blocked_obstacle` | 玩家尝试撞障碍被阻挡。 |
| `blocked_player` | 玩家被其它玩家单位阻挡。 |

NPC 生成器应只产生合法动作；NPC 越界或撞障碍在 simulator 中是 fail-fast，不应作为正常 replay 事件出现。

### `interactions`

每个移动步骤对应一个交互事件容器：

```json
{
  "pickups": [],
  "bomb_triggers": [],
  "tramples": []
}
```

拾金事件：

```json
{
  "actor": {
    "kind": "player_unit",
    "player_id": 1,
    "unit_id": 0,
    "npc_id": null
  },
  "position": [8, 8],
  "available_gold": 10,
  "picked_gold": 7,
  "remaining_gold": 3
}
```

炸弹触发事件：

```json
{
  "actor": {
    "kind": "npc",
    "player_id": null,
    "unit_id": null,
    "npc_id": -1
  },
  "position": [6, 7],
  "lost_gold": 0
}
```

踩踏处罚事件：

```json
{
  "actor": {
    "kind": "player_unit",
    "player_id": 1,
    "unit_id": 0,
    "npc_id": null
  },
  "position": [8, 8],
  "npc_count": 3,
  "penalty": 1
}
```

事件顺序与规则执行顺序一致。可视化器若只做帧展示，可以先忽略细粒度事件；若做 step playback，应按 `movement` 与 `interactions` 的数组顺序同步展示。

### `dispatch_order`

`dispatch_order` 记录本轮 actor 执行顺序：

```json
[1, -3, -1, -2, 2]
```

正数表示玩家 ID，负数表示 NPC ID。玩家条目代表该玩家两个单位按其 `GameOutput.order/k/actions` 完整执行；NPC 条目代表该 NPC 的三步动作执行。

## Snapshot

`snapshot` 字段为 simulator 生成的规则层快照，结构与 `simulator.types.Snapshot` 对齐：

```json
{
  "window_begin": 0,
  "window_end": 4,
  "regions": [
    {
      "id": 1,
      "enter": 0,
      "leave": 0,
      "gold_generated": 0,
      "gold_collected": 0,
      "gold_remaining": 0,
      "occupants": 0
    }
  ]
}
```

无新 snapshot 的回合为 `null`。可视化器可直接展示该对象，不需要重新计算窗口统计。

## 可视化器兼容建议

可视化器建议把输入归一化为内部模型：

```text
ReplayDocument
- kind: official | merged | simulator_full
- players
- maps
- rounds
```

Simulator full replay 的渲染建议：

- 基础资源层使用 `round.start.grid` 或 `round.end.grid`。
- 静态障碍可使用 `maps=1` 或状态 `grid=-1`，两者应一致。
- 玩家单位从 `state.players[].units[]` 渲染。
- NPC 从 `state.npcs[]` 渲染。
- 若需要表现重叠，在同一 cell 上叠加 actor 徽标，不要依赖 `grid=-2/-4`。
- 若需要玩家视野预览，可根据 `active_vision_radius` 和单位位置派生 mask；不要把 simulator full replay 当作带迷雾输入。
- 若需要逐步播放，可用 `actions` 从 `start` 重放，也可用 `events.movement` 做调试展示；以 `end` 作为最终状态校验。

当前生成的示例文件位于：

```text
temp/simulator_smoke_replays/stay_players_default_mechanisms_map1_seed2026073101.json
temp/simulator_smoke_replays/stay_players_default_mechanisms_map2_seed2026073102.json
temp/simulator_smoke_replays/stay_players_default_mechanisms_map3_seed2026073103.json
```

这些文件是本地临时观测资产，不属于稳定数据集路径；开发 visualizer 时可用于 smoke test。

## 版本演进

`format_version=1` 的稳定承诺：

- 顶层 `format` / `format_version` / `source` / `players` / `maps` / `rounds` 保持含义不变。
- `grid` 只表达障碍、炸弹、空地和金币，不写入玩家/NPC/迷雾派生标记。
- NPC 不出现 `gold` 字段。
- `rounds[].start` 和 `rounds[].end` 都是完整上帝视角状态帧。
- `actions.players` 和 `actions.npcs` 的 key 保持字符串形式。

未来若需要压缩存储、逐步动作帧、visualizer 专用派生层、或 official-like adapter，应新增字段或新增 adapter 格式；不要改变上述 canonical 字段的既有语义。
