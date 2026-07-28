# 双视角合并 Replay Schema

本文定义离线合并同一对局两个玩家视角 replay 时使用的派生格式。该格式服务于后续数据分析和可视化器，不是评测平台官方 `GET /api/user/get_game_log` 的返回格式。

官方 replay 格式仍以 `gamerules/replay.md` 为准。本文只描述在保留官方 replay 语义的前提下，如何表达“双视角合并回放”。

## 设计目标

- 兼容普通官方 replay 和双视角合并 replay。
- 通过顶层 `source.views[*].log_ref` 引用两个原始玩家视角，便于复查和重跑合并逻辑。
- 显式记录每个合并值的观测来源，避免把联合视野误当作上帝视角。
- 对理论上应一致的字段采用严格校验；若出现不一致，脚本应直接失败并暴露样本，而不是在合并产物中长期保存冲突状态。
- 允许可视化器在单视角、双视角、联合视野和来源 mask 间切换。

## 基本判断

官方平台下载的 replay 是“基于玩家视角的”日志：

- `viewer_side=1` 时，`player1` 信息较完整，`player2` 受视野裁剪。
- `viewer_side=2` 时，`player2` 信息较完整，`player1` 受视野裁剪。
- 不可见格通常在 `grid` 中表现为 `-5`。
- 不可见对方单位可能表现为 `position:null`，并缺少 `actions` / `pickup`。
- `viewer_side` 来自 `get_game_info`，不在 `get_game_log` NDJSON 正文中。

因此，不能简单把两个 `grid` 拼成一个普通 `grid` 后仍称为官方 replay。双视角合并 replay 应视为官方 replay 的派生 superset。

## 文件组织

合并脚本应贴合 `mechanism/data/raw` 中已有下载格式。默认输入：

```text
mechanism/data/raw/observer_runs/<run_id>/main/replays/<game_id>.ndjson
mechanism/data/raw/observer_runs/<run_id>/observer/replays/<game_id>.ndjson
```

默认输出：

```text
mechanism/data/processed/merged_replays/<run_id>/<game_id>.json
```

第一版不需要抽象成任意输入布局。若后续有独立可视化器、外部样本导入或批量数据集发布需求，再增加显式 `--view1-log` / `--view2-log` 等入口。

## 顶层格式

推荐使用单个 compact JSON 文档保存合并结果，而不是继续伪装成官方 NDJSON。合并产物不嵌入完整原始 round；原始双视角 replay 通过 `source.views[*].log_ref` 外部引用。

```json
{
  "format": "goldrush2_merged_replay",
  "format_version": 1,
  "source": {
    "game_id": 12345,
    "map_id": 2,
    "run_id": "symobs-a-map1-100-20260727-231446",
    "views": {
      "1": {
        "viewer_side": 1,
        "source_type": "platform_user_replay",
        "account_dir": "observer",
        "log_ref": "mechanism/data/raw/observer_runs/<run_id>/observer/replays/<game_id>.ndjson"
      },
      "2": {
        "viewer_side": 2,
        "source_type": "platform_user_replay",
        "account_dir": "main",
        "log_ref": "mechanism/data/raw/observer_runs/<run_id>/main/replays/<game_id>.ndjson"
      }
    }
  },
  "players": {
    "player1": "modelA",
    "player2": "modelB"
  },
  "maps": [],
  "rounds": [],
  "forfeit": null
}
```

字段说明：

| 字段 | 必填 | 含义 |
| --- | --- | --- |
| `format` | 是 | 固定为 `goldrush2_merged_replay`。 |
| `format_version` | 是 | 当前为 `1`。 |
| `source.game_id` | 建议 | 平台对局 ID。若来自 SDK 样本或脱敏数据，可为空或省略。 |
| `source.map_id` | 建议 | 平台地图 ID。 |
| `source.run_id` | 建议 | `observer_runs` 下的采集批次 ID。 |
| `source.views` | 是 | 两个原始玩家视角的来源信息。key 使用玩家编号字符串 `"1"` / `"2"`。 |
| `players` | 是 | 与官方 replay 第 1 行一致的玩家名映射。 |
| `maps` | 是 | 与官方 replay 第 2 行一致的静态地图。 |
| `rounds` | 是 | 合并后的轮次数组。 |
| `forfeit` | 否 | 判负信息。若两个视角存在，应校验一致后保留。 |

## 轮次格式

每个轮次只包含派生合并层：

```json
{
  "round": 0,
  "merged": {
    "start": {},
    "end": {},
    "snapshot": {}
  }
}
```

规则：

- 原始轮次对象不嵌入合并产物；如需复查，按顶层 `source.views[*].log_ref` 读取原始 NDJSON。
- `round` 编号必须在两个视角中一致，否则拒绝生成。
- `merged` 是派生数据，不能反向覆盖原始 replay。

## 合并帧格式

`merged.start` 和 `merged.end` 使用同一结构：

```json
{
  "grid": [],
  "visible_by": [],
  "players": [],
  "npcs": [],
  "overlap_events": [],
  "trample_events": [],
  "burned": 0,
  "dispatch_order": []
}
```

### `grid`

`merged.grid` 是双方可见格的联合结果：

- 双方都不可见：保留 `-5`。
- 仅一方可见：取该方 `grid` 值。
- 双方都可见且值一致：取该值。
- 双方都可见但值不一致：视为数据异常，脚本应直接失败并报告 `game_id`、`round`、`phase`、坐标和两侧取值。

注意：双视角联合 `grid` 仍不是上帝视角。双方都不可见的金币、炸弹、NPC 和单位状态仍然缺失。

### `visible_by`

`visible_by` 是 `17x17` 整数矩阵，记录每个格子的观测来源：

| 值 | 含义 |
| --- | --- |
| `0` | 双方都不可见。 |
| `1` | 仅 `viewer_side=1` 的 replay 可见。 |
| `2` | 仅 `viewer_side=2` 的 replay 可见。 |
| `3` | 两个 replay 都可见。 |

判断可见性的默认规则：

```text
cell_visible = grid[row][col] != -5
```

若未来发现平台在特殊字段中提供更精确的 visibility mask，应新增字段保留原始 mask，不应改变 `grid=-5` 的历史含义。

## 实体合并

实体包括 `players[].units[]` 和 `npcs[]`。合并实体时应按实体稳定 ID 对齐：

- 玩家按 `players[].id` 和 unit 下标对齐。
- NPC 按 `npcs[].id` 对齐。

合并后的实体建议增加来源字段：

```json
{
  "id": 1,
  "units": [
    {
      "position": [0, 0],
      "gold": 0,
      "actions": [1, 1, 4],
      "pickup": 0,
      "visible_by": 1,
      "position_source": 1,
      "actions_source": 1,
      "is_complete": true
    }
  ]
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `visible_by` | 该实体本帧被哪些视角观察到，编码同格子 `visible_by`。 |
| `position_source` | 当前 `position` 取自哪个视角；双方一致可用 `3`。 |
| `actions_source` | 当前 `actions` 取自哪个视角；双方一致可用 `3`。 |
| `is_complete` | 是否可认为该实体本帧信息完整。玩家本方视角的本方单位通常可为 `true`；仅被对方局部看见的单位或 NPC 不应默认完整。 |

对玩家单位：

- `viewer_side=1` 的 replay 通常能完整提供 `player1` 单位。
- `viewer_side=2` 的 replay 通常能完整提供 `player2` 单位。
- 对方单位即使可见，也可能只提供局部动作信息；不要用可见片段推断整轮完整轨迹。

对 NPC：

- 只合并实际出现在某个视角 `npcs[]` 中的 NPC。
- 某个 NPC 不在任一视角中出现时，表示本合并 replay 无法观察，不表示该 NPC 不存在。
- NPC `actions` 只代表可见 replay 提供的信息，不应自动视为全局完整轨迹。

## 事件与统计字段

事件字段应优先保留官方原始视角，并在 `merged` 中提供校验后的派生值。

建议规则：

- `dispatch_order`：两个视角应一致；不一致时拒绝生成。
- `burned`：两个视角应一致；不一致时拒绝生成。
- `trample_events`：按事件内容去重合并，同时保留来源。
- `overlap_events`：当前语义未确认，保留原始视角，合并层可为空或按完全相同事件去重。
- `snapshot`：如果两个视角提供同一轮 `snapshot` 且内容一致，写入 `merged.snapshot`；不一致时拒绝生成。不要用对称性补齐单局单轮真实状态。

## 严格校验

第一版合并格式不设置 `conflicts` 字段。若两个视角在理论上应一致的位置出现不一致，合并脚本应 fail fast。

必须严格一致的字段：

- 玩家名映射。
- 静态地图。
- 轮次数量与每轮 `round` 编号。
- 双方共同可见格的 `grid` 值。
- 双方同时提供的玩家级标量字段：`gold`、`vision_spent`、`cost`、`order`。
- `dispatch_order`、`burned`、`snapshot` 等全局字段。

本地已对 `mechanism/data/raw/observer_runs` 中 201 个双视角对局做过初步扫描：双方共同可见的 `grid` cell 共 `3225372` 个，未发现取值冲突；双方同时提供的玩家级 `gold`、`vision_spent`、`cost`、`order` 也未发现冲突。因此第一版采用 assert 语义是合理的。

## 可视化器兼容建议

可视化器建议先把输入归一化为内部模型：

```text
ReplayDocument
- kind: official | merged
- players
- maps
- rounds
```

普通官方 replay：

- `kind = official`
- 只存在一个 `views[viewer_side]`
- 可按需即时生成单视角 `visible_by`

双视角合并 replay：

- `kind = merged`
- 顶层 `source.views` 引用原始双视角 replay；每轮只保存派生 `merged`
- UI 可提供 `player1视角`、`player2视角`、`联合视野`、`来源mask` 模式

可视化器不应假设 `merged.grid` 是完整世界状态；展示联合视野时应能区分“未知”和“已观察为空地”。

## 版本演进

`format_version=1` 的稳定承诺：

- 顶层 `format` / `format_version` / `source` / `players` / `maps` / `rounds` 保持含义不变。
- 顶层 `source.views` 保留原始官方 replay 的来源引用。
- `rounds[].merged` 只保存派生数据。
- `visible_by` 编码 `0/1/2/3` 不改变。

未来若需要引入逐步动作级 visibility、压缩存储、实体历史索引或更精确的冲突解析，应增加新字段或提升 `format_version`，不要改变上述字段的既有语义。
