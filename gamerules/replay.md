# GoldRush2.0 回放日志格式分析

本文记录 `/game/?id=<game_id>` 回放页使用的 `GET /api/user/get_game_log` 返回格式，以及前端如何解析和展示该日志。

当前样本来源：

- `game_id=22240`
- 对局：`sdkplayerA` vs `sdkplayerB`
- 地图：`map_id=2`，`地图2`
- 日志接口：`GET /api/user/get_game_log?id=22240`
- `Content-Type`: `application/x-ndjson; charset=utf-8`
- 样本长度：`1412182` 字符
- 样本行数：`502` 行

辅助对照样本：

- `official_sdk/data/full/g*.txt`
- `official_sdk/data/user/g*.txt`

## 字段分级

本文按字段含义来源分为四级：

| 分级 | 含义 |
| --- | --- |
| 官方对应 | 可与 `gamerules/gamerules.md` 中官方 PDF 规则或官方 SDK 结构直接对应。 |
| 平台确认 | 可由 `/game/` 前端解析逻辑、接口返回结构或页面展示方式确认，但不一定是官方规则语义。 |
| 数据推断 | 由样本日志、前端行为和游戏规则交叉推断，尚未由官方文本直接说明。 |
| 未确认 | 当前只能观察到字段存在，缺少足够样本或规则依据来确定含义。 |

当前字段总览：

| 字段 | 分级 | 当前含义 |
| --- | --- | --- |
| `player1` / `player2` | 平台确认 | 玩家名映射。 |
| 地图行 `0` / `1` | 平台确认 | `1` 被前端绘制为静态障碍，`0` 未绘制为障碍。 |
| 地图行 `2` | 未确认 | 样本存在，但前端未按静态障碍绘制；后续帧中可与玩家、NPC、金币、炸弹重合。 |
| `round` / `start` / `end` | 平台确认 | 回合日志结构。 |
| `snapshot` | 官方对应 | 对应官方区域快照，但 replay 字段形态与 SDK 结构略有差异。 |
| `grid` 中 `-5` / `-3` / `-1` / `0` / `>0` | 官方对应 | 迷雾、炸弹、障碍、空地、金币。 |
| `grid` 中 `-2` / `-4` | 数据推断 | full/replay 日志中的视觉占用标记，分别对应玩家角色、NPC；`official_sdk/data/user` 中未出现。 |
| `players[].id` | 平台确认 | 玩家编号，`1`/`2` 映射到 `player1`/`player2`。 |
| `players[].cost` | 数据推断 | 决策耗时，前端按时间单位展示；精确计量口径待确认。 |
| `players[].gold` | 官方对应 | 玩家毛金币总数。 |
| `players[].vision_spent` | 官方对应 | 视野花费。 |
| `players[].order` | 官方对应 | 玩家内两个角色执行顺序。 |
| `players[].units` | 官方对应 | 玩家控制的两个角色。 |
| `units[].position` | 官方对应 | 角色位置；不可见时平台可返回 `null`。 |
| `units[].gold` | 官方对应 | 单角色持有金币。 |
| `units[].actions` | 官方对应 | 本轮动作序列。 |
| `units[].pickup` | 数据推断 | 本轮拾取金币数量。 |
| `npcs[].id` / `position` | 官方对应 | NPC 标识和位置，对应官方 SDK 可见 NPC 信息。 |
| `npcs[].actions` / `pickup` / `gold` | 数据推断 | NPC 本轮动作、拾取和持币，对应规则中的 NPC 行动和争夺金币。 |
| `npcs[].cost` | 未确认 | 字段存在但所有已查样本均为 `0`；前端仅在缺少 `dispatch_order` 时用于行动排序兜底。 |
| `dispatch_order` | 数据推断 | 本轮玩家和 NPC 的实体行动顺序。 |
| `vision_r` | 数据推断 | 当前可见方的视野半径，`2/3/4` 可能对应 `5x5/7x7/9x9`。 |
| `overlap_events` | 未确认 | 字段存在，但 `official_sdk/data/user` 和 `official_sdk/data/full` 中均未观察到非空样本。 |
| `trample_events` | 数据推断 | 踩踏事件数组，字段可与拥挤踩踏规则对应。 |
| `burned` | 数据推断 | 本回合玩家角色金币损耗；full 样本中等于玩家角色持币流失，包含炸弹和踩踏损失。 |
| `forfeit` | 官方对应 | 判负信息，对应异常、超时、格式非法等判负规则。 |

## 总体格式

日志不是单个 JSON，而是 NDJSON：每一行都是一个独立 JSON 值。

行结构：

1. 第 1 行：玩家名对象。
2. 第 2 行：`17x17` 地图数组。
3. 后续每行：一轮日志对象。
4. 可选行：`forfeit` 对象。

前端解析后形成如下结构：

```json
{
  "players": {},
  "maps": [],
  "rounds": [],
  "forfeit": null
}
```

前端解析规则：

- 第一个非轮次对象作为 `players`。
- 第一个 `17x17` 数组作为 `maps`。
- 包含 `round`、`start`、`end` 的对象进入 `rounds`。
- 包含 `forfeit` 的对象写入 `forfeit`。

## 不包含选手程序输出

已用 `game_id=22240` 做过一次日志回传检查：该对局提交的 `official_sdk/code/player.py` 中包含多处 `print()`，输出特征包括 `round            =`、`my_units`、`snapshot_valid`、`grid (行=row, 列=col...)` 和分隔线。

检查结果：

- `GET /api/user/get_game_log?id=22240` 返回的 `502` 行均可按 JSON 解析，没有非 JSON stdout/stderr 行。
- 原始 NDJSON 中未出现上述 `print()` 输出特征。
- 回放对象字段中也未发现 `stdout`、`stderr`、`print`、`log`、`message` 等可承载选手程序输出的字段。
- 同步检查 `get_game_info` 和 `get_game_list_1`，除玩家名与对局元信息外，也没有选手程序标准输出内容。

因此，当前可见回放日志中没有选手程序 `print()` 输出的影子。后续异常模型实验也显示 stdout/stderr marker 未进入 `error_msg` 或 `get_game_log`；但异常 message 和非法返回值可进入回放末尾的 `forfeit` 对象，见“判负显示”。

## 玩家名行

分级：平台确认。

样本第 1 行：

```json
{
  "player1": "sdkplayerA",
  "player2": "sdkplayerB"
}
```

前端通过 `player1` / `player2` 映射日志中的玩家 ID：

- `id=1` 对应 `player1`
- `id=2` 对应 `player2`

## 地图行

分级：平台确认，`2` 为未确认。

样本第 2 行为 `17x17` 数组。

已观察取值：

- `0`: 空地或非静态障碍格。
- `1`: 前端按静态障碍物绘制。
- `2`: 样本中存在，但当前前端静态地图绘制逻辑未把它当障碍绘制。

样本计数：

```json
{
  "0": 245,
  "1": 24,
  "2": 20
}
```

前端绘制逻辑：

- 遍历 `maps[row][col]`。
- 仅当 `maps[row][col] === 1` 时绘制静态障碍物。
- 静态障碍物素材从 `方1`、`方2`、`方3`、`木箱`、`石头`、`花坛`、`障碍` 中随机选择。

静态 `2` 排查记录：

- `official_sdk/data/user/g*.txt` 和 `official_sdk/data/full/g*.txt` 的地图行中，静态 `2` 坐标集合一致，共 `20` 个。
- 这些坐标为：`(0,4)`、`(0,5)`、`(0,12)`、`(1,6)`、`(1,10)`、`(5,0)`、`(5,16)`、`(7,1)`、`(7,15)`、`(9,2)`、`(9,14)`、`(11,0)`、`(11,16)`、`(12,3)`、`(12,13)`、`(15,6)`、`(15,10)`、`(16,4)`、`(16,5)`、`(16,12)`。
- 在 `official_sdk/data/user/g*.txt` 的 `3940000` 次静态 `2` 坐标帧检查中，观察到金币 `111439` 次、玩家 `28140` 次、炸弹 `16595` 次、NPC `13586` 次。
- 在 `official_sdk/data/full/g0..g2.txt` 的 `60000` 次静态 `2` 坐标帧检查中，观察到金币 `13470` 次、炸弹 `2884` 次、NPC `1831` 次、玩家 `591` 次。
- 因此静态地图行 `2` 不表示障碍，也不表示与玩家、NPC、金币、炸弹互斥的格子。

## 轮次对象

分级：平台确认。

样本共有 `500` 个轮次对象，`round=0..499`。

每轮对象顶层字段：

```json
{
  "round": 0,
  "start": {},
  "end": {}
}
```

`start` 常见字段：

- `grid`
- `players`
- `npcs`
- `overlap_events`
- `npc`
- `vision_r`

`end` 常见字段：

- `grid`
- `players`
- `npcs`
- `overlap_events`
- `npc`
- `dispatch_order`
- `trample_events`
- `burned`

轮次顶层可选字段：

- `snapshot`

## Grid

分级：

- 官方对应：`-5`、`-3`、`-1`、`0`、正数。
- 数据推断：`-2`、`-4`。

`start.grid` 和 `end.grid` 都是 `17x17` 数组。

已确认或可对应官方规则的格子含义：

- `-5`: 迷雾。
- `-3`: 炸弹。
- `-1`: 障碍物。
- `0`: 空地。
- `> 0`: 金币数量。

样本中额外观察到的回放标记：

- `-2`: 推断为玩家角色所在格。
- `-4`: 推断为 NPC 所在格。

排查记录：

- `official_sdk/data/user/g*.txt` 共 `197` 个日志、`197000` 个 `start` / `end` 帧，未观察到任何 `-2` 或 `-4`。
- 因此 `-2` / `-4` 不应视作官方 SDK 玩家视角 `GameInput.grid` 的输入值。
- `official_sdk/data/full/g0..g2.txt` 共 `3000` 个 `start` / `end` 帧中，`-2` 均落在玩家角色位置上，未落在 NPC 位置上。
- 同一批 full 样本中，`-4` 均落在 NPC 位置上；若玩家与 NPC 重叠，玩家所在格也可能显示为 `-4`。
- full 样本中，实体所在格有时保留正数金币值而不是显示 `-2` / `-4`。因此判断实体位置应优先使用 `players[].units[].position` 和 `npcs[].position`，不要只依赖 `grid` 标记。

前端绘制逻辑：

- 炸弹格绘制炸弹素材。
- 迷雾格绘制遮罩。
- 正数格绘制金币和金币数量。
- 静态障碍物如果所在格变为迷雾，会被隐藏。

## Players

分级：

- 官方对应：`gold`、`vision_spent`、`order`、`units`。
- 平台确认：`id`。
- 数据推断：`cost` 的用途为耗时展示，精确计量口径待确认。

`start.players[]` 和 `end.players[]` 常见字段：

- `id`: 玩家 ID，通常为 `1` 或 `2`。
- `cost`: 决策耗时，前端会格式化为 `ns` / `μs` / `ms` / `s`。
- `gold`: 该玩家毛金币总数。
- `vision_spent`: 视野花费。
- `order`: 玩家内两个角色执行顺序。
- `units`: 角色数组。

页面当前成绩显示公式：

```text
netGold = gold - vision_spent
```

胜负排序：

- 无 `forfeit` 时，按 `netGold` 从高到低排序。
- 有 `forfeit` 时，被判负玩家排名为 2，另一方排名为 1。

样本最终净金币：

- `sdkplayerA`: `142 - 198 = -56`
- `sdkplayerB`: `127 - 198 = -71`

这与 `get_game_list_1.players[].coin_num` 中的 `-56`、`-71` 对应。

## Units

分级：

- 官方对应：`position`、`gold`、`actions`。
- 数据推断：`pickup`。

`units[]` 常见字段：

- `position`: `[row, col]`，角色位置；不可见或不可用时可能为 `null`。
- `gold`: 单角色金币。
- `actions`: 本轮动作数组。
- `pickup`: 本轮拾取金币数量。

动作编码：

| 值 | 前端显示 |
| --- | --- |
| `0` | 上 |
| `1` | 下 |
| `2` | 左 |
| `3` | 右 |
| `4` | 停 |

前端日志面板会计算每个角色本轮金币变化：

```text
delta = end_unit.gold - start_unit.gold
```

## NPCs

分级：

- 官方对应：`id`、`position`。
- 数据推断：`actions`、`pickup`、`gold`。
- 未确认：`cost`。

`start.npcs` 和 `end.npcs` 为 NPC 数组。兼容旧格式时，前端也会读取单个 `npc` 字段并转成数组。

NPC 常见字段：

- `id`: NPC ID，样本中为负数，如 `-5`、`-3`。
- `position`: `[row, col]`。
- `actions`: 本轮动作数组。
- `pickup`: 本轮拾取金币数量。
- `gold`: NPC 持有金币。
- `cost`: NPC 行动耗时或成本字段。

`cost` 排查记录：

- `official_sdk/data/user/g*.txt` 中，`npcs[]` 共 `279989` 条，全部包含 `cost`，取值全部为 `0`。
- `official_sdk/data/full/g0..g2.txt` 中，`npcs[]` 共 `21000` 条，全部包含 `cost`，取值全部为 `0`。
- 平台可见已解析对局 `22240`、`22096`、`12425`、`12121` 中，`npcs[]` 共 `2716` 条，全部包含 `cost`，取值全部为 `0`。
- 兼容旧格式的单个 `npc` 字段也全部为 `cost=0`。
- `/game/` 前端在存在 `end.dispatch_order` 时优先按 `dispatch_order` 排行动画；缺少 `dispatch_order` 时才回退为按 `cost` 排序。
- 因此目前只能确认 `cost` 是 replay NPC 对象中的保留排序字段；不能确认它是否有真实规则语义。

## 行动顺序

分级：数据推断。

`end.dispatch_order` 是本轮实体执行顺序数组。

样本第 1 轮末尾：

```json
{
  "dispatch_order": [2, -6, -5, -3, -1, -4, -7, -2, 1]
}
```

前端动画排序优先使用 `dispatch_order`：

- `1`、`2` 表示玩家。
- 负数表示 NPC。
- 找不到顺序时，前端回退到玩家内 `order` 和单位序号排序。

## 事件字段

分级：

- 数据推断：`trample_events`、`burned`。
- 未确认：`overlap_events`。

已观察事件字段：

- `overlap_events`: 事件数组；当前未观察到非空样本，具体含义未确认。
- `trample_events`: 踩踏事件数组。
- `burned`: 本回合玩家角色金币损耗，前端指标面板会累计各回合 `end.burned`。

`overlap_events` 排查记录：

- `official_sdk/data/user/g*.txt` 共 `197` 个日志、`98500` 个轮次。
- 每个轮次的 `start.overlap_events` 和 `end.overlap_events` 字段都存在，但事件数均为 `0`。
- 对照 `official_sdk/data/full/g0..g2.txt` 共 `3` 个日志、`1500` 个轮次，同样未观察到非空 `overlap_events`。
- 同一批 user 日志中，`end.trample_events` 观察到 `201` 条事件，说明 `overlap_events` 全空不是因为事件字段整体缺失。
- 因此当前不能从 SDK 样本推断 `overlap_events[]` 的字段结构或规则语义，也不应把它等同于踩踏事件。

已观察 `trample_events[]` 字段：

- `round`: 事件关联轮次；是否总是等于外层 `round` 待确认。
- `pos`: 触发踩踏的位置。
- `unit_owner`: 被处罚玩家 ID。
- `npc_count`: 该位置 NPC 数量。
- `penalty`: 处罚金币数。

`burned` 排查记录：

- `burned` 只出现在 `end`，未在 `start` 中观察到。
- `official_sdk/data/user/g*.txt` 共 `98500` 个轮次，`end.burned` 全部存在；非零 `16388` 次，总和 `234835`。
- `official_sdk/data/full/g0..g2.txt` 共 `1500` 个轮次，`end.burned` 全部存在；非零 `239` 次，总和 `5736`。
- 平台可见已解析对局 `22240`、`22096`、`12425`、`12121` 共 `1500` 个轮次，`end.burned` 全部存在；非零 `197` 次，总和 `1534`。
- 在 full 样本中，对每轮按玩家角色计算 `sum(start_unit.gold + end_unit.pickup - end_unit.gold)`，结果与 `end.burned` 在 `1500/1500` 个轮次完全一致。
- full 样本中唯一观察到的踩踏处罚轮，`trample_events[].penalty` 合计 `30`，同轮 `end.burned=30`，说明踩踏损失被计入 `burned`。
- 由于官方规则中还存在炸弹损失，且大量 `burned>0` 轮没有 `trample_events`，这些非踩踏损耗可推断为炸弹导致的玩家金币损失。
- 因此 `burned` 应理解为“本回合玩家角色金币损耗总额”，不是累计值；累计损耗需对每轮 `end.burned` 求和。

## Snapshot

分级：官方对应。

`snapshot` 可出现在轮次顶层，而不是 `start` 或 `end` 内。该字段对应官方 `Snapshot` / `RegionStat` 快照规则。

已观察字段：

- `round`: 快照发布轮。
- `window`: 统计窗口，形如 `[0, 4]`。
- `regions`: 5 个区域统计对象。

`regions[]` 字段与官方 `RegionStat` 对应：

- `id`: 区域编号 `1..5`。
- `enter`: 窗口内进入该区域的角色次数。
- `leave`: 窗口内离开该区域的角色次数。
- `gold_generated`: 窗口内该区域生成的金币总量。
- `gold_collected`: 窗口内该区域被拾取的金币总量。
- `gold_remaining`: 该区域地面当前剩余金币。
- `occupants`: 该区域当前角色数。

## 样本末轮摘要

样本第 500 轮末尾摘要：

```json
{
  "round": 499,
  "players": [
    {
      "id": 1,
      "cost": 3533362,
      "gold": 142,
      "vision_spent": 198,
      "order": 0,
      "units": [
        {
          "position": [15, 14],
          "gold": 96,
          "actions": [3, 4, 0, 4],
          "pickup": 2
        },
        {
          "position": [9, 12],
          "gold": 46,
          "actions": [4, 0],
          "pickup": 0
        }
      ]
    },
    {
      "id": 2,
      "cost": 443125,
      "gold": 127,
      "vision_spent": 198,
      "units": [
        {
          "gold": 96,
          "position": null
        },
        {
          "gold": 31,
          "position": null
        }
      ]
    }
  ],
  "npcs_count": 2,
  "burned": 0,
  "overlap_events": [],
  "trample_events": []
}
```

## 回放交互

回放页面行为：

- 画布：`canvas#gameCanvas`，`1000x1000`。
- 默认按每 20 轮分组显示日志选择项。
- 点击单轮“回放”会跳转到对应轮。
- 上一轮、播放/暂停、下一轮按钮仅在游戏加载完成后显示。
- 下载日志按钮会将当前日志保存为 `game_<id>.log`。
- 管理员可选择播放速度：`1x`、`2x`、`3x`、`5x`、`8x`、`10x`、`15x`、`20x`。

## 判负显示

日志中若存在 `forfeit` 行，前端会显示判负横幅。

`forfeit` 解析结果：

```json
{
  "round": 123,
  "player_id": 1,
  "reason": "runtime_error"
}
```

原因映射：

- `exception` / `runtime_error` / `segfault`：运行时错误。
- `exit`：主动退出。
- `time_pool_exhausted`：超时判负。
- 其他：返回格式错误。

错误详情仅在当前用户是判负方或管理员时显示，可能包含：

- `exception_repr`
- `raw_return`
- `exit_code`

异常回传实验：

- 实验资产位于 `temp/error_msg_probe_20260724/`。
- A: `game_id=22373`，`raise RuntimeError("GR2_EXC_MARKER_A1_R0")` 后，`get_game_info.error_msg` 为空；`get_game_log` 第 3 行为 `forfeit`，其中 `exception_repr` 包含该 marker。
- B: `game_id=22375`，先写 stdout/stderr 再抛异常；`error_msg` 为空，`forfeit.exception_repr` 仅包含异常 marker，stdout/stderr marker 均未出现。
- C: `game_id=22376`，返回 `["GR2_RAW_RETURN_MARKER_C1_R0"]`；`error_msg` 为空，`forfeit.raw_return` 包含非法返回 marker，`reason` 为 `length_mismatch`。
- LEN_EXC_256K: `game_id=22497`，抛出 `262144` 字符 ASCII payload；`forfeit.exception_repr` 字段长度为 `1024`，保留 `BEGIN`，不保留 `END`。
- LEN_RAW_256K: `game_id=22498`，返回 `262144` 字符 ASCII payload；`forfeit.raw_return[0]` 字段长度为 `64`，保留 `BEGIN`，不保留 `END`。

当前结论：列表级 `error_msg` 不是本次异常模型的回传通道；可见通道是可解析回放末尾的 `forfeit.exception_repr` 与 `forfeit.raw_return`。该通道会导致立即判负；在 256KB 纯 ASCII 长 payload 实验中，`exception_repr` 最终字段截断到 `1024` 字符，`raw_return[0]` 最终字段截断到 `64` 字符。转义规则和可见性边界仍需单独测试。
