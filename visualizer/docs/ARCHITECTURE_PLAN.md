# GoldRush2 可视化器架构方案

本文是 `visualizer/` 当前方案冻结点。若后续实现与本文冲突，应优先修正本文，而不是在代码中长期保留隐式约定。

## 目标

可视化器服务于本地 replay 回看与策略失败分析。第一版只读取 replay 文件，并把官方单视角与双账号合并格式统一投影为可显示的回合帧。

核心目标：

- 同时支持官方 NDJSON replay 和双账号合并 JSON replay。
- 用唯一 auto 视角展示 replay：官方文件展示文件自身视角，合并文件展示联合视野。
- 清楚展示未知信息边界。
- 用静态标注表达移动、拾取、炸弹、踩踏等关键事件。
- 保持实现足够薄，不引入 AntWar 那套特征工程和 evaluator overlay 复杂度。

## 数据流

```text
local replay file
  -> replay_io parser
  -> ReplayDocument
  -> DerivedStateBuilder
  -> FrameBundle[]
  -> PlaybackController
  -> UI scene layers
```

### `replay_io`

职责：

- 读取本地文件。
- 判断 replay 类型。
- 解析官方 NDJSON 与合并 JSON。
- 规范化静态地图中的字符串数字。
- 保留原始 round/view 对象，避免丢失未来分析所需字段。
- 输出 `ReplayDocument`。

不负责：

- 生成显示路径。
- 解释游戏事件。
- 绘制 UI。

### `model`

职责：

- 将 `ReplayDocument` 转为 `FrameBundle`。
- 自动选择显示数据源：official 使用 `raw_round`，merged 使用 `rounds[].merged`。
- 生成 `FrameState.start` 与 `FrameState.end`。
- 生成移动箭头、浮动文字、地图格标注和事件摘要。
- 管理置信度：`certain`、`partial`、`unknown`。

不负责：

- Qt 图元。
- 具体颜色、线宽、按钮状态。
- 平台接口。

### `ui`

职责：

- 播放、暂停、跳转、倍速。
- 图层开关。
- 地图、实体、箭头、标注和事件面板绘制。
- 选择实体并显示当前帧详情。

不负责：

- 解析原始 replay。
- 自己推导动作路径或事件来源。

## 内部模型

`ReplayDocument` 是 replay I/O 层输出的中立模型：

- `kind`: `official` 或 `merged`
- `players`: 玩家名映射
- `static_map`: `17x17` 静态地图，值已转为整数
- `rounds`: 规范化回合列表
- `forfeit`: 判负对象或空

`FrameBundle` 是 UI 消费的显示模型：

- `round_index`
- `start`: 回合开始帧
- `end`: 回合结束帧
- `motions`: 回合级路径/箭头
- `events`: 事件摘要

`FrameState` 包含：

- `grid`: 当前显示地面层
- `visible_by`: 观测来源矩阵；用于详情与未知边界，不作为第一版独立视图
- `players`: 毛金币、视野花费、净金币
- `entities`: 玩家单位和 NPC 的显示状态
- `cell_annotations`: 炸弹触发、踩踏等格子标记
- `floating_labels`: `+n`、`-n` 等实体头顶文字

## Auto 视角

第一版只支持 auto 视角，不提供玩家视角或来源 mask 切换。

- 官方 NDJSON replay：展示该文件本身包含的单视角数据。`viewer_side` 若由外部元信息提供，只作为详情信息，不触发视图切换。
- 双账号合并 JSON replay：只读取每轮 `merged` 层，展示联合视野。
- 合并 replay 顶层 `source.views[*].log_ref` 只作为来源引用，第一版可视化器不追读这些原始 NDJSON。

重要约束：

- `merged.grid` 不是上帝视角。
- `visible_by=0` 的格子必须显示为未知。
- 不可见实体不能通过规则猜位置。

## 动画标注

第一版不做连续 tween，而做回合级静态标注。

路径标注：

- `solid_polyline`: 起点、终点和逐格路径可信；用实线折线箭头。
- `dashed_endpoint`: 只可信起终点；用虚线箭头。
- `endpoint_only`: 路径不足以连线；只显示实体位置。

实体浮动文字：

- 拾取金币：`+n`
- 炸弹损失：`-n`
- 踩踏损失：`-n`

地图格标注：

- 炸弹触发：红色感叹号。
- 踩踏：拥挤或惩罚 marker。

这些语义必须由 `model` 输出 annotation；scene 只负责按 annotation 画出来。

## 当前实现状态

已实现：

- `replay_io` 能解析官方 NDJSON 与 `goldrush2_merged_replay` JSON。
- `model` 能按 auto 视角输出每回合 start/end 的地图、实体、分数和 `visible_by`。
- `model` 能生成实线/虚线箭头、`+n/-n`、炸弹触发和踩踏格标注。
- UI 能打开本地 replay、跳转回合、播放/暂停、切换 start/end、显示分数、实体和事件摘要。
- 已有标准库 `unittest` 覆盖 parser、derived state 和 playback。

后续优先事项：

1. 增加实体点击选择与右侧详情联动。
2. 增加 snapshot 区域表格。
3. 优化同格多实体、长路径箭头和文字避让。
4. 为真实 merged replay 样本补稳定测试资产或更完整的合成样本。

## 依赖说明

AntWar 使用 PySide6。当前 `goldrush` conda 环境尚未安装 PySide6，因此 GUI 实现前需要明确是否引入该依赖。

在引入 GUI 依赖之前，`replay_io` 和 `model` 应保持纯 Python，可在无 Qt 环境中测试。
