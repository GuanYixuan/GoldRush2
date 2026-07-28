# GoldRush2 Replay Visualizer

`visualizer/` 是 GoldRush2.0 的本地 replay 可视化器工作区。第一版只面向回放文件，不接入策略输入、不运行选手代码、不访问评测平台。

当前已实现一个 PySide6 MVP，可打开本地 official NDJSON 或 merged JSON replay。

## 输入范围

支持两类本地 replay：

- 官方 replay：`gamerules/replay.md` 描述的 NDJSON 格式。
- 双账号合并 replay：`docs/merged_replay_schema.md` 描述的 `goldrush2_merged_replay` JSON 格式。

暂不支持：

- 策略 `GameInput` 可视化。
- 本地策略运行或模拟器。
- 平台在线下载、提交或发起对局。
- 预测模型、特征工程或 evaluator overlay。

## 分层

第一版压缩为三层：

```text
replay_io/  # 读取 official NDJSON / merged JSON，并规范化为 ReplayDocument
model/      # ReplayDocument -> FrameBundle；生成路径、事件和静态标注
ui/         # 播放状态、窗口、scene 图层和用户交互
```

硬边界：

- `ui/` 不直接解析原始 replay。
- scene 绘制代码不重新推导游戏规则，只消费 `FrameBundle`。
- 所有“实线/虚线箭头、+n、红色感叹号”等显示语义都先在 `model/` 中变成 annotation。

## 动画与标注口径

第一版采用静态可解释标注，而不是连续补间动画：

- 行动路径可逐格还原：画实线折线箭头。
- 只有起终点或路径不完整：画虚线箭头连接起终点。
- 路径完全不可知：只显示终点或当前可见位置。
- 拾取金币：实体头顶显示 `+n`。
- 炸弹或踩踏损失：实体头顶显示 `-n`，并在事件面板说明来源。
- 炸弹触发格：地图格叠加红色感叹号。
- 合并 replay 的联合视野：必须区分未知格与已观察空地；`visible_by` 保留在详情数据中，第一版不做来源 mask 视图切换。

## 当前文件

```text
visualizer/
  main.py
  app.py
  replay_open.py
  replay_io/replay_types.py
  replay_io/replay_loader.py
  model/annotations.py
  model/derived_state.py
  ui/playback.py
  ui/main_window.py
  ui/board_view.py
  docs/ARCHITECTURE_PLAN.md
```

## 运行

需要在 `goldrush` conda 环境中运行。当前 GUI 依赖为 PySide6。

打开官方样本：

```bash
conda run -n goldrush python -m visualizer.main official_sdk/data/user/g3.txt
```

打开合并 replay：

```bash
conda run -n goldrush python -m visualizer.main mechanism/data/processed/merged_replays/symobs-a-map1-100-20260727-231446/36300.json
```

运行最小测试：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n goldrush python -m unittest tests.visualizer.test_replay_visualizer -v
```

## 后续实现顺序

1. 改进实体选择与右侧详情联动。
2. 增加 snapshot 区域表格。
3. 优化同格多实体、长路径箭头和文字避让。
4. 为真实 merged replay 样本补固定测试资产或更稳定的测试构造。
