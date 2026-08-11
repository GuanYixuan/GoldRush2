# GoldRush2 Replay Visualizer

`visualizer/` 是 GoldRush2.0 的本地 replay 可视化器工作区。它只面向本地回放文件，不接入策略输入、不运行选手代码、不访问评测平台。

当前已实现 PySide6 GUI，可打开本地 official NDJSON、merged JSON 或 simulator full JSON replay。

## 输入范围

支持三类本地 replay：

- 官方 replay：`gamerules/replay.md` 描述的 NDJSON 格式。
- 双账号合并 replay：`docs/merged_replay_schema.md` 描述的 `goldrush2_merged_replay` JSON 格式。
- simulator replay：`docs/simulator_replay_schema.md` 描述的 `goldrush2_simulator_full_replay` JSON 格式。

暂不支持：

- 策略 `GameInput` 可视化。
- 本地策略运行。
- 平台在线下载、提交或发起对局。
- 预测模型、特征工程或 evaluator overlay。

## 分层

当前实现分为三层：

```text
replay_io/  # 读取 official NDJSON / merged JSON / simulator JSON，并规范化为 ReplayDocument
model/      # ReplayDocument -> FrameBundle；生成路径、事件和静态标注
ui/         # 播放状态、窗口、scene 图层和用户交互
```

硬边界：

- `ui/` 不直接解析原始 replay。
- scene 绘制代码不重新推导游戏规则，只消费 `FrameBundle`。
- 所有“实线/虚线箭头、+n、红色感叹号”等显示语义都先在 `model/` 中变成 annotation。

## 动画与标注口径

当前采用静态可解释标注，而不是连续补间动画：

- 行动路径可逐格还原：画实线折线箭头。
- 只有起终点或路径不完整：画虚线箭头连接起终点。
- 路径完全不可知：只显示终点或当前可见位置。
- 拾取金币：实体头顶显示 `+n`。
- 炸弹或踩踏损失：实体头顶显示 `-n`，并在事件面板说明来源。
- 炸弹触发格：地图格叠加红色感叹号。
- 合并 replay 的联合视野：必须区分未知格与已观察空地；`visible_by` 保留在详情数据中，当前不做来源 mask 视图切换。
- simulator replay 使用上帝视角，不显示迷雾；移动、拾金、炸弹和踩踏优先使用 `events` 中的确定事件。

## 运行

需要在 `goldrush` conda 环境中运行。当前 GUI Python 依赖为 PySide6；在 Ubuntu/Debian 容器中还需要 Qt/PySide6 的系统运行库：

```bash
apt-get update && apt-get install -y \
  libgl1 libegl1 \
  libxkbcommon0 libxkbcommon-x11-0 \
  libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libxcb-xkb1
```

可视化器需要图形显示端。若在容器中运行，请先配置桌面、X11 forwarding 或 VNC；仅做无界面加载检查时可临时使用 `QT_QPA_PLATFORM=offscreen`。

打开官方样本：

```bash
conda run --no-capture-output -n goldrush \
  python -m visualizer.main official_sdk/data/user/g3.txt
```

打开合并 replay：

```bash
conda run --no-capture-output -n goldrush \
  python -m visualizer.main mechanism/data/processed/merged_replays/symobs-a-map1-100-20260727-231446/36300.json
```

打开 simulator replay：

```bash
conda run --no-capture-output -n goldrush \
  python -m visualizer.main temp/simulator_smoke_replays/stay_players_default_mechanisms_map1_seed2026073101.json
```

运行最小测试：

```bash
PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n goldrush \
  python -m unittest tests.visualizer.test_replay_visualizer -v
```

## 维护注意

- 新增 replay 输入格式时，先扩展 `replay_io/` 的规范化层，再让 `model/` 和 `ui/` 消费统一结构。
- 显示语义应先在 `model/` 中形成 annotation，不要在 scene 绘制代码里重新推导游戏规则。
- 若 simulator replay schema 或 merged replay schema 改动，需要同步检查本 README 和对应 loader 测试。
