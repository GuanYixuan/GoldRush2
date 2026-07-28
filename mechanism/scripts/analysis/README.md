# 离线分析脚本

本目录存放机理建模使用的离线 replay 解析、合并和统计脚本。默认不访问评测平台，不读取登录 key。

## `merge_dual_view_replay.py`

合并同一 `observer_runs/<run_id>` 下 `main` 与 `observer` 两个玩家视角 replay，输出双视角合并 replay。格式定义见 `docs/merged_replay_schema.md`。

默认输入：

```text
mechanism/data/raw/observer_runs/<run_id>/main/replays/<game_id>.ndjson
mechanism/data/raw/observer_runs/<run_id>/observer/replays/<game_id>.ndjson
```

默认输出：

```text
mechanism/data/processed/merged_replays/<run_id>/<game_id>.json
```

常用命令：

```bash
conda run -n goldrush python mechanism/scripts/analysis/merge_dual_view_replay.py --run-id <run_id>
```

只校验不写出：

```bash
conda run -n goldrush python mechanism/scripts/analysis/merge_dual_view_replay.py --run-id <run_id> --dry-run
```

只处理指定对局：

```bash
conda run -n goldrush python mechanism/scripts/analysis/merge_dual_view_replay.py --run-id <run_id> --game-id <game_id>
```

脚本会从 replay 的 `start.vision_r` 推断 `main` / `observer` 分别对应的 `viewer_side`，不硬编码账号目录到玩家编号的映射。双方共同可见格和全局字段若出现不一致，脚本会直接失败。

## `extract_gold_features.py`

从双视角合并 replay 抽取金币生成建模特征表。脚本使用 `end[t-1] -> start[t]` 的干净可见格 transition 作为 cell 级观测，并按 `snapshot.window` 的实测对齐方式写入窗口聚合：

```text
window_begin = floor((t - 1) / 5) * 5
```

默认输入：

```text
mechanism/data/processed/merged_replays/<run_id>/*.json
```

默认输出：

```text
mechanism/data/processed/gold_features/<run_id>/
```

输出文件：

- `cell_transitions.csv`：逐干净可见格 transition 明细，包含 `delta`、区域、静态地图值、可见来源和窗口编号。
- `snapshot_windows.csv`：官方 snapshot 的区域窗口统计。
- `window_observed.csv`：由 cell transition 聚合得到的区域窗口观测值。
- `cell_summary.csv`：按格子聚合的 exposure、事件数、生成量和金额分布。
- `summary.json`：本次抽取计数与输出路径。

常用命令：

```bash
conda run -n goldrush python mechanism/scripts/analysis/extract_gold_features.py --run-id <run_id>
```

如果只需要窗口和格子聚合，不写逐 transition 明细：

```bash
conda run -n goldrush python mechanism/scripts/analysis/extract_gold_features.py --run-id <run_id> --skip-transitions
```

## `extract_bomb_features.py`

从双视角合并 replay 抽取炸弹生成建模特征表。脚本使用 `end[t-1] -> start[t]` 的连续可见格 transition 作为生成相位观测，`prev_grid != -3 && curr_grid == -3` 记为可见新增炸弹。

默认输入：

```text
mechanism/data/processed/merged_replays/<run_id>/*.json
```

默认输出：

```text
mechanism/data/processed/bomb_features/<run_id>/
```

输出文件：

- `bomb_cell_transitions.csv`：逐连续可见格 transition 明细，包含炸弹出现/消失、金币和可见占用信息。
- `bomb_round_region.csv`：按 game / round / region 聚合的炸弹出现、消失和存量计数。
- `bomb_cell_summary.csv`：按格子聚合的 exposure、炸弹出现/消失次数和约束检查计数。
- `summary.json`：本次抽取计数与输出路径。

常用命令：

```bash
conda run -n goldrush python mechanism/scripts/analysis/extract_bomb_features.py --run-id <run_id>
```

如果只需要聚合表，不写逐 transition 明细：

```bash
conda run -n goldrush python mechanism/scripts/analysis/extract_bomb_features.py --run-id <run_id> --skip-transitions
```

## `extract_npc_features.py`

从双视角合并 replay 抽取 NPC 行为建模特征表。脚本只使用可连续观察的 NPC 轨迹做 round / step 级样本，即同一轮 `start.npcs[id]` 与 `end.npcs[id]` 均可见且 `end.npcs[id].actions` 存在。NPC 与可见炸弹的交互按 `dispatch_order` 保守重放，未观测到的炸弹不会被补全。

默认输入：

```text
mechanism/data/processed/merged_replays/<run_id>/*.json
```

默认输出：

```text
mechanism/data/processed/npc_features/<run_id>/
```

输出文件：

- `npc_round_observations.csv`：每个连续可见 NPC round 一行，包含起终点、区域、动作、调度序号、拾金和局部环境特征。
- `npc_step_events.csv`：每个 NPC step 一行，包含动作前后位置、目标格、到可见金币/炸弹/玩家的距离变化。
- `npc_dispatch_orders.csv`：每轮 NPC 内部顺序和首/慢玩家。
- `npc_interactions.csv`：NPC pickup、可见炸弹触发、终点重叠等事件。
- `summary.json`：本次抽取计数与输出路径。

常用命令：

```bash
conda run -n goldrush python mechanism/scripts/analysis/extract_npc_features.py --run-id <run_id>
```
