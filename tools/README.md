# 本地工具说明

本目录放置仓库内可复用的辅助脚本。新增或修改工具时，应同步更新本文档，说明用途、是否会访问评测平台、是否会发起提交/对局，以及常用命令。

## 安全原则

- 默认优先做只读工具；会提交策略、发起对局或修改平台状态的工具必须在文档中明确标注。
- 平台登录 key 来自仓库根目录 `.env` 中的 `LOGIN_KEY`，工具不得打印 key 或 token。
- 下载的 replay 日志建议缓存到 `/tmp` 或其它 gitignore 路径，不要把大量平台日志直接写入仓库。

## `platform_games.py`

只读平台数据工具，用于查询公测对局列表、下载 replay 日志并计算常用统计指标。该工具不会提交策略，也不会发起对局。

### 常用命令

列出某模型相关对局：

```bash
python3 tools/platform_games.py list --since '2026-07-26 18:58:48' --model player142
```

统计指定对局双方指标：

```bash
python3 tools/platform_games.py stats 31240 31276 32590
```

只显示某个模型的指标：

```bash
python3 tools/platform_games.py stats 32586 31535 32431 --focus player142
```

强制重新下载日志：

```bash
python3 tools/platform_games.py stats 31240 --refresh
```

### 输出指标

`list` 输出：

- `game_id`
- 北京时间
- 地图
- 结果
- 对手
- 金币
- 平台解析状态

`stats` 输出：

- 毛金币、视野花费、净金币
- P50 / P90 / P99 / Avg / Max 决策耗时
- `forfeit` 行数
- 先手回合数与先手比

先手比口径：每回合优先使用 replay 的 `end.dispatch_order`，取其中第一个正数玩家 id 作为该回合先手；缺少 `dispatch_order` 时，用双方 `cost` 较小者兜底。

### 缓存

默认将 replay 日志缓存到：

```text
/tmp/goldrush_logs/
```

缓存路径可通过 `--cache-dir` 修改。缓存只用于减少重复下载，不作为长期数据源。
