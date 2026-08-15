# GoldRush2 本地模拟器

`simulator/` 是 GoldRush2.0 的本地游戏环境包，服务于规则单测、replay 对齐、策略调试、PPO rollout 和本地 replay 生成。实现时以 `gamerules/gamerules.md` 为规则事实来源；官方未公开机制只能作为可替换 approximation，不得写死成官方规则。

## 模块边界

模拟器负责：

- 已确认规则层：移动、碰撞、拾取、炸弹触发、踩踏、视野、snapshot、净金币和胜负结算。
- 未公开机制层的可替换近似：地图池、初始金币、中心/外围金币生成、炸弹刷新和 NPC 策略。
- 玩家视角 observation：生成接近官方 SDK 形状的 `GameInput`。
- 本地环境接口：round-level stepping、duel runner、结构化事件和 simulator full replay。

## 当前结构

```text
simulator/
  constants.py      # 棋盘尺寸、动作编码、grid 值、规则常量
  types.py          # GameInput/GameOutput、位置、事件等轻量对象
  state.py          # 完整上帝视角 GameState / PlayerState / UnitState / NpcState
  config.py         # RulesConfig、EpisodeConfig、机制配置
  errors.py         # 规则、配置和对齐错误
  rules/            # deterministic 官方规则层
  mechanisms/       # 地图、金币、炸弹、NPC 等 approximation
  observation/      # 从 full state 生成玩家视角 SDK observation
  envs/             # round_step 和 duel runner
  replay/           # simulator full replay recorder/export/schema
```

测试位于 `tests/simulator/`。

## 规则层

`rules/` 只实现已确认规则，输入必须是完整状态和已给定的动作/机制事件，不在内部随机采样金币、炸弹、NPC 动作或地图。

当前关键口径：

- 玩家一回合总共 6 个 primitive step，由 `GameOutput.k` 和 `GameOutput.order` 分配给两个角色。
- 玩家非法移动、撞障碍、撞任一玩家角色时跳过当前 step，后续 step 继续执行。
- NPC 可与玩家或 NPC 重叠；NPC 越界或撞障碍视为机制模型错误并 fail-fast。
- 成功移动后立即结算金币、炸弹和拥挤踩踏。
- 金币拾取为当前地面金币的 `ceil(65%)`，停留或非法移动不触发脚下金币。
- 玩家踩炸弹扣当前持币 `ceil(10%)` 并移除炸弹；NPC 踩炸弹只移除炸弹。
- 玩家成功移动到 `>=3` 个 NPC 的格子后触发踩踏，扣当前持币 `ceil(5%)`。
- `net_gold = gross_gold - vision_spent`；同净金币时必须传入双方 P90 延迟才能判胜，否则 fail-fast。

`rules.transition.transition_one_round()` 编排一整回合：

```text
resource mechanisms already applied
-> first player steps
-> NPC steps
-> second player steps
-> vision/snapshot/scoring bookkeeping
```

`first_player_id` 由调用方显式传入；训练中的先后手假设不写入规则层。

## 机制近似层

`mechanisms/` 承载官方未公开或无法完全复刻的部分。默认实现用于训练和本地回放，不代表官方真实随机源。

当前实现：

- `maps.py`：内置地图池、静态障碍、`static_grid == 2` 特殊非障碍格和固定出生配置。`built_in_public_map_pool()` 保留当前已观测公开图；`built_in_training_map_pool()` 是训练/本地对局默认池，包含公开图和满足规则约束的 synthetic training 图。
- `gold.py`：中心小额金币生成、外围 `static_grid == 2` high batch 和伴随外围小额金币生成；公开 map 4 的外围 `static_grid == 2` 数量少于旧三图，生成候选按地图静态层实际格子处理，零合法 static2 候选时将 high total fallback 到 high region 外外围普通格。
- `bombs.py`：固定 20 回合刷新周期，按候选格 Bernoulli 采样。
- `npc.py`：默认 NPC path-level softmax approximation，当前推荐口径见 `mechanism/results/npc/npc_behavior_modeling_overview.md`。
- `scripted.py`：固定金币/NPC 行为脚本，用于 deterministic transition 测试。

机制参数和证据入口：

- 金币：`mechanism/results/gold/gold_generation_observation.md`
- 初始金币：`mechanism/results/gold/initial_gold_analysis.md`
- 炸弹：`mechanism/results/bomb/bomb_generation_observation.md`
- NPC：`mechanism/results/npc/npc_behavior_modeling_overview.md`

重要约束：

- 机制生成器返回事件，不直接绕过规则层修改结算逻辑。
- 已有金币不阻挡金币生成；新增金额叠加到地面金币。
- `static_grid == 2` 是可通行特殊格，不是障碍。
- synthetic training 图必须满足静态障碍至少上下对称或左右对称之一；每个外围 region 必须恰好有 5 个 `static_grid == 2` 特殊格；出生点和 NPC 点不得位于障碍；四个出生点应能静态连通到中心区域。公开官方图按平台 replay 观测静态层保留，不强行套用 synthetic 图约束。
- 未观测清楚的极端机制行为应 fail-fast 或显式标注 approximation，不在规则层伪装成确定事实。

## Observation

`observation.sdk.make_game_input()` 从完整 `GameState` 生成玩家视角输入。它可以读取 full state，但输出必须遵守官方可见性约束：

- 不可见 `grid` 为 `-5`。
- 玩家角色位置通过 `my_units` 给出，不写入 `grid`。
- 可见敌人只通过 `visible_enemies` 给出，不暴露敌方 unit id。
- NPC 只通过 `visible_npcs` 给出。
- 视野购买下回合生效，持续 1 回合。

训练 actor feature extractor 不在 simulator 内实现，而在 `policy_runtime/`；训练期 privileged critic feature 由 `training/` 从 full state 派生。

## 环境接口

`envs.round_step.RoundStepEnv` 是训练栈使用的核心薄接口：

- `reset()` 返回行动时 observation。
- `step(player_outputs, first_player_id)` 接收双方已生成 `GameOutput` 和显式先手方，推进完整一回合。
- 不定义 reward，不调用 policy，不固定训练身份，不决定先后手。

`envs.duel` 是便利 runner，用于本地 smoke、固定策略对战和 replay 生成。训练主线不应把规则假设写死到 `duel`。

核心数据流：

```text
EpisodeConfig
  -> map/mechanisms reset
  -> resource generation at round start
  -> make player observations
  -> caller/player policies produce GameOutput
  -> transition_one_round()
  -> next observations + trace + optional replay
```

## Replay

`simulator.replay` 导出 `goldrush2_simulator_full_replay` 单 JSON 文档，schema 见 `docs/simulator_replay_schema.md`。

Replay 口径：

- 上帝视角，无迷雾。
- `grid` 只表达地形和资源，不写玩家/NPC 叠层。
- 玩家和 NPC 通过独立实体字段表达。
- NPC 不维护也不导出 `npc.gold`。
- 规则事件、机制事件、双方输出、NPC actions、snapshot 和 start/end full state 都应结构化记录。

可视化器通过顶层 `format` 区分 official NDJSON、merged replay 和 simulator full replay。

## 测试

运行 simulator 测试时需要显式加入仓库根目录到 `PYTHONPATH`：

```bash
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python -m pytest tests/simulator
```

直接运行不带 `PYTHONPATH=.` 的 `pytest tests/simulator` 可能在某些环境下找不到本地 `simulator` 包。

修改规则层时优先补小而确定的单测；修改机制 approximation 时应同时检查对应 `mechanism/results/` 证据文档是否需要更新。
