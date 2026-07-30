# GoldRush2 本地模拟器

`simulator/` 是 GoldRush2.0 的本地游戏环境包，服务于规则单测、replay 对齐、策略调试和后续 RL 训练。实现时以 `gamerules/gamerules.md` 为规则事实来源；官方未公开的机制必须放在可替换的近似模型中，不得写死成确定规则。

## 设计目标

- 复刻已确认的官方规则层：移动、碰撞、拾取、炸弹、踩踏、视野、快照、胜负结算。
- 隔离未公开机制层：金币生成、炸弹刷新、NPC 策略、地图/障碍采样。
- 生成贴近官方 SDK 的 `GameInput` / `GameOutput`，便于本地策略调试。
- 为 RL 主链路提供稳定环境接口，支持快速 opponent 先手、NPC 居中、agent 后手的训练分布。
- 输出结构化事件，方便单测、replay 对齐和可视化排查。

## 非目标

- 不访问评测平台，不提交策略，不发起对局。
- 不承载最终提交策略的 feature extractor；训练和部署共享的特征逻辑应放在 `policy_runtime/`。
- 不把双视角 replay 合并逻辑搬进模拟器；合并 replay 仍属于 `mechanism/scripts/analysis/` 和 `visualizer/` 的输入生态。
- 不声称复刻官方随机分布；所有未公开机制都应显式标记为 approximation。

## 推荐目录结构

```text
simulator/
  __init__.py
  README.md

  constants.py      # 棋盘尺寸、动作编码、grid 值、规则常量
  types.py          # Position、Action、GameOutput、事件等轻量值对象
  state.py          # 完整上帝视角 GameState / PlayerState / UnitState / NpcState
  config.py         # RulesConfig、EpisodeConfig、机制模型配置
  errors.py         # 非法动作、配置错误、对齐失败等异常类型

  rules/
    __init__.py
    movement.py     # 玩家/NPC 单步移动、非法移动、角色阻挡与交换判定
    interaction.py  # 金币拾取、炸弹触发、拥挤踩踏
    vision.py       # 官方视野裁剪、visible_enemies、visible_npcs
    snapshot.py     # 5 区域统计窗口与 Snapshot 生成
    scoring.py      # 毛金币、视野花费、净金币、胜负结算
    transition.py   # 单回合官方规则编排，不直接采样未知机制

  mechanisms/
    __init__.py
    maps.py         # 静态地图/出生点提供器
    gold.py         # 中央/外围金币生成模型接口与默认近似实现
    bombs.py        # 炸弹刷新模型接口与默认近似实现
    npc.py          # NPC 策略接口与默认近似实现
    scripted.py     # 测试用固定脚本机制，保证单测可复现

  observation/
    __init__.py
    sdk.py          # 生成官方 SDK 形状的 GameInput
    replay_frame.py # 生成 official-like 调试帧，不替代正式 replay schema

  envs/
    __init__.py
    duel.py         # 双策略本地对战 runner
    rl.py           # 单智能体 against-league 环境接口

  replay/
    __init__.py
    recorder.py     # 记录模拟器局内状态与事件，导出调试日志
    align.py        # 用 full/user replay 样本做规则层对齐检查
```

测试放在：

```text
tests/simulator/
  test_movement.py
  test_interaction.py
  test_vision.py
  test_snapshot.py
  test_transition.py
```

## 分层边界

### `rules/`

只实现已确认规则，输入必须是完整状态和已给定的动作/事件上下文，不在内部随机采样金币、炸弹、NPC 动作或地图。该层应尽量 deterministic，单测覆盖优先级最高。

当前已实现：

- `rules.movement.apply_player_step()`：玩家单步移动；越界、撞障碍、撞任一玩家角色时跳过当前步并记录 `MovementEvent`。
- `rules.movement.apply_player_turn()`：按 `GameOutput.k` 和 `GameOutput.order` 执行玩家 6 步动作；同队角色不能直接穿过或交换位置，但可以进入队友已经腾出的格子。
- `rules.movement.apply_npc_step()`：NPC 单步移动；NPC 可与玩家或 NPC 重叠。根据当前 replay 证据，NPC 尝试越界或撞障碍视为机制模型错误，直接抛 `SimulatorRuleError`。
- `rules.interaction.apply_step_interactions()`：在一次成功移动已经写入 `GameState` 后，按落点结算金币、炸弹和踩踏，并返回 `InteractionEvents`。
- `rules.scoring`：提供毛金币、净金币、视野购买、pending 视野生效和最终胜负判定。
- `rules.snapshot`：提供官方 5 区域划分、窗口统计累积和 `Snapshot` 生成。

NPC 状态口径：

- `GameState.npcs` 只维护 NPC 的稳定 ID 和位置，不维护 `npc.gold`。
- 原因：训练玩家策略时需要的是地面金币、炸弹、NPC 位置和玩家资产；NPC 内部持币不可观测、不参与胜负结算，且会引入无法确认的 NPC 炸弹扣款规则。

`rules.interaction` 先检查落点状态不变量，再对合法状态按固定顺序结算：

1. 金币拾取：玩家或 NPC 移动到金币格时拾取 `ceil(65%)`，剩余金币留在原格；玩家拾取增加玩家角色持币，NPC 拾取只减少地面金币并记录事件；停留或非法移动不触发脚下金币。
2. 炸弹触发：玩家移动到炸弹格时扣除当前持币 `ceil(10%)` 并移除炸弹；NPC 移动到炸弹格时只移除炸弹。
3. 拥挤踩踏：仅玩家成功移动到 `>=3` 个 NPC 的格子后触发，扣除当前持币 `ceil(5%)`；NPC 自身不会触发踩踏处罚。

禁止状态：

- 同一位置同时存在炸弹和 `>=3` 个 NPC 时，`rules.interaction` 直接抛 `SimulatorRuleError`，不把“玩家同一步落到炸弹 + 踩踏格”作为正常可达分支处理。依据：规则文档已确认“炸弹不会刷新在玩家、NPC、金币或已有炸弹所在格”和“同一个位置不会同时有炸弹和大于等于 3 个 NPC”；机制数据扫描 `mechanism/data/processed/merged_replays` 下 4 个 run、400 局、400000 个 start/end 合并帧，也未发现稳定帧中炸弹与 NPC 或 `>=3` NPC 重合。

`rules.transition` 只负责编排一轮中已知规则：

1. 应用机制层提前采样好的金币/炸弹刷新事件。
2. 根据指定行动顺序执行先手玩家；每个成功移动步后立即调用 `rules.interaction`。
3. 执行机制层给出的 NPC 动作；每个成功移动步后立即调用 `rules.interaction`。
4. 执行后手玩家；每个成功移动步后立即调用 `rules.interaction`。
5. 更新视野花费、snapshot 窗口、回合计数和结构化事件。

`rules.scoring` 口径：

- `gross_gold(state, player_id)` 等于该玩家两个角色持币之和。
- `net_gold(state, player_id)` 等于毛金币减 `vision_spent`。
- `apply_vision_purchase(state, player_id, vp)` 只累计视野费用并设置下回合视野半径，不检查当前毛金币是否足额。
- `activate_pending_vision(state)` 在回合推进时让 pending 视野生效 1 回合，并把下一回合 pending 半径重置为默认 `5x5` 半径。
- `determine_winner(state, p90_latency_ns=None)` 先按净金币判胜；净金币相等时必须传入双方 P90 延迟，低者胜。若净金币和 P90 都相等，官方未明确后续 tie-break，当前 fail-fast。

`rules.snapshot` 口径：

- `region_id(position)` 按 `gamerules/gamerules.md` 的 5 区域坐标划分返回区域编号 `1..5`。
- `SnapshotAccumulator.record_round(round_index, start_state, end_state, gold_generated=(), interactions=())` 累积一个完整回合。
- `enter/leave` 通过同一批稳定 actor 的回合起点区域和终点区域比较得到；actor 包括 4 个玩家角色和全部 NPC。
- `gold_generated` 来自机制层传入的 `GoldGenerationEvent`。
- `gold_collected` 来自每步 `InteractionEvents.pickups` 的 `picked_gold`。
- `gold_remaining` 和 `occupants` 在 snapshot 发布时基于当前 `end_state` 计算。
- 默认每 `snapshot_period=5` 个回合返回一个 `Snapshot`；未到窗口末尾时返回 `None`。

### `mechanisms/`

承载官方未公开或无法完全复刻的部分。每个机制模型都应能被固定脚本实现替换，以便单测和 replay 对齐。

核心接口应保持窄：

```python
gold_events = gold_model.generate(state, rng)
bomb_events = bomb_model.refresh(state, rng)
npc_actions = npc_policy.decide(state, npc_id, rng)
static_map, spawn_points = map_provider.sample(rng)
```

### `observation/`

负责从完整 `GameState` 生成玩家视角输入。它可以读取上帝视角状态，但输出必须符合官方可见性约束：

- `grid` 中不可见格为 `-5`。
- 玩家角色位置不写进 `grid`，通过 `my_units` 给出。
- 敌方角色只通过 `visible_enemies` 给出，且不暴露敌方 unit id。
- NPC 只通过 `visible_npcs` 给出。
- 视野购买下回合生效，持续 1 回合。

当前已实现：

- `observation.sdk.make_game_input()`：生成对齐官方 Python `GameInput` 字段的本地 dataclass。
- 可见范围为己方两个角色当前 `active_vision_radius` 方形视野的并集；默认半径 `2`，对应 `5x5`。
- `grid` 只表达纯地形和资源：迷雾 `-5`、障碍 `-1`、炸弹 `-3`、空地 `0`、金币正数；玩家和 NPC 不写入 `grid`。
- `visible_enemies` 固定返回 2 个位置槽。可见敌人按 `(row, col)` 排序后紧凑排列，尾部用 `(-1, -1)` 填充，避免通过 unit 下标泄露敌方角色身份。
- `visible_npcs` 返回可见 NPC 的 `id,row,col`，按 NPC ID 排序，长度不超过 `MAX_NPCS`。
- `snapshot` 由调用方传入并透传；通常来自 `rules.snapshot.SnapshotAccumulator`。`snapshot=None` 时 `snapshot_valid=False`。
- 若观测层发现金币与炸弹、金币与障碍、炸弹与障碍重叠，会抛 `SimulatorRuleError`，避免在 `grid` 中静默选择覆盖顺序。

### `envs/`

环境层调用 `rules/`、`mechanisms/` 和 `observation/`，但不实现规则细节。

第一阶段至少支持这些行动顺序模式：

| 模式 | 用途 |
| --- | --- |
| `agent_after_opponent` | 主训练分布：快速 opponent 行动，NPC 行动，agent 行动。 |
| `agent_before_opponent` | 反事实评估。 |
| `random_order` | 少量鲁棒性评估。 |
| `cost_based` | 预留真实耗时排序接口。 |

注意：即使 agent 后手，双方策略收到的 observation 也必须来自同一个行动前状态。

## 核心数据流

```text
EpisodeConfig
  -> map_provider.sample()
  -> GameState
  -> make_observation(state, player_id)
  -> policy.move_decision(GameInput)
  -> mechanisms generate gold/bomb/npc actions
  -> rules.transition(state, player_actions, mechanism_events, order_mode)
  -> next_state + events
```

## 第一阶段实现顺序

1. 已完成：建立 `constants.py`、`types.py`、`state.py`、`config.py` 的最小数据模型。
2. 已完成：实现 `rules.movement`，优先覆盖非法移动、碰撞、同队阻挡、同队交换。
3. 已完成：实现 `rules.interaction`，覆盖金币拾取、炸弹触发、踩踏处罚和事件记录。
4. 已完成：实现 `observation.sdk`，对齐官方 Python `GameInput` 字段。
5. 已完成：实现 `rules.snapshot` 和 `rules.scoring`。
6. 用 `mechanisms.scripted` 组装 deterministic transition 测试。
7. 再引入默认金币、炸弹和 NPC 近似模型。
8. 最后补 `envs.rl`，不要在规则层尚未稳定时先接训练接口。

## 质量要求

- 规则层遇到未明确规则时 fail fast，并在代码或文档中指向不确定项。
- 单测优先构造小而精的完整状态，不依赖随机种子验证核心规则。
- replay 对齐工具只用于验证和发现偏差；若发现官方规则文档缺口，应回写到 `gamerules/gamerules.md` 或相关机制分析文档。
- 新增通用脚本时同步维护 `tools/README.md`；`simulator/` 内部库代码不应隐式访问平台或 `.env`。
