# GoldRush2 本地模拟器

`simulator/` 是 GoldRush2.0 的本地游戏环境包，服务于规则单测、replay 对齐、策略调试和后续训练实验。实现时以 `gamerules/gamerules.md` 为规则事实来源；官方未公开的机制必须放在可替换的近似模型中，不得写死成确定规则。

## 设计目标

- 复刻已确认的官方规则层：移动、碰撞、拾取、炸弹、踩踏、视野、快照、胜负结算。
- 隔离未公开机制层：金币生成、炸弹刷新、NPC 策略、地图/障碍采样。
- 生成贴近官方 SDK 的 `GameInput` / `GameOutput`，便于本地策略调试。
- 为上层训练/实验系统提供稳定 round-step 接口；先后手、策略调用和训练目标由上层决定。
- 输出结构化事件，方便单测、replay 对齐和可视化排查。
- 导出 simulator 专用上帝视角 replay，便于本地整局复盘、训练失败分析和机制调参。

## 非目标

- 不访问评测平台，不提交策略，不发起对局。
- 不承载最终提交策略的 feature extractor；训练和部署共享的特征逻辑应放在 `policy_runtime/`。
- 不把双视角 replay 合并逻辑搬进模拟器；合并 replay 仍属于 `mechanism/scripts/analysis/` 和 `visualizer/` 的输入生态。
- 不把 simulator 生成的上帝视角 replay 伪装成官方 NDJSON 或 `goldrush2_merged_replay`；它应使用独立格式标识。
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
    round_step.py   # 双玩家整回合 step 接口，不定义 reward/policy/order

  replay/
    __init__.py
    schema.py       # simulator full replay 的 JSON schema / dataclass 约定
    recorder.py     # 记录模拟器局内状态与事件，导出调试日志
    export.py       # 写出 goldrush2_simulator_full_replay JSON
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
- `rules.transition.transition_one_round()`：用给定玩家输出和机制脚本推进一个完整回合，返回移动、交互、金币生成、dispatch order 和可选 snapshot。

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

当前 `transition_one_round()` 约定：

- 输入 `state` 表示本回合行动前状态，函数原地修改并返回同一个 `GameState`，返回后表示下一回合开始状态。
- `first_player_id` 显式指定本回合先手玩家；后续 `cost_based` 模式再由耗时模型决定该值。
- 玩家行动按 `first_player -> NPC -> second_player` 执行，每个成功移动步后立即调用 `rules.interaction`。
- 本回合 `vp` 在回合结尾计入 `vision_spent`，并通过 `activate_pending_vision()` 激活为下一回合视野；未购买方恢复默认视野。
- snapshot 若传入 `SnapshotAccumulator`，使用回合开始前状态和行动结束状态统计本回合窗口；生成金币和拾取事件由 transition 传入 accumulator。

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
npc_order = npc_policy.sample_order(state, rng)
npc_actions = npc_policy.decide_all(decision_state, static_map, npc_order, rng)
static_map, spawn_points = map_provider.sample(rng)
```

当前已实现：

- `mechanisms.scripted.ScriptedMechanisms`：按 `round_index` 返回固定 `GoldGenerationEvent` 和每个 NPC 的动作序列，用于 deterministic transition 测试和后续规则对齐。
- `mechanisms.scripted.ScriptedRound`：单回合脚本容器，字段为 `gold_generated` 和 `npc_actions`。
- `mechanisms.maps.MapTemplate`：冻结一张静态地图模板，保存 `static_grid`、障碍格和特殊非障碍格。
- `mechanisms.maps.SpawnConfig`：冻结跨图稳定出生配置；默认 P1 在 `(0,0)/(16,16)`，P2 在 `(0,16)/(16,0)`，7 个 NPC 在 `(8,8)`。
- `mechanisms.maps.MapPool`：管理地图池并用外部传入的 `random.Random` 做均匀 seeded 抽样。
- `mechanisms.maps.built_in_public_map_pool()`：内置当前已观测的 3 张公开地图，按 map_id `1/2/3` 排列；默认抽样均匀，不从样本数量推断官方真实权重。
- `mechanisms.maps.build_initial_state()`：由 `MapTemplate + SpawnConfig` 构造初始 `GameState`，并校验出生点不落在障碍上。
- `mechanisms.gold.CenterGoldGenerator`：中心区域小额金币生成模型；返回 `GoldGenerationEvent`，不直接修改 `GameState.gold`。
- `mechanisms.gold.CenterGoldConfig`：中心金币模型参数，默认 `A=0.0596`、`B=0.06735`、金额均匀整数 `1..11`，来自 `mechanism/results/gold/` 的中心生成分析。
- `mechanisms.gold.OuterGoldGenerator`：外围 `static_grid == 2` high batch 与伴随外围 `static_grid == 0` 小额金币生成模型；返回 `GoldGenerationEvent`，不直接修改 `GameState.gold`。
- `mechanisms.gold.OuterGoldState`：记录下一次 `static2` high batch 的计划回合；外围生成是 stateful renewal process，不是逐回合 Bernoulli。
- `mechanisms.gold.OuterGoldConfig`：冻结第一版经验分布参数，包括首次触发 offset、gap、region、`static2` batch total、伴随普通格事件数和金额分布；首次 offset 暂定为 `UniformInteger(8, 14)`。
- `mechanisms.npc.M4aNpcPolicy`：默认 NPC 策略 profile；使用 simultaneous-start 口径一次性为 7 个 NPC 采样 actions，再由 transition/env 按随机 permutation 执行结算。
- `mechanisms.npc.NpcPolicyConfig` / `NpcEpisodeProfile`：冻结 M4a 默认权重、`temperature`、`bomb_blind_p` 和 episode-level randomization 配置。
- `mechanisms.bombs.BernoulliBombRefresher`：每 `20` 回合按候选格独立 Bernoulli 采样刷新炸弹；刷新周期是已确认规则常量，不向机制配置暴露。
- `mechanisms.bombs.BombConfig`：第一版只暴露 `spawn_probability`，默认 `0.0795`，来自 `mechanism/results/bomb/bomb_generation_observation.md` 的地图1/2候选格采样率。

`ScriptedMechanisms` 当前只覆盖 transition 已经接入的机制项：金币生成和 NPC 动作。炸弹刷新、地图采样等后续接入时再扩展，避免先造未使用接口。

地图口径：

- `static_grid == 1` 是障碍，进入 `GameState.obstacles`。
- `static_grid == 2` 是特殊非障碍格，保留在 `MapTemplate.special_cells`，不进入障碍集合；后续金币/外围机制可选择使用。
- 出生点不写入地图模板，统一由 `SpawnConfig` 管理。

炸弹刷新口径：

- 非刷新轮不修改 `state.bombs`。
- 刷新轮条件固定为 `state.round_index % 20 == 0`。
- 刷新时先清空全部旧炸弹，再对候选格独立采样；旧炸弹格清空后若合法，可以被重新抽中。
- 候选格排除障碍、金币、玩家、NPC；`static_grid == 2` 不排除。
- 已有炸弹若位于障碍、金币、玩家或 NPC 格，刷新前直接 fail-fast，暴露坏状态。

中心金币生成口径：

- 第一版只覆盖中心 `9x9` 区域内的 `static_grid == 0` 格；障碍、已有金币、炸弹、玩家和 NPC 位置全部排除。
- 每个候选格独立采样，概率为 `A * exp(-B * ((row - 8)^2 + (col - 8)^2))`。
- 命中后金额从闭区间 `[1, 11]` 均匀整数采样。
- 生成器只返回事件，不直接落盘到状态；由 transition/env 层统一应用，以便单测和 replay 对齐复用同一事件流。

外围金币生成口径：

- 外围主事件是 `static_grid == 2` high batch，使用 `OuterGoldState.next_static2_round` 触发；非计划回合不生成，错过计划回合时 fail-fast。
- high batch 触发后按 gap 经验分布计划下一次触发；第一版 gap 主体为 `8..16`，不建成每回合 Bernoulli。
- high region 第一版在外围区域 `2..5` 中均匀采样；同轮只触发一个 high region。
- high region 内必须有 5 个 `static_grid == 2` 模板格，否则认为地图模板与当前机制不匹配并 fail-fast。
- `static2` high batch 先采样 batch total，再把 total 近似均分到 high region 内全部合法 `static_grid == 2` 候选格，余数随机分配到合法候选格。
- high region 之外的外围 `static_grid == 0` 普通格按经验事件数采样，并按经验金额分布生成小额金币；high region 内普通格不生成伴随项。
- 候选格动态排除已有金币、炸弹、玩家和 NPC。
- 若 high region 内部分 `static_grid == 2` 候选格被动态对象、炸弹或已有金币占用，本次 high batch 仍在其它合法候选格上生成。
- TODO：全部 `static_grid == 2` 候选格均不可用时官方行为未观测；第一版跳过本次 `static2` 分配并继续推进下一次 gap。
- 首次 high batch 相位分布暂定为 `UniformInteger(8, 14)`；后续若 replay 首触发分布给出更强证据，再替换该默认参数。

NPC 策略默认口径：

- 第一版实现 `mechanisms.npc.M4aNpcPolicy`，作为 simulator 默认 NPC profile。
- 每轮 NPC 内部顺序用 7 个 NPC ID 的均匀随机 permutation；外层仍由 transition 按 `first_player -> NPC -> second_player` 编排。
- 每个 NPC 行动时枚举当前位置出发的所有合法 3-step path；合法性只排除边界和 `static_grid == 1` 障碍。NPC 与玩家、NPC 之间可重叠，不作为阻挡。
- 候选 path 应按 `(map_id/static_grid, start_pos)` 缓存；每个起点候选数很小，允许全枚举。
- 对每条 path 计算 M4a score 后按 `softmax(score / temperature)` 采样，不做 argmax 硬选择。
- 当前证据不能区分 NPC 是否看到先手玩家执行后的状态；第一版按“玩家与 NPC 基于同一行动前局面同时 plan”建模。
- 决策阶段使用同一个 NPC 决策快照同时为 7 个 NPC 生成 actions；该快照取本回合资源生成/炸弹刷新后的行动前状态，先手玩家尚未执行。不要因为前序玩家或 NPC 之后执行过动作就修改已生成的 NPC score。
- 执行阶段按随机 permutation 依次执行已生成的 actions，并按真实规则逐步结算金币和炸弹。前序 NPC 的拾金/踩炸弹只影响后序 NPC 的执行结算，不影响同轮已生成的决策 score。
- 第一版随机化采用 episode-level profile：每局开始时采样一套全局 M4a 权重、`temperature` 和 `bomb_blind_p`，整局内固定。
- per-NPC jitter 作为可选项，默认可关闭；若开启，只在 episode reset 时对每个 NPC 采样固定小扰动，幅度约为全局扰动的 `20%`。不要每次 NPC 决策重抽权重。

默认 M4a + center score：

```text
score(path) =
    2.106687 * dynamic_reward_div10(path)
  - 2.796592 * enter_bomb_count(path)
  - 3.126171 * stay_count(path)
  + 1.212827 * straight3(path)
  - 0.580930 * backtrack(path)
  + 10.882944 * bomb_trapped_stay(path)
  + 0.180587 * center_delta_chebyshev(path)
```

特征定义：

- `dynamic_reward_div10 = dynamic_pickup_reward / 10`；`dynamic_pickup_reward` 按 path 内每次移动进入金币格时的 `ceil(0.65 * remaining_gold)` 累加，并同步扣减该 path 的局部剩余金币。该局部扣减只用于评估当前候选 path，不影响同轮其它 NPC 的决策 score。
- `enter_bomb_count`：path 中移动进入当前仍存在炸弹格的次数；候选 path 内进入炸弹后，该局部炸弹从后续步移除。
- `stay_count`：三步动作中 `action=4` 的次数。
- `straight3`：三步同向且非停。
- `backtrack`：存在相邻两步反向移动，即 `(上,下)`、`(下,上)`、`(左,右)`、`(右,左)`。
- `bomb_trapped_stay`：起点所有合法相邻非停格都是当前炸弹，且候选 path 为 `[停,停,停]`。
- `center_delta_chebyshev = start_chebyshev - end_chebyshev`，其中 `chebyshev(pos) = max(abs(row - 8), abs(col - 8))`。该项修正 baseline M4a 对中心区域 NPC 聚集的低估。

M5c 暂不作为默认实现。它的 held-out NLL 更低、形状统计更贴近 replay，但首动作和前两步 argmax 正确率低于 M4a，且参数更多。后续若需要统计复现 profile，可在 M4a 基础上增加 `first_two_same_nonstay`、`last_two_same_nonstay` 和 `sandwich` 三项；不要把它们混入默认 M4a。

M4a randomization 配置建议：

```text
episode reset:
    w_gold *= LogUniform(0.5, 2.0)
    w_enter_bomb *= LogUniform(0.5, 2.0)
    w_stay += Uniform(-0.8, 0.8)
    w_straight += Uniform(-0.8, 0.8)
    w_backtrack += Uniform(-0.5, 0.5)
    w_bomb_trapped_stay += Uniform(-3.0, 3.0)
    w_center += Uniform(-0.14, 0.14)
    temperature *= LogUniform(0.7, 1.5)
    bomb_blind_p ~ Uniform(0.0, 0.15)
```

`bomb_blind_p` 只影响决策阶段是否屏蔽炸弹风险，执行阶段仍按真实状态触发炸弹。

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

### `replay/`

负责把 simulator 自身运行过程导出为本地调试 replay。该层读取完整 `GameState`、玩家输出、`TransitionResult` 和机制事件，但不实现游戏规则。完整格式说明见 `docs/simulator_replay_schema.md`。

推荐 canonical 格式为单个 JSON 文档：

```json
{
  "format": "goldrush2_simulator_full_replay",
  "format_version": 1,
  "source": {
    "seed": 123,
    "map_id": 1,
    "mechanisms": {}
  },
  "players": {
    "player1": "policy_a",
    "player2": "policy_b"
  },
  "maps": [],
  "rounds": []
}
```

每轮记录：

- `round`：回合编号。
- `start`：行动前完整上帝视角状态。
- `end`：行动后完整上帝视角状态。
- `actions`：双方本回合 `GameOutput`，以及机制层给出的 NPC 动作。
- `events`：移动、交互、金币生成、炸弹刷新、视野购买、dispatch order 等结构化事件。
- `snapshot`：若本轮发布 snapshot，则记录 simulator 生成的 `Snapshot`。

上帝视角 replay 口径：

- `grid` 只表达地形和资源：障碍、炸弹、空地、金币；不使用迷雾 `-5`。
- 玩家和 NPC 不写入 `grid`，必须通过独立的 `players` / `npcs` 结构表达。
- 不使用官方 full replay 中的 `grid=-2/-4` 作为 canonical 表达；这些标记可由 visualizer 派生。
- NPC 仍不维护 `npc.gold`；导出格式不得伪造该字段。
- 文件必须通过 `format="goldrush2_simulator_full_replay"` 与官方 NDJSON、`goldrush2_merged_replay` 明确区分。

当前已实现：

- `replay.recorder` 记录初始地图、seed、机制配置和每回合 start/end 状态及事件。
- `replay.schema` 固定 JSON 字段和版本号。
- `replay.export` 只写出 simulator full JSON；若后续为了现有 visualizer 快速兼容需要 official-like NDJSON，应作为 adapter 明确标注，不作为 canonical schema。

当前 `serialize_state()` 会 fail-fast 检查资源格不变量：金币数必须为正，金币/炸弹不能落在障碍上，炸弹不能和金币重合。NPC 只导出 `id` 与 `position`，不包含 `gold`。

### `envs/`

环境层调用 `rules/`、`mechanisms/` 和 `observation/`，但不实现规则细节。

`envs.round_step` 是面向上层实验/训练系统的核心薄适配接口。它不定义 reward，不调用 policy，不给 P1/P2 附加训练身份，也不决定先后手；调用方必须显式传入 P1/P2 的 `GameOutput` 和 `first_player_id`。每次 `step()` 推进完整一回合，返回双方下一轮 observation、trace、终局状态和可选 replay。

`envs.duel` 是便利 runner，用于本地 smoke、固定策略对战和 replay 生成。它可以调用 policy/provider 并按配置选择先后手，但不应成为上层训练系统的强依赖。

## 核心数据流

```text
EpisodeConfig
  -> map_provider.sample()
  -> GameState
  -> mechanisms generate gold/bomb before action-time observation
  -> make_observation(state, player_id)
  -> NPC actions sampled from the same action-time state
  -> caller provides P1/P2 GameOutput and first_player_id
  -> rules.transition_started_round()
  -> next observations + trace + optional replay
```

## 第一阶段实现顺序

1. 已完成：建立 `constants.py`、`types.py`、`state.py`、`config.py` 的最小数据模型。
2. 已完成：实现 `rules.movement`，优先覆盖非法移动、碰撞、同队阻挡、同队交换。
3. 已完成：实现 `rules.interaction`，覆盖金币拾取、炸弹触发、踩踏处罚和事件记录。
4. 已完成：实现 `observation.sdk`，对齐官方 Python `GameInput` 字段。
5. 已完成：实现 `rules.snapshot` 和 `rules.scoring`。
6. 已完成：用 `mechanisms.scripted` 组装 deterministic transition 测试。
7. 引入默认机制近似模型。该步骤拆成如下子阶段：
   - 7.1 已完成：`mechanisms.maps` 提供 3 张公开地图 pool、固定出生点配置、seeded 均匀抽样和初始 `GameState` 构造。
   - 7.2 基本完成：`mechanisms.gold.CenterGoldGenerator` 已实现中心区域小额金币生成 approximation；`OuterGoldGenerator` 已实现 stateful `static_grid == 2` high batch 与伴随外围 `static_grid == 0` 小额金币，首次 high batch offset 暂定 `UniformInteger(8, 14)`。已覆盖同 seed 可复现、不生成在障碍/炸弹/玩家/NPC/已有金币格、不直接修改状态、非法配置 fail-fast。剩余参数观察项是动态占用导致候选不足时的官方处理。
   - 7.3 已完成机制本体：`mechanisms.npc` 提供 seeded NPC 策略 approximation，默认实现 simultaneous-start M4a path-level softmax。必须只产生不会越界/撞障碍的动作，因为 `rules.movement.apply_npc_step()` 对 NPC 非法动作 fail-fast。已覆盖同 seed 可复现、每个 NPC 每回合 3 个动作、合法 path 枚举、动态拾金/炸弹风险特征、`bomb_blind_p`、同一轮 7 个 NPC 基于同一决策快照生成 actions。transition/env 接入时应先生成全体 NPC actions，再按随机 dispatch order 执行结算。
   - 7.4 已完成：`mechanisms.bombs` 提供固定 20 回合刷新周期和 `p_bomb=0.0795` 的 Bernoulli 候选格采样，严格执行刷新约束。
   - 7.5 已完成：`simulator.replay` 实现 `goldrush2_simulator_full_replay` recorder/exporter，记录完整上帝视角 start/end、双方输出、机制事件、规则事件和 snapshot。该格式是 simulator 调试产物，不伪装成官方 NDJSON 或 merged replay。已覆盖固定小局输出 JSON 可稳定复现，且不会写出 `npc.gold` 或 `grid=-2/-4` canonical 标记。
   - 7.6 已完成：`envs.duel` 提供最小本地对战 runner。输入两个 policy/output provider，支持 `agent_after_opponent`、`agent_before_opponent`、`random_order`、`cost_based` 先后手模式；每轮先执行资源生成，再用同一行动前状态生成双方 observation，随后按先手玩家、NPC、后手玩家顺序调用 transition 规则入口。输出最终 `GameState`、`GameResult`、每回合 trace、snapshot 和可选 simulator full replay。已覆盖固定机制 seed 下 replay 可复现和 500 回合最小 smoke。
8. 已完成：`envs.round_step` 提供无训练假设的双玩家 round-level stepping API。`reset()` 返回行动时 observation；`step(player_outputs, first_player_id)` 接收双方已生成输出和显式先手方，推进完整一轮。该层不定义 reward、不调用 policy、不决定先后手、不固定任何玩家的训练身份。已覆盖资源生成先于 observation、显式先手、可选 replay、终局结果和 500 回合最小 smoke。

## 质量要求

- 规则层遇到未明确规则时 fail fast，并在代码或文档中指向不确定项。
- 单测优先构造小而精的完整状态，不依赖随机种子验证核心规则。
- replay 对齐工具只用于验证和发现偏差；若发现官方规则文档缺口，应回写到 `gamerules/gamerules.md` 或相关机制分析文档。
- 新增通用脚本时同步维护 `tools/README.md`；`simulator/` 内部库代码不应隐式访问平台或 `.env`。
