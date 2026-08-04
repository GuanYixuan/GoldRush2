# Feature Extractor v1 设计

本文冻结 GoldRush2.0 初版 feature extractor 的网络输入语义。v1 按 CNN 主线设计，只输出 spatial planes 与 scalars，不引入 tokens。RL 栈总览见 `docs/rl_architecture.md`，运行时工程入口见 `policy_runtime/README.md`。

## 定位

feature extractor 属于 `policy_runtime/`，由 C++ 单实现承担，Python 训练侧通过 binding 调用同一套状态机。它负责把官方形状 `GameInput` 转换为网络输入，同时维护 agent 自身在部分可观测环境中的记忆。

它不负责：

- 推进真实游戏状态。
- 调用 simulator 规则。
- 计算 reward。
- 决定 opponent 或 NPC 的真实隐藏状态。
- 输出最终 `GameOutput`。

## 设计目标

初版 feature extractor 应满足：

- 训练/部署一致：C++ core 直接使用官方 `game_api.h` 的 `GameInput` / `GameOutput`。
- 形状稳定：输出 shape 在一次训练 run 内固定，便于 PyTorch、ONNX 和 C++ 部署对齐。
- 状态可测：`reset -> observe -> commit_action` 时序清晰，跨 episode 不串状态。
- 信息合法：actor 输入只能来自官方 observation 或由历史 observation 推断出的 belief，不能泄漏 simulator privileged state。
- 计算轻量：单回合 feature extraction 应明显小于神经网络推理耗时，不引入复杂搜索作为必需路径。

## 状态机

每局应遵循：

```text
extractor.reset(player_id)
for each round:
    features = extractor.observe(game_input)
    action, side_output = policy(features)
    extractor.commit_action(action, side_output)
```

当前最小实现只有 `commit_action(GameOutput)`；后续若策略输出辅助头或 fast option，可扩展 `side_output`。

原则：

- `observe()` 是唯一写入新观测事实的入口。
- `commit_action()` 只能记录上一动作和策略意图，不能推演移动结果。
- 下一轮真实位置、碰撞、拾取、被对手抢金币、NPC 干扰等必须等下一次 `observe()` 确认。
- round 回退应 fail-fast 或显式 reset；部署侧可额外实现新局自动 reset 检测。

## 输入边界

C++ core 输入：

- `GameInput`：直接来自 `official_sdk/code/game_api.h`。
- `GameOutput`：直接来自 `official_sdk/code/game_api.h`。

Python binding 只负责把 simulator 的 Python dataclass 转成官方 C++ struct，不定义第二套 `GameInputLike` schema。

snapshot 通过 `GameInput.snapshot_valid` / `GameInput.snapshot` 进入 feature extractor；v1 不要求额外 batch extraction API。部署时若需要自动识别新局，可在 `round` 回到 0 或 `player_id` 变化时显式 reset，具体工程策略不改变本文件冻结的 feature schema。

## 输出形态

v1 输出：

```text
feature_schema: "goldrush2_feature_v1"
spatial_planes: 38 x 17 x 17
scalars: 10
tokens: none
```

`spatial_planes` 分为两类：

- `temporal_spatial_planes`：少数关键观测按固定时间窗口展开后并入 channel 维。
- `nontemporal_spatial_planes`：表达当前 observation、长期记忆或确定派生量，不按时间窗口展开。

Python 返回格式应能稳定暴露 `spatial_planes`、`scalars`、`feature_schema`、`channel_names`、`scalar_names`。具体是否使用 dict、dataclass 或 numpy tuple 是 binding 工程细节，但名字和顺序必须与本文件一致。

## Channel Order

`spatial_planes` 的 channel 顺序固定为：

```text
visible_mask_t0
visible_mask_t1
visible_mask_t2
visible_mask_t3
visible_mask_t4
gold_count_t0
gold_count_t1
gold_count_t2
gold_count_t3
gold_count_t4
bomb_mask_t0
bomb_mask_t1
bomb_mask_t2
bomb_mask_t3
bomb_mask_t4
visible_enemy_count_t0
visible_enemy_count_t1
visible_enemy_count_t2
visible_enemy_count_t3
visible_enemy_count_t4
obstacle_known_mask
obstacle_mask
visible_npc_count
npc_crowded_mask
own_unit0_mask
own_unit1_mask
unit0_manhattan_distance
unit1_manhattan_distance
unit0_known_obstacle_distance
unit1_known_obstacle_distance
region_1_center_mask
region_2_up_mask
region_3_left_mask
region_4_down_mask
region_5_right_mask
last_snapshot_gold_remaining_map
prev_snapshot_gold_remaining_map
last_snapshot_occupants_map
```

temporal 部分采用 key-major、time-minor 的 flatten 顺序，即同一个 key 的 `t0..t4` 连续排列。`scalars` 的顺序见 “Scalar Features”。

## Temporal Spatial Planes

v1 将最关键、最容易从短期变化中受益的观测项独立成 temporal 组：

```text
temporal_keys = [
    visible_mask,
    gold_count,
    bomb_mask,
    visible_enemy_count,
]
time_window = 5
```

语义：

- `t0` 表示当前 `observe(game_input)` 看到的局面。
- `t1..t4` 表示更早的 1 到 4 次己方决策前 observation。
- 缺失历史用 0 填充。
- 所有 temporal plane 都只记录玩家视角下实际观测到的信息，不记录 simulator full state 或官方不可见真值。

具体通道：

- `visible_mask[t-k]`：第 `t-k` 次 observation 中该格是否可见。
- `gold_count[t-k]`：第 `t-k` 次 observation 中可见格的金币数量，取值为 `clip(raw_gold / 20, 0, 3)`；不可见格填 0。
- `bomb_mask[t-k]`：第 `t-k` 次 observation 中可见格是否为炸弹；不可见格填 0。
- `visible_enemy_count[t-k]`：第 `t-k` 次 observation 中每格可见敌方角色数量；玩家角色之间不能重叠，正常取值为 0 或 1；不可见格填 0，并由 `visible_mask[t-k]` 区分未知。

设计约束：

- `visible_mask` 必须与 `gold_count` / `bomb_mask` / `visible_enemy_count` 同时保留，否则 `0` 无法区分“可见且没有目标”和“不可见未知”。
- 当前 `visible_mask`、当前可见金币数量、当前炸弹 mask、当前可见敌方角色数量不再作为单独 nontemporal plane 重复出现，由 `t0` 承担当前信息。
- temporal 窗口按“己方收到 observation 的次数”推进，不按全局真实回合补齐隐藏过程；这与官方回放视角一致。
- `gold_count` 的 scaling 不是严格归一化，而是固定除以 20 后截断到 3，避免少数高金币格产生过大的输入幅度。
- `visible_enemy_count` 聚合敌方角色，不区分敌方角色 0 / 1，避免给网络引入不稳定的敌方 unit 编号排列敏感性。
- C++ 实现可用固定长度 ring buffer，`observe()` 写入 `t0`，导出时按 `t0 -> t4` 的顺序排列。

v1 不加入 NPC 的 temporal 版本；若后续证明短期 NPC 轨迹对策略收益明显，再作为 schema v2 扩展。

## Nontemporal Spatial Planes

v1 固定加入：

```text
nontemporal_keys = [
    obstacle_known_mask,
    obstacle_mask,
    visible_npc_count,
    npc_crowded_mask,
    own_unit0_mask,
    own_unit1_mask,
    unit0_manhattan_distance,
    unit1_manhattan_distance,
    unit0_known_obstacle_distance,
    unit1_known_obstacle_distance,
    region_1_center_mask,
    region_2_up_mask,
    region_3_left_mask,
    region_4_down_mask,
    region_5_right_mask,
    last_snapshot_gold_remaining_map,
    prev_snapshot_gold_remaining_map,
    last_snapshot_occupants_map,
]
```

语义：

- `obstacle_known_mask`：该格障碍状态是否已知。
- `obstacle_mask`：该格是否为障碍；只有在 `obstacle_known_mask=1` 时语义可靠，未知格填 0。
- `visible_npc_count`：当前 observation 中每格可见 NPC 数量，取值为 `clip(count / 3, 0, 1)`；不可见或无 NPC 的格子为 0。
- `npc_crowded_mask`：当前 observation 中该格可见 NPC 数量是否 `>=3`。
- `own_unit0_mask` / `own_unit1_mask`：当前己方角色 0 / 1 的位置 one-hot。
- `unit0_manhattan_distance` / `unit1_manhattan_distance`：每格到己方角色 0 / 1 的曼哈顿距离，取值为 `distance / 32`。
- `unit0_known_obstacle_distance` / `unit1_known_obstacle_distance`：每格到己方角色 0 / 1 的已知障碍 BFS 距离，取值为 `clip(distance / 32, 0, 2)`。
- `region_1_center_mask` / `region_2_up_mask` / `region_3_left_mask` / `region_4_down_mask` / `region_5_right_mask`：官方 5 区域划分 one-hot。
- `last_snapshot_gold_remaining_map`：最近一次有效 snapshot 中对应区域的 `gold_remaining`，广播到该区域，取值为 `clip(gold_remaining / 100, 0, 2)`。
- `prev_snapshot_gold_remaining_map`：上上次有效 snapshot 中对应区域的 `gold_remaining`，广播到该区域，取值为 `clip(gold_remaining / 100, 0, 2)`。
- `last_snapshot_occupants_map`：最近一次有效 snapshot 中对应区域的 `occupants`，广播到该区域，取值为 `clip(occupants / 4, 0, 2)`。

更新规则：

- 当前或历史 observation 中直接看到 `grid=-1` 的格子，记为已知障碍。
- 当前或历史 observation 中直接看到非迷雾且非障碍的格子，记为已知非障碍。
- 对每个已知格 `(r, c)`，可按 180 度中心对称推断其镜像格 `(16-r, 16-c)` 的障碍状态。
- 直接观测事实优先于对称推断；如果后续直接观测与之前的对称推断冲突，应以直接观测覆盖。
- `visible_npc_count` / `npc_crowded_mask` 只使用当前 `visible_npcs`，v1 不对不可见 NPC 做历史位置 belief。
- `own_unit0_mask` / `own_unit1_mask` 直接来自当前 `my_units`。
- 曼哈顿距离不考虑障碍，用 `abs(row - unit.row) + abs(col - unit.col)` 计算。
- 已知障碍 BFS 距离只把 `obstacle_known_mask=1 && obstacle_mask=1` 的格子作为阻挡；未知格按可通行处理。
- 若某格在已知障碍 BFS 下不可达，距离 plane 取 clip 后最大值 `2`。
- 区域 one-hot 固定使用官方坐标划分：1=中央，2=上，3=左，4=下，5=右。
- 当 `snapshot_valid=1` 时，先将当前最近一次 snapshot 记忆移入 `prev_snapshot_*`，再写入新的最近一次 snapshot 记忆；无对应历史 snapshot 时，相关 map 全 0。

设计依据与边界：

- 当前三张公开地图的障碍格均满足 180 度中心对称，因此该推断对 obstacle 有较强实用价值。
- `static_map=2` 特殊格不应使用同一对称推断：现有公开地图中地图1/2的 `static_map=2` 并不中心对称。
- 对称推断属于地图先验/数据推断，不是官方接口直接暴露的信息；若正式赛地图不满足障碍中心对称，该特征可能带来误导。
- `npc_crowded_mask` 对应官方拥挤踩踏阈值：玩家行动后走到 NPC 个数 `>=3` 的格子会触发踩踏。
- 敌方角色在 temporal 组中聚合；己方角色仍需区分，因为 `k/order/actions` 与己方角色编号强绑定。
- `unit0/1_known_obstacle_distance` 是基于当前障碍记忆的近似可达性，不表达对手角色阻挡、NPC、炸弹或未来资源变化。
- `last_snapshot_*_map` / `prev_snapshot_*_map` 表达的是最近两次有效快照的记忆，不是当前真值；最近一次 snapshot 是否存在由 scalar `last_snapshot_valid` 区分。v1 暂不加入 `prev_snapshot_valid`，因此上上次 snapshot 缺失与其数值为 0 存在轻微歧义；也暂不加入其它 snapshot gold 派生面，避免把间接统计过早铺宽。

## Scalar Features

v1 固定加入：

```text
scalar_keys = [
    game_phase_sin,
    game_phase_cos,
    own_gold_scaled,
    opp_gold_scaled,
    gold_margin_scaled,
    outer_gold_phase_sin,
    outer_gold_phase_cos,
    snapshot_sin,
    snapshot_cos,
    last_snapshot_valid,
]
```

语义：

- `game_phase_sin` / `game_phase_cos`：整局 500 回合进度的半周期相位。`phase=-pi/2+pi*round/(round_count-1)`，分别取 `sin(phase)` 与 `cos(phase)`；官方固定 `round_count=500` 时分母为 `499`。
- `own_gold_scaled`：己方两个角色当前毛金币总和，取值为 `clip((my_units_gold[0]+my_units_gold[1]) / 2000, 0, 3)`。
- `opp_gold_scaled`：对手当前毛金币总和，取值为 `clip(gold_opp / 2000, 0, 3)`。
- `gold_margin_scaled`：己方毛金币减对手毛金币，取值为 `clip((own_gold-gold_opp) / 500, -3, 3)`。
- `outer_gold_phase_sin` / `outer_gold_phase_cos`：外围金币生成机制近似的 20 回合相位。`phase=(round % 20)/20`，分别取 `sin(2*pi*phase)` 与 `cos(2*pi*phase)`。
- `snapshot_sin` / `snapshot_cos`：官方 `D=5` snapshot 周期相位。`phase=(round % 5)/5`，分别取 `sin(2*pi*phase)` 与 `cos(2*pi*phase)`。
- `last_snapshot_valid`：是否已经收到过至少一次有效 snapshot。

设计依据与边界：

- 金币相关 scalar 均使用毛金币口径；对手视野花费在局中不可知，因此不尝试构造净金币差。
- `game_phase_sin` 在整局中近似单调从 `-1` 到 `1`，承担进度信号；`game_phase_cos` 在中局最高，表达中期资源争夺窗口。
- `outer_gold_phase_*` 来自当前机制近似，不是官方直接公开的精确生成周期；如果正式赛外围金币机制变化，该特征可能失真。

v1 不加入当前/上一轮视野购买、上一轮 `k/order/actions/vp` 或 player id / 出生侧编码。这些项留作后续策略结构或数据增强方案确定后再评估。

## Tokens

v1 不引入 tokens，网络主线按 CNN 输入 `spatial_planes + scalars` 设计。后续如果网络结构需要 token，可在新 schema 中加入己方角色、可见敌人、可见 NPC、snapshot 区域或候选目标 token；不能在 `goldrush2_feature_v1` 中改变 shape 或语义。

## 记忆与 Belief

v1 只维护三类状态记忆：

- temporal observation ring buffer：最近 5 次 `visible_mask`、`gold_count`、`bomb_mask`、`visible_enemy_count`。
- obstacle memory：历史直接观测到的障碍/非障碍事实，以及基于中心对称的障碍状态推断。
- snapshot memory：最近一次和上上次有效 snapshot 中部分区域统计。

v1 不加入 `explored_mask`、`last_seen_age`、`remembered_gold_count`、`remembered_bomb_mask`、opponent hidden position belief、NPC hidden transition belief、金币生成概率图、炸弹刷新概率图或对手策略风格估计。

关键原则：belief 可以基于历史 observation 和已冻结的数据推断生成，但不能使用 simulator full state。

## 派生特征策略

v1 派生特征只包括：

- 曼哈顿距离图：`unit0_manhattan_distance` / `unit1_manhattan_distance`。
- 已知障碍 BFS 距离图：`unit0_known_obstacle_distance` / `unit1_known_obstacle_distance`。
- 区域 one-hot：`region_1_center_mask` 到 `region_5_right_mask`。
- snapshot broadcast maps：`last_snapshot_gold_remaining_map`、`prev_snapshot_gold_remaining_map`、`last_snapshot_occupants_map`。

这些派生均在 feature extractor 内计算，确保训练和部署一致。v1 不加入 frontier、remembered resource、static_map2 prior、legal destination 或 reachable-in-6 等派生面。

## 版本与兼容

v1 schema 固定为：

```text
feature_schema = "goldrush2_feature_v1"
```

训练 checkpoint、ONNX 导出和 C++ 部署必须记录并校验该版本。任何改变 feature shape、channel/scalar 顺序或语义的修改都必须使用新 schema 名称，不能复用 `goldrush2_feature_v1`。

## 测试要求

已完成的最小测试覆盖：

- C++ binding 可接收 simulator Python `GameInput` / `GameOutput`。
- `observe()` 输出基础 shape。
- 可见格更新当前基础 planes。
- `commit_action()` 只记录上一动作。
- reset 清空 memory。
- round regression fail-fast。
- runtime 可接入真实 rollout/evaluation。

v1 设计落地后还应增加：

- `feature_schema`、`channel_names`、`scalar_names` 与本文顺序完全一致的测试。
- `spatial_planes.shape == (38, 17, 17)`、`scalars.shape == (10,)` 的测试。
- 每个 channel/scalar 的定点样例测试。
- temporal ring buffer 的 `t0..t4` 顺序、缺失历史填 0 测试。
- obstacle 直接观测、中心对称推断、直接观测覆盖推断的测试。
- known-obstacle BFS 距离图的阻挡、未知格可通行、不可达 clip 测试。
- 跨 P1/P2 出生侧的 feature 对齐测试。
- snapshot 更新、last/prev 轮转、无历史 snapshot 填 0 测试。
- schema version 测试。
- 真实 rollout 中 memory 单调性和 reset 不串局测试。

## 后续可选扩展

以下方向不属于 `goldrush2_feature_v1`，后续若加入应使用新 schema：

- tokens：己方角色、可见敌人、可见 NPC、snapshot 区域或目标 token。
- temporal NPC：短期 `visible_npc_count` 窗口。
- `prev_snapshot_valid` 或更多 snapshot 统计 broadcast map。
- 当前/上一轮视野购买、上一轮 `k/order/actions/vp`、player id / 出生侧编码。
- `static_map=2` / 外围高额金币 prior；该信息不能从局中 `GameInput.grid` 直接观测得到，只有在地图模板识别方案冻结后再考虑。
- `explored_mask`、`last_seen_age`、`remembered_gold_count`、`remembered_bomb_mask`、frontier、legal destination、reachable-in-6 等其它 belief 或派生特征。
