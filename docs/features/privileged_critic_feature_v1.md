# Privileged Critic Feature v1 设计

本文冻结当前 PPO privileged critic 的训练期输入语义。对应实现位于 `training/rl/privileged_critic_features.py`，网络消费语义见 `docs/policy_network_design.md`。

## 定位

privileged critic feature 只服务训练期 value function，不进入 actor、BC、ONNX/C++ 导出或平台提交路径。

输入来源为 simulator full state，而不是官方 `GameInput`：

```text
source: GameState / RoundStepEnv full state
critic_feature_schema: goldrush2_privileged_critic_feature_v1
```

它可以包含全图金币、全图炸弹、全图障碍、双方真实角色位置、NPC 真实位置、双方金币和回合进度等信息。critic feature 的时态应与 actor observe 对齐：使用本回合行动前的 `round.start` 状态，即本回合开始时的金币生成和炸弹刷新已经应用后的状态。actor 仍只能使用 `goldrush2_feature_v2`。

## 输出形态

critic feature 不绑定 actor feature 的 plane 数量。当前 v1 固定为：

```text
critic_planes: Cc x 17 x 17
critic_scalars: Sc
tokens: none
```

当前已确认 `Cc = 26`、`Sc = 17`。即使某些表达方式沿用 actor feature 的缩放，critic planes 与 actor planes 仍属于不同 schema，名称、顺序和语义在本文单独冻结。

## 设计原则

- 不沿用 actor feature 中带局部观察语义的名称，例如 `visible_mask_t*`、`visible_enemy_count_t*`、`obstacle_known_mask`。
- 如果保留 temporal planes，必须表示 full-state 历史或明确的统计趋势，不能伪装成 visible history。
- 如果当前 full state 已足够表达某项信息，优先使用当前真值；temporal stack 只在能降低 value prediction 难度时加入。
- scalar 优先承载全局局势和时间信息，plane 优先承载空间分布和位置局部性。
- 所有缩放必须固定、可测、与 actor feature 独立。
- 缺失 privileged feature 时训练应 fail-fast，禁止静默退回 actor feature。
- critic feature 必须以当前训练样本的 `agent_player_id` 为视角构造；`first_player_id` 只表示本回合执行先后，不能用于决定己方/敌方身份。
- unit 编号必须与 actor `GameInput.my_units` 和动作槽位一致，禁止按位置、金币、距离、先后手或 replay 数组顺序重排。

## 已确认 Spatial Planes

当前已确认 plane 顺序如下：

```text
gold_count
bomb_mask
obstacle_mask
npc_count
npc_crowded_mask
static_2_mask
static_2_available_mask
next_static2_region_mask
center_gold_spawn_prob
own_unit0_mask
own_unit1_mask
enemy_unit0_mask
enemy_unit1_mask
unit0_manhattan_distance
unit1_manhattan_distance
unit0_obstacle_distance
unit1_obstacle_distance
enemy_unit0_manhattan_distance
enemy_unit1_manhattan_distance
enemy_unit0_obstacle_distance
enemy_unit1_obstacle_distance
region_1_center_mask
region_2_up_mask
region_3_left_mask
region_4_down_mask
region_5_right_mask
```

语义与缩放：

- `gold_count`：当前全图真实金币数量，取值为 `clip(raw_gold / 20, 0, 3)`；无金币格为 0。
- `bomb_mask`：当前全图真实炸弹位置，炸弹格为 1，否则为 0。
- `obstacle_mask`：全图真实障碍，障碍格为 1，否则为 0；不再区分 known/unknown。
- `npc_count`：当前全图每格 NPC 数量，取值为 `clip(count / 3, 0, 1)`。
- `npc_crowded_mask`：当前全图每格 NPC 数量是否 `>=3`。
- `static_2_mask`：外围 `static_map=2` 特殊非阻挡格，候选格为 1，否则为 0。
- `static_2_available_mask`：当前状态下 static2 high batch 的合法候选格，要求 `static_2_mask=1` 且当前无炸弹、无玩家、无 NPC 占用。该 plane 是 action-time 状态派生量；若本回合 high batch 已经触发，它不回放触发前的候选集合。
- `next_static2_region_mask`：下一次未来 static2 high batch 的目标外围区域 mask。若 `outer_state.next_static2_region == r`，则 `region_r_mask` 对应格为 1，其它格为 0；只允许 `r in {2,3,4,5}`。该信息来自 simulator 机制状态中随 `next_static2_round` 一起预采样的外生随机量。
- `center_gold_spawn_prob`：中心小额金币生成概率图的归一化版本。中心候选格取 `exp(-B * ((row - 8)^2 + (col - 8)^2))`，其它格为 0；当前默认 `B=0.06735`。该 plane 不含总体强度 `A`，范围约为 `0..1`。
- `own_unit0_mask` / `own_unit1_mask`：当前己方角色 0 / 1 真实位置 one-hot。
- `enemy_unit0_mask` / `enemy_unit1_mask`：当前敌方角色 0 / 1 真实位置 one-hot。
- `unit0_manhattan_distance` / `unit1_manhattan_distance`：每格到己方角色 0 / 1 的曼哈顿距离，取值为 `distance / 32`。
- `unit0_obstacle_distance` / `unit1_obstacle_distance`：每格到己方角色 0 / 1 的真实障碍 BFS 距离，取值为 `clip(distance / 32, 0, 2)`；不可达格取最大值 2。
- `enemy_unit0_manhattan_distance` / `enemy_unit1_manhattan_distance`：每格到敌方角色 0 / 1 的曼哈顿距离，取值为 `distance / 32`。
- `enemy_unit0_obstacle_distance` / `enemy_unit1_obstacle_distance`：每格到敌方角色 0 / 1 的真实障碍 BFS 距离，取值为 `clip(distance / 32, 0, 2)`；不可达格取最大值 2。
- `region_1_center_mask` / `region_2_up_mask` / `region_3_left_mask` / `region_4_down_mask` / `region_5_right_mask`：官方 5 区域划分 one-hot。

设计说明：

- `gold_count`、`bomb_mask`、`npc_count`、`npc_crowded_mask`、角色位置、距离图和区域 mask 的 scaling/clipping 沿用 actor feature 中对应表达。
- `obstacle_mask` 使用 full-state 真值，不需要 actor feature 的 `obstacle_known_mask`。
- 敌方角色使用 `enemy_unit0_mask` / `enemy_unit1_mask` 分开表达，不使用 actor feature 的 `visible_enemy_count_t0` 聚合语义。
- 敌方距离图与己方距离图对称表达，帮助 critic 显式比较双方对金币、中心和风险格的相对可达性。
- `static_2_mask`、`next_static2_region_mask` 和 `center_gold_spawn_prob` 是机制先验类 privileged plane。它们来自地图模板、机制状态和已冻结机制建模结论，不来自 actor 可见 observation。
- `static_2_available_mask` 只使用 action-time state，不读取本回合执行结果；它表达当前状态下哪些 static2 格满足生成候选条件，不表达本回合开始生成前的历史候选集合。
- `next_static2_region_mask` 必须由 simulator 在计划下一次 high batch 时预先采样并保存，不能在 feature 提取时根据未来实际生成事件、未来占用结果或未来金币分配反推。critic feature 提取发生在本回合开始生成之后，因此若本回合刚触发 high batch，该 plane 应指向下一次未来 high batch 的区域。
- 第一版不保留 actor 的 temporal stack；上述 planes 均为 action-time full state 当前真值或由当前真值确定的派生量。

## 待讨论 Plane 类别

- 到中心、高金币格、炸弹或 NPC 风险区的派生距离图。
- 区域金币统计 broadcast map。
- 可选趋势：金币变化、角色位置历史、最近 pickup/bomb 事件。

## 已确认 Scalar Features

当前已确认 scalar 顺序如下：

```text
game_phase_sin
game_phase_cos
remaining_rounds_scaled
bomb_refresh_phase_sin
bomb_refresh_phase_cos
static2_high_imminence
own_gross_gold_scaled
own_net_gold_scaled
opp_gross_gold_scaled
opp_net_gold_scaled
net_gold_margin_scaled
own_unit0_gold_scaled
own_unit1_gold_scaled
enemy_unit0_gold_scaled
enemy_unit1_gold_scaled
total_gold_on_map_scaled
center_gold_on_map_scaled
```

语义与缩放：

- `game_phase_sin` / `game_phase_cos`：整局进度的半周期相位，沿用 actor feature 口径。`phase=-pi/2+pi*round/(round_count-1)`，分别取 `sin(phase)` 与 `cos(phase)`。
- `remaining_rounds_scaled`：剩余回合比例，`(round_count - round) / round_count`。
- `bomb_refresh_phase_sin` / `bomb_refresh_phase_cos`：炸弹刷新 20 回合相位。`phase=(round % 20)/20`，分别取 `sin(2*pi*phase)` 与 `cos(2*pi*phase)`。
- `static2_high_imminence`：下一次未来 static2 high batch 的临近程度，使用机制状态中的 `next_static2_round` 计算。critic feature 提取发生在本回合开始生成之后，因此 `next_static2_round` 应指向严格晚于当前 `round` 的下一次触发；若 `next_static2_round <= round` 应 fail-fast 暴露时序错误。令 `rounds_to_next_static2_high = next_static2_round - round`，取值为 `exp(-(rounds_to_next_static2_high - 1) / 8)`；下回合即将触发时为 1，距离越远越接近 0。
- `own_gross_gold_scaled`：己方两个角色当前毛金币总和，取值为 `clip(own_gross / 2000, 0, 3)`。
- `own_net_gold_scaled`：己方当前净金币，取值为 `clip(own_net / 2000, -1, 3)`。
- `opp_gross_gold_scaled`：对手两个角色当前毛金币总和，取值为 `clip(opp_gross / 2000, 0, 3)`。
- `opp_net_gold_scaled`：对手当前净金币，取值为 `clip(opp_net / 2000, -1, 3)`。
- `net_gold_margin_scaled`：己方净金币减对手净金币，取值为 `clip((own_net - opp_net) / 500, -3, 3)`。
- `own_unit0_gold_scaled` / `own_unit1_gold_scaled`：己方角色 0 / 1 当前携带金币，取值为 `clip(unit_gold / 1000, 0, 3)`。
- `enemy_unit0_gold_scaled` / `enemy_unit1_gold_scaled`：敌方角色 0 / 1 当前携带金币，取值为 `clip(unit_gold / 1000, 0, 3)`。
- `total_gold_on_map_scaled`：当前全图未拾取金币总量，取值为 `clip(total_map_gold / 2000, 0, 3)`。
- `center_gold_on_map_scaled`：当前中心区域未拾取金币总量，取值为 `clip(center_gold / 1000, 0, 3)`。

设计说明：

- critic 使用 full-state 净金币，因此不复用 actor scalar 中基于可见信息限制的 `gold_margin_scaled` 语义。
- `snapshot_sin`、`snapshot_cos` 和 `last_snapshot_valid` 不进入 critic v1；snapshot 是 actor 局部视角补偿机制，privileged critic 直接读取 full state。
- 单位携带金币第一版放入 scalar，不重复添加 position-local carry plane。若后续 EV 仍不足，再讨论是否增加对应 plane。
- vision spent 暂不单独放入 scalar，因为 gross/net 已经隐含其差值；如后续 value 需要解释视野购买行为，可在 v2 添加。
- 早期 actor feature 中曾把同一 `round % 20` 相位命名为 `outer_gold_phase_*`。机制建模显示 static2 high batch 没有固定 20 回合相位；该相位主要对应炸弹刷新，因此 critic schema 使用 `bomb_refresh_phase_*` 命名。
- `static2_high_imminence` 不直接暴露原始倒计时 round，而是用单调衰减表征“未来临近程度”。本回合若刚刚触发 high batch，其生成出的金币已经进入 `gold_count`、`total_gold_on_map_scaled` 和 `center/static2` 相关空间分布，不需要再用该 scalar 标记“刚生成”。该 scalar 只读取 action-time 已存在的机制状态，不包含本回合 action 后的 high batch 分配结果，因此不破坏 value function 的 Bellman 一致性。
- `static2_high_imminence` 与 `next_static2_region_mask` 共同描述下一次未来 static2 high batch 的时间和区域。二者属于训练期 critic 降噪信息；actor、BC、ONNX/C++ 导出路径不得读取。

## 角色与 Unit 编号

critic feature 从 full state 构造时必须使用 agent 视角：

```text
own_unit0 = state.players[agent_player_id].units[0]
own_unit1 = state.players[agent_player_id].units[1]

enemy_unit0 = state.players[3 - agent_player_id].units[0]
enemy_unit1 = state.players[3 - agent_player_id].units[1]
```

对应关系必须同时用于位置 mask、距离图和携带金币 scalar：

```text
own_unit{i}_mask
unit{i}_manhattan_distance
unit{i}_obstacle_distance
own_unit{i}_gold_scaled

enemy_unit{i}_mask
enemy_unit{i}_manhattan_distance
enemy_unit{i}_obstacle_distance
enemy_unit{i}_gold_scaled
```

设计约束：

- `agent_player_id` 来自 rollout 样本或 env reset 配置，是训练 actor 的玩家身份。
- `first_player_id` 只影响本回合 transition 执行顺序，不改变 feature 中 own/enemy 的定义。
- actor feature 中 `own_unit0_mask` / `own_unit1_mask` 来自 `GameInput.my_units[0/1]`；critic 的 `own_unit0/1` 必须与同一 `agent_player_id` 下的 `GameInput.my_units[0/1]` 对齐。
- 敌方 unit 可以使用 privileged 的真实 `unit.id`，但 enemy mask、enemy 距离图和 enemy carry scalar 必须使用同一编号。
- 如果从 replay JSON 或离线样本读取 full state，必须按显式 `player id` 和 `unit id` 解析，不能依赖数组出现顺序。

## 待讨论 Scalar 类别

- 若后续需要更细表达 static2 high batch 倒计时，可考虑少量 RBF countdown basis；第一版先只冻结 `static2_high_imminence` 单 scalar。

## 与网络和 PPO 的关系

privileged critic feature 应进入独立的 critic encoder：

```text
critic_planes/scalars -> GoldRushPrivilegedCriticEncoder -> critic_mlp -> value
```

actor 路径继续使用：

```text
actor_planes/scalars -> GoldRushActorFeatureEncoder -> action/logprob/entropy
```

PPO buffer 应同时保存 actor feature 和 critic feature。policy loss 只读取 actor feature；value loss 只读取 critic feature。

## 已冻结的 Simulator 约束

- `OuterGoldState` 显式保存 `next_static2_round` 与 `next_static2_region`；二者在 reset 和每次 high batch 结束后同时预采样、同时更新。
- outer gold、center gold、bomb 与 NPC 使用独立的命名随机流。static2 schedule 的预采样不得扰动中心金币、炸弹或 NPC 的随机序列。
- critic 只能使用本回合开始资源生成完成后的 action-time state；不得读取 terminal 结果、本回合执行后的状态或未来实际生成结果。
