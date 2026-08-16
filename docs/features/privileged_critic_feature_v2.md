# Privileged Critic Feature v2 设计草案

本文记录下一版 PPO privileged critic 输入方案。v2 的核心变化是：critic 不只读取 simulator full state，还显式读取 actor 当时的输入级信息态，用于降低部分可观测、视野购买和 actor belief 带来的 value aliasing。

对应 actor feature 版本固定为 `goldrush2_feature_v2`，v1 critic 语义见 `docs/features/privileged_critic_feature_v1.md`。

## 定位

v2 仍然只服务训练期 value function，不进入 actor、BC、ONNX/C++ 导出或平台提交路径。

输入来源分成两块：

```text
privileged block:
    simulator GameState / MapTemplate / OuterGoldState full state

actor-observation block:
    同一 agent_player_id、同一 action-time observation 下的 goldrush2_feature_v2 actor 输入
```

schema 名称：

```text
critic_feature_schema: goldrush2_privileged_critic_feature_v2
required_actor_feature_schema: goldrush2_feature_v2
```

critic feature 的时态继续与 actor observe 对齐：使用本回合行动前的 `round.start` 状态，即本回合开始时金币生成和炸弹刷新已经应用后的状态。actor-observation block 必须来自同一回合、同一玩家视角的 actor feature extractor 输出。

## 设计目标

v1 critic 只看到真实世界状态，但看不到 actor 的信息态。在 POMDP 下，这会把以下样本混在一起：

- full state 相同，但 actor 是否见过某炸弹不同。
- full state 相同，但 actor 是否在线推断出 static2 候选不同。
- full state 相同，但 actor 当前视野半径和可见边界不同。
- full state 相同，但 actor temporal observation ring buffer 不同。

这些样本在真实 return 上可能不同，因为当前 policy 的动作分布依赖 actor 输入，而不只依赖 full state。v2 的目标是让 critic 同时知道：

- 真实世界里有什么。
- actor 当时知道什么、记住什么、相信什么。

## 输出形态

第一版 v2 不整体附加 actor feature，而是只加入一组精选 actor 信息态 block：

```text
critic_planes: 39 x 17 x 17
critic_scalars: 20
tokens: none
```

拆分如下：

```text
privileged_planes: 26 x 17 x 17   # 与 v1 完全一致
actor_observation_planes: 13 x 17 x 17

privileged_scalars: 17            # 与 v1 完全一致
actor_observation_scalars: 3
```

设计取舍：

- actor-observation block 只保留会改变 policy 行动分布、且 full-state block 无法表达的 actor 信息态。
- 不重复加入 actor 的金币 temporal plane、敌人/NPC 可见 plane、己方位置、距离图、区域 mask、中心金币先验和金币 scalar；这些要么已有更精确 full-state 表达，要么不是第一版要解决的 VP/bomb/static2 credit 核心。
- actor-observation block 直接使用输入级 feature 派生，不使用 actor encoder、actor trunk hidden、decoder hidden 或 action head 中间结果。
- critic encoder 仍是独立参数。value loss 不得反传进 actor encoder 或 actor action head。
- snapshot 相关项虽然在真实世界状态上可被 full-state 金币分布部分替代，但它们表达 actor 收到过的区域级历史统计，因此保留。

## Plane 顺序

v2 plane 顺序固定为：

```text
# 0..25: privileged v1 block
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

# 26..38: actor-observation block
actor_visible_mask_t0
actor_visible_mask_t1
actor_visible_mask_t2
actor_visible_mask_t3
actor_visible_mask_t4
actor_bomb_belief_mask
actor_obstacle_known_mask
actor_obstacle_mask
actor_static_2_mask
actor_to_static2_distance
actor_last_snapshot_gold_generated_map
actor_last_snapshot_gold_remaining_map
actor_prev_snapshot_gold_remaining_map
```

前 26 个 plane 的语义、缩放、角色编号和时态完全沿用 `goldrush2_privileged_critic_feature_v1`。

后 13 个 plane 来自同一 transition 中保存给 actor 的 `goldrush2_feature_v2.spatial_planes`，只是按本文精选并加上 `actor_` 前缀。critic 文档不重新冻结这些 plane 的内部语义；其来源语义以 `docs/features/actor_feature_v2.md` 为准。

## Scalar 顺序

v2 scalar 顺序固定为：

```text
# 0..16: privileged v1 block
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

# 17..19: actor-observation block
actor_snapshot_sin
actor_snapshot_cos
actor_last_snapshot_valid
```

前 17 个 scalar 完全沿用 v1。后 3 个 scalar 来自同一 transition 中保存给 actor 的 `goldrush2_feature_v2.scalars`。

第一版不额外增加 `vision_spent` scalar。v1 已经提供 `own_gross_gold_scaled` 与 `own_net_gold_scaled`，critic MLP 可以从二者差值中恢复己方累计视野成本；若后续仍发现 VP value 难学，再讨论显式加入 `own_vision_spent_scaled` / `opp_vision_spent_scaled`。

## Actor 信息态的作用

actor-observation block 中最关键的新增信号是：

- `actor_visible_mask_t0..t4`：critic 知道 actor 当前和短期历史真正看见过哪里。
- `actor_bomb_belief_mask`：critic 知道 actor 当前对炸弹风险的综合 belief。该 plane 已覆盖 actor 能推断并记忆的炸弹信息，第一版不再附加 `actor_bomb_mask_t0..t4`。
- `actor_obstacle_known_mask` / `actor_obstacle_mask`：critic 知道 actor 的地图认知边界，而不是只看真实障碍。
- `actor_static_2_mask` / `actor_to_static2_distance`：critic 知道 actor 是否已经在线发现外围高额金币格。
- `actor_snapshot_*`：critic 知道 actor 从官方 snapshot 得到的区域统计。

这不是为了让 critic 变成 actor 的副本，而是为了估计“当前 policy 在当前信息态下会如何行动”。full-state block 解释真实机会和风险；actor-observation block 解释 policy 能否感知、记忆并利用这些机会和风险。

第一版刻意不加入：

- `actor_gold_count_t0..t4`：critic 已有真实全图 `gold_count`；actor 是否看见某金币可由 `actor_visible_mask_t*` 与 full-state 金币共同表达。
- `actor_visible_enemy_count_t0..t4`：critic 已有敌方真实位置；是否需要建模 actor 看见敌方，留作后续消融。
- `actor_visible_npc_count` / `actor_npc_crowded_mask`：critic 已有 NPC 真实位置和拥挤 mask。
- actor 己方位置、距离图和区域 mask：与 privileged block 重复。
- actor 金币 scalar 和 game phase scalar：与 privileged block 重复。
- `actor_center_gold_spawn_prob`：critic 已有完整中心金币先验；actor 已知障碍过滤差异第一版暂不加入。

## 不使用 Actor Trunk Feature

v2 明确不把以下张量作为 critic 输入：

- actor encoder 输出 `H_actor`。
- actor pooled context。
- GRU decoder hidden。
- action logits、vp logits、threshold residual hidden。

原因：

- 这些表征是为 policy logits 优化的压缩特征，不保证保留 value 需要的空间细节。
- 如果 value loss 反传进 actor trunk，会重新耦合 actor/critic，破坏当前分离主干设计。
- 如果 detach actor trunk feature，则 critic 依赖一个自己不能优化且会随 actor 更新漂移的中间表示。
- 输入级 actor feature 是稳定 schema，便于 checkpoint、shared memory、测试和文档冻结。

## 构造边界

每条 rollout transition 应按同一顺序构造：

```text
actor_feature = actor_extractor.observe(game_input)
critic_v1_feature = extract_privileged_critic_features(
    state=round_start_state,
    map_template=map_template,
    outer_gold_state=outer_gold_state,
    agent_player_id=agent_player_id,
)
actor_info_state = select_actor_info_state_planes_and_scalars(actor_feature)
critic_v2_feature = concat(critic_v1_feature, actor_info_state)
```

约束：

- `agent_player_id` 必须一致。actor block 的 `my_units[0/1]` 必须对应 privileged block 的 `own_unit0/1`。
- `round` 必须一致。禁止用下一回合 actor observation 拼接当前回合 full state。
- actor feature schema 必须等于 `goldrush2_feature_v2`，否则 fail-fast。
- actor planes/scalars 必须使用 PPO buffer 中同一 transition 的 actor feature，避免重复调用 extractor 造成状态推进或历史 ring buffer 偏移。
- 拼接只发生在训练/rollout 数据路径；部署 actor-only 路径不构造 critic v2。

## 与网络和 PPO 的关系

网络消费形态更新为：

```text
critic_spatial_planes: B x 39 x 17 x 17
critic_scalars: B x 20
fast_scalars: B x 2
critic_feature_schema: goldrush2_privileged_critic_feature_v2
```

critic encoder 仍使用独立 stem、独立 scalar tower 和独立 residual backbone。actor 路径不读取 critic feature。

PPO buffer 和 multiprocess shared memory 需要保存：

```text
actor:  43 x 17 x 17, scalars 10
fast:   scalars 2
critic: 39 x 17 x 17, scalars 20
```

实现上可以选择在 worker 侧直接写入完整 critic v2 feature，或在主进程组 batch 时由 `critic_v1_feature + actor_info_state` 拼接生成。但不管采用哪种方式，训练 batch 中的 critic tensor 必须是固定 shape，且 checkpoint metadata 必须记录 v2 schema 和 count。

## 兼容与 Checkpoint

v2 改变 critic input shape，但前 26 个 plane 和前 17 个 scalar 与 v1 完全一致，因此默认采用显式零扰动 critic inflation，而不是重新初始化 critic。

推荐迁移路径：

- actor 参数可以从旧 checkpoint 加载。
- critic 参数从 v1 checkpoint 零扰动 inflation 到 v2。
- optimizer state 不继承。
- 不允许隐式部分加载；schema 不匹配时必须通过显式 inflation 工具。
- critic v2 inflation 工具只处理 critic 输入 shape 迁移，要求 action head schema 已经是当前主线版本；更旧动作头 checkpoint 应先走对应动作头 inflation。
- inflation 后 checkpoint metadata 必须包含：

```text
critic_feature_schema = "goldrush2_privileged_critic_feature_v2"
critic_plane_count = 39
critic_scalar_count = 20
required_actor_feature_schema = "goldrush2_feature_v2"
inflated_from_critic_feature_schema = "goldrush2_privileged_critic_feature_v1"
critic_inflation = "v1_to_v2_zero_actor_info"
```

零扰动规则：

- critic stem 第一层卷积从 `in_channels=26` 扩展到 `in_channels=39`。
  - 旧 26 个 input channel 对应权重逐元素照抄。
  - 新增 13 个 actor-info input channel 权重置 0。
  - bias 照抄。
- critic scalar tower 第一层线性层从 `in_features=17` 扩展到 `in_features=20`。
  - 旧 17 个 input scalar 对应列逐元素照抄。
  - 新增 3 个 actor-info scalar 对应列置 0。
  - bias 照抄。
- critic encoder 后续卷积、SE block、FiLM 后续层、critic context MLP、value head 和 fast scalar 相关权重全部照抄。
- actor 参数、普通动作头、VP head、fast threshold head 不因 critic feature v2 inflation 改变。

如果网络实现中 critic stem 或 scalar tower 命名发生变化，inflation 工具必须 fail-fast，而不能用 fuzzy key 匹配。

该 inflation 的目标是：

```text
same privileged v1 input
same actor-info input 任意值
新增通道/新增 scalar 权重为 0
=> inflated v2 critic value == old v1 critic value
```

允许的数值误差仅来自浮点计算顺序，测试应使用严格容差，例如 `atol <= 1e-6`。

重新初始化 critic 只作为后续消融项，不作为主线默认迁移方式。原因是完全重置 critic 会显著改变早期 advantage，难以判断表现变化来自 actor 信息态本身，还是来自 critic 重新学习期。

## 测试要求

v2 至少需要新增以下测试：

- `critic_feature_schema == "goldrush2_privileged_critic_feature_v2"`。
- `critic_planes.shape == (39, 17, 17)`、`critic_scalars.shape == (20,)`。
- 前 26 个 plane 与 v1 critic extractor 在同一 full state 下逐元素一致。
- 前 17 个 scalar 与 v1 critic extractor 在同一 full state 下逐元素一致。
- 后 13 个 plane 与同 transition actor feature v2 中对应 channel 逐元素一致。
- 后 3 个 scalar 与同 transition actor feature v2 中 `snapshot_sin`、`snapshot_cos`、`last_snapshot_valid` 逐元素一致。
- actor/critic 的 `agent_player_id` 不一致时 fail-fast 或测试能捕获 unit mask 不对齐。
- actor feature schema 非 `goldrush2_feature_v2` 时 fail-fast。
- v1 checkpoint 经零扰动 inflation 后，在 actor-info 新增输入任意变化但 privileged v1 输入相同的情况下，critic value 与旧 checkpoint 一致。
- inflation 后 critic stem 新增 13 个 channel 权重全 0，scalar tower 新增 3 个 input 列全 0。
- inflation 后旧 26 个 stem channel 权重、旧 17 个 scalar input 列、后续 critic/value 参数逐元素照抄。
- PPO 双输入检查：policy loss 只更新 actor 参数，value loss 只更新 critic 参数。
- multiprocess rollout shared memory shape 与 batch schema 更新后，serial/mp rollout 输出 logprob/value finite。

## 待讨论

- 是否需要显式添加 `own_vision_spent_scaled`、`opp_vision_spent_scaled` 或当前 active vision radius scalar。
- 是否需要在 critic v2 之后增加辅助 value head，单独预测 VP 后的信息收益或炸弹损失。
