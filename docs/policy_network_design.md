# Policy Network 设计

本文冻结当前训练与部署共享的网络结构和动作概率语义。Actor feature 输入见 `docs/features/actor_feature_v2.md`，PPO 栈见 `docs/rl_architecture.md`，游戏动作规则以 `gamerules/gamerules.md` 为准。

## 定位

网络使用 actor/critic 双输入。actor 输入使用可提交路径 feature v2：

```text
actor_spatial_planes: B x 43 x 17 x 17
actor_scalars: B x 10
fast_scalars: B x 2
actor_feature_schema: goldrush2_feature_v2
action_head_schema: candidate_cell_residual_v1_fast_threshold_calibrated_base_v1_vp_final_position_v1
```

critic 输入使用训练期 privileged critic feature：

```text
critic_spatial_planes: B x 39 x 17 x 17
critic_scalars: B x 20
fast_scalars: B x 2
critic_feature_schema: goldrush2_privileged_critic_feature_v2
```

网络输出官方动作字段和 value：

```text
actions: B x 6，取值 0..4
k: B，取值 0..6
order: B，取值 0..1
vp: B，取值 0..2
threshold_raw: B，连续 stochastic action
threshold: B，映射到 `[4, 30]` 后供 fast controller 使用
logprob/value/entropy diagnostics: B
```

网络不维护 observation memory，不执行金币、炸弹和终局结算。actor 路径不访问 replay/full-state privileged 信息，也不模拟敌方或 NPC 行动。critic 路径只在训练期读取 privileged critic feature，部署、BC、ONNX/C++ 提交路径不得依赖 critic 输入。

## Encoder

actor 和 critic 使用两套参数独立的 encoder。第一版两套 encoder 主体结构相同，但输入 channel/scalar 数量和 schema 不同，不共享任何可训练参数：

```text
Actor stem:
    Conv3x3(43 -> 96)
    SiLU
    Conv3x3(96 -> 96)
    SiLU

Actor scalar tower:
    Linear(10 -> 96)
    SiLU
    Linear(96 -> 96)
    SiLU
    Linear(96 -> 192)  # FiLM gamma/beta

Fusion:
    H = H * (1 + gamma) + beta

Backbone:
    8 x SE-ResidualBlock(width=96, reduction=4)
```

critic encoder 使用同型结构，但 stem 输入为 `39`，scalar tower 输入为 `20`：

```text
Critic stem:
    Conv3x3(39 -> 96)
    SiLU
    Conv3x3(96 -> 96)
    SiLU

Critic scalar tower:
    Linear(20 -> 96)
    SiLU
    Linear(96 -> 96)
    SiLU
    Linear(96 -> 192)  # FiLM gamma/beta
```

SE residual block 不使用 normalization。FiLM 最后一层零初始化；residual branch 第二个卷积使用小 gain；SE gate 最后一层零初始化。默认激活为 SiLU，ReLU 只保留为部署实验配置。

第一版 FiLM 只在 stem 后、进入 residual blocks 前作用一次。多层 FiLM/per-block FiLM 暂缓，不进入本版默认结构；如后续 critic EV 仍不足，可作为独立网络消融项加入配置。

actor encoder 输出 `H_actor: B x 96 x 17 x 17`。actor 使用全局 average/max pooling 和两个己方角色初始位置处的 feature gather：

```text
actor_context = MLP(concat(avg, max, unit0_local, unit1_local))
actor_hidden = 256
```

critic encoder 输出 `H_critic: B x 96 x 17 x 17`。critic 使用全局 average/max pooling，以及四个真实角色位置处的 feature gather：

```text
critic_context = MLP(concat(
    avg,
    max,
    own_unit0_local,
    own_unit1_local,
    enemy_unit0_local,
    enemy_unit1_local,
))
critic_context_dim = width * 6
critic_hidden = (256, 128)
```

fast option 接入后，critic value MLP 额外接收 `fast_scalars`：

```text
critic_value_input = concat(critic_context, fast_scalars)
critic_mlp_input = critic_context_dim + 2
```

这两个 scalar 只进入 threshold head 和 critic value，不进入 actor encoder FiLM 或普通动作 decoder。旧 critic checkpoint inflation 时，critic 第一层新增的 2 个输入列初始化为 `0`，使初始 value 输出与旧 checkpoint 完全一致；之后 PPO 再让 critic 学会使用 fast 有效性 belief。

critic gather 使用 privileged critic schema 中的角色位置 channel：

```text
own_unit0_mask:   critic channel 9
own_unit1_mask:   critic channel 10
enemy_unit0_mask: critic channel 11
enemy_unit1_mask: critic channel 12
```

critic value 不依赖采样动作、actor local gather 或 decoder hidden。critic 的四角色 gather 只服务 value function，不能进入 actor action/logprob 路径。

PPO 中 `policy_loss` 和 entropy 只更新 actor encoder、actor heads、embedding 和 decoder；`value_loss` 只更新 critic encoder 和 `critic_mlp`。启动时会 fail-fast 检查 actor/critic 参数集合不重叠且覆盖全部模型参数。

## 动作概率模型

`k` 决定六个官方动作槽位的角色归属，`order` 决定两个角色的实际执行顺序。当前策略先将二者合成 14 类：

```text
ko = 2 * k + order
k = ko // 2
order = ko % 2
```

联合概率分解为：

```text
p(ko, vp, a_exec[0:6], threshold_raw | state)
= p(ko | state)
  * p(vp | state)
  * product_t p(a_exec[t] | state, ko, prefix[0:t], simulated_positions[t])
  * p(threshold_raw | state, ko, vp, a_exec[0:6])
```

`vp` 不影响本回合移动，首版保持独立。六步动作使用共享 GRU decoder，默认 `decoder_hidden=128`、embedding width `16`。每步输入：

```text
actor context
ko embedding
当前执行角色 embedding
当前角色模拟位置处的 H feature
另一己方角色模拟位置处的 H feature
前一步 action embedding（首步 BOS）
step embedding
```

GRU hidden 先经过旧 action head 得到 base logits：

```text
base_logits = decoder_action_head(hidden)
```

随后对五个候选落点做 candidate-cell gather，并用零初始化 residual head 逐动作评分：

```text
candidate_local[action] = H_actor[candidate_position[action]]
candidate_delta[action] = candidate_action_head(
    hidden,
    candidate_local[action],
    action_candidate_embedding[action],
)
logits = base_logits + candidate_delta
```

`candidate_action_head` 最后一层零初始化。旧 `autoregressive_head_v1` checkpoint 经过显式 inflation 后，初始 `candidate_delta == 0`，策略行为与旧 head 完全一致。该 residual 路径只提供对 `to_static2_distance`、`static_2_mask`、`bomb_belief_mask`、局部金币/障碍等 spatial feature 的直接动作消费路径，不改变官方动作空间或 `k/order` 语义。

动作生成按真实执行顺序进行，最终通过 `official_slot_table[ko]` scatter 回官方槽位。特别地：

- `order=0`：先生成角色 0 的 `actions[0:k]`，再生成角色 1 的 `actions[k:6]`。
- `order=1`：先生成角色 1 的 `actions[k:6]`，再生成角色 0 的 `actions[0:k]`。
- `k=0/6` 仍保留两个合法 `order` 类，不合并官方动作。

六步共享同一 decoder 参数，禁止重新拆成六套独立 head。

## Fast Threshold Head

threshold 是下一回合 fast controller 的参数，第一版不进入 actor scalar schema，避免扰动现有 actor encoder FiLM。模型额外接收：

```text
fast_scalars: B x 2
  p_fast_full_realization
  fast_effective_confidence
```

这两个量来自部署侧同样可计算的 Fast Full-Realization Belief，而不是 simulator 的隐藏真实先手率。`p_fast_full_realization` 是 strict proxy：只要 fast 后的实际持币增量小于根据上次可见金币模拟出的 `expected_gain`，就按失败样本更新；它不是 ratio 均值。

threshold head 接在动作采样之后。六步 decoder 完成后，使用模拟出的两个己方最终位置从 `H_actor` gather 局部 feature。当前 head 采用 calibrated base + neural residual：

```text
threshold_input = concat(
    actor_context,
    final_unit0_local,
    final_unit1_local,
    fast_scalars,
)

threshold_mlp:
    Linear(256 + 96 + 96 + 2 -> 128)
    SiLU
    Linear(128 -> 64)
    SiLU
    Linear(64 -> 1)  # residual_raw

base_T =
    11, if p_fast_full_realization <= 0.3
    6,  if p_fast_full_realization >= 0.6
    linear ramp from 11 to 6 otherwise
base_raw = logit((base_T - 4) / (30 - 4))
mu_raw = base_raw + residual_raw
```

训练时采样 raw action：

```text
threshold_raw ~ Normal(mu_raw, std_raw)
threshold = 4 + (30 - 4) * sigmoid(threshold_raw)
```

PPO logprob 在 `threshold_raw` 的 unbounded Normal 空间计算；`threshold` 只是 simulator/runtime 使用的确定性变换。这样 threshold 可以通过标准 policy gradient 学习，不依赖 `gold >= threshold` 的不可导触发条件。

给 fast controller 的整数阈值统一为：

```text
threshold_int = floor(threshold + 0.5)
threshold_int = clamp(threshold_int, 4, 30)
```

训练和 C++ runtime 必须使用同一口径，不能使用 Python `round()` 的 banker rounding。

第一版使用全局 learnable `threshold_log_std`，不做 state-dependent std。推荐初始化：

```text
residual_raw = 0
初始 p_fast_full_realization = 0.8 -> base_T = 6
threshold_log_std ~= -1.3
```

`threshold_mlp` 最后一层权重和 bias 都为 `0`，使接入初期完全等于 calibrated base。`threshold_log_std` 训练和导出时应 clamp 到稳定范围，例如 `[-3.0, 0.0]`。

部署默认也使用 stochastic action 口径。C++ runtime 应从模型输出的分布参数采样普通动作和 `threshold_raw`，再执行同一 sigmoid 与整数化逻辑；随机流必须由 runtime 显式维护，避免训练、评估和平台提交之间出现隐式语义漂移。

## Belief 位置模拟

decoder 从现有 planes 读取：

```text
obstacle_known_mask: channel 20
obstacle_mask:       channel 21
own_unit0_mask:      channel 24
own_unit1_mask:      channel 25
```

已知障碍定义为两个障碍 plane 的交集；未知格按可通行处理。每一步在 GPU 上同时计算五个候选位置，确定阻挡包括：

- 越过 `17x17` 边界。
- 进入已知障碍。
- 进入另一己方角色当前模拟位置。

NPC 不阻挡玩家。可见敌人的执行时位置不确定，不进入 hard mask。被确定阻挡的动作 logits 被 mask；`STAY` 始终提供合法等待。采样动作后只更新当前执行角色的位置，后一步继续使用更新后的两个位置。

这里模拟的是当前 obstacle belief，不是真实地图保证。障碍推断与直接观测的区分能力由 actor feature schema 本身决定。

实现约束：

- 六步只允许固定长度 Python 循环，不允许按 batch 样本循环。
- 位置、候选动作、mask、hidden 和调度表必须保持为 device Tensor。
- 六步内部禁止 `.cpu()`、`.item()`、NumPy 或 worker 往返。
- `execution_role_table`、`official_slot_table` 和 action delta 注册为 model buffer。

## 模型 API

公共训练接口：

```text
act(
    actor_spatial_planes, actor_scalars,
    fast_scalars,
    critic_spatial_planes, critic_scalars,
    deterministic=False,
) -> PolicyAction

evaluate_actions(
    actor_spatial_planes, actor_scalars,
    fast_scalars,
    critic_spatial_planes, critic_scalars,
    actions, k, order, vp, threshold_raw,
) -> PolicyEvaluation
```

`act()` 使用 actor feature 按 `deterministic` 参数采样或取贪心 `ko/vp/actions`，再基于已选动作后的模拟终点采样或取中心值 `threshold_raw`，并使用 critic feature 通过 critic encoder 输出 value。训练和默认部署均使用 stochastic 路径；deterministic 只用于等价测试、导出 smoke 或明确指定的消融。`evaluate_actions()` 从保存的 `k/order` 恢复 `ko`，按保存动作 teacher force 同一个 decoder，并用保存的 `threshold_raw` 重算 threshold logprob；value 由 critic feature 重算。

部署/BC/ONNX 路径应保留 actor-only 入口。该入口只能返回 action、threshold、logprob/entropy 或使用占位 value，不得要求 privileged critic feature。PPO 训练路径必须使用双输入接口。

`forward()` 是导出友好的 actor-only 包装，不用于 PPO 更新。默认提交路径应支持 stochastic 采样；若导出图本身只输出 logits/mu/log_std，则采样由 C++ runtime 在 ONNX 推理后完成。

若 teacher-forced 动作被同一规则 mask 判为非法，立即报错。这通常表示槽位映射、位置递推或 feature 不一致，禁止静默赋予极小概率。

## PPO 概率与熵

rollout 保存的联合 logprob：

```text
log p(ko | state)
+ log p(vp | state)
+ sum_t log p(a_exec[t] | prefix_t, simulated_positions[t])
+ log p(threshold_raw | state, ko, vp, a_exec[0:6])
```

PPO buffer 和 multiprocess shared memory 保存：

```text
actor_spatial_planes/actor_scalars
fast_scalars
critic_spatial_planes/critic_scalars
actions/k/order/vp/threshold_raw/threshold_int/old_logprob/value
```

decoder hidden、执行顺序动作、`ko`、连续 `threshold` 和 threshold head 输入均可由现有字段确定，不进入 buffer。`threshold_int` 保存用于对齐环境执行和诊断。PPO 更新必须 teacher force buffer 动作和 `threshold_raw`，禁止重新采样或用当前 argmax/均值作为 prefix。模型参数不变时，rollout logprob 与重算 logprob 必须在浮点误差内一致。

threshold 是 delayed action component：第 `t` 条 transition 保存 `threshold_raw_t` 和对应 logprob，第 `t+1` 回合 fast controller 的后果通过标准 reward/return/GAE 回传到 `advantage_t`。第一版不拆 `threshold_loss`，也不把 `threshold_logprob_t` 搬到第 `t+1` 条 transition；所有 actor component 共用同一个 PPO clipped objective 和同一个 `advantage_t`。

fast option 的 success/miss/path-fail/fallback、belief、first-rate 和 one-step delta 等诊断按 update 聚合，不作为模型输入或 PPO logprob 重算字段保存。第一版不加入逐次 debug-only 字段。

policy loss、KL 和 entropy 只读取 actor feature；value loss、old value 对齐和 explained variance 只读取 critic feature。`old_logprob` 仍来自 actor 路径，`old_value` 来自 critic 路径。

PPO entropy bonus 按动作子头分别加权，默认值保持旧量级；实验中可通过 `PpoConfig.entropy_action_coef`、`PpoConfig.entropy_ko_coef`、`PpoConfig.entropy_vp_coef` 单独调整三路探索强度：

```text
action_entropy = mean_t H(action_t | prefix_t) / log(5)
ko_entropy = H(ko) / log(14)
vp_entropy = H(vp) / log(3)
threshold_entropy = H(Normal(mu_raw, std_raw))

entropy_bonus =
    entropy_action_coef * action_entropy
  + entropy_ko_coef * ko_entropy
  + entropy_vp_coef * vp_entropy
  + beta_threshold_entropy * threshold_entropy
```

默认值为 `entropy_action_coef=0.0100`、`entropy_ko_coef=0.0040`、`entropy_vp_coef=0.0003`。

`beta_threshold_entropy` 第一版应很小或为 `0`，先通过 `threshold_log_std` 初始化提供探索，避免 threshold 噪声长期主导 fast 行为。action entropy 暂以完整五类 `log(5)` 归一化；mask 后只有少量合法动作时指标会自然下降。masked logits 使用 dtype 有限最小值，避免 `0 * -inf` 产生 NaN。

## Rollout 与性能

serial 和 multiprocess rollout 均应在 action-time state 下同时构造 actor feature 和 critic feature，再调用 PPO 双输入 `model.act()`。一次 feature request 仍只返回完整 `GameOutput`、logprob 和 value；六步 decoder 不与 worker 中途通信。

multiprocess rollout 的 worker 负责提取两套 feature：

```text
policy_runtime.FeatureExtractor.observe(GameInput) -> actor feature
Fast Full-Realization Belief runtime state -> fast_scalars
extract_privileged_critic_features(GameState, MapTemplate, OuterGoldState, agent_player_id, actor_features) -> critic feature
```

主进程/GPU 只负责批量推理。shared memory 与 PPO batch 保存两套 feature，shape 固定为：

```text
actor:  43 x 17 x 17, scalars 10
fast:   scalars 2
critic: 39 x 17 x 17, scalars 20
```

自回归前的完整模型双实例基线约为：

```text
pair_count=32, round_count=500, workers=32
update wall:                 ~40.7s
worker action wait:          ~16.3ms/transition
mean feature batch size:     ~31.85
```

decoder 引入六步串行 GPU 数据依赖。修改 decoder hidden、worker 数或 batching 参数后必须重新 profile，不能假定旧参数仍最优。`inference_model_sample_ms` 必须在 CUDA synchronize 后结束计时。

## 初始化与兼容性

- `ko/vp/decoder_action` 输出层使用 `std=0.01` 小初始化。
- `candidate_action_head` 最后一层权重和 bias 为 0，使 residual 初始严格为 0。
- `threshold_mlp` 最后一层权重和 bias 为 0，使 threshold residual 初始严格为 0；初始 threshold 由 calibrated base 决定。
- `vp_head` 接收 `actor_context` 与动作解码后的两个己方角色 final-position local feature；从旧 fast-threshold checkpoint inflation 时，旧 `actor_context` 权重照抄，新增 final-position 列置 0。
- `threshold_log_std` 初始约为 `-1.3`，训练时 clamp 到稳定范围。
- `vp` bias 保持初始先验 `(0.90,0.07,0.03)`。
- GRU input weight 使用 Xavier，hidden weight 使用 orthogonal，bias 为 0。
- embedding 使用小正态初始化。
- value 输出层使用小初始化，使初始 value 接近 0。
- BC 训练只优化 actor 参数。PPO 从 BC checkpoint 初始化时，只加载 actor 路径；critic encoder 与 critic head 按 privileged critic schema 随机初始化或从专门 critic checkpoint 加载，不能默认拷贝 actor encoder，因为 actor/critic 输入 channel 与语义不同。

主线不维护隐式部分加载。当前迁移链路由显式 inflation 工具承担：

1. fast threshold calibrated-base inflation 只支持 feature v2 / `action_head_schema=candidate_cell_residual_v1` checkpoint：补齐 fast threshold residual head，并在 critic value 第一层追加 2 个 `fast_scalars` 输入列且置零，使初始普通动作分布和旧 value 输出保持不变。输出 schema 为 `candidate_cell_residual_v1_fast_threshold_calibrated_base_v1`。
2. VP final-position inflation 只支持 `candidate_cell_residual_v1_fast_threshold_calibrated_base_v1` checkpoint：默认把 `vp_head.weight` 从 `[3, actor_hidden]` 扩展到 `[3, actor_hidden + 2 * width]`，旧列照抄，新增 final-position 列置零，使初始 VP logits 完全不变。若旧 VP head 已饱和到几乎永不买视野，可显式使用 `--reset-vp-head-prior P0 P1 P2`，将 `vp_head.weight` 全置 0、bias 设为 `log([P0, P1, P2])`，用非零 VP 先验换取探索样本。输出 schema 为当前主线 schema。
3. privileged critic feature v2 inflation 应在动作头 schema 已经升级到当前主线后执行，只支持 critic feature v1 shape 的 PPO checkpoint：把 critic stem 从 `26 -> 39`、critic scalar tower 从 `17 -> 20`，旧输入权重照抄，新增 actor-info 通道/列置零，使初始 value 与旧 critic 完全一致。输出 `critic_feature_schema=goldrush2_privileged_critic_feature_v2`。

旧 factorized checkpoint、旧 shared-encoder PPO checkpoint、缺少 `action_head_schema=candidate_cell_residual_v1` 的旧 autoregressive head checkpoint、旧 `candidate_cell_residual_v1_fast_threshold_v1` checkpoint，以及 schema 元信息缺失或不匹配的 checkpoint 均不兼容当前模型，应 fail-fast。inflation 后不继承旧 optimizer state。

## 部署与验证

部署使用 stochastic 路径：ONNX 输出普通动作 logits、threshold `mu_raw/log_std` 等分布参数，C++ runtime 负责采样、动作选择、`threshold_raw -> threshold_int` 转换和 fast controller 调用。固定六步循环可在 ONNX 导出时展开，优先使用 Gather、ScatterElements、Where、Sigmoid 和基础整数/布尔算子；若采样放在 C++ 侧，训练用 `torch.multinomial`、Normal sampling 和 threshold logprob 不进入 ONNX 图。

最低测试要求：

- 14 个 `ko` 的执行角色和官方槽位表。
- `k=0/6`、两种 order、边界、障碍和己方碰撞。
- 非法动作不更新位置的规则参考对照。
- rollout 与 teacher forcing logprob 等价。
- candidate-cell residual head 初始为零扰动；旧 head checkpoint inflation 后在同一随机种子或 deterministic 检查下官方动作完全等价。
- fast threshold residual head 初始为零扰动；`p_fast_full_realization` 为 0.8 时 calibrated base 输出 `threshold_int=6`，为 0.3/0.6 时分别输出约 11/6。
- VP final-position head inflation 后旧 `actor_context` logits 完全等价，新增 final-position 列为 0；teacher-forced actions 改变 final positions 时 VP logits 能条件化变化。
- threshold raw teacher forcing logprob 等价，部署 stochastic 采样语义与训练分布一致。
- PPO 双输入中 actor feature 只影响 policy/logprob，critic feature 只影响 value。
- critic 四角色 gather 的 channel index、shape 和 P1/P2 视角。
- masked entropy、前向和反向 finite。
- serial/multiprocess rollout、checkpoint resume 和 CUDA smoke。
- 动作连续性 diagnostics：无效移动、相邻反向、有效位移和己方角色冲突。
