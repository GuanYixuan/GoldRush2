# Policy Network 设计

本文冻结当前训练与部署共享的网络结构和动作概率语义。Actor feature 输入见 `docs/features/actor_feature_v2.md`，PPO 栈见 `docs/rl_architecture.md`，游戏动作规则以 `gamerules/gamerules.md` 为准。

## 定位

网络使用 actor/critic 双输入。actor 输入使用可提交路径 feature v2：

```text
actor_spatial_planes: B x 43 x 17 x 17
actor_scalars: B x 10
actor_feature_schema: goldrush2_feature_v2
```

critic 输入使用训练期 privileged critic feature：

```text
critic_spatial_planes: B x 26 x 17 x 17
critic_scalars: B x 17
critic_feature_schema: goldrush2_privileged_critic_feature_v1
```

网络输出官方动作字段和 value：

```text
actions: B x 6，取值 0..4
k: B，取值 0..6
order: B，取值 0..1
vp: B，取值 0..2
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

critic encoder 使用同型结构，但 stem 输入为 `26`，scalar tower 输入为 `17`：

```text
Critic stem:
    Conv3x3(26 -> 96)
    SiLU
    Conv3x3(96 -> 96)
    SiLU

Critic scalar tower:
    Linear(17 -> 96)
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
critic_mlp_input = width * 6
critic_hidden = (256, 128)
```

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
p(ko, vp, a_exec[0:6] | state)
= p(ko | state)
  * p(vp | state)
  * product_t p(a_exec[t] | state, ko, prefix[0:t], simulated_positions[t])
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

动作生成按真实执行顺序进行，最终通过 `official_slot_table[ko]` scatter 回官方槽位。特别地：

- `order=0`：先生成角色 0 的 `actions[0:k]`，再生成角色 1 的 `actions[k:6]`。
- `order=1`：先生成角色 1 的 `actions[k:6]`，再生成角色 0 的 `actions[0:k]`。
- `k=0/6` 仍保留两个合法 `order` 类，不合并官方动作。

六步共享同一 decoder 参数，禁止重新拆成六套独立 head。

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

这里模拟的是当前 obstacle belief，不是真实地图保证。feature v1 的对称障碍推断无法与直接观测区分；该风险由 feature schema 本身决定。

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
    critic_spatial_planes, critic_scalars,
    deterministic=False,
) -> PolicyAction

evaluate_actions(
    actor_spatial_planes, actor_scalars,
    critic_spatial_planes, critic_scalars,
    actions, k, order, vp,
) -> PolicyEvaluation
```

`act()` 使用 actor feature 采样或逐步 argmax `ko/vp/actions`，并使用 critic feature 通过 critic encoder 输出 value。`evaluate_actions()` 从保存的 `k/order` 恢复 `ko`，按保存动作 teacher force 同一个 decoder，并用 critic feature 重算 value。

部署/BC/ONNX 路径应保留 actor-only 入口。该入口只能返回 action/logprob/entropy 或使用占位 value，不得要求 privileged critic feature。PPO 训练路径必须使用双输入接口。

`forward()` 是部署友好的 deterministic `act()` 包装，不用于 PPO 更新。

若 teacher-forced 动作被同一规则 mask 判为非法，立即报错。这通常表示槽位映射、位置递推或 feature 不一致，禁止静默赋予极小概率。

## PPO 概率与熵

rollout 保存的联合 logprob：

```text
log p(ko | state)
+ log p(vp | state)
+ sum_t log p(a_exec[t] | prefix_t, simulated_positions[t])
```

PPO buffer 和 multiprocess shared memory 保存：

```text
actor_spatial_planes/actor_scalars
critic_spatial_planes/critic_scalars
actions/k/order/vp/old_logprob/value
```

decoder hidden、执行顺序动作和 `ko` 均可由现有字段确定，不进入 buffer。PPO 更新必须 teacher force buffer 动作，禁止重新采样或用当前 argmax 作为 prefix。模型参数不变时，rollout logprob 与重算 logprob 必须在浮点误差内一致。

policy loss、KL 和 entropy 只读取 actor feature；value loss、old value 对齐和 explained variance 只读取 critic feature。`old_logprob` 仍来自 actor 路径，`old_value` 来自 critic 路径。

当前 entropy 权重保持旧量级：

```text
action_entropy = mean_t H(action_t | prefix_t) / log(5)
ko_entropy = H(ko) / log(14)
vp_entropy = H(vp) / log(3)

entropy_bonus =
    0.0100 * action_entropy
  + 0.0040 * ko_entropy
  + 0.0003 * vp_entropy
```

action entropy 暂以完整五类 `log(5)` 归一化；mask 后只有少量合法动作时指标会自然下降。masked logits 使用 dtype 有限最小值，避免 `0 * -inf` 产生 NaN。

## Rollout 与性能

serial 和 multiprocess rollout 均应在 action-time state 下同时构造 actor feature 和 critic feature，再调用 PPO 双输入 `model.act()`。一次 feature request 仍只返回完整 `GameOutput`、logprob 和 value；六步 decoder 不与 worker 中途通信。

multiprocess rollout 的 worker 负责提取两套 feature：

```text
policy_runtime.FeatureExtractor.observe(GameInput) -> actor feature
extract_privileged_critic_features(GameState, MapTemplate, OuterGoldState, agent_player_id) -> critic feature
```

主进程/GPU 只负责批量推理。shared memory 与 PPO batch 保存两套 feature，shape 固定为：

```text
actor:  43 x 17 x 17, scalars 10
critic: 26 x 17 x 17, scalars 17
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
- `vp` bias 保持初始先验 `(0.90,0.07,0.03)`。
- GRU input weight 使用 Xavier，hidden weight 使用 orthogonal，bias 为 0。
- embedding 使用小正态初始化。
- value 输出层使用小初始化，使初始 value 接近 0。
- BC 训练只优化 actor 参数。PPO 从 BC checkpoint 初始化时，只加载 actor 路径；critic encoder 与 critic head 按 privileged critic schema 随机初始化或从专门 critic checkpoint 加载，不能默认拷贝 actor encoder，因为 actor/critic 输入 channel 与语义不同。

旧 factorized checkpoint 和旧 shared-encoder PPO checkpoint 不兼容当前模型。主线不维护隐式部分加载；需要迁移 actor 或 critic 时必须使用显式转换实验并记录缺失字段。

## 部署与验证

部署使用逐步 argmax 路径。固定六步循环应在 ONNX 导出时展开，优先使用 Gather、ScatterElements、Where、ArgMax 和基础整数/布尔算子；训练用 `torch.multinomial` 不进入确定性部署图。

最低测试要求：

- 14 个 `ko` 的执行角色和官方槽位表。
- `k=0/6`、两种 order、边界、障碍和己方碰撞。
- 非法动作不更新位置的规则参考对照。
- rollout 与 teacher forcing logprob 等价。
- PPO 双输入中 actor feature 只影响 policy/logprob，critic feature 只影响 value。
- critic 四角色 gather 的 channel index、shape 和 P1/P2 视角。
- masked entropy、前向和反向 finite。
- serial/multiprocess rollout、checkpoint resume 和 CUDA smoke。
- 动作连续性 diagnostics：无效移动、相邻反向、有效位移和己方角色冲突。
