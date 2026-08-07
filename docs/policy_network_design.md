# Policy Network 设计

本文冻结当前训练与部署共享的网络结构和动作概率语义。Feature 输入见 `docs/feature_extractor_design.md`，PPO 栈见 `docs/rl_architecture.md`，游戏动作规则以 `gamerules/gamerules.md` 为准。

## 定位

固定输入保持 feature v1 不变：

```text
spatial_planes: B x 38 x 17 x 17
scalars: B x 10
feature_schema: goldrush2_feature_v1
```

网络输出官方动作字段和 value：

```text
actions: B x 6，取值 0..4
k: B，取值 0..6
order: B，取值 0..1
vp: B，取值 0..2
logprob/value/entropy diagnostics: B
```

网络不维护 observation memory，不访问 replay/full-state privileged 信息，不模拟敌方或 NPC 行动，也不执行金币、炸弹和终局结算。当前 critic 虽然已与 actor 参数分离，但仍使用同一份 feature v1 输入；privileged critic 属于后续结构升级。

## Encoder

actor 和 critic 使用两套参数独立的 encoder。两套 encoder 结构相同，输入也相同，但不共享任何可训练参数：

```text
Stem:
    Conv3x3(38 -> 96)
    SiLU
    Conv3x3(96 -> 96)
    SiLU

Scalar tower:
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

SE residual block 不使用 normalization。FiLM 最后一层零初始化；residual branch 第二个卷积使用小 gain；SE gate 最后一层零初始化。默认激活为 SiLU，ReLU 只保留为部署实验配置。

actor encoder 输出 `H_actor: B x 96 x 17 x 17`。actor 使用全局 average/max pooling 和两个己方角色初始位置处的 feature gather：

```text
actor_context = MLP(concat(avg, max, unit0_local, unit1_local))
actor_hidden = 256
```

critic encoder 输出 `H_critic: B x 96 x 17 x 17`。critic 只使用 `concat(avg, max)`，经 `256,128` MLP 输出标量 value，不依赖采样动作、actor local gather 或 decoder hidden。

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
act(spatial_planes, scalars, deterministic=False) -> PolicyAction

evaluate_actions(
    spatial_planes, scalars,
    actions, k, order, vp,
) -> PolicyEvaluation
```

`act()` 采样或逐步 argmax `ko/vp/actions`，并同时通过 critic encoder 输出 value。`evaluate_actions()` 从保存的 `k/order` 恢复 `ko`，按保存动作 teacher force 同一个 decoder，并用 critic encoder 重算 value。actor 路径与 critic 路径只共用输入 feature、执行表、位置转移和 mask 规则，不共享可训练参数。

`forward()` 是部署友好的 deterministic `act()` 包装，不用于 PPO 更新。

若 teacher-forced 动作被同一规则 mask 判为非法，立即报错。这通常表示槽位映射、位置递推或 feature 不一致，禁止静默赋予极小概率。

## PPO 概率与熵

rollout 保存的联合 logprob：

```text
log p(ko | state)
+ log p(vp | state)
+ sum_t log p(a_exec[t] | prefix_t, simulated_positions[t])
```

PPO buffer 和 multiprocess shared memory 继续只保存：

```text
spatial_planes/scalars/actions/k/order/vp/old_logprob/value
```

decoder hidden、执行顺序动作和 `ko` 均可由现有字段确定，不进入 buffer。PPO 更新必须 teacher force buffer 动作，禁止重新采样或用当前 argmax 作为 prefix。模型参数不变时，rollout logprob 与重算 logprob 必须在浮点误差内一致。

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

serial 和 multiprocess rollout 均直接调用 `model.act()`。一次 feature request 仍只返回完整 `GameOutput`、logprob 和 value；六步 decoder 不与 worker 中途通信。

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
- BC 训练只优化 actor 参数。PPO 从 BC checkpoint 初始化时，只加载 actor 路径；critic encoder 显式拷贝 actor encoder 初值，critic head 保持随机初始化。

旧 factorized checkpoint 和旧 shared-encoder PPO checkpoint 不兼容当前模型。主线不维护隐式部分加载；需要迁移 actor 或 critic 时必须使用显式转换实验并记录缺失字段。

## 部署与验证

部署使用逐步 argmax 路径。固定六步循环应在 ONNX 导出时展开，优先使用 Gather、ScatterElements、Where、ArgMax 和基础整数/布尔算子；训练用 `torch.multinomial` 不进入确定性部署图。

最低测试要求：

- 14 个 `ko` 的执行角色和官方槽位表。
- `k=0/6`、两种 order、边界、障碍和己方碰撞。
- 非法动作不更新位置的规则参考对照。
- rollout 与 teacher forcing logprob 等价。
- masked entropy、前向和反向 finite。
- serial/multiprocess rollout、checkpoint resume 和 CUDA smoke。
- 动作连续性 diagnostics：无效移动、相邻反向、有效位移和己方角色冲突。
