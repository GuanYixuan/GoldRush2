# Policy Network 设计草案

本文用于逐项讨论 GoldRush2.0 初版 policy network 的设计。其目标是冻结训练和部署共享的网络输入输出语义、网络主干形态和导出约束。Feature 输入语义见 `docs/feature_extractor_design.md`，RL 栈总览见 `docs/rl_architecture.md`。

## 定位

policy network 负责把 feature extractor 产生的固定输入转换为策略分布和价值估计。v1 网络主线按 CNN 设计，不依赖 token 输入。

输入：

```text
spatial_planes: 38 x 17 x 17
scalars: 10
feature_schema: "goldrush2_feature_v1"
```

v1 输出采用 factorized policy heads，并覆盖官方 `GameOutput`：

```text
action_logits: B x 6 x 5
k_logits: B x 7
order_logits: B x 2
vp_logits: B x 3
value: B
```

它不负责：

- 维护局内 observation memory。
- 执行游戏规则结算。
- 生成 opponent/NPC 行为。
- 直接访问 replay/full-state privileged 信息。

## 设计目标

v1 policy network 应满足：

- 推理可部署：能导出 ONNX，并在 C++ `.so` 内通过 ONNX Runtime 推理。
- 速度可控：单次推理应显著低于 300ms，给 feature extraction、候选生成和后处理留余量。
- 体积可控：模型权重、runtime 代码和必要 assets 合计必须适配 C++ `.so` 16MB 限制。
- 训练一致：Python 训练使用的 forward 语义与部署侧完全一致。
- 输出可约束：后处理能生成合法或尽量低风险的官方 `GameOutput`。
- 可扩展：后续可加入 candidate plan、auxiliary heads、BC auxiliary loss 或 fast option，但不破坏 v1 的核心 I/O。

## 输入融合

当前主线方案：

```text
spatial_planes
    -> shallow CNN stem
    -> feature map H

scalars
    -> scalar MLP
    -> FiLM gamma/beta

H
    -> FiLM modulation
    -> SE-Residual CNN backbone
```

FiLM 用于让全局状态调制整张空间特征图。例如对局阶段、金币差、snapshot/外围生成相位会改变同一空间 pattern 的策略价值，适合用 scalar tower 生成通道级调制参数。

建议 FiLM 采用 residual 风格：

```text
H = H * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
```

其中生成 `gamma/beta` 的最后一层可零初始化，使初始网络等价于未调制的 CNN，降低训练早期不稳定性。

v1 固定：

- scalar MLP hidden 为 `96, 96`。
- FiLM 只在 shallow CNN stem 与 SE-Residual backbone 之间作用一次。
- `spatial_planes` 不做额外 normalization，依赖 feature extractor 的固定 scaling。
- temporal group / nontemporal group 在网络内不分支处理，全部作为普通 channel 输入 stem。

## Backbone

当前主线方案：

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
    Linear(96 -> 192)  # gamma, beta

Fusion:
    FiLM(H, gamma, beta)

Backbone:
    8 x SE-ResidualBlock(96, r=4)
```

SE-Residual block 用于在 `17x17` 小图上继续建立空间关联，并让网络根据局面动态调整通道权重。当前 feature planes 中包含资源、炸弹、敌人、NPC、距离图、区域和 snapshot 统计，通道间交互较强，因此 SE 的收益/成本比合理。

v1 的 `SE-ResidualBlock(W, r)` 固定写法：

```text
input: x, shape = B x W x 17 x 17

residual branch:
    Conv3x3(W -> W, padding=1)
    SiLU
    Conv3x3(W -> W, padding=1)

SE gate:
    GlobalAvgPool(residual)          # B x W
    Linear(W -> W / r)
    SiLU
    Linear(W / r -> W)
    Sigmoid                          # B x W

output:
    y = SiLU(x + residual * gate[:, :, None, None])
```

v1 不在该 block 内使用 BatchNorm / LayerNorm / GroupNorm，也不额外加入 residual scale。默认激活函数使用 SiLU；ReLU 保留为部署/量化 fallback 实验，GELU 不进入 v1 默认候选。后续若训练稳定性需要，再以新实验配置评估 normalization 或 residual scale。

v1 固定规模：

```text
W = 96
N = 8
SE reduction = 4
scalar MLP hidden = 96, 96
```

实现约束：

- v1 不使用 token/attention 主干。
- 优先使用 ONNX Runtime 部署稳定的基础算子。
- 避免 BatchNorm 作为默认选择，因为 RL batch 分布变化大，部署 batch 可能为 1。

v1 不加入 dilation 或额外全局上下文模块。ReLU fallback 只作为部署/量化对照实验，不改变 SiLU 主线配置。

## 初始化

v1 默认初始化策略：

- Stem、scalar MLP、普通 hidden linear 和 residual branch 第一个 Conv 使用 orthogonal initialization，gain 近似取 `sqrt(2)`。
- FiLM 最后一层 `Linear(96 -> 192)` 使用零初始化，`weight=0`、`bias=0`，使初始调制等价于 `H'=H`。
- `SE-ResidualBlock` 的 residual branch 第二个 Conv 使用小 gain 初始化，例如 `0.1`，避免无 normalization 条件下 residual 初始幅度过大。
- SE gate 最后一层 `Linear(W/r -> W)` 使用 `weight=0`、`bias=0`，使初始 gate 为 `0.5`。
- action / k / order 输出层使用小初始化，例如 `weight std=0.01`、`bias=0`。
- vp 输出层使用小初始化，并设置 no-buy prior bias，使初始 `P(vp=0,1,2)=(0.90,0.07,0.03)`。
- value 输出层使用小初始化，例如 `weight std=0.01`、`bias=0`，使初始 value 接近 0。

训练诊断至少记录 residual block 输出 RMS、FiLM `gamma/beta` 均值/标准差/最大绝对值、shared trunk 和各 head gradient norm、logits 最大绝对值、value 输出分布。

## Head Architecture

共享 CNN 输出：

```text
H: B x 96 x 17 x 17
```

actor head 使用全局池化和两个己方角色位置 gather：

```text
avg = GlobalAvgPool(H)             # B x 96
max = GlobalMaxPool(H)             # B x 96
u0  = Gather(H, own_unit0_pos)      # B x 96
u1  = Gather(H, own_unit1_pos)      # B x 96

actor_input = concat(avg, max, u0, u1)  # B x 384
actor_mlp:
    Linear(384 -> 256)
    SiLU
    Linear(256 -> 256)
    SiLU

actor outputs:
    Linear(256 -> 30) -> reshape B x 6 x 5
    Linear(256 -> 7)
    Linear(256 -> 2)
    Linear(256 -> 3)
```

角色位置 gather 不改变 feature schema；实现上可优先用 `own_unit0_mask` / `own_unit1_mask` 对 `H` 做空间加权求和，避免额外坐标输入：

```text
u_i = sum_xy H[:, :, x, y] * own_unit_i_mask[:, x, y]
```

critic head 使用全局池化，不使用角色 gather：

```text
critic_input = concat(avg, max)  # B x 192
critic_mlp:
    Linear(192 -> 256)
    SiLU
    Linear(256 -> 128)
    SiLU
    Linear(128 -> 1)
```

actor/value 共享 CNN trunk，但 pooling 后使用独立 MLP。actor 更依赖角色局部上下文，critic 更依赖全局局势；分离 head MLP 的参数成本较低，能降低 head-level 目标干扰。

## Policy Output

v1 先采用 factorized action head，用于最快跑通 PPO：

```text
policy_outputs:
    action_logits: B x 6 x 5
    k_logits: B x 7
    order_logits: B x 2
    vp_logits: B x 3
    value: B
```

采样语义：

```text
actions[i] ~ Categorical(action_logits[:, i, :])  # i=0..5
k          ~ Categorical(k_logits)
order      ~ Categorical(order_logits)
vp         ~ Categorical(vp_logits)
```

对应官方 `GameOutput`：

```text
actions[6]: 每步动作，取值 0..4
k: 0..6
order: 0..1
vp: 0..2
```

PPO 记录的 policy logprob 为各 head logprob 之和：

```text
logprob_total =
    sum_i logprob(actions[i])
    + logprob(k)
    + logprob(order)
    + logprob(vp)
```

v1 entropy bonus 使用归一化分 head形式，而不是直接把九个 categorical entropy 无权重相加：

```text
H_action = (1/6) * sum_i H(actions[i]) / log(5)
H_k      = H(k) / log(7)
H_order  = H(order) / log(2)
H_vp     = H(vp) / log(3)

entropy_bonus =
    0.0100 * H_action
  + 0.0030 * H_k
  + 0.0010 * H_order
  + 0.0003 * H_vp
```

`vp` entropy 权重最低，因为随机买视野有直接经济成本且收益延迟较长；但保留非零权重，避免策略永久不探索视野购买。

部署 greedy 策略先使用各 head 独立 argmax。v1 不在网络输出层处理复杂合法性 mask；越界、撞已知障碍、互撞等非法移动由游戏规则自然跳过。必要时部署侧可增加轻量 fallback，但 PPO 第一版应先忠实执行采样结果，避免把后处理策略和网络学习混在一起。

该 factorized head 是 PPO 闭环 baseline，不是长期最优 action 表达。后续主方案倾向 candidate plan head：

```text
candidate_plans: M x GameOutput
shared_repr + candidate_features -> score[M]
sample/argmax candidate index
```

candidate plan head 可以更好地过滤明显低质或非法计划，但会引入 variable candidate set、masked categorical、候选生成训练/部署一致性等复杂度，因此不阻塞 v1。

## Value Output

v1 先实现最小 `V_rl` critic，用于跑通 PPO/GAE：

```text
V_rl: scalar
```

语义：

- `V_rl` 预测当前训练 reward schedule 下的 discounted return。
- PPO advantage / GAE 使用 `V_rl`，不使用辅助 value head。
- 在当前最小 `WinLossReward` 阶段，`V_rl` 近似预测终局胜负回报。
- 后续引入 dense margin reward 后，`V_rl` 必须跟随实际训练 reward 定义，而不是固定预测最终胜负。
- value head 最后一层不加 `tanh`，输出 unconstrained scalar；训练使用 Huber value loss，并配合 value clipping。

v1 critic 与 actor 共享同一个 observable CNN 表示，在 shared representation 后接 value head。这是为了先完成 PPO 闭环，降低实现复杂度；它不是长期最优结构判断。

后续扩展方向：

- actor/critic 主干分离的 asymmetric history-state critic。
- critic-only privileged state branch。
- `V_win_aux`：最终胜负辅助 value。
- `V_margin_aux`：最终净金币差压缩辅助 value。
- 使用 `detach(actor_hidden)` 作为 critic 输入的 ablation。

## Auxiliary Outputs

可选但不急于实现：

- spatial value map。
- risk map。
- path-direction logits。
- candidate score / candidate reweighting。
- BC auxiliary heads。
- fast option trigger / fast action。

待讨论：

- 哪些 auxiliary head 对 v1 训练收益大。
- 哪些 head 会增加部署复杂度。
- auxiliary output 是否进入 ONNX 部署图，还是只训练期使用。

## Action Masking 与后处理

v1 原则：

- 不在网络内做复杂 action mask。
- 不枚举 candidate plan。
- `k/order/vp/actions` 的取值范围由 categorical head 保证。
- 越界、撞已知障碍、己方两角色互撞、可见敌方占位等非法移动，先交由游戏规则跳过。
- PPO 第一版 rollout 忠实执行采样结果，不把 scripted 修正混入训练数据。

部署侧可保留最小 fallback：如果网络输出张量异常、包含 NaN/Inf 或后处理失败，则返回固定安全动作，例如全停留且 `vp=0`。该 fallback 只处理工程异常，不承担策略优化。

candidate plan、规则 mask、轻量搜索和 candidate rerank 均作为后续 action head 方案，不属于 v1 PPO 闭环。

## PPO v1 训练配置

v1 默认 PPO 配置：

```yaml
gamma: 0.9999
gae_lambda: 0.995

clip_range: 0.20
value_clip_range: 0.20
value_loss: huber
huber_delta: 1.0
value_coef: 0.50

optimizer: Adam
learning_rate: 2.0e-4
adam_eps: 1.0e-5
weight_decay: 0
lr_schedule: linear_decay
max_grad_norm: 0.5

rollout_horizon: 500
parallel_envs: 32
transitions_per_update: 16000
minibatch_size: 1024
update_epochs: 2

normalize_advantage: true
normalize_reward: false
target_joint_kl: 0.05
early_stop_on_kl: true
```

设计依据：

- `gamma=0.9999` 使 500 回合后的终局 reward 对开局仍有约 `0.95` 的折扣权重。
- `gae_lambda=0.995` 保留长时信用分配，同时比纯 Monte Carlo 方差更低。
- `rollout_horizon=500` 使用完整 episode，避免早期 rollout chunk 看不到 terminal reward。
- `update_epochs=2` 保守更新，避免 factorized joint policy 的联合 ratio 变化过快。
- KL 监控以 joint policy 为主，同时记录各 factor KL；early stop 可由 `joint_kl > 0.05` 触发。
- 第一版 PPO 不使用 BC warm start；reward 使用 `terminal_win_plus_margin_potential_v1`，默认 `beta=0.2`，smoke/debug 可设 `beta=0` 退化为纯终局胜负 reward，详见 `docs/reward_design.md`。

## 训练接口

v1 模型代码落点：

```text
training/models/
  policy_network.py
```

模型目录独立于 `training/rl/`，因为同一网络会被 PPO、后续 BC、评估、导出和部署链路复用。

PyTorch forward 签名固定为：

```text
forward(spatial_planes, scalars) -> PolicyNetworkOutput
```

输入 batch shape：

```text
spatial_planes: B x 38 x 17 x 17
scalars: B x 10
```

`PolicyNetworkOutput` 字段固定为：

```text
action_logits: B x 6 x 5
k_logits: B x 7
order_logits: B x 2
vp_logits: B x 3
value: B
```

forward 不返回 hidden features；diagnostics 由 trainer hook 或独立 debug API 采集，避免训练主路径和部署导出输出漂移。

训练/evaluation 侧提供 policy wrapper：

- train mode：从各 categorical head 采样，生成官方 `GameOutput`，并返回 `logprob_total`、`value`、分 head entropy / KL 所需统计。
- eval mode：各 head 独立 argmax，生成官方 `GameOutput`。
- fallback：若网络输出包含 NaN/Inf 或后处理失败，返回固定安全动作，例如 `actions=[4,4,4,4,4,4]`、`k=3`、`order=0`、`vp=0`。该 fallback 只处理工程异常，不承担策略优化。

PPO rollout buffer 字段固定为：

```text
spatial_planes
scalars
actions[6]
k
order
vp
old_logprob
value
reward
done
episode_id / round / map_id / opponent metadata
```

实现文件建议：

```text
training/rl/ppo.py
training/rl/ppo_buffer.py
training/scripts/train_ppo.py
```

PPO v1 reward class 建议实现为：

```text
TerminalWinPlusMarginPotentialReward
```

并记录 `reward_schema = "terminal_win_plus_margin_potential_v1"`。

## 导出与部署

待讨论：

- ONNX opset。
- dynamic axes 是否需要。
- fp32 / fp16 / int8 QDQ 版本路线。
- C++ 输入输出 tensor layout。
- 模型文件是否内嵌进 `.so`。
- schema version 校验位置。

## 测试要求

v1 设计落地后应覆盖：

- forward shape 测试。
- deterministic eval 模式测试。
- ONNX export/import 数值一致性测试。
- C++ ONNX Runtime 推理 smoke test。
- policy output 到 `GameOutput` 的后处理测试。
- 与 `training/rl` rollout/evaluation harness 的集成测试。

## 当前待讨论清单

1. ONNX 导出与量化路线。
