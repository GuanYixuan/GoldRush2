# Conditional Fast Option 设计

本文记录 conditional fast option 的未来扩展设计。它用于缓解“神经网络策略常态后手”的问题，不属于第一阶段 MVP。

## 核心想法

网络在第 `t` 回合正常输出当前动作，同时可选输出一个第 `t+1` 回合可微秒级执行的条件预案：

```text
normal_action_t = (k, order, actions[0:6], vp)
next_fast_option_t = None | FastOptionSpec
```

第 `t+1` 回合收到新 observation 后：

```text
if armed_fast_option and ultra_fast_trigger(observation):
    return ultra_fast_action(observation, armed_fast_option)
else:
    run_neural_policy()
```

它不是普通推理缓存，而是 one-step conditional option：网络提前生成下一回合的压缩策略，让 fast controller 在简单、高时效、高置信局面中绕过神经网络，争取先手。

## 训练影响

引入 fast option 后，策略不再是“永远后手”的单一角色：

```text
常态：neural action，较慢，通常后手
触发：fast action，较快，可能先手
```

这会改变训练问题：

- 第 `t+1` 回合 fast action 的后果应归因给第 `t` 回合输出的 option，而不是错误地当作第 `t+1` 回合网络采样动作。
- critic 和 policy state 需要看到当前是否挂载 option，否则同一 observation 会对应不同未来。
- simulator 必须建模 `tau_fast << tau_neural` 带来的行动顺序变化，否则网络无法学到 fast option 的真实价值。
- rollout schema 需要记录 option 的采样、触发、执行和收益归因。

## 设计预留

第一阶段不实现 fast option，但架构上应避免把接口设计死。建议预留：

- `Action` 中的可选 `next_fast_option` 字段。
- `PolicyState` 或 observation 中的 `armed_fast_option` 表达。
- `Env` 中的 `compute_mode` / `decision_latency` 概念。
- rollout 中的 option 归因字段：

```text
option_armed_at_t
option_triggered_at_t1
option_logprob
executed_by_fast_path
```

`policy_runtime` 的 feature extractor 也应避免假设每回合都完整运行 neural policy；fast path 可能在完整 `observe()` 前短路。

## 后续训练路线

未来若实现，推荐路线：

1. 先训练完整慢速后手策略。
2. 从慢策略、专家策略或 opponent league 中蒸馏 fast templates 和 gate。
3. 再在包含真实执行顺序的环境中联合 RL 微调。
4. 单独评估 trigger precision、trigger regret、fast 使用率、先手收益和被 exploit 风险。

fast option 不应允许网络输出任意程序。更稳的做法是预定义少量模板，例如抢最大可达金币、抢争议金币、安全大额金币、就近收割、紧急避险和固定区域进入；网络只选择 template 和少量离散参数。
