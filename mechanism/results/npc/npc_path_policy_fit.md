# NPC Path Policy M1-M5c 拟合结果

本文记录 NPC path-level softmax 的逐项拟合实验结果。推荐建模与 simulator 实现口径见 `npc_behavior_modeling_overview.md`。

## 数据与口径

输入数据：

- `symobs-a-map1-100-20260727-231446`
- `symobs-a-map2-100-20260727-231704`
- `symobs-m3a-map3-100-20260728-200953`
- `symobs-m3b-map3-100-20260728-202647`

运行命令：

```bash
conda run -n goldrush python mechanism/scripts/analysis/fit_npc_path_policy.py \
  --fit-id maps123_ordered_m1_m5_path_shape_sample75k_allgames_l2_1e-5_20260730 \
  --sample-mod 10 \
  --max-samples-per-run 0 \
  --maxiter 100 \
  --l2 1e-5
```

输出目录：

```text
mechanism/data/processed/npc_policy_fit/maps123_ordered_m1_m5_path_shape_sample75k_allgames_l2_1e-5_20260730/
```

本次使用 ordered 口径：

1. 只使用同一轮 7 个 NPC 都能按 `start -> actions -> end` 连续观察的样本。
2. 按 `dispatch_order` 依次处理 NPC。
3. 对当前 NPC 计算候选 path 特征前，先用更早 NPC 的真实 action 扣减可见金币和炸弹。
4. 候选集为当前起点上所有合法 3-step path；合法性只排除边界和静态障碍。
5. train / valid 按 `run_id:game_id` 的稳定 hash 切分。

样本量：

| split | samples |
| --- | ---: |
| train | `68849` |
| valid | `18113` |
| total | `86962` |

各 run 可用 ordered 样本与抽样样本：

| run_id | eligible ordered samples | used samples |
| --- | ---: | ---: |
| `symobs-a-map1-100-20260727-231446` | `301553` | `30156` |
| `symobs-a-map2-100-20260727-231704` | `308840` | `30884` |
| `symobs-m3a-map3-100-20260728-200953` | `126378` | `12638` |
| `symobs-m3b-map3-100-20260728-202647` | `132832` | `13284` |

## 模型序列

```text
M0:  uniform

M1:
    dynamic_reward_div10

M2a:
    dynamic_reward_div10
    enter_bomb_count

M2:
    dynamic_reward_div10
    enter_bomb_count
    adjacent_bomb_count

M3a:
    dynamic_reward_div10
    enter_bomb_count
    stay_count

M3b:
    dynamic_reward_div10
    enter_bomb_count
    stay_count
    straight3

M3c:
    dynamic_reward_div10
    enter_bomb_count
    stay_count
    straight3
    backtrack

M4a:
    dynamic_reward_div10
    enter_bomb_count
    stay_count
    straight3
    backtrack
    bomb_trapped_stay

M5a:
    M4a
    first_two_same_nonstay

M5b:
    M5a
    last_two_same_nonstay

M5c:
    M5b
    sandwich
```

特征含义：

- `dynamic_reward_div10 = dynamic_pickup_reward / 10`。
- `enter_bomb_count`：三步 path 中实际移动进入可见炸弹格的次数。
- `adjacent_bomb_count`：三步位置中与当前可见炸弹 Manhattan 距离为 1 的次数。
- `stay_count`：三步动作中 `action=4` 的次数。
- `straight3`：三步同向且非停。
- `backtrack`：存在相邻两步反向移动。
- `bomb_trapped_stay`：起点所有合法相邻非停格都是当前可见炸弹，且候选 path 为 `[4,4,4]`。
- `first_two_same_nonstay`：前两步同向且非停。
- `last_two_same_nonstay`：后两步同向且非停。
- `sandwich`：`a0 == a2 != a1`，且 `a0/a1` 都非停。

注意：`bomb_trapped_stay` 是稀有特征。默认正则 `l2=1e-3` 会明显压低该项权重，因此本次主结果使用 `l2=1e-5`。

## 拟合权重

| model | dynamic_reward_div10 | enter_bomb_count | adjacent_bomb_count | stay_count | straight3 | backtrack | bomb_trapped_stay | first_two_same_nonstay | last_two_same_nonstay | sandwich |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M1 | `2.819947` |  |  |  |  |  |  |  |  |  |
| M2a | `2.854113` | `-2.458719` |  |  |  |  |  |  |  |  |
| M2 | `2.851996` | `-2.451756` | `0.058965` |  |  |  |  |  |  |  |
| M3a | `2.243166` | `-2.729309` |  | `-3.093344` |  |  |  |  |  |  |
| M3b | `2.135307` | `-2.778070` |  | `-2.912154` | `1.467275` |  |  |  |  |  |
| M3c | `2.064451` | `-2.841216` |  | `-3.095453` | `1.188170` | `-0.568594` |  |  |  |  |
| M4a | `2.064294` | `-2.800031` |  | `-3.131701` | `1.186625` | `-0.569309` | `10.777999` |  |  |  |
| M5a | `2.042112` | `-2.801638` |  | `-3.034541` | `0.863517` | `-0.498504` | `10.720481` | `0.512085` |  |  |
| M5b | `2.029310` | `-2.800418` |  | `-2.917844` | `0.320291` | `-0.419714` | `10.633031` | `0.708062` | `0.570831` |  |
| M5c | `2.021454` | `-2.789653` |  | `-2.815642` | `0.196427` | `-0.384252` | `10.516389` | `0.847191` | `0.707631` | `0.322904` |

解释：

- 金币收益权重始终为正，动态 `65%` path reward 是稳定的一阶信号。
- 直接进入炸弹格权重稳定为强负。
- `adjacent_bomb_count` 在 M2 中只有很小正权重，且 M2 相比 M2a 的增益极小，不建议进入默认模拟器策略。
- `stay_count` 权重强负，说明 NPC 几乎走满三步这一形状先验非常关键。
- M5 系列显示 NPC 对 path 形状有更细的偏好：前两步同向、后两步同向、`a0 == a2 != a1` 都是正项。
- 加入 M5 细形状项后，`straight3` 权重被明显分解，说明原 M3/M4 的 `straight3` 部分吸收了“两步连续同向”的偏好。
- `bomb_trapped_stay` 权重很大，表示它是 `stay_count` 强负先验下的例外项。

## 验证指标

| model | valid NLL | top1 | top3 | MRR | actual path prob |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 uniform | `4.370542` | `7.70%` | `14.09%` | `15.21%` | `1.352%` |
| M1 dynamic reward | `4.122165` | `17.06%` | `28.77%` | `26.91%` | `2.964%` |
| M2a dynamic + enter bomb | `4.052143` | `18.51%` | `30.43%` | `28.54%` | `3.260%` |
| M2 dynamic + bomb | `4.051946` | `15.98%` | `27.42%` | `25.90%` | `3.259%` |
| M3a + stay | `3.443826` | `18.52%` | `31.78%` | `30.33%` | `5.304%` |
| M3b + straight3 | `3.317798` | `16.89%` | `36.10%` | `31.53%` | `6.634%` |
| M3c + backtrack | `3.287138` | `17.51%` | `36.75%` | `32.10%` | `6.838%` |
| M4a + trapped stay | `3.285297` | `17.52%` | `36.76%` | `32.11%` | `6.852%` |
| M5a + first two same | `3.269292` | `16.47%` | `34.57%` | `30.98%` | `6.973%` |
| M5b + last two same | `3.252415` | `16.52%` | `35.02%` | `31.21%` | `7.124%` |
| M5c + sandwich | `3.250092` | `16.63%` | `35.22%` | `31.37%` | `7.114%` |

关键增益：

- M1 相比 M0：NLL 改善 `0.2484 nats / trajectory`。
- M2a 相比 M1：NLL 改善 `0.0700`，主要来自直接踩炸弹惩罚。
- M2 相比 M2a：NLL 只改善 `0.0002`，`adjacent_bomb_count` 暂不值得保留。
- M3a 相比 M2a：NLL 改善 `0.6083`，`stay_count` 是极强 shape 特征。
- M3b 相比 M3a：NLL 改善 `0.1260`，`straight3` 有明确增益。
- M3c 相比 M3b：NLL 改善 `0.0307`，`backtrack` 有小但稳定的增益。
- M4a 相比 M3c：整体 NLL 只改善 `0.0018`，但其目标是修正稀有 trapped 场景。
- M5a/M5b/M5c 相比 M4a：NLL 合计改善 `0.0352`，主要来自两步连续同向，`sandwich` 额外改善较小。

M5 系列的 top1 和 MRR 低于 M4a，但 NLL、actual path probability 和关键形状特征均值更好。若目标是复现 replay 统计分布，M5c 优于 M4a；若目标是更小的可解释策略，M4a 仍是可接受简化版。

## 特征均值对照

valid split 中，实际 path 与模型期望的关键特征：

| model | dynamic_reward_div10 | enter_bomb_count | stay_count | straight3 | backtrack | bomb_trapped_stay | first_two_same_nonstay | last_two_same_nonstay | sandwich |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actual | `0.257859` | `0.011594` | `0.030862` | `0.214376` | `0.312262` | `0.000166` | `0.450947` | `0.415503` | `0.142108` |
| uniform expected | `0.085571` | `0.093556` | `0.697126` | `0.028699` | `0.314086` | `0.000004` | `0.150127` | `0.144712` | `0.088450` |
| M2a expected | `0.255891` | `0.011808` | `0.643599` | `0.036065` | `0.318868` | `0.000022` | `0.169986` | `0.158921` | `0.096416` |
| M3c expected | `0.254963` | `0.011850` | `0.039022` | `0.210994` | `0.313973` | `0.000000` | `0.382101` | `0.370373` | `0.165255` |
| M4a expected | `0.255025` | `0.012210` | `0.037948` | `0.211077` | `0.314104` | `0.000099` | `0.382336` | `0.370595` | `0.165496` |
| M5c expected | `0.254972` | `0.012209` | `0.037894` | `0.211441` | `0.313602` | `0.000102` | `0.452085` | `0.413331` | `0.143970` |

观察：

- M2a 已能同时匹配动态收益和直接踩炸弹频率。
- M2a 仍严重高估停留次数，低估三步直线和两步连续同向比例。
- M3c/M4a 已能较好匹配停留、直线和折返，但仍低估两步连续同向。
- M5c 明显修正 `first_two_same_nonstay`、`last_two_same_nonstay` 和 `sandwich` 的模型期望，同时没有破坏金币、踩炸弹、停留和直线比例。

## Trapped 条件专项

在本次 valid 抽样中，满足：

```text
all legal adjacent non-stay moves are visible bombs
```

的样本只有 `3` 条，且真实动作全部为 `[4,4,4]`。

| model | P([4,4,4] \| trapped) | actual path prob | trapped NLL | top1 |
| --- | ---: | ---: | ---: | ---: |
| M3c | `0.007%` | `0.007%` | `10.189` | `0.00%` |
| M4a | `59.47%` | `59.47%` | `0.716` | `66.67%` |

这解释了为什么 M4a 的全局 NLL 改善很小：trapped 样本太稀有。但在目标条件下，M4a 修正非常明显。M5 系列继承 `bomb_trapped_stay`，不会改变该机制解释。

## Prefix Argmax 指标

在 valid split 上，取模型打分最高的 path 作为预测 path，并检查 action prefix 是否与真实 path 一致。输出文件：

```text
mechanism/data/processed/npc_policy_fit/maps123_ordered_m1_m5_path_shape_sample75k_allgames_l2_1e-5_20260730/prefix_accuracy.csv
```

| model | path top1 | 首动作正确 | 前两个动作正确 | 正确首动作概率质量 | 正确前两步概率质量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M4a | `17.52%` | `45.06%` | `27.91%` | `36.12%` | `15.14%` |
| M5a | `16.47%` | `43.98%` | `26.71%` | `36.20%` | `15.85%` |
| M5b | `16.52%` | `43.95%` | `26.86%` | `36.21%` | `15.91%` |
| M5c | `16.63%` | `44.02%` | `26.92%` | `36.21%` | `15.88%` |

观察：

- M4a 在 path top1、首动作正确、前两个动作正确上均高于 M5 系列。
- M5 系列给正确 prefix 的 softmax 概率质量略高，尤其前两步；这与 M5 更偏向统计分布校准的定位一致。

## 决策时序对照

为区分“按 `dispatch_order` 执行结算”和“NPC 决策是否也逐个看到前序 NPC 的状态变化”，补充比较两种特征口径：

- `ordered`：按 `dispatch_order` 用前序 NPC 的真实 action 先扣减可见金币和炸弹，再为当前 NPC 候选 path 计算特征。
- `simultaneous_start`：同一轮 7 个 NPC 都基于同一个 round-start NPC 决策快照计算候选 path 特征，不扣减前序 NPC；执行结算仍可按随机 `dispatch_order` 发生。

输出文件：

```text
temp/npc_decision_timing_comparison.csv
```

两种口径使用相同 run、相同抽样和相同 train/valid split：

| timing | model | valid NLL | top1 | actual path prob |
| --- | --- | ---: | ---: | ---: |
| ordered | M1 | `4.122165` | `17.06%` | `2.964%` |
| simultaneous_start | M1 | `4.093327` | `17.94%` | `3.174%` |
| ordered | M2a | `4.052143` | `18.51%` | `3.260%` |
| simultaneous_start | M2a | `4.023021` | `19.39%` | `3.484%` |
| ordered | M3c | `3.287138` | `17.51%` | `6.838%` |
| simultaneous_start | M3c | `3.268881` | `18.06%` | `7.083%` |
| ordered | M4a | `3.285297` | `17.52%` | `6.852%` |
| simultaneous_start | M4a | `3.267048` | `18.08%` | `7.097%` |

`simultaneous_start` 在上述所有模型上均优于 `ordered`。这说明当前数据更支持“同轮 NPC 同时基于同一决策快照产出 actions，随后按随机顺序执行结算”，而不是“后序 NPC 决策时看到前序 NPC 已经执行后的状态”。

simultaneous-start M4a 拟合权重为：

```text
score(path) =
    2.073989 * dynamic_reward_div10(path)
  - 2.795283 * enter_bomb_count(path)
  - 3.120374 * stay_count(path)
  + 1.183404 * straight3(path)
  - 0.562663 * backtrack(path)
  + 10.776788 * bomb_trapped_stay(path)
```

## 实验结论

- 动态 `65%` path reward 是稳定的一阶信号，M1 相比 uniform 显著改善 NLL。
- `enter_bomb_count` 足以校正直接踩炸弹频率；`adjacent_bomb_count` 增益极小且权重为小正值，暂不值得进入默认策略。
- path shape 是最强的后续增益来源：`stay_count`、`straight3`、`backtrack` 分别修正停留、直线和折返统计。
- `bomb_trapped_stay` 的全局 NLL 增益很小，但在炸弹封路条件下能把 `[4,4,4]` 概率从近似 `0` 提升到约 `59%`。
- M5c 的 held-out NLL 最低，形状特征均值最接近 replay；M4a 的 prefix argmax 指标更好且参数更少。
- 决策时序对照中，`simultaneous_start` 明确优于 `ordered`，因此第一版 simulator 默认应同时生成 7 个 NPC actions，再按随机 `dispatch_order` 执行。

推荐建模结论不在本文冻结，统一见 `npc_behavior_modeling_overview.md`。
