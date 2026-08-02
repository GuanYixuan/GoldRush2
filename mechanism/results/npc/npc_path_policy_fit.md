# NPC Path Policy Canonical 拟合结果

本文记录 NPC path-level softmax 的 canonical 口径拟合结果。本文是正式 NPC 拟合结论来源。

## Canonical 口径

本项目后续统一采用如下 NPC plan 口径：

1. 回合开始时先刷新炸弹、生成金币，并把资源变化落到 round-start state。
2. 玩家与 NPC 都基于该同一个 round-start state 规划动作。
3. 7 个 NPC 的 actions 同时产出，不看先手玩家行动后的状态，也不看同轮前序 NPC 已执行后的状态。
4. 之后按 `first_player -> NPC permutation -> second_player` 执行和结算。

这与当前 simulator 实现一致。

## 数据与命令

输入数据：

- `symobs-a-map1-100-20260727-231446`
- `symobs-a-map2-100-20260727-231704`
- `symobs-m3a-map3-100-20260728-200953`
- `symobs-m3b-map3-100-20260728-202647`

运行命令：

```bash
conda run --no-capture-output -n goldrush python mechanism/scripts/analysis/fit_npc_path_policy.py \
  --fit-id maps123_canonical_simstart_m1_m5_static_pickup_center_allgames_l2_1e-5_20260802 \
  --sample-mod 10 \
  --max-samples-per-run 0 \
  --maxiter 200 \
  --l2 1e-5 \
  --pickup-mode static-count
```

输出目录：

```text
mechanism/data/processed/npc_policy_fit/maps123_canonical_simstart_m1_m5_static_pickup_center_allgames_l2_1e-5_20260802/
```

样本量：

| split | samples |
| --- | ---: |
| train | `68849` |
| valid | `18113` |
| total | `86962` |

各 run 样本：

| run_id | eligible samples | used samples |
| --- | ---: | ---: |
| `symobs-a-map1-100-20260727-231446` | `301553` | `30156` |
| `symobs-a-map2-100-20260727-231704` | `308840` | `30884` |
| `symobs-m3a-map3-100-20260728-200953` | `126378` | `12638` |
| `symobs-m3b-map3-100-20260728-202647` | `132832` | `13284` |

## 特征

- `dynamic_reward_div10`：按 path 内动态 `ceil(65%)` 拾取结算得到的 pickup reward，再除以 `10`。
- `pickup_count`：path 中发生金币 pickup 的步数；只计是否吃到金币，不按 amount 放大。
- `static_pickup_count`：path 中每次移动进入 round-start 时有金币的格子计 `1`；该特征不模拟 path 内金币扣减，因此即使金额为 `1`，重复进出同一格也可计多次。金额收益仍由 `dynamic_reward_div10` 按真实动态扣减计算。
- `enter_bomb_count`：path 中实际移动进入可见炸弹格的次数。
- `adjacent_bomb_count`：path 三步位置中与可见炸弹 Manhattan 距离为 `1` 的次数。
- `stay_count`：三步动作中 `action=4` 的次数。
- `straight3`：三步同向且非停。
- `backtrack`：存在相邻两步反向移动。
- `bomb_trapped_stay`：起点所有合法相邻非停格都是可见炸弹，且候选 path 为 `[4,4,4]`。
- `first_two_same_nonstay`：前两步同向且非停。
- `last_two_same_nonstay`：后两步同向且非停。
- `sandwich`：`a0 == a2 != a1`，且 `a0/a1` 都非停。
- `center_delta_chebyshev`：`start_chebyshev - end_chebyshev`，越大表示越靠近中心；`chebyshev(pos)=max(abs(row-8), abs(col-8))`。

## 验证指标

| model | valid NLL | top1 | top3 | MRR | actual path prob |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 uniform | `4.370542` | `7.70%` | `14.09%` | `15.21%` | `1.352%` |
| M1 dynamic reward | `4.093327` | `17.94%` | `30.17%` | `27.96%` | `3.174%` |
| M1b + pickup count | `3.995739` | `17.38%` | `29.76%` | `27.55%` | `3.553%` |
| M2a dynamic + enter bomb | `4.023021` | `19.39%` | `31.87%` | `29.59%` | `3.484%` |
| M2b + pickup count | `3.932054` | `18.69%` | `31.39%` | `29.10%` | `3.834%` |
| M3a + stay | `3.423387` | `19.42%` | `33.15%` | `31.34%` | `5.547%` |
| M3a + pickup count | `3.379153` | `18.82%` | `32.64%` | `30.89%` | `5.852%` |
| M3b + straight3 | `3.298712` | `17.37%` | `37.05%` | `32.17%` | `6.881%` |
| M3c + backtrack | `3.268881` | `18.06%` | `37.38%` | `32.67%` | `7.083%` |
| M4a + trapped stay | `3.267048` | `18.08%` | `37.39%` | `32.68%` | `7.097%` |
| M4b + pickup count | `3.210355` | `18.97%` | `38.43%` | `33.65%` | `7.619%` |
| M4c + center chebyshev | `3.246296` | `17.87%` | `38.29%` | `32.95%` | `7.402%` |
| M4d + pickup + center | `3.192912` | `19.35%` | `39.24%` | `34.28%` | `7.903%` |
| M4e + static pickup + center | `3.177952` | `19.59%` | `40.47%` | `34.86%` | `8.043%` |
| M5a + first two same | `3.251115` | `17.03%` | `35.44%` | `31.58%` | `7.221%` |
| M5b + last two same | `3.234455` | `17.02%` | `35.80%` | `31.75%` | `7.375%` |
| M5c + sandwich | `3.232150` | `17.15%` | `36.00%` | `31.93%` | `7.365%` |
| M5d + pickup count | `3.174828` | `18.69%` | `37.42%` | `33.45%` | `7.921%` |
| M5e + center chebyshev | `3.210594` | `17.27%` | `36.71%` | `32.39%` | `7.684%` |
| M5f + pickup + center | `3.156817` | `18.53%` | `38.30%` | `33.66%` | `8.216%` |
| M5g + static pickup + center | `3.142093` | `18.57%` | `38.84%` | `33.96%` | `8.354%` |

关键增益：

- `pickup_count` 是强特征：M4a -> M4b 改善 `0.0567` NLL；M5c -> M5d 改善 `0.0573`。
- `center_delta_chebyshev` 仍有效：M4a -> M4c 改善 `0.0208`；M5c -> M5e 改善 `0.0216`。
- `pickup_count` 与中心项可叠加：M4a -> M4d 改善 `0.0741`；M5c -> M5f 改善 `0.0753`。
- `static_pickup_count` 进一步优于 `pickup_count`：M4d -> M4e 改善 `0.0150`；M5f -> M5g 改善 `0.0147`。
- `adjacent_bomb_count` 增益极小，仍不建议进入默认策略。

## 拟合权重

重点模型权重：

| feature | M4a | M4d pickup+center | M4e static+center | M5f pickup+center | M5g static+center |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dynamic_reward_div10` | `2.073989` | `0.945710` | `0.907331` | `0.909601` | `0.875531` |
| `pickup_count` |  | `0.876940` |  | `0.871307` |  |
| `static_pickup_count` |  |  | `0.912740` |  | `0.905037` |
| `enter_bomb_count` | `-2.795283` | `-2.720128` | `-2.716949` | `-2.711302` | `-2.710227` |
| `stay_count` | `-3.120374` | `-3.073668` | `-3.065158` | `-2.760196` | `-2.758740` |
| `straight3` | `1.183404` | `1.209525` | `1.209899` | `0.217320` | `0.219614` |
| `backtrack` | `-0.562663` | `-0.647360` | `-0.709615` | `-0.462810` | `-0.526579` |
| `bomb_trapped_stay` | `10.776788` | `10.936934` | `10.916492` | `10.659804` | `10.652026` |
| `first_two_same_nonstay` |  |  |  | `0.848344` | `0.826686` |
| `last_two_same_nonstay` |  |  |  | `0.707010` | `0.716532` |
| `sandwich` |  |  |  | `0.302928` | `0.274567` |
| `center_delta_chebyshev` |  | `0.167486` | `0.167728` | `0.167818` | `0.167938` |

观察：

- 加入 pickup 基础效用后，`dynamic_reward_div10` 从约 `2.07` 降到约 `0.88-0.95`，说明原动态收益项吸收了“进入金币目标格本身有固定效用”的一部分。
- `static_pickup_count` 权重稳定在 `0.90-0.91`，略强于动态 `pickup_count` 口径。
- 中心项在 pickup 后仍稳定为正，权重约 `0.167-0.168`；它不是 pickup 的替代项。

## 特征均值对照

valid split 中，实际 path 与模型期望的关键特征：

| model | dynamic_reward_div10 | pickup feature | center_delta_chebyshev | stay_count | straight3 | first_two_same | last_two_same |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actual | `0.285348` | count `0.683763` / static `0.720919` | `-0.012422` | `0.030862` | `0.214376` | `0.450947` | `0.415503` |
| uniform expected | `0.094210` | count `0.277797` / static `0.285981` | `-0.118141` | `0.697126` | `0.028699` | `0.150127` | `0.144712` |
| M4a expected | `0.281683` | count `0.552524` | `-0.243084` | `0.037959` | `0.210769` | `0.382313` | `0.370718` |
| M4b expected | `0.282087` | count `0.683563` | `-0.221463` | `0.038105` | `0.210810` | `0.381174` | `0.370247` |
| M4c expected | `0.281915` | count `0.556775` | `-0.008219` | `0.038030` | `0.210275` | `0.381521` | `0.369700` |
| M4d expected | `0.282236` | count `0.684183` | `-0.007218` | `0.038155` | `0.210305` | `0.380557` | `0.369397` |
| M4e expected | `0.282295` | static `0.720879` | `-0.007271` | `0.038149` | `0.210303` | `0.382382` | `0.366743` |
| M5c expected | `0.281527` | count `0.550935` | `-0.251077` | `0.037888` | `0.211168` | `0.451499` | `0.413252` |
| M5f expected | `0.282063` | count `0.683644` | `-0.007940` | `0.038079` | `0.210663` | `0.450428` | `0.412793` |
| M5g expected | `0.282140` | static `0.720270` | `-0.007956` | `0.038099` | `0.210694` | `0.450421` | `0.412748` |

解释：

- M4a/M5c 已能匹配动态 pickup reward，但明显低估进入金币目标格的次数。`pickup_count` 和 `static_pickup_count` 都能修正该问题。
- `static_pickup_count` 下，模型期望的入口次数几乎贴合实际：M4e actual `0.720919`，expected `0.720879`。
- 加入中心项后，`center_delta_chebyshev` 也从明显偏负修正到接近实际。
- M5 系列主要修正两步连续同向等 path shape 细节；M4 系列更简洁。

## 推荐结论

第一版 simulator 默认 NPC 建议从 M4e 开始：

```text
score(path) =
    0.907331 * dynamic_reward_div10(path)
  + 0.912740 * static_pickup_count(path)
  - 2.716949 * enter_bomb_count(path)
  - 3.065158 * stay_count(path)
  + 1.209899 * straight3(path)
  - 0.709615 * backtrack(path)
  + 10.916492 * bomb_trapped_stay(path)
  + 0.167728 * center_delta_chebyshev(path)
```

M4e 的理由：

- 参数量仍较小，保留 M4a 的可解释结构。
- 相比 M4c center-only，valid NLL 从 `3.246296` 降到 `3.177952`。
- 相比 M4d 动态 `pickup_count`，valid NLL 继续改善 `0.0150`。
- 同时修正 pickup 次数和中心吸引两个已在 rollout/观测中暴露的问题。
- top1、top3、MRR 均高于 M4c。

若目标是更强的统计分布复现，可使用 M5g：

```text
score(path) =
    0.875531 * dynamic_reward_div10(path)
  + 0.905037 * static_pickup_count(path)
  - 2.710227 * enter_bomb_count(path)
  - 2.758740 * stay_count(path)
  + 0.219614 * straight3(path)
  - 0.526579 * backtrack(path)
  + 10.652026 * bomb_trapped_stay(path)
  + 0.826686 * first_two_same_nonstay(path)
  + 0.716532 * last_two_same_nonstay(path)
  + 0.274567 * sandwich(path)
  + 0.167938 * center_delta_chebyshev(path)
```

M5g 的 valid NLL 最低，但参数更多，且相比 M4e 的增益只有 `0.0359`。面向第一版 simulator/RL 训练，M4e 更适合作为默认；M5g 可作为统计复现 profile 或后续 ablation 对照。
