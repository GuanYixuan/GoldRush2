# NPC 重复消耗 ablation

本文是历史 rollout ablation，解释为何需要给 NPC path score 增加进入金币目标格的基础效用。当前 NPC 默认实现已更新为 `npc_behavior_modeling_overview.md` 中的 M4e static-pickup+center；本文不冻结当前默认参数。

## 实验目的

上一轮 map1 layout A 对齐实验发现：当前 simulator 默认机制会留下过多小金币堆，并且平均地面金币堆数明显高于真实 replay。由于金币生成模型已有较强独立证据，本实验固定金币/炸弹/玩家策略，只替换 NPC path score，验证“NPC 对残堆重复消耗不足”是否能解释该偏差。

## 实验设置

- 真实目标：`symobs-a-map1-100-20260727-231446` 的 `real_visible` 统计。
- 仿真口径：map1 + layout A 双方策略，每个变体 rollout `100` 局，每局 `500` 回合。
- baseline：实验当时的默认 NPC，即 M4a + center chebyshev；该 baseline 已被后续 canonical M4e static-pickup+center 拟合替换。
- 临时脚本：`temp/simulator_alignment/map1_a_default/run_npc_ablation_experiment.py`
- 详细表：`temp/simulator_alignment/map1_a_default/npc_ablation/`

运行命令：

```bash
conda run --no-capture-output -n goldrush python temp/simulator_alignment/map1_a_default/run_npc_ablation_experiment.py --games 100 --rounds 500
```

补充中间变体命令：

```bash
conda run --no-capture-output -n goldrush python temp/simulator_alignment/map1_a_default/run_npc_ablation_experiment.py --games 100 --rounds 500 --variants repeat_1p00 cleanup_small_0p75_repeat_0p50 cleanup_small_0p75_repeat_1p00 cleanup_small_1p00 cleanup_small_1p00_repeat_0p50
```

基础金币项补充命令：

```bash
conda run --no-capture-output -n goldrush python temp/simulator_alignment/map1_a_default/run_npc_ablation_experiment.py --games 100 --rounds 500 --variants pickup_0p25 pickup_0p50 pickup_0p75 pickup_1p00 pickup_0p50_cleanup_small_0p50
```

## Ablation 项

在 baseline score 之外临时加入：

```text
score += pickup_bonus * count(path 中发生金币 pickup 的步数)
score += small_bonus  * count(path 中进入当前 amount <= 3 的金币格)
score += repeat_bonus * count(path 中重复进入同一金币格)
score += clear_bonus  * count(path 中把金币堆清空的 pickup)
```

这些项在本实验时只用于 ablation。后续正式 path-level 拟合确认进入金币目标格的基础效用是稳定强特征，并已纳入 simulator 默认 M4e static-pickup+center profile。

## 总体结果

end frame 主指标：

| variant | score | avg_piles | avg_total | avg_mean | `amount<=3` | reductions | pickups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real | - | `24.13` | `150.89` | `6.17` | `48.97%` | `204689` | - |
| baseline | `1.0138` | `36.16` | `180.46` | `5.00` | `59.82%` | `161557` | `177721` |
| pickup_0p25 | `0.7777` | `33.00` | `170.87` | `5.18` | `58.70%` | `167726` | `186325` |
| pickup_0p50 | `0.5704` | `30.22` | `162.63` | `5.37` | `57.32%` | `172567` | `193730` |
| pickup_0p75 | `0.3553` | `27.27` | `153.87` | `5.61` | `55.19%` | `175381` | `199103` |
| pickup_1p00 | `0.2089` | `25.15` | `150.05` | `5.94` | `52.17%` | `178267` | `204850` |
| pickup_0p50_cleanup_small_0p50 | `0.3291` | `26.30` | `162.51` | `6.14` | `47.44%` | `174635` | `199881` |
| repeat_1p00 | `1.0117` | `35.68` | `180.99` | `5.09` | `59.22%` | `157409` | `180071` |
| cleanup_small_0p75 | `0.6652` | `29.99` | `185.09` | `6.15` | `45.88%` | `170936` | `192652` |
| cleanup_small_0p75_repeat_0p50 | `0.6490` | `29.77` | `182.32` | `6.12` | `45.97%` | `168447` | `194618` |
| cleanup_small_0p75_repeat_1p00 | `0.6073` | `28.76` | `178.75` | `6.19` | `45.24%` | `165100` | `196017` |
| cleanup_small_1p00 | `0.6067` | `28.25` | `181.59` | `6.39` | `42.32%` | `170699` | `194687` |
| cleanup_small_1p00_repeat_0p50 | `0.6123` | `27.86` | `182.07` | `6.51` | `41.65%` | `168260` | `197558` |
| cleanup_small_1p50 | `0.6376` | `26.30` | `187.19` | `7.08` | `34.51%` | `171461` | `200737` |
| cleanup_small_1p50_repeat_1p00 | `0.6201` | `25.27` | `184.81` | `7.29` | `33.61%` | `164882` | `204154` |
| cleanup_small_1p50_clear_0p75 | `0.6747` | `24.83` | `191.49` | `7.66` | `28.50%` | `169450` | `204362` |
| cleanup_small_3p00 | `0.7876` | `22.59` | `189.66` | `8.35` | `22.61%` | `163147` | `206756` |

`score` 是临时综合距离，包含平均金币堆数、地面金币总量、`amount<=3` 占比和 visible reductions 与真实值的相对差异；只用于排序，不是机制拟合 NLL。

## 关键观察

### repeat-only 基本无效

`repeat_1p00` 与 baseline 几乎一致：

- 平均金币堆数：`36.16 -> 35.68`
- `amount<=3`：`59.82% -> 59.22%`
- visible reductions：`161557 -> 157409`

这说明“同一个 3-step path 内重复进入同一格”不是主要缺失项。真实中的重复消耗更可能是跨回合/跨邻域地持续清残堆，而不是单轮路径内部来回。

### 基础 pickup bonus 解释力最强

`pickup_bonus` 表示“只要路径中发生一次金币 pickup，就额外给一项基础分”，不按金币金额线性放大。该项比硬阈值 small cleanup 更有效：

- `pickup_1p00`：avg_piles `25.15`，接近真实 `24.13`。
- `pickup_1p00`：avg_total `150.05`，几乎贴合真实 `150.89`。
- `pickup_1p00`：pickups `204850`，也接近真实 visible reductions `204689`，虽然两者口径并不完全相同。
- `pickup_1p00` 的 `amount<=3=52.17%`，仍略高于真实 `48.97%`，但远优于 baseline `59.82%`。

这说明此前“残堆重复消耗不足”的现象，更可能来自 M4a 里缺少“发生 pickup 本身的固定效用”，而不一定是专门针对低金额残堆的规则。

### 低金额金币堆 bonus 有效但不如基础 pickup

`small_bonus=0.75..1.0` 能显著改善小堆残留问题：

- `cleanup_small_0p75`：`amount<=3` 从 `59.82%` 降到 `45.88%`，已接近真实 `48.97%`；平均每堆金额从 `5.00` 拉到 `6.15`，接近真实 `6.17`。
- `cleanup_small_1p00`：平均金币堆数从 `36.16` 降到 `28.25`，但 `amount<=3=42.32%` 已低于真实。

这支持“NPC 更愿意清理低金额残堆”这一方向，但基础 pickup 项能同时改善堆数、总量和 pickup 事件数，解释力更强。

### 基础 pickup + 弱 small cleanup 分布更均衡

`pickup_0p50_cleanup_small_0p50` 的总体 score 不如 `pickup_1p00`，但 amount 分布更接近真实：

- avg_piles：`26.30`，接近真实 `24.13`。
- avg_total：`162.51`，仍高于真实 `150.89`。
- avg_mean：`6.14`，接近真实 `6.17`。
- `amount<=3`：`47.44%`，接近真实 `48.97%`。

若目标是让地面金币堆 amount 分布更像真实，这个组合比单独 `pickup_1p00` 更稳；若目标是总库存和堆数，`pickup_1p00` 更强。

### 强 cleanup 会过度清残堆

`small_bonus>=1.5` 后，平均金币堆数接近真实，但小金币堆占比被压得过低：

- `cleanup_small_1p50_repeat_1p00`：avg_piles `25.27` 接近真实 `24.13`，但 `amount<=3=33.61%`。
- `cleanup_small_3p00`：avg_piles `22.59`，`amount<=3=22.61%`，明显过清。

因此不能简单把 cleanup 权重调大作为默认机制。

### 地面总金币偏高主要被基础 pickup 解释

只加 small cleanup 时，`avg_total` 仍明显高于真实：

- real：`150.89`
- baseline：`180.46`
- best score `cleanup_small_1p00`：`181.59`
- 堆数接近真实的 `cleanup_small_1p50_repeat_1p00`：`184.81`

但 `pickup_1p00` 将 `avg_total` 拉到 `150.05`，几乎贴合真实。因此总金币偏高不再优先指向金币生成过强，也不必急着引入高价值区域驻留项；更直接的解释是 NPC path score 缺少“吃到金币就值得”的基础效用。

## 区域分解

end frame 各区域 `amount<=3`：

| dataset | center9 | other | static2 |
| --- | ---: | ---: | ---: |
| real | `58.86%` | `46.40%` | `35.18%` |
| baseline | `69.27%` | `55.26%` | `50.63%` |
| pickup_0p75 | `65.66%` | `52.12%` | `44.01%` |
| pickup_1p00 | `64.49%` | `48.65%` | `39.77%` |
| pickup_0p50_cleanup_small_0p50 | `57.85%` | `45.64%` | `35.01%` |
| cleanup_small_1p00 | `51.64%` | `42.99%` | `27.66%` |
| cleanup_small_0p75_repeat_1p00 | `54.31%` | `44.10%` | `33.33%` |
| cleanup_small_1p50_repeat_1p00 | `42.16%` | `35.77%` | `18.58%` |

区域分解进一步支持基础 pickup 项：

- `pickup_1p00` 能明显修正 `static2`，从 baseline `50.63%` 降到 `39.77%`，接近真实 `35.18%`。
- `pickup_0p50_cleanup_small_0p50` 的区域 `amount<=3` 最接近真实：center9 `57.85%` vs `58.86%`，other `45.64%` vs `46.40%`，static2 `35.01%` vs `35.18%`。
- 单独 small cleanup 容易把 static2 过度清理到低于真实。

## 结论

本实验更新了对该问题的判断：当前 simulator 小金币堆偏差的更强解释不是“专门缺少低金额残堆规则”，而是 NPC path score 缺少“发生金币 pickup 的基础效用”。

1. 基础 pickup bonus 是目前解释力最强的单项，`pickup_1p00` 同时改善平均金币堆数、地面总金币、pickup 事件数和小堆占比。
2. 单轮 path 内 repeat bonus 基本无效，说明需要建模的是跨回合/局部区域的残堆清理倾向。
3. 低金额 cleanup bonus 仍有价值，尤其能修正 amount 分布；但单独使用时不能解释地面总金币偏高。
4. 当前还不急着优先加入高价值区域驻留项；应先把基础 pickup 项纳入候选 NPC 模型，并在 map2/map3 上交叉验证。

后续 canonical 拟合已确认更强方向为 `static_pickup_count`：每次移动进入 round-start 时有金币的格子计 `1`，不模拟 path 内金币扣减。该方向已进入当前 NPC 默认实现；具体公式和权重见 `mechanism/results/npc/npc_behavior_modeling_overview.md`。

若还希望 amount 分布更贴近真实，可测试弱组合：

```text
pickup_weight = 0.50
cleanup_small_weight = 0.50
```

不建议把单轮 repeat bonus 作为核心项，也不建议直接加入强 clear bonus。
