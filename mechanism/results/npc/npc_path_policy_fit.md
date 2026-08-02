# NPC Path Policy 拟合产物索引

本文只记录 NPC path-level softmax 拟合的复现实验信息和产物位置，不冻结 simulator 默认实现。当前推荐模型、公式、特征定义和 domain randomization 以 `npc_behavior_modeling_overview.md` 为准。

## 口径

canonical NPC plan 口径：

1. 回合开始先刷新炸弹、生成金币，并把资源变化落到 round-start state。
2. 玩家与 NPC 都基于该同一个 round-start state 规划动作。
3. 7 个 NPC actions 同时产出，不看先手玩家行动后的状态，也不看同轮前序 NPC 已执行后的状态。
4. 之后按 `first_player -> NPC permutation -> second_player` 执行和结算。

## 输入

- `symobs-a-map1-100-20260727-231446`
- `symobs-a-map2-100-20260727-231704`
- `symobs-m3a-map3-100-20260728-200953`
- `symobs-m3b-map3-100-20260728-202647`

样本量：

| split | samples |
| --- | ---: |
| train | `68849` |
| valid | `18113` |
| total | `86962` |

各 run 抽样：

| run_id | eligible samples | used samples |
| --- | ---: | ---: |
| `symobs-a-map1-100-20260727-231446` | `301553` | `30156` |
| `symobs-a-map2-100-20260727-231704` | `308840` | `30884` |
| `symobs-m3a-map3-100-20260728-200953` | `126378` | `12638` |
| `symobs-m3b-map3-100-20260728-202647` | `132832` | `13284` |

## 运行

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

关键产物：

- `feature_ablation.csv`：各模型 train/valid NLL、top1、top3、MRR、actual path probability。
- `weights.json`：各模型拟合权重。
- `validation_summary.csv`：actual/model/uniform 的特征均值对照。
- `metrics.json`：完整元数据、样本计数、模型配置和指标。

## 最小指标

valid split 关键结果：

| model | valid NLL | top1 | top3 | actual path prob |
| --- | ---: | ---: | ---: | ---: |
| M4a dynamic+shape+trapped | `3.267048` | `18.08%` | `37.39%` | `7.097%` |
| M4d + dynamic pickup + center | `3.192912` | `19.35%` | `39.24%` | `7.903%` |
| M4e + static pickup + center | `3.177952` | `19.59%` | `40.47%` | `8.043%` |
| M5g + static pickup + center + shape | `3.142093` | `18.57%` | `38.84%` | `8.354%` |

结论：M4e 是当前 M4 系列中最强的简洁模型；M5g 的 NLL 最低但参数更多。默认实现选择见 `npc_behavior_modeling_overview.md`。
