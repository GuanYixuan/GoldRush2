# BC 基础设施

`training/bc/` 提供 GoldRush2.0 行为克隆 warm start 所需的数据采集、审计、加载和纯 BC 训练工具。当前主线 teacher 是 `fast_probe_v3_like` self-play 的外部后手方，但数据 schema 和训练接口不绑定单一 teacher。

Shard 中的 `teacher_diagnostic_tag` 只用于 audit 和观察 teacher 行为分布，不参与默认 `BcTensorDataset`、BC loss 或训练采样。该字段高度依赖 teacher，不应作为 BC 框架的通用训练语义。

## 采集语义

采集直接驱动 `RoundStepEnv`。每个 paired seed 生成两局：

- `first_player_id=1`，记录 P2 后手。
- `first_player_id=2`，记录 P1 后手。

每回合双方 teacher 都必须基于同一份行动前 observation 决策；禁止先执行先手再为后手重新生成 observation。后手方 `FeatureExtractor` 按 episode 顺序 `observe -> record -> commit_action` 推进。

## 常用命令

```bash
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python -m training.scripts.collect_bc_dataset \
  --output-dir temp/bc_datasets/v3_like_map1_smoke \
  --teacher fast_probe_v3_like \
  --map-ids 1 \
  --train-seeds 10000:10009 \
  --round-count 500 \
  --num-workers 4 \
  --task-chunk-size 2
```

```bash
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python -m training.scripts.audit_bc_dataset \
  --dataset-dir temp/bc_datasets/v3_like_map1_smoke \
  --device cpu
```

```bash
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python -m training.scripts.train_bc \
  --dataset-dir temp/bc_datasets/v3_like_map1_smoke \
  --output-dir temp/bc_runs/v3_like_map1_smoke \
  --device cpu \
  --epochs 1 \
  --batch-size 512
```
