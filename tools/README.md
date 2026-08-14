# 本地工具说明

本目录放置仓库内可复用的辅助脚本。新增或修改工具时，应同步更新本文档，说明用途、是否会访问评测平台、是否会发起提交/对局，以及常用命令。

## 安全原则

- 默认优先做只读工具；会提交策略、发起对局或修改平台状态的工具必须在文档中明确标注。
- 平台登录 key 来自仓库根目录 `.env` 中的 `LOGIN_KEY`，工具不得打印 key 或 token。
- 下载的 replay 日志建议缓存到 `/tmp` 或其它 gitignore 路径，不要把大量平台日志直接写入仓库。

## `platform_games.py`

只读平台数据工具，用于查询公测对局列表、下载 replay 日志并计算常用统计指标。该工具不会提交策略，也不会发起对局。

### 常用命令

列出某模型相关对局：

```bash
conda run --no-capture-output -n goldrush \
  python tools/platform_games.py list --since '2026-07-26 18:58:48' --model player142
```

统计指定对局双方指标：

```bash
conda run --no-capture-output -n goldrush \
  python tools/platform_games.py stats 31240 31276 32590
```

只显示某个模型的指标：

```bash
conda run --no-capture-output -n goldrush \
  python tools/platform_games.py stats 32586 31535 32431 --focus player142
```

强制重新下载日志：

```bash
conda run --no-capture-output -n goldrush \
  python tools/platform_games.py stats 31240 --refresh
```

### 输出指标

`list` 输出：

- `game_id`
- 北京时间
- 地图
- 结果
- 对手
- 金币
- 平台解析状态

`stats` 输出：

- 毛金币、视野花费、净金币
- P50 / P90 / P99 / Avg / Max 决策耗时
- `forfeit` 行数
- 先手回合数与先手比

先手比口径：每回合优先使用 replay 的 `end.dispatch_order`，取其中第一个正数玩家 id 作为该回合先手；缺少 `dispatch_order` 时，用双方 `cost` 较小者兜底。

### 缓存

默认将 replay 日志缓存到：

```text
/tmp/goldrush_logs/
```

缓存路径可通过 `--cache-dir` 修改。缓存只用于减少重复下载，不作为长期数据源。

## `submission/`

策略导出后的提交资产组装与平台操作工具。完整流程见 `docs/submission_pipeline.md`。

### 只写本地资产

`build_cpp_policy.py` 将 actor ONNX 组装为 C++ `.so`，会写入指定 `--output-dir`，不会访问平台：

```bash
PYTHONPATH=. conda run --no-capture-output -n goldrush \
  python tools/submission/build_cpp_policy.py \
  --onnx temp/submission_builds/example/actor.onnx \
  --output-dir temp/submission_builds/example/cpp \
  --module-name PlayerExample \
  --fast-runtime-mode release
```

`--fast-runtime-mode` 必须显式填写。正式平台提交使用 `release`；训练、eval 或本地调试需要保留 fast debug 结果时使用 `debug`。

`smoke_cpp_policy.py` 在本地 simulator 中加载 `.so` 做短局 smoke，不会访问平台。

### 会访问或修改平台

以下工具会读取 `.env` 中的 `LOGIN_KEY`，不得打印 key 或 token：

- `platform_submit_selfplay.py`：上传同一 `.so` 两份并发起一局公测 self-play，随后下载 replay。
- `platform_publish_challenge.py`：上传 `.so` 为供他人挑战版本，会修改平台上当前 `player<user.id>` 挑战代码。

平台提交、发起对局或修改挑战版本只能在用户明确要求时执行。

## 多进程本地 eval

训练代码中的 `training.rl.eval_mp.evaluate_parallel()` / `evaluate_fast_runtime_crn_pair()` 是本地 simulator 评估入口，不访问评测平台。fast option 实验优先使用 `evaluate_fast_runtime_crn_pair()` 做 fast runtime on/off 的 CRN paired 对照；它会复用 stochastic policy sampling 的 common random numbers，并输出净金币、margin、炸弹、踩踏、视野等 paired delta。fast runtime 打开时还会报告 threshold 分布、fast success/miss/path-fail/fallback、belief effective score 和 fast 回合事件统计。

## `metrics_tensorboard.py`

只读训练指标查看工具。它将命令行明确指定的一个或多个实验目录中的 `metrics.jsonl` 导出为 TensorBoard event 文件；不会扫描整个 `temp/`，不会访问评测平台，也不会修改训练 checkpoint 或原始日志。

TensorBoard 依赖与当前 `goldrush` 的 `protobuf 3.20.2` 和 NumPy 2 需要保持兼容。`tensorboard 2.19.0` 已兼容 NumPy 2；其启动代码仍依赖已被新版 setuptools 移除的 `pkg_resources`，因此需要同时将 setuptools 限制在 `<81`。若环境未配置或无法启动 TensorBoard，执行：

```bash
conda run --no-capture-output -n goldrush python -m pip install \
  'tensorboard==2.19.0' 'setuptools<81'
```

### 常用命令

比较两个已完成实验：

```bash
conda run --no-capture-output -n goldrush python tools/metrics_tensorboard.py \
  temp/ppo_separate_critic_calibration/runs/separate_critic_20260807_114240 \
  temp/ppo_separate_critic_calibration/runs/linear_ramp_20260807_122717

conda run --no-capture-output -n goldrush tensorboard \
  --logdir temp/tensorboards_logs/生成的时间戳目录
```

查看持续训练的实验：

```bash
conda run --no-capture-output -n goldrush python tools/metrics_tensorboard.py \
  temp/your_experiment/runs/current_run \
  --watch-seconds 10
```

每个 `metrics.jsonl` 是一个 TensorBoard run，名称为 `实验目录名/相对日志目录`。未传 `--logdir` 时，工具会相对仓库根目录新建 `temp/tensorboards_logs/YYYYMMDD_HHMMSS`；无论从仓库根目录还是 `tools/` 启动，位置都一致。同秒发生冲突时自动添加序号。需要固定 event 目录时可显式传入 `--logdir`，覆盖旧 event 文件时才传入 `--overwrite`。默认集包含核心 PPO 指标、advantage/value 诊断、净金币 reward 分解、双方经济/交互事件，以及 batch assembly、总推理、总 scheduler、每 transition worker wall 四项性能汇总。`--metric FIELD` 可重复传入以选择特定字段；`--all-metrics` 用于专项排查全部有限数值与布尔字段。

传入 `--watch-seconds` 时，工具除跟随已发现文件的追加行外，还会在每轮轮询递归扫描指定实验目录。训练中新出现的 `metrics.jsonl` 会自动注册为新的 TensorBoard run；已有文件保持读取 offset，不会被重复导出。输入实验目录在启动时必须已经存在，但可以暂时不含 `metrics.jsonl`。

### 曲线分类

工具为每条标量写入稳定的 tag 前缀，TensorBoard 的 Scalars 页可按此折叠或筛选：

- `optimization`：actor/critic 学习率。
- `policy`：KL、clip fraction、policy loss、熵项，以及 actor/critic/联合梯度范数。
- `critic`：value loss、explained variance、value 统计。
- `return`：reward、return、advantage 统计。
- `match_outcome`：rollout 内训练对局的胜率、净金币差等双方对局结果。
- `agent_outcome`：agent 的净金币、视野花费、拾取、炸弹与踩踏事件。
- `opponent_outcome`：opponent 的净金币、拾取、炸弹与踩踏事件。
- `behavior`：动作、动作顺序、视野和移动相关分布。
- `performance`：worker、scheduler、inference 与其它耗时/吞吐剖析。
- `system`：transition 数、batch 数、可训练参数量等规模统计。
- `other`：尚未归类的数值字段，便于发现 metrics schema 新增项。
