# 正式文档入口

`docs/` 存放项目内需要长期维护的架构、schema、feature、reward、网络和工程路线文档。游戏规则本身不在本目录冻结，统一以 `gamerules/gamerules.md` 为准。

## 当前主线文档

- `rl_architecture.md`：训练栈、rollout、PPO batch、BC warm start 和 privileged critic 的总体口径。
- `policy_network_design.md`：当前 actor/critic 分离网络、自回归动作头、critic gather 和 ONNX 约束。
- `reward_design.md`：当前 PPO reward 项、dense gold/net gold、BC 与 PPO 的关系。
- `features/README.md`：actor/critic feature schema 索引。
- `features/actor_feature_v2.md`：当前可部署 actor feature schema；v1 文档仅用于历史追溯和 checkpoint inflation。
- `features/privileged_critic_feature_v2.md`：当前训练期 privileged critic feature schema；v1 文档仅用于历史追溯和 checkpoint inflation。

## Replay 与派生格式

- `merged_replay_schema.md`：双账号玩家视角合并后的 replay 派生格式。
- `simulator_replay_schema.md`：本地 simulator 上帝视角 replay 格式。

官方平台 replay 格式见 `gamerules/replay.md`。

## 部署与研究归档

- `submission_pipeline.md`：checkpoint 到 ONNX、C++ `.so`、本地 smoke、平台 self-play 和供挑战上传的正式流程。
- `neural_inference.md`：C++/ONNX Runtime 提交路线、平台速度和量化实验归档。
- `fast_option_design.md`：比赛后期进入主线的 conditional fast option、threshold head、runtime 状态与赛后冻结评估结论。

## 维护原则

- 同一条规则或 schema 只在一个文档中冻结；索引文档只做导航。
- 修改 feature、reward、网络或 replay schema 时，同步更新对应正式文档。
- 实验观察和一次性分析优先放在 `temp/`；只有会影响后续实现或复现实验的结论才迁入本目录。
