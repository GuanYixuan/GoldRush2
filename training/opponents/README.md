# Opponent League

`training/opponents/` 负责训练和评估用的 opponent league。它只统一对手接口、对手参数随机化和 league 采样，不实现游戏规则、reward、feature extractor、平台接口或 RL learner。

## 接口

所有 opponent runner 统一为：

```python
runner.reset(seed, episode_context)
action = runner.act(game_input)
```

其中 `game_input` 来自 simulator 的官方 SDK 形状 observation，`action` 必须是 `simulator.types.GameOutput`。

runner 也实现 `__call__(game_input)`，因此可以直接传给 `simulator.envs.duel.run_duel()`。

## 当前实现

当前只实现 Python opponent：

- `stay`：永远不动，`vp=0`。
- `random`：随机动作，支持 `stay_prob` 和 `vp_policy`。
- `greedy_visible_gold`：只基于当前可见 `grid` 贪心抢金币，支持 `target_score`、`avoid_bombs`、`risk_weight` 和 `vp_policy`。
- `fast_probe_v3_like`：复刻 `fast_probing/v3` 的轻量行为骨架；扫描两个角色视野内最大可达金币，主角色贪心抢金币，副角色用剩余步数向中心移动，连续坏轮后购买 `9x9` 视野。

这些策略是训练/评估基础件，不追求平台强度。需要新增对手时，优先通过独立 runner 或 scripted 文件扩展 league，而不是把游戏规则、feature extractor 或 reward 逻辑放进 opponent 层。

不同 scripted opponent 放在 `training/opponents/scripted/` 下的独立文件中，避免单个 `scripted.py` 持续膨胀。外部仍通过 `training.opponents.scripted` 导入公开类和 `build_scripted_opponent()`。

## 参数随机化

league 使用两层随机化：

```text
先按 weight 采样 LeagueEntry
再从该 entry 的 ParamSpace 采样本局参数
```

`ParamSpace` 支持：

- `constant`
- `choice`
- `categorical`
- `uniform`
- `log_uniform`
- `int_uniform`
- `maybe`

训练 split 可以逐局随机化；eval 和 holdout 应使用固定 entry、固定参数范围和固定 seed 集，以保证评估曲线可比较。

## C++ Opponent 未接入

`OpponentSpec.kind` 预留：

- `cpp_subprocess`
- `cpp_binding`

当前没有实现 C++ runner。若需要接 C++ opponent，优先考虑 `cpp_subprocess` 用于评估/回归，协议采用 line-delimited JSON：

```json
{"type":"reset","seed":123,"params":{}}
{"type":"act","game_input":{}}
{"type":"close"}
```

返回：

```json
{"type":"action","output":{"actions":[0,4,4,4,4,4],"k":1,"order":0,"vp":0}}
```

高吞吐训练若需要 C++ opponent，再实现 `cpp_binding`。
