# Policy Runtime

`onnxruntime_c_api/` 保存已验证可用于提交 `.so` 组装的 ONNX Runtime C API headers。`tools/submission/build_cpp_policy.py` 会复制这些 headers 到生成目录，并在 C++ 侧通过 `dlopen` / `dlsym` 动态调用平台环境中的 `libonnxruntime.so`，避免编译期链接。

`policy_runtime/` 承载训练和最终 C++ 提交策略共享的确定性运行时逻辑，包括 actor 侧 `goldrush2_feature_v2`、conditional fast controller、跨回合 `FastRuntimeState` 及 pending/backfill。具体 feature 语义见 `docs/features/actor_feature_v2.md`，fast 行为与 threshold 语义见 `docs/fast_option_design.md`。

训练期 privileged critic feature 不在本 runtime 中实现。critic feature 读取 simulator full state，由训练栈 Python 侧提供，当前 schema 见 `docs/features/privileged_critic_feature_v2.md`。

## Conditional fast runtime

比赛后期主线已使用这套 runtime。每回合先尝试依据上一回合保存的 `threshold_int` 执行 fast controller；命中时直接返回动作并保存最小 pending，未命中时由 `prepare_neural()` 补齐历史、更新 fast belief、生成 actor feature，再进入神经网络。普通神经回合结束后由 `commit_neural()` 提交动作状态并保存下一回合 threshold。

训练入口的 `--enable-fast-runtime-features` 控制 rollout/eval 是否接入这套行为，默认关闭。提交构建参数 `--fast-runtime-mode release/debug` 只选择是否保留调试信息和热路径实现方式，不是 fast option 的开关；正式提交使用 `release`。

## 边界

- C++ core 直接使用官方 `official_sdk/code/game_api.h` 的 `GameInput` / `GameOutput`。
- Python binding 只负责把 simulator 的 Python `GameInput` / `GameOutput` dataclass 转成官方 C++ struct。
- `observe()` 只能写入 observation 中可见事实。
- 普通 feature-only 调用通过 `commit_action()` 记录动作；conditional fast 路径由 `FastRuntimeState` 的 `prepare_neural()`、`commit_neural()` 和 pending backfill 维护同一份跨回合语义。

## 构建

```bash
conda run --no-capture-output -n goldrush python setup.py build_ext --inplace
```

`setup.py` 会显式以 `POLICY_RUNTIME_FAST_DEBUG=1` 构建 Python binding，供训练、eval 和本地调试使用。正式平台 `.so` 由 `tools/submission/build_cpp_policy.py --fast-runtime-mode release` 生成，并使用 `POLICY_RUNTIME_FAST_DEBUG=0`；`fast_option.h` 要求该宏必须显式设置，避免混用训练调试版和正式导出版。

## Python 用法

```python
from policy_runtime import FeatureExtractor

extractor = FeatureExtractor(player_id=1)
features = extractor.observe(game_input)
extractor.commit_action(game_output)
```

`features` 是 dict：

- `feature_schema`: `"goldrush2_feature_v2"`
- `planes`: numpy array, shape `(43, 17, 17)`
- `scalars`: numpy array, shape `(10,)`
- `channel_names`
- `scalar_names`

训练 checkpoint、ONNX 导出和部署策略应记录并校验 `feature_schema`。actor feature shape 或语义变化必须使用新的 schema 名称，不能在 `goldrush2_feature_v2` 下隐式改变。
