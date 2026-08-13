# Policy Runtime

`onnxruntime_c_api/` 保存已验证可用于提交 `.so` 组装的 ONNX Runtime C API headers。`tools/submission/build_cpp_policy.py` 会复制这些 headers 到生成目录，并在 C++ 侧通过 `dlopen` / `dlsym` 动态调用平台环境中的 `libonnxruntime.so`，避免编译期链接。

`policy_runtime/` 承载训练和最终 C++ 提交策略共享的确定性运行时逻辑。当前实现冻结为 actor 侧 `goldrush2_feature_v2`，具体 feature 语义见 `docs/features/actor_feature_v2.md`。

训练期 privileged critic feature 不在本 runtime 中实现。critic feature 读取 simulator full state，由训练栈 Python 侧提供，schema 见 `docs/features/privileged_critic_feature_v1.md`。

## 边界

- C++ core 直接使用官方 `official_sdk/code/game_api.h` 的 `GameInput` / `GameOutput`。
- Python binding 只负责把 simulator 的 Python `GameInput` / `GameOutput` dataclass 转成官方 C++ struct。
- `observe()` 只能写入 observation 中可见事实。
- `commit_action()` 预留给跨回合策略状态使用，当前 actor v1 feature 不输出上一动作相关特征。

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
