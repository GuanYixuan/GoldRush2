# Policy Runtime

`policy_runtime/` 承载训练和最终 C++ 提交策略共享的确定性运行时逻辑。当前实现冻结为 `goldrush2_feature_v1`，具体 feature 语义见 `docs/features/actor_feature_v1.md`。

## 边界

- C++ core 直接使用官方 `official_sdk/code/game_api.h` 的 `GameInput` / `GameOutput`。
- Python binding 只负责把 simulator 的 Python `GameInput` / `GameOutput` dataclass 转成官方 C++ struct。
- `observe()` 只能写入 observation 中可见事实。
- `commit_action()` 预留给后续策略状态使用，当前 v1 feature 不输出上一动作相关特征。

## 构建

```bash
conda run --no-capture-output -n goldrush python setup.py build_ext --inplace
```

## Python 用法

```python
from policy_runtime import FeatureExtractor

extractor = FeatureExtractor(player_id=1)
features = extractor.observe(game_input)
extractor.commit_action(game_output)
```

`features` 是 dict：

- `feature_schema`: `"goldrush2_feature_v1"`
- `planes`: numpy array, shape `(38, 17, 17)`
- `scalars`: numpy array, shape `(10,)`
- `channel_names`
- `scalar_names`

训练 checkpoint、ONNX 导出和部署策略应记录并校验 `feature_schema`。
