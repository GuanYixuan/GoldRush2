# Policy Runtime

`policy_runtime/` 承载训练和最终 C++ 提交策略共享的确定性运行时逻辑。第一阶段只实现最小 C++ feature extractor 状态机，后续再扩展 belief、snapshot、batch extraction 和 fast option。

## 边界

- C++ core 直接使用官方 `official_sdk/code/game_api.h` 的 `GameInput` / `GameOutput`。
- Python binding 只负责把 simulator 的 Python `GameInput` / `GameOutput` dataclass 转成官方 C++ struct。
- `observe()` 只能写入 observation 中可见事实。
- `commit_action()` 只能记录策略上一回合输出，不能推演移动结果或写入未观测事实。

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

- `planes`: numpy array, shape `(11, 17, 17)`
- `scalars`: numpy array, shape `(14,)`
- `channel_names`
- `scalar_names`

当前 feature 是最小 baseline，不代表最终网络输入结构。
