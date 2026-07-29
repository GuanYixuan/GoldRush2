# GoldRush2.0 策略提交约束

本文集中记录策略提交、运行环境、文件大小、日志回传和工程路线相关结论。平台接口字段细节见 `gamerules/platform.md`，回放字段结构见 `gamerules/replay.md`；神经网络推理测速和量化实验见 `docs/neural_inference.md`。

## 提交入口

公测阶段使用 `POST /api/user/add_model_1` 上传代码并发起对局。

基础约束：

- 请求类型：`multipart/form-data`。
- Python 策略：上传单个 `.py`，`model_langs=1`。
- C++ 策略：上传单个 `.so`，`model_langs=2`。
- 自己两份代码对战：上传两组 `model_files` / `model_langs` / `model_names`。
- 选择他人代码：上传自己一份代码，并额外传 `model_id`。
- 上传成功以业务字段 `code=0` 且 `data.game_id` 存在为准，不应只看 HTTP `200`。
- `code=0` 只表示已创建对局，不表示模型运行正常；运行异常、格式非法或超时仍可能在可解析回放末尾出现 `forfeit`。

模型名约束：

- `model_names` 需要字母开头，且仅包含字母和数字。
- 不接受下划线。
- 模型名过长时也可能返回同一条校验提示：`Model名称仅限字母和数字, 字母开头`。
- 自动化探测建议使用短字母数字名，例如 `ProbeA01`、`Norm01`。

提交后建议流程：

1. 调用 `add_model_1`，确认 `code=0` 和 `data.game_id`。
2. 轮询 `GET /api/user/get_game_info?id=<game_id>`。
3. 等待 `is_upload_log=2`，或同时满足 `is_upload_log=1` 且 `is_parse_log=1`。
4. 只有 `is_parse_log=1` 后才请求 `GET /api/user/get_game_log?id=<game_id>`。
5. 即使 `game_info.error_msg` 为空，也要检查回放末尾是否存在 `forfeit`。

## 文件大小

当前实测边界：

| 类型 | 最大成功样本 | 失败样本 | 结论 |
| --- | ---: | ---: | --- |
| Python `.py` | `16760822` B | `16777206` B | 单文件可行/失败区间差值 `16384` B。 |
| C++ `.so` | `16776128` B | `16792504` B | 单文件可行/失败区间差值 `16376` B。 |

注意：

- 失败样本表现为发送 multipart body 时连接被对端重置，且最近对局列表中没有对应模型，说明未创建对局。
- 这些边界不应理解为纯文件大小硬上限；平台接收的是 multipart 请求体，字段、边界、文件名和第二个模型也占用字节。
- 实际策略应保留明显余量，不要贴近 `16MiB`。

实验来源：

- `temp/neural_submit_probe_20260729/experiment_a_summary.md`
- `temp/neural_submit_probe_20260729/experiment_b_summary.md`

## Python 运行环境

当前 Python 评测环境可导入：

- `numpy`
- `torch`
- `onnxruntime`
- `scipy`
- `sklearn`

这些实验只验证“可 import”，不验证具体版本、模型加载、张量推理速度、CPU 指令兼容或内存上限。真实 Python 神经网络路线还需要继续测试：

- `onnxruntime.InferenceSession` 构造耗时。
- `session.run()` 每回合推理耗时。
- `torch` / `numpy` 首次调用和后续调用耗时。
- 内嵌权重解码、解压和数组构造成本。

实验来源：

- `temp/neural_submit_probe_20260729/experiment_c_summary.md`

## C++ 动态库环境

通过 C++ `dlopen("<library>", RTLD_LAZY | RTLD_LOCAL)` 探测得到：

| library | `dlopen` 推断 |
| --- | --- |
| `libonnxruntime.so` | 成功 |
| `libonnxruntime.so.1` | 成功 |
| `libpython3.so` | 成功 |
| `libpython3.10.so.1.0` | 失败 |
| `libpython3.11.so.1.0` | 失败 |
| `libopenblas.so` | 失败 |
| `libgomp.so.1` | 成功 |
| `libstdc++.so.6` | 成功 |

探测编码方式：

- C++ probe 只链接 `libdl`，不在编译期链接目标库。
- `dlopen` 成功时每回合返回 `vp=2`，失败时返回 `vp=0`。
- 500 回合完整运行时，probe 侧最终 `vision_spent=1500` 推断成功，`vision_spent=0` 推断失败。

局限：

- 该实验只证明 `dlopen` 二值可用性。
- 不验证 headers、C API ABI、符号版本或推理性能。
- 后续实验已进一步确认 ONNX Runtime C API 入口和最小 session 可用，见下节。

实验来源：

- `temp/neural_submit_probe_20260729/experiment_e_summary.md`

## C++ ONNX Runtime C API

进一步探测结果：

- `libonnxruntime.so` 和 `libonnxruntime.so.1` 均可通过 `dlsym("OrtGetApiBase")` 获取 C API 入口。
- `GetVersionString()` 返回非空，说明 C API base 可用。
- `GetApi(8/12/16/18/20/22)` 成功，`GetApi(29)` 失败。
- 当前评测环境支持 ONNX Runtime C API 至少到 version `22`；不应直接使用本地最新版 header 的 `ORT_API_VERSION=29`。
- 使用 C API version `8` 时，`CreateEnv`、`CreateSessionOptions`、`CreateSessionFromArray` 和一次 `Run` 最小 Identity ONNX 均成功。
- 最小 Identity ONNX session 从内嵌 bytes 创建成功；该模型体积为 `87` B。

工程建议：

- C++ + ONNX Runtime 路线应通过 `dlopen` / `dlsym` 动态获取 C API，避免编译期链接。
- 代码应显式请求已验证的 C API version，例如 `8` 或更高但不超过 `22`，不要直接使用未验证的最新 `ORT_API_VERSION`。
- session 应缓存并复用；本实验中首次创建 env/session/run 的首轮 cost 约 `9e6..1.1e7` 原始值，后续回合回到几十量级。
- 真实 CNN / Transformer / 端到端 ONNX 模型已进一步测速；详见 `docs/neural_inference.md`。

实验来源：

- `temp/neural_submit_probe_20260729/experiment_fgh_summary.md`

## 神经网络推理路线

`players[].cost` 是 replay 中的玩家策略本轮决策耗时字段，原始单位为纳秒；前端会格式化为 `ns` / `μs` / `ms` / `s` 展示。以下数值均为 replay 原始纳秒值。

当前与提交直接相关的结论：

- 当前推荐路线：C++ `.so` 内嵌 ONNX bytes，通过 `dlopen("libonnxruntime.so")` 调用 ONNX Runtime C API。
- `OrtEnv` / `OrtSession` / 常用 input tensor 应在 C++ 全局对象构造阶段初始化，避免计入首轮 `moveDecision`。
- static INT8 QDQ 是当前首选量化路线；实测同时显著降低 `.so` 体积和推理耗时。
- 目前更先撞线的是提交体积而非推理时间；`.so` 应保留明显余量，不要贴近 `16MiB`。
- 详细测速矩阵、量化对照、参数量分布和扩容建议见 `docs/neural_inference.md`。

性能相关实验来源：

- `docs/neural_inference.md`

## 日志与回传

已确认不可用或受限的通道：

- 选手程序 `print()` / stdout / stderr 不进入可见回放日志。
- `get_game_info.error_msg` 不是选手程序日志或异常详情的可靠回传通道。
- Python 异常 message 可进入回放末尾 `forfeit.exception_repr`，但会立即判负。
- Python 非法返回值可进入 `forfeit.raw_return`，但会立即判负。
- C++ `std::runtime_error::what()` 未进入 `forfeit.exception_repr`。
- C++ 非法 `GameOutput` 的 `raw_return` 只是 wrapper repr，不包含结构体字段值。

当前长度观察：

| 通道 | 样本 | 可见长度 |
| --- | --- | --- |
| Python `forfeit.exception_repr` | 256KB ASCII payload | 截断到 `1024` 字符，保留前缀。 |
| Python `forfeit.raw_return[0]` | 256KB ASCII payload | 截断到 `64` 字符，保留前缀。 |
| C++ `std::runtime_error::what()` | 约 4KB payload | 不可见。 |
| C++ 非法 `GameOutput` | 整数字段 marker | 字段值不可见。 |

因此：

- Python 可以用判负回传做短 probe，但会牺牲该局。
- C++ probe 结果应优先使用 replay 可观察行为编码，例如成功时购买视野、失败时不购买视野。

详细字段语义和历史样本见 `gamerules/replay.md` 的“判负显示”。

## 推荐工程路线

当前判断：

- 首选：C++ + ONNX Runtime C API + static INT8 QDQ。
- 备选：C++ 内嵌权重 + 自写推理；该路线不依赖外部库，但需要将模型结构翻译为 C++。
- 不优先：Python + ONNX Runtime；开发成本低，但 Python 调用和初始化成本更高，除非另做针对性测速。
