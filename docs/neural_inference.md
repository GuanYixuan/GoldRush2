# 神经网络推理实验归档

本文记录 GoldRush2.0 神经网络策略在评测平台上的推理路线、速度、体积和量化实验结论。平台提交硬约束见 `gamerules/submission.md`；本文只讨论神经网络工程路线。

## 当前推荐

当前首选路线：

- C++ `.so` 提交。
- ONNX 内容内嵌进 `.so`。
- 通过 `dlopen("libonnxruntime.so")` / `dlsym("OrtGetApiBase")` 调用 ONNX Runtime C API。
- 显式使用已验证的 C API version `8`。
- `OrtEnv` / `OrtSession` / 常用 input tensor 在 C++ 全局对象构造阶段初始化，避免计入首轮 `moveDecision`。
- 模型使用 static INT8 QDQ，per-channel，优先量化 `Conv/MatMul/Gemm`。

结论摘要：

- 评测端 ONNX Runtime 可运行 FP32、FP16、dynamic INT8 和 static INT8 QDQ ONNX。
- static INT8 QDQ 同时显著降低体积和推理耗时，是当前最优路线。
- 当前实测下，体积比推理时间更先成为瓶颈。
- `.so` 应按提交大小留余量；本实验采用约 `15.50MB` 作为安全阈值。

## 工具链

本地导出和校验环境：

| 项 | 版本 |
| --- | --- |
| Python | `3.12.13` |
| PyTorch | `2.6.0+cu124` |
| ONNX | `1.17.0` |
| ONNX Runtime | `1.20.1` |
| ONNX Script | `0.1.0` |
| ONNX Converter Common | `1.14.0` |
| NumPy | `2.1.3` |
| Protobuf | `3.20.2` |

注意：安装 `onnxconverter-common==1.14.0` 时 protobuf 曾从 `7.35.1` 降到 `3.20.2`。后续若 PyTorch 导出异常，应优先排查 protobuf 版本漂移。

## 平台测量方法

- 对局通过 `POST /api/user/add_model_1` 发起。
- 地图固定为 `map_id=2`。
- 对手使用静止 Python 轻逻辑策略。
- C++ probe 成功执行 ORT `Run` 时返回 `vp=2`；最终 `vision_spent=1500` 作为每回合成功运行的辅助信号。
- 速度从 replay 的 `players[].cost` 统计，原始单位为纳秒；本文表格统一换算为毫秒展示。
- 重点记录 `first`、`p50`、`p90`、`p99`、`max` 和是否 `forfeit`。

## ONNX Runtime 路线

已确认：

- `libonnxruntime.so` 可 `dlopen`。
- `OrtGetApiBase` 可通过 `dlsym` 获取。
- `GetApi(8/12/16/18/20/22)` 成功，`GetApi(29)` 失败。
- 使用 C API version `8` 时，`CreateEnv`、`CreateSessionOptions`、`CreateSessionFromArray` 和一次最小 Identity ONNX `Run` 均成功。
- 真实 CNN / Transformer / 端到端模型均可通过同一路线运行完整 `500` 回合。

工程建议：

- 不要编译期链接 ONNX Runtime；使用 `dlopen` 更稳。
- 不要直接使用本地最新版 header 的 `ORT_API_VERSION=29`。
- session 必须缓存复用；模型加载不要放进 `moveDecision`。

## 模型结构实验

目标结构：

```text
belief map: C x 17 x 17
  -> Residual CNN
  -> spatial heads: value / risk / path logits / dense feature
  -> gather 己方角色与其它关键位置
  -> pool 若干区域或 belief mask
  -> 轻量 Transformer
  -> k / order / vp / plan logits
```

### CNN 主干

| probe | game_id | C | width | blocks | 参数量 | SO 大小 | p50 | p90 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `NsCnnS` | 50467 | 32 | 48 | 4 | 180.2K | 741.3KB | 1.14ms | 1.17ms | 1.21ms | 1.29ms |
| `NsCnnM` | 50469 | 48 | 64 | 8 | 618.6K | 2.50MB | 3.65ms | 3.67ms | 3.77ms | 4.25ms |
| `NsCnnL` | 50470 | 64 | 96 | 12 | 2.05M | 8.22MB | 11.25ms | 11.28ms | 11.40ms | 11.88ms |

CNN 主干是主要耗时来源；在 `17x17` 输入上，耗时随参数量近似增长。`NsCnnL` 仍远低于 `300ms` 限时，但已经接近 FP32 路线下的体积压力。

### 空间 heads 与 gather/pool

| probe | game_id | 结构 | 参数量 | SO 大小 | p50 | p90 | p99 | max |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `NsHead1` | 50486 | M backbone + value/risk/path | 619.0K | 2.50MB | 3.66ms | 3.68ms | 3.75ms | 4.21ms |
| `NsHead2` | 50487 | M backbone + value/risk/path + dense32 | 621.1K | 2.51MB | 3.68ms | 3.70ms | 3.78ms | 4.32ms |
| `NsHead3` | 50488 | L backbone + value/risk/path + dense64 | 2.06M | 8.25MB | 11.29ms | 11.32ms | 11.42ms | 11.90ms |
| `NsPoolFixed` | 50493 | fixed gather 11 + fixed pool 5 | 619.6K | 2.51MB | 3.66ms | 3.68ms | 3.74ms | 3.80ms |
| `NsPoolDyn` | 50497 | dynamic idx gather 11 + fixed pool 5 | 619.6K | 2.51MB | 3.67ms | 3.69ms | 3.72ms | 4.26ms |
| `NsPoolRich` | 50499 | dynamic idx gather 16 + dynamic mask pool 8 | 620.1K | 2.51MB | 3.66ms | 3.68ms | 3.76ms | 3.87ms |

这些模块没有观察到明显额外开销；主要成本仍由 CNN 卷积主干决定。

### Transformer

| probe | game_id | line | shape | 参数量 | nodes | opset | SO 大小 | p50 | p90 | p99 | max |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `NsTrPrimS` | 50534 | primitive | 16x64, 1L, 2H | 33.5K | 51 | 13 | 155.7KB | 0.05ms | 0.06ms | 0.07ms | 0.17ms |
| `NsTrTorchS` | 50535 | torch module | 16x64, 1L, 2H | 33.5K | 101 | 14 | 163.8KB | 0.05ms | 0.06ms | 0.14ms | 0.18ms |
| `NsTrPrimM` | 50536 | primitive | 24x96, 2L, 4H | 149.7K | 97 | 13 | 626.7KB | 0.14ms | 0.15ms | 0.25ms | 0.29ms |
| `NsTrTorchM` | 50537 | torch module | 24x96, 2L, 4H | 149.7K | 197 | 14 | 643.1KB | 0.14ms | 0.15ms | 0.16ms | 0.27ms |
| `NsTrPrimL` | 50538 | primitive | 32x128, 3L, 4H | 397.6K | 143 | 13 | 1.63MB | 0.33ms | 0.34ms | 0.38ms | 0.52ms |
| `NsTrTorchL` | 50539 | torch module | 32x128, 3L, 4H | 397.6K | 293 | 14 | 1.65MB | 0.33ms | 0.35ms | 0.37ms | 0.49ms |

当前 token 数较小，Transformer 模块本身很轻。`32x128, 3L, 4H` 的 p90 约 `0.34ms`，远低于 M 级 CNN 主干约 `3.67ms`。

导出注意：

- `nn.TransformerEncoderLayer` 在 PyTorch `2.6.0+cu124` legacy exporter 下会落到不支持的 `aten::_transformer_encoder_layer_fwd`。
- 可行写法是 `nn.MultiheadAttention + nn.LayerNorm + FFN`。
- 需要关闭 `torch.backends.mha` fastpath，否则会落到不支持的 `aten::_native_multi_head_attention`。
- 关闭 fastpath 后，`scaled_dot_product_attention` 需要 opset `14`。

### 端到端 FP32

| probe | game_id | 结构 | 参数量 | nodes | SO 大小 | p50 | p90 | p99 | max |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `NsE2ES` | 50574 | C32 W48 4res, 16tok D64 1L | 220.1K | 154 | 921.6KB | 1.21ms | 1.23ms | 1.24ms | 1.37ms |
| `NsE2EM` | 50575 | C48 W64 8res, 24tok D96 2L | 782.6K | 272 | 3.19MB | 3.83ms | 3.84ms | 3.93ms | 4.44ms |
| `NsE2EL` | 50576 | C64 W96 12res, 32tok D128 3L | 2.47M | 388 | 9.98MB | 11.65ms | 11.67ms | 11.75ms | 12.00ms |

端到端 FP32 中，heads、dynamic gather、mask pool 和 `32x128, 3L` Transformer 没有引入明显额外延迟；耗时仍主要来自 CNN 主干。

## 量化实验

### 本地兼容性

| baseline | variant | ONNX 大小 | local ORT | key ops |
| --- | --- | ---: | --- | --- |
| `NsCnnL` | `fp32` | 8.21MB | ok | Conv:25, Gemm:1 |
| `NsCnnL` | `fp16` | 4.11MB | ok | Cast:2, Conv:25, Gemm:1 |
| `NsCnnL` | `dynamic_int8` | 8.21MB | ok | Conv:25, DynamicQuantizeLinear:1, MatMulInteger:1 |
| `NsCnnL` | `static_int8_qdq` | 2.15MB | ok | Conv:25, DequantizeLinear:104, Gemm:1, QuantizeLinear:52 |
| `NsE2EL` | `fp32` | 9.97MB | ok | Cast:6, Conv:29, Gemm:4, MatMul:17 |
| `NsE2EL` | `fp16` | 5.05MB | ok | Cast:9, Conv:29, Gemm:4, MatMul:17 |
| `NsE2EL` | `dynamic_int8` | 8.77MB | ok | DynamicQuantizeLinear:14, MatMulInteger:14 |
| `NsE2EL` | `static_int8_qdq` | 2.75MB | ok | Conv:29, DequantizeLinear:180, Gemm:4, MatMul:17, QuantizeLinear:104 |

### 平台量化对照

| probe | variant | game_id | ONNX 大小 | SO 大小 | nodes | p50 | p90 | p99 | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q2Fp32` | FP32 | 50795 | 9.97MB | 9.98MB | 388 | 11.64ms | 11.68ms | 11.95ms | 12.17ms |
| `Q2Fp16` | FP16 | 50797 | 5.05MB | 5.06MB | 391 | 11.94ms | 11.97ms | 12.46ms | 13.85ms |
| `Q2DynI8` | dynamic INT8 | 50799 | 8.77MB | 8.79MB | 448 | 11.52ms | 11.56ms | 11.69ms | 11.83ms |
| `Q2StaI8` | static INT8 QDQ | 50800 | 2.75MB | 2.76MB | 672 | 2.94ms | 2.96ms | 3.02ms | 3.44ms |

量化结论：

- FP16 约为 FP32 一半体积，但 CPU ORT 路线下没有加速，本次略慢。
- dynamic INT8 主要量化 `MatMul/Gemm`，对 CNN 主体帮助有限，体积和速度收益都不大。
- static INT8 QDQ `.so` 约为 FP32 的 `27.7%`，p90 `2.96ms`，约为 FP32 `11.68ms` 的 `25.3%`，当前最优。

## 更大 static INT8 QDQ

| probe | game_id | C | width | blocks | tokens | token_dim | transformer | 参数量 | SO 大小 | p50 | p90 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Q3S` | 51176 | 80 | 128 | 16 | 32 | 160 | 3L / 4H | 5.47M | 5.83MB | 6.29ms | 6.31ms | 6.53ms | 6.93ms |
| `Q3M` | 51178 | 96 | 160 | 20 | 40 | 192 | 4L / 4H | 10.61M | 11.09MB | 11.87ms | 11.94ms | 12.48ms | 12.68ms |
| `Q3LTrimA` | 51179 | 112 | 176 | 22 | 48 | 224 | 4L / 8H | 14.15M | 14.67MB | 15.92ms | 15.95ms | 16.23ms | 16.51ms |

最大 `Q3LTrimA` 完整运行 `500` 回合，无 forfeit，p90 为 `15.95ms`。这说明当前 static INT8 QDQ 路线下，推理时间仍不是主要限制；`.so` 体积已经接近安全提交阈值。

## 参数量分布

当前 Q3 结构中，参数几乎都在 CNN backbone：

| 模型 | 总参数 | CNN backbone | Transformer | 其它 |
| --- | ---: | ---: | ---: | ---: |
| `Q3S` | 5.47M | 4.81M, 88.0% | 0.62M, 11.3% | <1% |
| `Q3M` | 10.61M | 9.36M, 88.2% | 1.19M, 11.2% | <1% |
| `Q3LTrimA` | 14.15M | 12.45M, 88.0% | 1.62M, 11.4% | <1% |

其中最大项是 residual blocks 中的 `3x3 conv`，Q3LTrimA 中约占 `86.8%`。在 `token_dim` 固定时，`num_heads` 基本不改变参数量；`tokens` 数量主要影响 attention 运行时间，而非参数量。

因此：

- 压体积优先降低 CNN `width` 和 residual `blocks`。
- 想提升效果上界时，可以考虑把部分参数预算从 CNN 转到 Transformer 层数或高层 token 表达。
- 当前 token 数不大，适度增加 Transformer 层数比继续扩大 CNN width 更值得测试。

## 未测风险

- 当前 static INT8 校准样本较少，只验证工程可运行和速度，不代表量化后策略效果。
- 未系统测试内存上限。
- 未测试正式比赛环境是否与公测环境完全一致。
- 未测试真实训练后模型中的全部算子组合；新增算子前应做小规模平台 probe。
- `.so` 体积边界来自平台上传行为，实际请求还包含 multipart 字段、文件名、第二模型等开销，应持续保留余量。

## 实验来源

- `temp/neural_submit_probe_20260729/experiment_fgh_summary.md`
- `temp/neural_speed_probe_20260729/experiment_a_summary.md`
- `temp/neural_speed_probe_20260729/experiment_bc_summary.md`
- `temp/neural_speed_probe_20260729/experiment_d_summary.md`
- `temp/neural_speed_probe_20260729/experiment_e_summary.md`
- `temp/neural_quant_probe_20260729/q0_summary.md`
- `temp/neural_quant_probe_20260729/q1_summary.md`
- `temp/neural_quant_probe_20260729/q2_summary.md`
- `temp/neural_quant_probe_20260729/q3_summary.md`
