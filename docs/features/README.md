# Feature 文档

本目录区分 actor 可提交 feature 与训练期 privileged critic feature。两者服务不同路径，不能因为 shape 相同就混用语义。

## 文档入口

- `actor_feature_v1.md`：第一版可提交 actor feature，schema 为 `goldrush2_feature_v1`，输入来源是官方 `GameInput`。该路径可以进入 ONNX/C++ 提交模型。
- `actor_feature_v2.md`：v1 的小增量版本，schema 为 `goldrush2_feature_v2`，新增 5 个 spatial planes：snapshot generated、中心金币先验、炸弹 belief、static2 mask 和 static2 距离图。
- `privileged_critic_feature_v1.md`：当前训练期 critic feature，schema 为 `goldrush2_privileged_critic_feature_v1`，输入来源是 simulator full state。该路径禁止进入提交模型。

## 边界

actor feature 只允许使用官方 observation 和由历史 observation 推断出的 belief。它的 fog、visible mask、temporal stack 和 obstacle memory 语义必须与提交环境一致。

privileged critic feature 允许使用训练时 simulator full state，目标是降低 value prediction 难度。它不承诺可提交，不应放在 `policy_runtime/` 的部署路径中。

后续若修改 actor feature shape、顺序或语义，必须使用新的 actor schema 名称。后续若修改 critic feature shape、顺序或语义，也必须使用新的 critic schema 名称。
