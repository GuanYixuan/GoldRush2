# Fast Probing 快速策略

`fast_probing/` 存放一系列 C++ 快速专家策略实验。它们的目标是用极低 `moveDecision` 耗时争取先手，并提供可解释的非神经策略基线。

这些策略不是当前神经网络主线，但已经被训练侧复用为先验来源：`training/opponents` 中的 `fast_probe_v3_like` 复刻了 `v3` 的轻量行为骨架，用于 opponent league 和 BC teacher。

## 版本定位

- `v1/`：BFS 半径 6 的保守抢金币策略，强调规则安全和可达性。
- `v2/`：移除 BFS，改用曼哈顿距离和逐步贪心，换取更低延迟。
- `v3/`：在 v2 基础上加入视野购买、剩余步数副角色向中心移动和有限 bounce 逻辑，是当前最重要的快速策略参考。

各版本 README 保留了设计细节、平台实测和历史对照；根 README 只做定位和索引。

## 构建

各版本都采用相对 include 引用官方接口 `official_sdk/code/game_api.h`。进入版本目录后使用对应 `Makefile`：

```bash
cd fast_probing/v3
make
```

进行平台提交或发起对局前，必须遵守根目录 `AGENTS.md` 中的平台安全边界。
