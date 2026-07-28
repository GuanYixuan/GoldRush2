# 机理建模

本目录是 GoldRush2.0 机理建模工作区，重点服务于金币/炸弹生成的主动数据采集与离线分析。

仓库内规则文档仍是事实来源，具体路径参见`AGENTS.md`

## 目录结构

```text
mechanism/
  scripts/
    observer/   # 观测账号专用平台脚本与代理检查
    probes/     # 用于受控数据采集的 probe 策略
    analysis/   # 离线 replay 解析、反推与统计汇总
  data/
    raw/        # 下载的 replay/log 原始数据，尽量保持平台返回原样
    processed/  # 派生表、特征与可供建模使用的输出
```

## 边界约定

- `mechanism/scripts/observer/` 只服务于观测账号。该目录下脚本只能读取自己的`.env`中的`OBSERVER_KEY`。不得读取主号 `LOGIN_KEY`，也不应与主号工具共用 session/cache 文件。
- `mechanism/scripts/probes/` 放实验策略。任何平台提交或发起对局操作都必须是显式且已授权的。
- `mechanism/scripts/analysis/` 默认应为离线脚本；除非脚本文档明确说明，否则不应访问评测平台。
- `mechanism/data/raw/` 存放不可变的输入资产（主要是官方log）。清洗后或反推得到的输出优先写入 `mechanism/data/processed/`。

大量下载数据和生成产物默认被 git 忽略。
