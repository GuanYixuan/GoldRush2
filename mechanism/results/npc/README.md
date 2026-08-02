# NPC 机制分析文档导航

本文只做导航，不冻结 NPC 默认实现细节。

当前 NPC 推荐实现的唯一权威入口是：

- `npc_behavior_modeling_overview.md`

该文档冻结 simulator/RL 当前应采用的 NPC plan 口径、默认模型、特征定义、权重和 domain randomization。后续若调整默认 NPC 策略，应优先更新该文档，再按需同步代码与实验记录。

其它文档定位：

- `npc_path_policy_fit.md`：path-level softmax 拟合实验结果和指标表，不作为默认实现 spec。
- `npc_behavior_feature_observation.md`：NPC 基础动作、调度、拾金和局部金币响应观察。
- `npc_feature_tables_and_bomb_interaction.md`：特征表导出和 NPC 与炸弹交互。
- `npc_id_independence_analysis.md`：不同 `npc_id` 是否需要独立策略参数。
- `npc_step_path_and_gold_target_analysis.md`：step 独立性、path 结构和金币目标函数。
- `npc_repeated_pickup_analysis.md`：动态 `65%` 拾取规则与反复进入金币格。
- `npc_step_count_and_zero_move_analysis.md`：走步数分布、0 步行为和炸弹封路例外。
- `npc_center_bias_analysis.md`：中心吸引项隔离实验。

历史 rollout 对齐实验位于 `mechanism/results/simulator_alignment/`。这些文档记录当时 simulator default 下的偏差诊断，不应直接理解为当前 M4e 默认实现的评估结果。
