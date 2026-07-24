// game_api.h — GoldRush 2.0 选手接口 (C++)
// 实现 moveDecision, 编译成 .so 提交。所有字段 4 字节对齐。
#pragma once

constexpr int GRID_SIZE = 17;   // 地图 17x17
constexpr int MAX_NPCS  = 7;    // 最多可见 NPC 数
constexpr int S         = 6;    // 每回合总步数
constexpr int REGION_COUNT = 5; // 快照区域数

// 坐标(行, 列)
struct Position {
    int row;
    int col;
};

// 可见 NPC
struct NpcInfo {
    int      id;   // NPC 编号, 跨回合不变; 空槽为 0
    Position pos;  // 位置(row,col); 不可见时 (-1,-1)
};

// 快照中单个区域的统计
struct RegionStat {
    int id;              // 区域编号 1-5
    int enter;           // 窗口内进入该区域的角色次数
    int leave;           // 窗口内离开该区域的角色次数
    int gold_generated;  // 窗口内该区域生成的金币总量
    int gold_collected;  // 窗口内该区域被拾取的金币总量
    int gold_remaining;  // 该区域地面当前剩余金币
    int occupants;       // 该区域当前角色数
};

// 每若干轮提供一次的全局区域快照
struct Snapshot {
    int        window_begin;         // 统计窗口起始轮; 无快照时为 -1
    int        window_end;           // 统计窗口结束轮
    RegionStat regions[REGION_COUNT];
};

// 决策输入
struct GameInput {
    int      round;                  // 当前回合(从 0 开始)
    int      grid[GRID_SIZE][GRID_SIZE];
                                     // 带迷雾纯地形: -5雾 -3炸弹 -1障碍 0空地 >=1金币
                                     // (角色不在网格中标记, 见下方 my_units / visible_enemies / visible_npcs)
    Position my_units[2];            // 我方角色0、角色1 的位置(row,col)
    int      my_units_gold[2];       // 我方角色0、角色1 各自持有金币
    int      gold_opp;               // 对手两角色金币总和
    Position visible_enemies[2];     // 视野内敌方角色位置, 从下标0紧凑排列; 空槽(-1,-1); 不告知是敌方哪个角色
    int      num_visible_npcs;       // visible_npcs 中有效条数
    NpcInfo  visible_npcs[MAX_NPCS]; // 可见 NPC 列表, 尾部用 id=0 / (-1,-1) 填充
    int      snapshot_valid;         // 1=本轮有新快照; 0=无
    Snapshot snapshot;               // 全局区域快照
};

// 决策输出
struct GameOutput {
    int actions[S];   // 6 步动作, 每个 ∈ [0,4]: 0=上 1=下 2=左 3=右 4=不动
    int k;            // 分割点 [0,6]: 角色0走 actions[0..k-1], 角色1走 actions[k..5]
    int order;        // 0=角色0先执行, 1=角色1先执行
    int vp;           // 视野购买: 0=不买 1=买7x7 2=买9x9 (费用对局结束后结算)
};

extern "C" GameOutput moveDecision(const GameInput* input);