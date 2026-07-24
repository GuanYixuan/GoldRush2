"""game_api.py — GoldRush 2.0 输入/输出类型定义(仅供本地调试参考)
比赛引擎会传给 MoveDecision 一个字段与本文件 GameInput 完全一致的对象;
无需在提交代码里 import 本文件, 实战直接对 game_input 属性访问即可。
用途: 1)字段速查; 2)本地构造 GameInput 测试你的 Player。
动作编码: 0=上 1=下 2=左 3=右 4=不动
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

GRID_SIZE = 17
MAX_NPCS = 7
S = 6
REGION_COUNT = 5

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_STAY = 4


@dataclass
class RegionStat:
    id: int = 0
    enter: int = 0
    leave: int = 0
    gold_generated: int = 0
    gold_collected: int = 0
    gold_remaining: int = 0
    occupants: int = 0


@dataclass
class Snapshot:
    window_begin: int = 0
    window_end: int = 0
    regions: List[RegionStat] = field(
        default_factory=lambda: [RegionStat() for _ in range(REGION_COUNT)])


@dataclass
class NpcInfo:
    id: int = 0
    row: int = -1
    col: int = -1


@dataclass
class GameInput:
    round: int                              # 当前回合(从0开始)
    grid: List[List[int]]                   # 17x17 带迷雾纯地形, grid[row][col]
    my_units: List[Tuple[int, int]]         # 我方两角色位置 [(row,col),(row,col)]
    my_units_gold: Tuple[int, int]          # 我方角色0、角色1 各自持有金币
    gold_opp: int                           # 对手两角色金币总和
    visible_enemies: List[Tuple[int, int]]  # 视野内敌方角色位置, 紧凑排列, 空位(-1,-1)
    visible_npcs: List[NpcInfo]             # 可见NPC列表(长度<=MAX_NPCS)
    snapshot_valid: bool                    # 本轮是否有新快照
    snapshot: Optional[Snapshot]            # 快照对象; 无快照时为 None


@dataclass
class GameOutput:
    actions: List[int]   # 长度6, 每个∈[0,4]: 0=上 1=下 2=左 3=右 4=不动
    k: int               # 分割点[0,6]: 角色0走 actions[0:k], 角色1走 actions[k:6]
    order: int           # 0=角色0先执行, 1=角色1先执行
    vp: int              # 0=不买 1=买7x7 2=买9x9

# grid 取值: -5雾 -3炸弹 -1障碍 0空地 >=1金币
# 我方位置看 my_units, 敌方看 visible_enemies, NPC 看 visible_npcs