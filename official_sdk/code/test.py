from game_api import GameInput, Snapshot, RegionStat, NpcInfo, GRID_SIZE, REGION_COUNT
from player import Player


def make_input(round_idx, with_snapshot):
    grid = [[-5] * GRID_SIZE for _ in range(GRID_SIZE)]
    for r in range(0, 3):
        for c in range(0, 3):
            grid[r][c] = 0
    grid[1][1] = 5      # 金币
    grid[2][2] = -1     # 障碍
    grid[0][2] = -3     # 炸弹

    snapshot = None
    if with_snapshot:
        snapshot = Snapshot(
            window_begin=round_idx - 4, window_end=round_idx,
            regions=[
                RegionStat(id=i + 1, enter=i, leave=0, gold_generated=10 * i,
                           gold_collected=i, gold_remaining=5 * i, occupants=i % 3)
                for i in range(REGION_COUNT)
            ],
        )

    return GameInput(
        round=round_idx, grid=grid,
        my_units=[(0, 0), (16, 16)],
        my_units_gold=(3, 0), gold_opp=7,
        visible_enemies=[(1, 2), (-1, -1)],
        visible_npcs=[NpcInfo(id=-1, row=1, col=0), NpcInfo(id=-3, row=0, col=1)],
        snapshot_valid=with_snapshot, snapshot=snapshot,
    )


def check_output(ret):
    assert isinstance(ret, list) and len(ret) == 9, f"返回必须是长度9的list, 得到 {ret}"
    actions, k, order, vp = ret[0:6], ret[6], ret[7], ret[8]
    assert all(isinstance(a, int) and 0 <= a <= 4 for a in actions), f"actions 非法: {actions}"
    assert isinstance(k, int) and 0 <= k <= 6, f"k 非法: {k}"
    assert order in (0, 1), f"order 非法: {order}"
    assert vp in (0, 1, 2), f"vp 非法: {vp}"
    return actions, k, order, vp


def main():
    p = Player()

    print("\n########## 回合 0 (无快照) ##########")
    ret = p.MoveDecision(make_input(0, with_snapshot=False))
    actions, k, order, vp = check_output(ret)
    print(f"返回: actions={actions} k={k} order={order} vp={vp}")
    assert vp == 0, "无快照回合 vp 应为 0"
    print("OK: 输出合法, 无快照 vp=0")

    print("\n########## 回合 1 (有快照) ##########")
    ret = p.MoveDecision(make_input(1, with_snapshot=True))
    actions, k, order, vp = check_output(ret)
    print(f"返回: actions={actions} k={k} order={order} vp={vp}")
    assert vp == 1, "有快照回合 vp 应为 1 (买7x7)"
    assert k == 4, "示例策略 k 应为 4"
    print("OK: 输出合法, 有快照 vp=1, k=4")

    print("\n全部通过。")


if __name__ == "__main__":
    main()