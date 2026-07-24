"""player.py — GoldRush 2.0 Python 选手示例(带输入打印, 便于调试)
策略: 角色0走4步、角色1走2步(k=4), 每步随机; 有快照的回合买7x7视野。
额外把引擎传入的 game_input 所有字段打印出来, 只打印前 DEBUG_ROUNDS 个回合,
避免500回合刷爆日志; 正式提交把 DEBUG_ROUNDS 设0关闭。
无需import任何库: 属性访问 + 返回长度9的list [a0..a5, k, order, vp]。
动作: 0=上 1=下 2=左 3=右 4=不动
"""
import random

DEBUG_ROUNDS = 3   # 打印前几个回合的输入; 设 0 关闭


def _print_input(gi):
    print("=" * 60)
    print(f"round            = {gi.round}")
    print(f"my_units         = {list(gi.my_units)}        # [(row,col) x2]")
    print(f"my_units_gold    = {tuple(gi.my_units_gold)}")
    print(f"gold_opp         = {gi.gold_opp}")
    print(f"visible_enemies  = {list(gi.visible_enemies)}  # 紧凑, 空位(-1,-1)")
    print(f"num_visible_npcs = {getattr(gi, 'num_visible_npcs', len(gi.visible_npcs))}")
    for n in gi.visible_npcs:
        print(f"    npc id={n.id} pos=({n.row},{n.col})")
    print(f"snapshot_valid   = {gi.snapshot_valid}")
    if gi.snapshot is not None:
        s = gi.snapshot
        print(f"snapshot window  = [{s.window_begin}, {s.window_end}]")
        for r in s.regions:
            print(f"    region {r.id}: enter={r.enter} leave={r.leave} "
                  f"gen={r.gold_generated} collected={r.gold_collected} "
                  f"remaining={r.gold_remaining} occupants={r.occupants}")
    else:
        print("snapshot         = None (本轮无快照)")
    print("grid (行=row, 列=col; -5雾 -3炸弹 -1障碍 0空 >=1金币):")
    for row in gi.grid:
        print("    " + " ".join(f"{v:>3}" for v in row))
    print("=" * 60)


class Player:
    def MoveDecision(self, game_input):
        if DEBUG_ROUNDS and game_input.round < DEBUG_ROUNDS:
            _print_input(game_input)

        actions = [random.randint(0, 4) for _ in range(6)]
        k = 4          # 角色0走 actions[0:4](4步), 角色1走 actions[4:6](2步)
        order = 0      # 角色0先执行
        vp = 1 if game_input.snapshot_valid else 0   # 有快照买7x7

        return actions + [k, order, vp]