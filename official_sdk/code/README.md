# GoldRush 2.0 选手示例代码

C++和Python的示例策略, 逻辑相同:
- 角色0走 4 步、角色1走 2 步(k=4), 每一步动作随机(0..4);
- 本回合有快照(snapshot_valid)时购买 7x7 视野(vp=1), 否则不买。

动作编码: 0=上 1=下 2=左 3=右 4=不动

## C++ (cpp/)
- 入口 extern "C" GameOutput moveDecision(const GameInput* input)
- Makefile 用 -O2 -march=native -fPIC -shared
- 提交: make编译出的so文件

## Python (python/)
- player.py 内含 class Player, 方法 MoveDecision(self, game_input)
- 提交: 直接提交 player.py

## 合法性(否则判负)
- actions 每个 ∈ [0,4], k ∈ [0,6], order ∈ {0,1}, vp ∈ {0,1,2}
- 每轮必须正确返回, 格式非法或运行时错误直接判负