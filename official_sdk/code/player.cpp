// player.cpp — GoldRush 2.0 C++ 选手示例
//
// 策略: 角色0走4步、角色1走2步(k=4), 每步随机(0..4);
//       本回合有快照(snapshot_valid==1)时购买 7x7 视野, 否则不买。
//
// 编译: make  ->  生成 player.so
#include <random>
#include "game_api.h"

static std::mt19937& rng() {
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

extern "C" GameOutput moveDecision(const GameInput* input) {
    GameOutput out = {};

    // 6 步随机动作, 每个 ∈ [0,4]: 0=上 1=下 2=左 3=右 4=不动
    std::uniform_int_distribution<int> act(0, 4);
    for (int i = 0; i < S; ++i) {
        out.actions[i] = act(rng());
    }

    // k=4: 角色0走 actions[0..3](4步), 角色1走 actions[4..5](2步)
    out.k = 4;
    out.order = 0;   // 角色0先执行

    // 有快照的回合买 7x7 视野, 其余回合不买
    out.vp = (input->snapshot_valid == 1) ? 1 : 0;

    return out;
}