#include <cstdio>
#include <dlfcn.h>

#include "../../official_sdk/code/game_api.h"

using MoveDecisionFn = GameOutput (*)(const GameInput*);

int main() {
    void* handle = dlopen("./player.so", RTLD_NOW);
    if (handle == nullptr) {
        std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    auto move_decision = reinterpret_cast<MoveDecisionFn>(dlsym(handle, "moveDecision"));
    if (move_decision == nullptr) {
        std::fprintf(stderr, "dlsym failed: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }

    GameInput input = {};
    input.round = 0;
    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            input.grid[r][c] = 0;
        }
    }
    input.my_units[0] = Position{1, 1};
    input.my_units[1] = Position{15, 15};
    input.visible_enemies[0] = Position{-1, -1};
    input.visible_enemies[1] = Position{-1, -1};
    input.num_visible_npcs = 0;
    input.snapshot_valid = 0;

    input.grid[1][3] = 10;
    input.grid[2][1] = -3;
    input.grid[1][2] = 5;

    GameOutput out = move_decision(&input);

    for (int i = 0; i < S; ++i) {
        if (out.actions[i] < 0 || out.actions[i] > 4) {
            std::fprintf(stderr, "invalid action[%d]=%d\n", i, out.actions[i]);
            dlclose(handle);
            return 1;
        }
    }
    if (out.k < 0 || out.k > S) {
        std::fprintf(stderr, "invalid k=%d\n", out.k);
        dlclose(handle);
        return 1;
    }
    if (out.order != 0 && out.order != 1) {
        std::fprintf(stderr, "invalid order=%d\n", out.order);
        dlclose(handle);
        return 1;
    }
    if (out.vp < 0 || out.vp > 2) {
        std::fprintf(stderr, "invalid vp=%d\n", out.vp);
        dlclose(handle);
        return 1;
    }

    std::printf("smoke ok: actions=[%d,%d,%d,%d,%d,%d] k=%d order=%d vp=%d\n",
                out.actions[0], out.actions[1], out.actions[2],
                out.actions[3], out.actions[4], out.actions[5],
                out.k, out.order, out.vp);

    dlclose(handle);
    return 0;
}
