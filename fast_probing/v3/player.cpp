// fast_probing/v3/player.cpp
//
// Greedy ultra-light strategy with vision probing:
// - no BFS and no full-board safety precomputation;
// - choose the largest visible gold pile with Manhattan distance <= S;
// - move one unit greedily toward it, only doing profitable bounce pairs;
// - give leftover steps to the other unit to move toward the center;
// - buy 9x9 vision after two rounds without reaching a >1 gold target.
#include <cstdint>

#include "../../official_sdk/code/game_api.h"

namespace {

constexpr int DR[4] = {-1, 1, 0, 0};
constexpr int DC[4] = {0, 0, -1, 1};
constexpr int OPPOSITE[4] = {1, 0, 3, 2};
constexpr Position CENTER{8, 8};
constexpr Position NO_BLOCK{-1, -1};

uint32_t g_rng = 0x9e3779b9u;
int g_bad_visible_rounds = 0;

struct ActionPlan {
    int len;
    bool reached;
    Position end;
};

inline bool inBounds(int r, int c) {
    return 0 <= r && r < GRID_SIZE && 0 <= c && c < GRID_SIZE;
}

inline int absInt(int x) {
    return x < 0 ? -x : x;
}

inline int md(int r, int c, Position p) {
    return absInt(r - p.row) + absInt(c - p.col);
}

inline bool validPos(Position p) {
    return inBounds(p.row, p.col);
}

inline uint32_t fastRand(const GameInput* input) {
    uint32_t mix = static_cast<uint32_t>(input->round + 1) * 0x85ebca6bu;
    mix ^= static_cast<uint32_t>((input->my_units[0].row + 1) * 31 + input->my_units[0].col);
    mix ^= static_cast<uint32_t>((input->my_units[1].row + 1) * 131 + input->my_units[1].col) << 1;
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng ^ mix;
}

inline bool visibleEnemyAt(const GameInput* input, int r, int c) {
    const Position e0 = input->visible_enemies[0];
    if (validPos(e0) && e0.row == r && e0.col == c) {
        return true;
    }
    const Position e1 = input->visible_enemies[1];
    return validPos(e1) && e1.row == r && e1.col == c;
}

inline bool crowdedNpcAt(const GameInput* input, int r, int c) {
    int n = input->num_visible_npcs;
    if (n < 0) {
        n = 0;
    } else if (n > MAX_NPCS) {
        n = MAX_NPCS;
    }

    int count = 0;
    for (int i = 0; i < n; ++i) {
        const Position p = input->visible_npcs[i].pos;
        if (validPos(p) && p.row == r && p.col == c) {
            ++count;
            if (count >= 3) {
                return true;
            }
        }
    }
    return false;
}

inline bool safeCell(const GameInput* input, int role, int r, int c, Position extra_block = NO_BLOCK) {
    if (!inBounds(r, c)) {
        return false;
    }
    const int cell = input->grid[r][c];
    if (cell == -5 || cell == -3 || cell == -1) {
        return false;
    }
    const Position other = input->my_units[1 - role];
    if (other.row == r && other.col == c) {
        return false;
    }
    if (extra_block.row == r && extra_block.col == c) {
        return false;
    }
    if (visibleEnemyAt(input, r, c)) {
        return false;
    }
    return !crowdedNpcAt(input, r, c);
}

inline void fillStay(int actions[S], int from) {
    for (int i = from; i < S; ++i) {
        actions[i] = 4;
    }
}

bool findBounce(const GameInput* input, int role, Position target, int* out_action) {
    for (int a = 0; a < 4; ++a) {
        const int nr = target.row + DR[a];
        const int nc = target.col + DC[a];
        if (safeCell(input, role, nr, nc)) {
            *out_action = a;
            return true;
        }
    }
    return false;
}

int usefulEntriesForGold(int gold) {
    if (gold <= 0) {
        return 0;
    }
    if (gold <= 2) {
        return 1;
    }
    if (gold <= 8) {
        return 2;
    }
    return 3;
}

int usefulBouncePairs(int gold, bool already_entered_target) {
    int entries = usefulEntriesForGold(gold);
    if (already_entered_target && entries > 0) {
        --entries;
    }
    return entries;
}

int chooseStepToward(const GameInput* input, int role, Position pos, Position target,
                     uint32_t rnd, Position extra_block = NO_BLOCK) {
    const int cur = md(pos.row, pos.col, target);
    int best_action = 4;
    int best_dist = cur;

    for (int i = 0; i < 4; ++i) {
        const int a = static_cast<int>((rnd + static_cast<uint32_t>(i)) & 3u);
        const int nr = pos.row + DR[a];
        const int nc = pos.col + DC[a];
        if (!safeCell(input, role, nr, nc, extra_block)) {
            continue;
        }
        const int d = md(nr, nc, target);
        if (d < best_dist) {
            best_dist = d;
            best_action = a;
        }
    }

    if (best_action != 4) {
        return best_action;
    }

    int fallback = 4;
    int fallback_dist = 1000000;
    for (int i = 0; i < 4; ++i) {
        const int a = static_cast<int>((rnd + static_cast<uint32_t>(i)) & 3u);
        const int nr = pos.row + DR[a];
        const int nc = pos.col + DC[a];
        if (!safeCell(input, role, nr, nc, extra_block)) {
            continue;
        }
        const int d = md(nr, nc, target);
        if (d < fallback_dist) {
            fallback_dist = d;
            fallback = a;
        }
    }
    return fallback;
}

ActionPlan buildMainActions(const GameInput* input, int role, Position target,
                            bool enable_bounce, int target_gold, int actions[S]) {
    Position pos = input->my_units[role];
    int len = 0;
    uint32_t rnd = fastRand(input);
    bool reached = false;
    bool entered_by_move = false;

    while (len < S) {
        if (pos.row == target.row && pos.col == target.col) {
            if (!enable_bounce) {
                break;
            }
            const int max_bounces = usefulBouncePairs(target_gold, entered_by_move);
            if (max_bounces <= 0) {
                break;
            }
            int bounce = 4;
            if (!findBounce(input, role, target, &bounce)) {
                break;
            }
            reached = true;
            int used_bounces = 0;
            while (len + 1 < S && used_bounces < max_bounces) {
                actions[len] = bounce;
                actions[len + 1] = OPPOSITE[bounce];
                len += 2;
                ++used_bounces;
            }
            break;
        }

        const int action = chooseStepToward(input, role, pos, target, rnd + static_cast<uint32_t>(len));
        if (action == 4) {
            break;
        }
        actions[len] = action;
        ++len;
        pos.row += DR[action];
        pos.col += DC[action];
        if (pos.row == target.row && pos.col == target.col) {
            reached = true;
            entered_by_move = true;
        }
    }

    return ActionPlan{len, reached, pos};
}

int buildCenterActions(const GameInput* input, int role, int max_steps,
                       Position extra_block, int actions[S]) {
    Position pos = input->my_units[role];
    uint32_t rnd = fastRand(input);
    int len = 0;
    while (len < max_steps) {
        const int action = chooseStepToward(input, role, pos, CENTER,
                                            rnd + static_cast<uint32_t>(len), extra_block);
        if (action == 4) {
            break;
        }
        actions[len] = action;
        ++len;
        pos.row += DR[action];
        pos.col += DC[action];
    }
    return len;
}

void composeOutput(int main_role, const int main_actions[S], int main_len,
                   const int side_actions[S], int side_len, GameOutput* out) {
    fillStay(out->actions, 0);
    if (main_role == 0) {
        for (int i = 0; i < main_len && i < S; ++i) {
            out->actions[i] = main_actions[i];
        }
        for (int i = 0; i < side_len && main_len + i < S; ++i) {
            out->actions[main_len + i] = side_actions[i];
        }
        out->k = main_len;
        out->order = 0;
    } else {
        for (int i = 0; i < side_len && i < S; ++i) {
            out->actions[i] = side_actions[i];
        }
        for (int i = 0; i < main_len && side_len + i < S; ++i) {
            out->actions[side_len + i] = main_actions[i];
        }
        out->k = side_len;
        out->order = 1;
    }
}

bool chooseGoldTarget(const GameInput* input, int* out_role, Position* out_target, int* out_gold) {
    int best_gold = -1;
    int best_dist = 1000000;
    int best_role = -1;
    Position best_target{-1, -1};

    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            const int gold = input->grid[r][c];
            if (gold <= 0) {
                continue;
            }

            const int d0 = md(input->my_units[0].row, input->my_units[0].col, Position{r, c});
            const int d1 = md(input->my_units[1].row, input->my_units[1].col, Position{r, c});
            int role = 0;
            int d = d0;
            if (d1 < d0) {
                role = 1;
                d = d1;
            }
            if (d > S) {
                continue;
            }

            if (gold > best_gold || (gold == best_gold && d < best_dist)) {
                best_gold = gold;
                best_dist = d;
                best_role = role;
                best_target = Position{r, c};
            }
        }
    }

    if (best_role < 0) {
        return false;
    }
    *out_role = best_role;
    *out_target = best_target;
    *out_gold = best_gold;
    return true;
}

}  // namespace

extern "C" GameOutput moveDecision(const GameInput* input) {
    GameOutput out = {};
    fillStay(out.actions, 0);
    out.k = 6;
    out.order = 0;
    out.vp = 0;

    if (input == nullptr) {
        return out;
    }
    if (input->round == 0) {
        g_bad_visible_rounds = 0;
    }

    int role = -1;
    Position target{-1, -1};
    int target_gold = 0;
    bool enable_bounce = true;
    bool has_gold_target = chooseGoldTarget(input, &role, &target, &target_gold);
    if (!has_gold_target) {
        role = static_cast<int>(fastRand(input) & 1u);
        target = CENTER;
        enable_bounce = false;
    }

    int main_actions[S] = {4, 4, 4, 4, 4, 4};
    int side_actions[S] = {4, 4, 4, 4, 4, 4};
    const ActionPlan main_plan = buildMainActions(input, role, target, enable_bounce,
                                                  target_gold, main_actions);
    const int side_role = 1 - role;
    const int spare_steps = S - main_plan.len;
    const int side_len = spare_steps > 0
                             ? buildCenterActions(input, side_role, spare_steps,
                                                  main_plan.end, side_actions)
                             : 0;
    composeOutput(role, main_actions, main_plan.len, side_actions, side_len, &out);

    const bool good_gold_seen = has_gold_target && target_gold > 1 && main_plan.reached;
    if (good_gold_seen) {
        g_bad_visible_rounds = 0;
    } else {
        ++g_bad_visible_rounds;
    }

    if (g_bad_visible_rounds >= 2) {
        out.vp = 2;
        g_bad_visible_rounds = 0;
    } else {
        out.vp = 0;
    }

    return out;
}
