// fast_probing/v1/player.cpp
//
// Fast baseline strategy:
// - two depth-limited BFS passes, one per unit, max depth S=6;
// - pick the largest visible gold pile reachable this turn;
// - send the nearest unit to it and bounce out/back when possible;
// - fallback: pick one unit and move toward center through known safe cells.
#include <cstdint>

#include "../../official_sdk/code/game_api.h"

namespace {

constexpr int INF = 1000000;
constexpr int DR[4] = {-1, 1, 0, 0};
constexpr int DC[4] = {0, 0, -1, 1};
constexpr int OPPOSITE[4] = {1, 0, 3, 2};
constexpr Position CENTER{8, 8};

struct BfsResult {
    int8_t dist[GRID_SIZE][GRID_SIZE];
    uint8_t prev_action[GRID_SIZE][GRID_SIZE];
};

struct TurnContext {
    bool safe[2][GRID_SIZE][GRID_SIZE];
};

uint32_t g_rng = 0x9e3779b9u;

bool inBounds(int r, int c) {
    return 0 <= r && r < GRID_SIZE && 0 <= c && c < GRID_SIZE;
}

int absInt(int x) {
    return x < 0 ? -x : x;
}

int manhattan(Position a, Position b) {
    return absInt(a.row - b.row) + absInt(a.col - b.col);
}

bool validPos(Position p) {
    return inBounds(p.row, p.col);
}

uint32_t fastRand(const GameInput* input) {
    uint32_t mix = static_cast<uint32_t>(input->round + 1) * 0x85ebca6bu;
    mix ^= static_cast<uint32_t>((input->my_units[0].row + 1) * 31 + input->my_units[0].col);
    mix ^= static_cast<uint32_t>((input->my_units[1].row + 1) * 131 + input->my_units[1].col) << 1;
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng ^ mix;
}

void buildTurnContext(const GameInput* input, TurnContext* ctx) {
    bool enemy[GRID_SIZE][GRID_SIZE] = {};
    uint8_t npc_count[GRID_SIZE][GRID_SIZE] = {};

    for (int i = 0; i < 2; ++i) {
        const Position p = input->visible_enemies[i];
        if (validPos(p)) {
            enemy[p.row][p.col] = true;
        }
    }

    int n = input->num_visible_npcs;
    if (n < 0) {
        n = 0;
    }
    if (n > MAX_NPCS) {
        n = MAX_NPCS;
    }
    for (int i = 0; i < n; ++i) {
        const Position p = input->visible_npcs[i].pos;
        if (validPos(p)) {
            ++npc_count[p.row][p.col];
        }
    }

    for (int role = 0; role < 2; ++role) {
        const Position other = input->my_units[1 - role];
        for (int r = 0; r < GRID_SIZE; ++r) {
            for (int c = 0; c < GRID_SIZE; ++c) {
                const int cell = input->grid[r][c];
                bool ok = (cell != -5 && cell != -3 && cell != -1);
                ok = ok && !enemy[r][c] && npc_count[r][c] < 3;
                ok = ok && !(other.row == r && other.col == c);
                ctx->safe[role][r][c] = ok;
            }
        }
    }
}

bool knownSafeCell(const TurnContext& ctx, int role, int r, int c) {
    if (!inBounds(r, c)) {
        return false;
    }
    return ctx.safe[role][r][c];
}

void initBfs(BfsResult* bfs) {
    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            bfs->dist[r][c] = -1;
        }
    }
}

void runBfs(const GameInput* input, const TurnContext& ctx, int role, BfsResult* bfs) {
    initBfs(bfs);

    int qr[GRID_SIZE * GRID_SIZE];
    int qc[GRID_SIZE * GRID_SIZE];
    int head = 0;
    int tail = 0;

    const Position start = input->my_units[role];
    if (!validPos(start)) {
        return;
    }

    bfs->dist[start.row][start.col] = 0;
    qr[tail] = start.row;
    qc[tail] = start.col;
    ++tail;

    while (head < tail) {
        const int r = qr[head];
        const int c = qc[head];
        ++head;

        const int d = bfs->dist[r][c];
        if (d >= S) {
            continue;
        }

        for (int a = 0; a < 4; ++a) {
            const int nr = r + DR[a];
            const int nc = c + DC[a];
            if (!knownSafeCell(ctx, role, nr, nc)) {
                continue;
            }
            if (bfs->dist[nr][nc] != -1) {
                continue;
            }
            bfs->dist[nr][nc] = d + 1;
            bfs->prev_action[nr][nc] = a;
            qr[tail] = nr;
            qc[tail] = nc;
            ++tail;
        }
    }
}

bool findBounce(const TurnContext& ctx, int role, Position target, int* out_action) {
    for (int a = 0; a < 4; ++a) {
        const int nr = target.row + DR[a];
        const int nc = target.col + DC[a];
        if (knownSafeCell(ctx, role, nr, nc)) {
            *out_action = a;
            return true;
        }
    }
    return false;
}

int effectiveReachCost(const TurnContext& ctx, int role, const BfsResult& bfs, Position target) {
    const int d = bfs.dist[target.row][target.col];
    if (d < 0 || d > S) {
        return INF;
    }
    if (d == 0) {
        if (!knownSafeCell(ctx, role, target.row, target.col)) {
            return INF;
        }
        int bounce = 4;
        return findBounce(ctx, role, target, &bounce) ? 2 : INF;
    }
    return d;
}

int reconstructPath(const BfsResult& bfs, Position target, int actions[S]) {
    int rev[S];
    int len = 0;
    int r = target.row;
    int c = target.col;

    while (inBounds(r, c) && bfs.dist[r][c] > 0 && len < S) {
        const int action = bfs.prev_action[r][c];
        rev[len] = action;
        ++len;
        r -= DR[action];
        c -= DC[action];
    }

    for (int i = 0; i < len; ++i) {
        actions[i] = rev[len - 1 - i];
    }
    return len;
}

void fillStay(int actions[S], int from) {
    for (int i = from; i < S; ++i) {
        actions[i] = 4;
    }
}

void buildGoldActions(const TurnContext& ctx, int role, const BfsResult& bfs,
                      Position target, int actions[S]) {
    int len = reconstructPath(bfs, target, actions);

    int bounce = 4;
    if (findBounce(ctx, role, target, &bounce)) {
        while (len + 1 < S) {
            actions[len] = bounce;
            actions[len + 1] = OPPOSITE[bounce];
            len += 2;
        }
    }

    fillStay(actions, len);
}

int chooseSafeFallbackAction(const TurnContext& ctx, int role, Position pos, uint32_t rnd) {
    int best_action = 4;
    int best_dist = manhattan(pos, CENTER);

    for (int pass = 0; pass < 2; ++pass) {
        for (int i = 0; i < 4; ++i) {
            const int a = static_cast<int>((rnd + static_cast<uint32_t>(i)) & 3u);
            const int nr = pos.row + DR[a];
            const int nc = pos.col + DC[a];
            if (!knownSafeCell(ctx, role, nr, nc)) {
                continue;
            }
            const int d = absInt(nr - CENTER.row) + absInt(nc - CENTER.col);
            if (pass == 0) {
                if (d < best_dist) {
                    best_dist = d;
                    best_action = a;
                }
            } else if (best_action == 4) {
                best_action = a;
            }
        }
        if (best_action != 4) {
            break;
        }
    }

    return best_action;
}

void buildFallbackActions(const GameInput* input, const TurnContext& ctx, int role, int actions[S]) {
    Position pos = input->my_units[role];
    uint32_t rnd = fastRand(input);

    for (int step = 0; step < S; ++step) {
        const int action = chooseSafeFallbackAction(ctx, role, pos, rnd + static_cast<uint32_t>(step));
        actions[step] = action;
        if (action != 4) {
            pos.row += DR[action];
            pos.col += DC[action];
        }
    }
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

    BfsResult bfs[2];
    TurnContext ctx;
    buildTurnContext(input, &ctx);
    runBfs(input, ctx, 0, &bfs[0]);
    runBfs(input, ctx, 1, &bfs[1]);

    Position best_target{-1, -1};
    int best_role = -1;
    int best_gold = -1;
    int best_cost = INF;

    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            const int gold = input->grid[r][c];
            if (gold <= 0) {
                continue;
            }

            const Position target{r, c};
            const int cost0 = effectiveReachCost(ctx, 0, bfs[0], target);
            const int cost1 = effectiveReachCost(ctx, 1, bfs[1], target);
            int role = 0;
            int cost = cost0;
            if (cost1 < cost0) {
                role = 1;
                cost = cost1;
            }
            if (cost >= INF) {
                continue;
            }

            if (gold > best_gold || (gold == best_gold && cost < best_cost)) {
                best_gold = gold;
                best_cost = cost;
                best_role = role;
                best_target = target;
            }
        }
    }

    if (best_role >= 0) {
        buildGoldActions(ctx, best_role, bfs[best_role], best_target, out.actions);
    } else {
        best_role = static_cast<int>(fastRand(input) & 1u);
        buildFallbackActions(input, ctx, best_role, out.actions);
    }

    if (best_role == 0) {
        out.k = 6;
        out.order = 0;
    } else {
        out.k = 0;
        out.order = 1;
    }
    out.vp = 0;

    return out;
}
