#include "policy_runtime/fast_option.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace policy_runtime {
namespace {

constexpr int GRID_FOG = -5;
constexpr int ACTION_UP = 0;
constexpr int ACTION_DOWN = 1;
constexpr int ACTION_LEFT = 2;
constexpr int ACTION_RIGHT = 3;
constexpr int ACTION_STAY = 4;
constexpr int SCAN_COUNT = 24;
constexpr int PAD2 = FastRuntimeState::PAD2;
constexpr int PAD2_STRIDE = FastRuntimeState::PAD2_STRIDE;
constexpr int PAD2_COUNT = FastRuntimeState::PAD2_COUNT;

constexpr int kScan5x5Offsets[SCAN_COUNT] = {
    -2 * PAD2_STRIDE - 2, -2 * PAD2_STRIDE - 1, -2 * PAD2_STRIDE, -2 * PAD2_STRIDE + 1, -2 * PAD2_STRIDE + 2,
    -1 * PAD2_STRIDE - 2, -1 * PAD2_STRIDE - 1, -1 * PAD2_STRIDE, -1 * PAD2_STRIDE + 1, -1 * PAD2_STRIDE + 2,
    -2, -1, 1, 2,
    PAD2_STRIDE - 2, PAD2_STRIDE - 1, PAD2_STRIDE, PAD2_STRIDE + 1, PAD2_STRIDE + 2,
    2 * PAD2_STRIDE - 2, 2 * PAD2_STRIDE - 1, 2 * PAD2_STRIDE, 2 * PAD2_STRIDE + 1, 2 * PAD2_STRIDE + 2,
};
constexpr int kScan5x5RowDelta[SCAN_COUNT] = {
    -2, -2, -2, -2, -2,
    -1, -1, -1, -1, -1,
    0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
};
constexpr int kScan5x5ColDelta[SCAN_COUNT] = {
    -2, -1, 0, 1, 2,
    -2, -1, 0, 1, 2,
    -2, -1, 1, 2,
    -2, -1, 0, 1, 2,
    -2, -1, 0, 1, 2,
};

constexpr int pad2_index(int row, int col) {
    return (row + PAD2) * PAD2_STRIDE + (col + PAD2);
}

inline bool in_bounds(Position p) {
    return static_cast<unsigned int>(p.row) < GRID_SIZE && static_cast<unsigned int>(p.col) < GRID_SIZE;
}

inline GameOutput stay_output() {
    return GameOutput{{ACTION_STAY, ACTION_STAY, ACTION_STAY, ACTION_STAY, ACTION_STAY, ACTION_STAY}, 0, 0, 0};
}

inline int opposite_action(int action) {
    if (action == ACTION_UP) {
        return ACTION_DOWN;
    }
    if (action == ACTION_DOWN) {
        return ACTION_UP;
    }
    if (action == ACTION_LEFT) {
        return ACTION_RIGHT;
    }
    if (action == ACTION_RIGHT) {
        return ACTION_LEFT;
    }
    return ACTION_STAY;
}

inline Position moved(Position p, int action) {
    if (action == ACTION_UP) {
        return Position{p.row - 1, p.col};
    }
    if (action == ACTION_DOWN) {
        return Position{p.row + 1, p.col};
    }
    if (action == ACTION_LEFT) {
        return Position{p.row, p.col - 1};
    }
    if (action == ACTION_RIGHT) {
        return Position{p.row, p.col + 1};
    }
    return p;
}

inline bool is_visible_enemy(const GameInput& input, int row, int col) {
    const Position enemy0 = input.visible_enemies[0];
    const Position enemy1 = input.visible_enemies[1];
    return (enemy0.row == row && enemy0.col == col) || (enemy1.row == row && enemy1.col == col);
}

inline int ceil_pickup(int value) {
    return (value * 65 + 99) / 100;
}

inline int clamp_threshold(int threshold) {
    return std::max(FAST_THRESHOLD_LOW, std::min(threshold, FAST_THRESHOLD_HIGH));
}

bool choose_target_padded(
    const std::int8_t* padded_grid,
    const GameInput& input,
    int threshold,
    Position* target,
    int* role) {
    const int u0r = input.my_units[0].row;
    const int u0c = input.my_units[0].col;
    const int u1r = input.my_units[1].row;
    const int u1c = input.my_units[1].col;

    const int u0_flat = pad2_index(u0r, u0c);
    for (int i = 0; i < SCAN_COUNT; ++i) {
        const int gold = static_cast<int>(padded_grid[u0_flat + kScan5x5Offsets[i]]);
        if (gold < threshold) {
            continue;
        }
        const int row = u0r + kScan5x5RowDelta[i];
        const int col = u0c + kScan5x5ColDelta[i];
        *target = Position{row, col};
        *role = 0;
        return true;
    }

    const int u1_flat = pad2_index(u1r, u1c);
    for (int i = 0; i < SCAN_COUNT; ++i) {
        const int gold = static_cast<int>(padded_grid[u1_flat + kScan5x5Offsets[i]]);
        if (gold < threshold) {
            continue;
        }
        const int row = u1r + kScan5x5RowDelta[i];
        const int col = u1c + kScan5x5ColDelta[i];
        *target = Position{row, col};
        *role = 1;
        return true;
    }

    return false;
}

bool append_greedy_path_to_output(const GameInput& input, Position start, Position target, GameOutput* output, int* action_count) {
    int row = start.row;
    int col = start.col;
    const int target_row = target.row;
    const int target_col = target.col;
    const Position enemy0 = input.visible_enemies[0];
    const Position enemy1 = input.visible_enemies[1];
    while ((row != target_row || col != target_col) && *action_count < S) {
        int next_row = row;
        int next_col = col;
        int action = ACTION_STAY;
        bool moved_one_step = false;

        if (target_row < row) {
            next_row = row - 1;
            action = ACTION_UP;
        } else if (target_row > row) {
            next_row = row + 1;
            action = ACTION_DOWN;
        }
        if (action != ACTION_STAY
            && input.grid[next_row][next_col] >= 0
            && !(enemy0.row == next_row && enemy0.col == next_col)
            && !(enemy1.row == next_row && enemy1.col == next_col)) {
            output->actions[*action_count] = action;
            *action_count += 1;
            row = next_row;
            col = next_col;
            moved_one_step = true;
        }

        if (!moved_one_step) {
            next_row = row;
            next_col = col;
            action = ACTION_STAY;
            if (target_col < col) {
                next_col = col - 1;
                action = ACTION_LEFT;
            } else if (target_col > col) {
                next_col = col + 1;
                action = ACTION_RIGHT;
            }
            if (action != ACTION_STAY
                && input.grid[next_row][next_col] >= 0
                && !(enemy0.row == next_row && enemy0.col == next_col)
                && !(enemy1.row == next_row && enemy1.col == next_col)) {
                output->actions[*action_count] = action;
                *action_count += 1;
                row = next_row;
                col = next_col;
                moved_one_step = true;
            }
        }

        if (!moved_one_step) {
            return false;
        }
    }
    return row == target_row && col == target_col;
}

void init_padded_grid(std::int8_t* dst) {
    for (int i = 0; i < PAD2_COUNT; ++i) {
        dst[i] = static_cast<std::int8_t>(GRID_FOG);
    }
}

void pack_padded_grid_interior(std::int8_t* dst, const GameInput& input) {
    for (int row = 0; row < GRID_SIZE; ++row) {
        std::int8_t* out = &dst[pad2_index(row, 0)];
        for (int col = 0; col < GRID_SIZE; ++col) {
            int cell = input.grid[row][col];
            if (cell > 127) {
                cell = 127;
            }
            out[col] = static_cast<std::int8_t>(cell);
        }
    }
}

void pack_padded_grid(std::int8_t* dst, const GameInput& input) {
    init_padded_grid(dst);
    pack_padded_grid_interior(dst, input);
}

void split_output(const GameOutput& output, int role, int* actions, int* count) {
    if (role == 0) {
        *count = output.k;
        for (int i = 0; i < *count; ++i) {
            actions[i] = output.actions[i];
        }
        return;
    }
    *count = S - output.k;
    for (int i = 0; i < *count; ++i) {
        actions[i] = output.actions[output.k + i];
    }
}

int count_nonstay_for_role(const GameOutput& output, int role) {
    int actions[S]{};
    int count = 0;
    split_output(output, role, actions, &count);
    int nonstay = 0;
    for (int i = 0; i < count; ++i) {
        if (actions[i] != ACTION_STAY) {
            ++nonstay;
        }
    }
    return nonstay;
}

}  // namespace

bool try_fast_gold_grab(const GameInput& input, int threshold_int, GameOutput* output, FastTryResult* result) {
    std::int8_t padded_grid[PAD2_COUNT]{};
    pack_padded_grid(padded_grid, input);

    Position target{0, 0};
    int role = 0;
    if (!choose_target_padded(padded_grid, input, clamp_threshold(threshold_int), &target, &role)) {
        if (result != nullptr) {
            result->status = FastStatus::MissNoTarget;
        }
        return false;
    }

    GameOutput fused = stay_output();
    int action_count = 0;
    if (!append_greedy_path_to_output(input, input.my_units[role], target, &fused, &action_count)) {
        if (result != nullptr) {
            result->status = FastStatus::PathFail;
            result->role = role;
            result->target = target;
        }
        return false;
    }
    if (action_count > 0 && action_count + 2 <= S) {
        const int last_action = fused.actions[action_count - 1];
        fused.actions[action_count++] = opposite_action(last_action);
        fused.actions[action_count++] = last_action;
    }
    fused.k = role == 0 ? action_count : 0;
    fused.order = role == 0 ? 0 : 1;
    fused.vp = 0;

    *output = fused;
    if (result != nullptr) {
        result->status = FastStatus::Success;
        result->output = fused;
        result->role = role;
        result->target = target;
        result->action_count = action_count;
    }
    return true;
}

FastStatus try_fast_output_core(
    std::int8_t* padded_grid,
    const GameInput& input,
    int threshold_int,
    GameOutput* output,
#if POLICY_RUNTIME_FAST_DEBUG
    int* role_out,
    Position* target_out,
    int* action_count_out
#else
    int*,
    Position*,
    int*
#endif
) {
    pack_padded_grid_interior(padded_grid, input);

    Position target{0, 0};
    int role = 0;
    if (!choose_target_padded(padded_grid, input, threshold_int, &target, &role)) {
        return FastStatus::MissNoTarget;
    }

    GameOutput fused = stay_output();
    int action_count = 0;
    if (!append_greedy_path_to_output(input, input.my_units[role], target, &fused, &action_count)) {
#if POLICY_RUNTIME_FAST_DEBUG
        if (role_out != nullptr) {
            *role_out = role;
        }
        if (target_out != nullptr) {
            *target_out = target;
        }
#endif
        return FastStatus::PathFail;
    }
    if (action_count > 0 && action_count + 2 <= S) {
        const int last_action = fused.actions[action_count - 1];
        fused.actions[action_count++] = opposite_action(last_action);
        fused.actions[action_count++] = last_action;
    }
    fused.k = role == 0 ? action_count : 0;
    fused.order = role == 0 ? 0 : 1;
    fused.vp = 0;

    *output = fused;
#if POLICY_RUNTIME_FAST_DEBUG
    if (role_out != nullptr) {
        *role_out = role;
    }
    if (target_out != nullptr) {
        *target_out = target;
    }
    if (action_count_out != nullptr) {
        *action_count_out = action_count;
    }
#endif
    return FastStatus::Success;
}

int simulate_known_gold_pickups(const GameInput& input, const GameOutput& output, int role) {
    int grid[GRID_SIZE][GRID_SIZE]{};
    for (int row = 0; row < GRID_SIZE; ++row) {
        for (int col = 0; col < GRID_SIZE; ++col) {
            grid[row][col] = input.grid[row][col] > 0 ? input.grid[row][col] : 0;
        }
    }

    int actions[S]{};
    int action_count = 0;
    split_output(output, role, actions, &action_count);
    Position pos = input.my_units[role];
    int gain = 0;
    for (int i = 0; i < action_count; ++i) {
        const Position next = moved(pos, actions[i]);
        if (!in_bounds(next) || input.grid[next.row][next.col] < 0 || is_visible_enemy(input, next.row, next.col)) {
            continue;
        }
        if (next.row == pos.row && next.col == pos.col) {
            continue;
        }
        pos = next;
        int& available = grid[pos.row][pos.col];
        if (available > 0) {
            const int picked = ceil_pickup(available);
            gain += picked;
            available -= picked;
        }
    }
    return gain;
}

int infer_fast_role(const GameInput& input, const GameOutput& output) {
    const int expected0 = simulate_known_gold_pickups(input, output, 0);
    const int expected1 = simulate_known_gold_pickups(input, output, 1);
    if (expected0 > 0 && expected1 <= 0) {
        return 0;
    }
    if (expected1 > 0 && expected0 <= 0) {
        return 1;
    }
    const int nonstay0 = count_nonstay_for_role(output, 0);
    const int nonstay1 = count_nonstay_for_role(output, 1);
    if (nonstay0 > 0 && nonstay1 == 0) {
        return 0;
    }
    if (nonstay1 > 0 && nonstay0 == 0) {
        return 1;
    }
    throw std::logic_error("cannot uniquely infer fast role from pending output");
}

Position replay_output_final_position(const GameInput& input, const GameOutput& output, int role) {
    int actions[S]{};
    int action_count = 0;
    split_output(output, role, actions, &action_count);
    Position pos = input.my_units[role];
    for (int i = 0; i < action_count; ++i) {
        const Position next = moved(pos, actions[i]);
        if (!in_bounds(next) || input.grid[next.row][next.col] < 0 || is_visible_enemy(input, next.row, next.col)) {
            continue;
        }
        if (next.row == pos.row && next.col == pos.col) {
            continue;
        }
        pos = next;
    }
    return pos;
}

bool npc_at_position(const GameInput& input, Position position) {
    for (int i = 0; i < input.num_visible_npcs; ++i) {
        const NpcInfo& npc = input.visible_npcs[i];
        if (npc.pos.row == position.row && npc.pos.col == position.col) {
            return true;
        }
    }
    return false;
}

FastRuntimeState::FastRuntimeState(int player_id) : extractor_(player_id) {
    reset(player_id);
}

void FastRuntimeState::reset(int player_id) {
    if (player_id != 1 && player_id != 2) {
        throw std::invalid_argument("player_id must be 1 or 2");
    }
    player_id_ = player_id;
    extractor_.reset(player_id);
    clear_pending();
    diagnostics_ = FastDiagnostics{};
    fast_alpha_ = FAST_INITIAL_ALPHA;
    fast_beta_ = FAST_INITIAL_BETA;
    threshold_int_ = 12;
}

std::array<float, FAST_SCALAR_FEATURES> FastRuntimeState::fast_scalars() const {
    const float total = fast_alpha_ + fast_beta_;
    return {
        total > 0.0F ? fast_alpha_ / total : 0.8F,
        std::min(total / FAST_CONFIDENCE_CAP, 1.0F),
    };
}

FastDiagnostics FastRuntimeState::diagnostics() const {
    return diagnostics_;
}

int FastRuntimeState::threshold_int() const {
    return threshold_int_;
}

bool FastRuntimeState::pending_valid() const {
    return pending_.valid;
}

void FastRuntimeState::set_next_threshold(int threshold_int) {
    threshold_int_ = clamp_threshold(threshold_int);
}

FastStatus FastRuntimeState::try_fast_output(const GameInput& input, GameOutput* output) {
    GameOutput fused{};
#if POLICY_RUNTIME_FAST_DEBUG
    int action_count = 0;
    const FastStatus status = try_fast_output_core(pending_.grid, input, threshold_int_, &fused, nullptr, nullptr, &action_count);
#else
    const FastStatus status = try_fast_output_core(pending_.grid, input, threshold_int_, &fused, nullptr, nullptr, nullptr);
#endif

#if POLICY_RUNTIME_FAST_DEBUG
    if (status == FastStatus::MissNoTarget) {
        diagnostics_.fast_miss_no_target += 1;
        diagnostics_.neural_fallback += 1;
    } else if (status == FastStatus::PathFail) {
        diagnostics_.fast_path_fail += 1;
        diagnostics_.neural_fallback += 1;
    } else {
        diagnostics_.fast_success += 1;
        if (action_count > 0) {
            diagnostics_.fast_nonstay += 1;
        }
    }
#endif
    if (status != FastStatus::Success) {
        return status;
    }

    *output = fused;
    store_pending_meta(input, fused);
    return FastStatus::Success;
}

FastTryResult FastRuntimeState::try_fast(const GameInput& input) {
    FastTryResult result{};
    GameOutput output{};
#if POLICY_RUNTIME_FAST_DEBUG
    int role = -1;
    Position target{0, 0};
    int action_count = 0;
    result.status = try_fast_output_core(pending_.grid, input, threshold_int_, &output, &role, &target, &action_count);
    result.role = role;
    result.target = target;
    result.action_count = action_count;

    if (result.status == FastStatus::MissNoTarget) {
        diagnostics_.fast_miss_no_target += 1;
        diagnostics_.neural_fallback += 1;
        return result;
    }
    if (result.status == FastStatus::PathFail) {
        diagnostics_.fast_path_fail += 1;
        diagnostics_.neural_fallback += 1;
        return result;
    }
    result.output = output;
    diagnostics_.fast_success += 1;
    if (action_count > 0) {
        diagnostics_.fast_nonstay += 1;
    }
    store_pending_meta(input, output);
#else
    result.status = try_fast_output(input, &output);
#endif
    return result;
}

NeuralPrepareResult FastRuntimeState::prepare_neural(const GameInput& input) {
    if (pending_.valid) {
        backfill_pending(input);
    }
    NeuralPrepareResult result;
    result.actor_features = extractor_.observe(input);
    result.fast_scalars = fast_scalars();
    result.diagnostics = diagnostics_;
    return result;
}

void FastRuntimeState::commit_neural(const GameOutput& output) {
    extractor_.commit_action(output);
}

void FastRuntimeState::clear_pending() {
    pending_ = PendingFastBackfill{};
    init_pending_grid();
}

void FastRuntimeState::init_pending_grid() {
    init_padded_grid(pending_.grid);
}

void FastRuntimeState::pack_grid(const GameInput& input) {
    pack_padded_grid_interior(pending_.grid, input);
}

void FastRuntimeState::store_pending_meta(const GameInput& input, const GameOutput& output) {
    pending_.valid = true;
    pending_.round = input.round;
    pending_.my_units[0] = input.my_units[0];
    pending_.my_units[1] = input.my_units[1];
    pending_.my_units_gold[0] = input.my_units_gold[0];
    pending_.my_units_gold[1] = input.my_units_gold[1];
    pending_.gold_opp = input.gold_opp;
    pending_.visible_enemies[0] = input.visible_enemies[0];
    pending_.visible_enemies[1] = input.visible_enemies[1];
    pending_.num_visible_npcs = input.num_visible_npcs;
    std::memcpy(pending_.visible_npcs, input.visible_npcs, sizeof(pending_.visible_npcs));
    pending_.snapshot_valid = input.snapshot_valid;
    if (input.snapshot_valid != 0) {
        for (int i = 0; i < REGION_COUNT; ++i) {
            pending_.snapshot_regions[i].id = input.snapshot.regions[i].id;
            pending_.snapshot_regions[i].gold_generated = input.snapshot.regions[i].gold_generated;
            pending_.snapshot_regions[i].gold_remaining = input.snapshot.regions[i].gold_remaining;
            pending_.snapshot_regions[i].occupants = input.snapshot.regions[i].occupants;
        }
    }
    pending_.threshold_int = threshold_int_;
    pending_.output = output;
}

GameInput FastRuntimeState::restore_pending_input() const {
    GameInput input{};
    input.round = pending_.round;
    for (int row = 0; row < GRID_SIZE; ++row) {
        const std::int8_t* src = &pending_.grid[pad2_index(row, 0)];
        for (int col = 0; col < GRID_SIZE; ++col) {
            input.grid[row][col] = static_cast<int>(src[col]);
        }
    }
    input.my_units[0] = pending_.my_units[0];
    input.my_units[1] = pending_.my_units[1];
    input.my_units_gold[0] = pending_.my_units_gold[0];
    input.my_units_gold[1] = pending_.my_units_gold[1];
    input.gold_opp = pending_.gold_opp;
    input.visible_enemies[0] = pending_.visible_enemies[0];
    input.visible_enemies[1] = pending_.visible_enemies[1];
    input.num_visible_npcs = pending_.num_visible_npcs;
    std::memcpy(input.visible_npcs, pending_.visible_npcs, sizeof(input.visible_npcs));
    input.snapshot_valid = pending_.snapshot_valid;
    if (pending_.snapshot_valid != 0) {
        input.snapshot.window_begin = -1;
        input.snapshot.window_end = -1;
        for (int i = 0; i < REGION_COUNT; ++i) {
            input.snapshot.regions[i].id = pending_.snapshot_regions[i].id;
            input.snapshot.regions[i].enter = 0;
            input.snapshot.regions[i].leave = 0;
            input.snapshot.regions[i].gold_generated = pending_.snapshot_regions[i].gold_generated;
            input.snapshot.regions[i].gold_collected = 0;
            input.snapshot.regions[i].gold_remaining = pending_.snapshot_regions[i].gold_remaining;
            input.snapshot.regions[i].occupants = pending_.snapshot_regions[i].occupants;
        }
    }
    return input;
}

void FastRuntimeState::backfill_pending(const GameInput& input_now) {
    const GameInput pending_input = restore_pending_input();
    int role = -1;
    int expected_gain = 0;
    Position target{0, 0};
    try {
        role = infer_fast_role(pending_input, pending_.output);
        expected_gain = simulate_known_gold_pickups(pending_input, pending_.output, role);
        target = replay_output_final_position(pending_input, pending_.output, role);
    } catch (const std::logic_error&) {
        expected_gain = 0;
    }
    if (expected_gain > 0) {
        const int actual_delta = input_now.my_units_gold[role] - pending_.my_units_gold[role];
        bool should_update = actual_delta < expected_gain;
        float score = 0.0F;
        if (!should_update) {
            const bool new_npc_at_target = npc_at_position(input_now, target) && !npc_at_position(pending_input, target);
            if (new_npc_at_target) {
                should_update = true;
                score = 1.0F;
            } else {
                diagnostics_.fast_effective_skipped_success_no_new_npc += 1;
            }
        }
        if (should_update) {
            if (score > 0.0F) {
                diagnostics_.fast_effective_positive_updates += 1;
            } else {
                diagnostics_.fast_effective_negative_updates += 1;
            }
            fast_alpha_ += score;
            fast_beta_ += 1.0F - score;
            diagnostics_.fast_effective_updates += 1;
            diagnostics_.fast_effective_score_sum += score;
            diagnostics_.fast_expected_gain_sum += static_cast<float>(expected_gain);
            diagnostics_.fast_actual_delta_sum += static_cast<float>(actual_delta);
            diagnostics_.one_step_fast_delta_sum += static_cast<float>(actual_delta - expected_gain);
        }
    }
    extractor_.observe(pending_input);
    extractor_.commit_action(pending_.output);
    clear_pending();
}

}  // namespace policy_runtime
