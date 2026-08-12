#include "policy_runtime/feature_extractor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <queue>
#include <stdexcept>

namespace policy_runtime {
namespace {

constexpr int CELL_COUNT = GRID_SIZE * GRID_SIZE;
constexpr int GRID_FOG = -5;
constexpr int GRID_BOMB = -3;
constexpr int GRID_OBSTACLE = -1;
constexpr int UNKNOWN_OBSTACLE_STATUS = 0;
constexpr int KNOWN_NON_OBSTACLE = 1;
constexpr int KNOWN_OBSTACLE = 2;
constexpr float BOMB_SPAWN_PROBABILITY = 0.0795F;
constexpr int BOMB_REFRESH_PERIOD = 20;
constexpr int STATIC2_GOLD_THRESHOLD = 16;
constexpr float CENTER_GOLD_B = 0.06735F;
constexpr float PI = 3.14159265358979323846F;
constexpr unsigned int SYMMETRY_UP_DOWN = 1U << 0;
constexpr unsigned int SYMMETRY_LEFT_RIGHT = 1U << 1;
constexpr unsigned int ALL_SYMMETRY_AXES = SYMMETRY_UP_DOWN | SYMMETRY_LEFT_RIGHT;

enum TemporalKey {
    VISIBLE_MASK = 0,
    GOLD_COUNT = 1,
    BOMB_MASK = 2,
    VISIBLE_ENEMY_COUNT = 3,
};

int index(int row, int col) {
    return row * GRID_SIZE + col;
}

int plane_index(int channel, int row, int col) {
    return (channel * GRID_SIZE + row) * GRID_SIZE + col;
}

int temporal_index(int key, int age) {
    return key * TEMPORAL_WINDOW + age;
}

bool in_bounds(Position pos) {
    return 0 <= pos.row && pos.row < GRID_SIZE && 0 <= pos.col && pos.col < GRID_SIZE;
}

float clip(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

float scaled_clip(float value, float denominator, float low, float high) {
    return clip(value / denominator, low, high);
}

int up_down_mirror_flat(int flat) {
    const int row = flat / GRID_SIZE;
    const int col = flat % GRID_SIZE;
    return index(GRID_SIZE - 1 - row, col);
}

int left_right_mirror_flat(int flat) {
    const int row = flat / GRID_SIZE;
    const int col = flat % GRID_SIZE;
    return index(row, GRID_SIZE - 1 - col);
}

int effective_obstacle_status(
    const std::array<int, CELL_COUNT>& direct,
    const std::array<int, CELL_COUNT>& inferred,
    int flat) {
    return direct[flat] != UNKNOWN_OBSTACLE_STATUS ? direct[flat] : inferred[flat];
}

void record_obstacle_status(std::array<int, CELL_COUNT>& direct, int flat, int status) {
    direct[flat] = status;
}

void update_possible_symmetry_axes(const std::array<int, CELL_COUNT>& direct, unsigned int& axes) {
    if ((axes & SYMMETRY_UP_DOWN) != 0U) {
        for (int flat = 0; flat < CELL_COUNT; ++flat) {
            const int mirrored = up_down_mirror_flat(flat);
            if (direct[flat] != UNKNOWN_OBSTACLE_STATUS && direct[mirrored] != UNKNOWN_OBSTACLE_STATUS
                && direct[flat] != direct[mirrored]) {
                axes &= ~SYMMETRY_UP_DOWN;
                break;
            }
        }
    }
    if ((axes & SYMMETRY_LEFT_RIGHT) != 0U) {
        for (int flat = 0; flat < CELL_COUNT; ++flat) {
            const int mirrored = left_right_mirror_flat(flat);
            if (direct[flat] != UNKNOWN_OBSTACLE_STATUS && direct[mirrored] != UNKNOWN_OBSTACLE_STATUS
                && direct[flat] != direct[mirrored]) {
                axes &= ~SYMMETRY_LEFT_RIGHT;
                break;
            }
        }
    }
}

int inferred_status_for_axis(const std::array<int, CELL_COUNT>& direct, int flat, unsigned int axis) {
    const int mirrored = axis == SYMMETRY_UP_DOWN ? up_down_mirror_flat(flat) : left_right_mirror_flat(flat);
    return direct[mirrored];
}

void rebuild_inferred_obstacle_status(
    const std::array<int, CELL_COUNT>& direct,
    unsigned int possible_axes,
    std::array<int, CELL_COUNT>& inferred) {
    inferred.fill(UNKNOWN_OBSTACLE_STATUS);
    for (int flat = 0; flat < CELL_COUNT; ++flat) {
        if (direct[flat] != UNKNOWN_OBSTACLE_STATUS) {
            continue;
        }
        if (possible_axes == SYMMETRY_UP_DOWN || possible_axes == SYMMETRY_LEFT_RIGHT) {
            inferred[flat] = inferred_status_for_axis(direct, flat, possible_axes);
            continue;
        }
        if (possible_axes == ALL_SYMMETRY_AXES) {
            const int up_down = inferred_status_for_axis(direct, flat, SYMMETRY_UP_DOWN);
            const int left_right = inferred_status_for_axis(direct, flat, SYMMETRY_LEFT_RIGHT);
            if (up_down != UNKNOWN_OBSTACLE_STATUS && up_down == left_right) {
                inferred[flat] = up_down;
            }
        }
    }
}

int region_index(int row, int col) {
    if (4 <= row && row <= 12 && 4 <= col && col <= 12) {
        return 0;
    }
    if (row <= 3 && col <= 12) {
        return 1;
    }
    if (4 <= row && col <= 3) {
        return 2;
    }
    if (13 <= row && 4 <= col) {
        return 3;
    }
    if (row <= 12 && 13 <= col) {
        return 4;
    }
    throw std::logic_error("cell is outside all official regions");
}

void require_unit_position(Position pos, int unit_id) {
    if (!in_bounds(pos)) {
        throw std::invalid_argument("my_units position out of bounds at unit " + std::to_string(unit_id));
    }
}

std::array<float, CELL_COUNT> known_obstacle_distances(
    Position start,
    const std::array<int, CELL_COUNT>& direct,
    const std::array<int, CELL_COUNT>& inferred) {
    std::array<int, CELL_COUNT> distance{};
    distance.fill(-1);

    std::queue<Position> frontier;
    const int start_flat = index(start.row, start.col);
    if (effective_obstacle_status(direct, inferred, start_flat) != KNOWN_OBSTACLE) {
        distance[start_flat] = 0;
        frontier.push(start);
    }

    constexpr std::array<Position, 4> deltas = {
        Position{-1, 0},
        Position{1, 0},
        Position{0, -1},
        Position{0, 1},
    };

    while (!frontier.empty()) {
        const Position current = frontier.front();
        frontier.pop();
        const int current_flat = index(current.row, current.col);
        for (const Position delta : deltas) {
            const Position next{current.row + delta.row, current.col + delta.col};
            if (!in_bounds(next)) {
                continue;
            }
            const int next_flat = index(next.row, next.col);
            if (distance[next_flat] >= 0) {
                continue;
            }
            if (effective_obstacle_status(direct, inferred, next_flat) == KNOWN_OBSTACLE) {
                continue;
            }
            distance[next_flat] = distance[current_flat] + 1;
            frontier.push(next);
        }
    }

    std::array<float, CELL_COUNT> scaled{};
    for (int flat = 0; flat < CELL_COUNT; ++flat) {
        scaled[flat] = distance[flat] < 0 ? 2.0F : scaled_clip(static_cast<float>(distance[flat]), 32.0F, 0.0F, 2.0F);
    }
    return scaled;
}

std::array<float, CELL_COUNT> known_obstacle_distances_to_targets(
    const std::array<float, CELL_COUNT>& targets,
    const std::array<int, CELL_COUNT>& direct,
    const std::array<int, CELL_COUNT>& inferred) {
    std::array<int, CELL_COUNT> distance{};
    distance.fill(-1);

    std::queue<Position> frontier;
    for (int row = 0; row < GRID_SIZE; ++row) {
        for (int col = 0; col < GRID_SIZE; ++col) {
            const int flat = index(row, col);
            if (targets[flat] <= 0.5F) {
                continue;
            }
            if (effective_obstacle_status(direct, inferred, flat) == KNOWN_OBSTACLE) {
                continue;
            }
            distance[flat] = 0;
            frontier.push(Position{row, col});
        }
    }

    constexpr std::array<Position, 4> deltas = {
        Position{-1, 0},
        Position{1, 0},
        Position{0, -1},
        Position{0, 1},
    };

    while (!frontier.empty()) {
        const Position current = frontier.front();
        frontier.pop();
        const int current_flat = index(current.row, current.col);
        for (const Position delta : deltas) {
            const Position next{current.row + delta.row, current.col + delta.col};
            if (!in_bounds(next)) {
                continue;
            }
            const int next_flat = index(next.row, next.col);
            if (distance[next_flat] >= 0) {
                continue;
            }
            if (effective_obstacle_status(direct, inferred, next_flat) == KNOWN_OBSTACLE) {
                continue;
            }
            distance[next_flat] = distance[current_flat] + 1;
            frontier.push(next);
        }
    }

    std::array<float, CELL_COUNT> scaled{};
    for (int flat = 0; flat < CELL_COUNT; ++flat) {
        scaled[flat] = distance[flat] < 0 ? 2.0F : scaled_clip(static_cast<float>(distance[flat]), 32.0F, 0.0F, 2.0F);
    }
    return scaled;
}

void write_snapshot_memory(
    std::array<float, REGION_COUNT>& dst_gold,
    std::array<float, REGION_COUNT>& dst_occupants,
    std::array<float, REGION_COUNT>& dst_generated,
    const Snapshot& snapshot) {
    for (const RegionStat& region : snapshot.regions) {
        if (region.id < 1 || region.id > REGION_COUNT) {
            throw std::invalid_argument("snapshot region id must be in [1, REGION_COUNT]");
        }
        const int region_idx = region.id - 1;
        dst_gold[region_idx] = scaled_clip(static_cast<float>(region.gold_remaining), 100.0F, 0.0F, 2.0F);
        dst_occupants[region_idx] = scaled_clip(static_cast<float>(region.occupants), 4.0F, 0.0F, 2.0F);
        dst_generated[region_idx] = scaled_clip(static_cast<float>(region.gold_generated), 100.0F, 0.0F, 2.0F);
    }
}

bool known_obstacle(
    const std::array<int, CELL_COUNT>& direct,
    const std::array<int, CELL_COUNT>& inferred,
    int flat) {
    return effective_obstacle_status(direct, inferred, flat) == KNOWN_OBSTACLE;
}

void clear_known_obstacle_beliefs(
    std::array<float, CELL_COUNT>& belief,
    const std::array<int, CELL_COUNT>& direct,
    const std::array<int, CELL_COUNT>& inferred) {
    for (int flat = 0; flat < CELL_COUNT; ++flat) {
        if (known_obstacle(direct, inferred, flat)) {
            belief[flat] = 0.0F;
        }
    }
}

void mark_visible_non_bomb(std::array<float, CELL_COUNT>& bomb_belief, Position pos) {
    if (in_bounds(pos)) {
        bomb_belief[index(pos.row, pos.col)] = 0.0F;
    }
}

}  // namespace

FeatureExtractor::FeatureExtractor(int player_id) {
    reset(player_id);
}

void FeatureExtractor::reset(int player_id) {
    if (player_id != 1 && player_id != 2) {
        throw std::invalid_argument("player_id must be 1 or 2");
    }
    player_id_ = player_id;
    initialized_ = true;
    last_round_ = -1;
    last_action_ = GameOutput{};
    for (auto& plane : temporal_) {
        plane.fill(0.0F);
    }
    direct_obstacle_status_.fill(UNKNOWN_OBSTACLE_STATUS);
    inferred_obstacle_status_.fill(UNKNOWN_OBSTACLE_STATUS);
    last_snapshot_gold_remaining_.fill(0.0F);
    prev_snapshot_gold_remaining_.fill(0.0F);
    last_snapshot_occupants_.fill(0.0F);
    last_snapshot_gold_generated_.fill(0.0F);
    bomb_belief_.fill(BOMB_SPAWN_PROBABILITY);
    observed_high_outer_gold_mask_.fill(0.0F);
    possible_symmetry_axes_ = ALL_SYMMETRY_AXES;
    last_snapshot_valid_ = false;
}

FeatureOutput FeatureExtractor::observe(const GameInput& input) {
    if (!initialized_) {
        throw std::logic_error("FeatureExtractor is not initialized");
    }
    if (input.round < last_round_) {
        throw std::logic_error("round regression detected; call reset() before a new episode");
    }

    const bool new_round = input.round != last_round_;
    if (new_round) {
        for (int key = 0; key < TEMPORAL_KEYS; ++key) {
            for (int age = TEMPORAL_WINDOW - 1; age >= 1; --age) {
                temporal_[temporal_index(key, age)] = temporal_[temporal_index(key, age - 1)];
            }
            temporal_[temporal_index(key, 0)].fill(0.0F);
        }
        if (input.snapshot_valid != 0) {
            if (last_snapshot_valid_) {
                prev_snapshot_gold_remaining_ = last_snapshot_gold_remaining_;
            }
            last_snapshot_gold_remaining_.fill(0.0F);
            last_snapshot_occupants_.fill(0.0F);
            last_snapshot_gold_generated_.fill(0.0F);
            write_snapshot_memory(
                last_snapshot_gold_remaining_,
                last_snapshot_occupants_,
                last_snapshot_gold_generated_,
                input.snapshot);
            last_snapshot_valid_ = true;
        }
        if (input.round % BOMB_REFRESH_PERIOD == 0) {
            bomb_belief_.fill(BOMB_SPAWN_PROBABILITY);
            clear_known_obstacle_beliefs(bomb_belief_, direct_obstacle_status_, inferred_obstacle_status_);
        }
        last_round_ = input.round;
    } else {
        for (int key = 0; key < TEMPORAL_KEYS; ++key) {
            temporal_[temporal_index(key, 0)].fill(0.0F);
        }
    }

    for (int unit_id = 0; unit_id < 2; ++unit_id) {
        require_unit_position(input.my_units[unit_id], unit_id);
    }

    for (int row = 0; row < GRID_SIZE; ++row) {
        for (int col = 0; col < GRID_SIZE; ++col) {
            const int cell = input.grid[row][col];
            const bool visible = cell != GRID_FOG;
            const int flat = index(row, col);
            if (!visible) {
                continue;
            }
            temporal_[temporal_index(VISIBLE_MASK, 0)][flat] = 1.0F;
            if (cell > 0) {
                temporal_[temporal_index(GOLD_COUNT, 0)][flat] = scaled_clip(static_cast<float>(cell), 20.0F, 0.0F, 3.0F);
            }
            if (cell == GRID_BOMB) {
                temporal_[temporal_index(BOMB_MASK, 0)][flat] = 1.0F;
            }
            const int obstacle_status = cell == GRID_OBSTACLE ? KNOWN_OBSTACLE : KNOWN_NON_OBSTACLE;
            if (region_index(row, col) != 0 && cell >= STATIC2_GOLD_THRESHOLD) {
                observed_high_outer_gold_mask_[flat] = 1.0F;
            }
            bomb_belief_[flat] = cell == GRID_BOMB ? 1.0F : 0.0F;
            record_obstacle_status(direct_obstacle_status_, flat, obstacle_status);
        }
    }

    update_possible_symmetry_axes(direct_obstacle_status_, possible_symmetry_axes_);
    rebuild_inferred_obstacle_status(
        direct_obstacle_status_,
        possible_symmetry_axes_,
        inferred_obstacle_status_);
    clear_known_obstacle_beliefs(bomb_belief_, direct_obstacle_status_, inferred_obstacle_status_);
    mark_visible_non_bomb(bomb_belief_, input.my_units[0]);
    mark_visible_non_bomb(bomb_belief_, input.my_units[1]);

    for (const Position pos : input.visible_enemies) {
        if (in_bounds(pos)) {
            temporal_[temporal_index(VISIBLE_ENEMY_COUNT, 0)][index(pos.row, pos.col)] += 1.0F;
            bomb_belief_[index(pos.row, pos.col)] = 0.0F;
        }
    }

    FeatureOutput output;
    output.planes.assign(FEATURE_CHANNELS * GRID_SIZE * GRID_SIZE, 0.0F);
    output.scalars.assign(FEATURE_SCALARS, 0.0F);

    for (int channel = 0; channel < TEMPORAL_KEYS * TEMPORAL_WINDOW; ++channel) {
        std::copy(temporal_[channel].begin(), temporal_[channel].end(), output.planes.begin() + channel * CELL_COUNT);
    }

    std::array<int, CELL_COUNT> visible_npc_counts{};
    const int npc_count = std::max(0, std::min(input.num_visible_npcs, MAX_NPCS));
    for (int npc_idx = 0; npc_idx < npc_count; ++npc_idx) {
        const Position pos = input.visible_npcs[npc_idx].pos;
        if (in_bounds(pos)) {
            visible_npc_counts[index(pos.row, pos.col)] += 1;
            bomb_belief_[index(pos.row, pos.col)] = 0.0F;
        }
    }

    const auto unit0_bfs = known_obstacle_distances(input.my_units[0], direct_obstacle_status_, inferred_obstacle_status_);
    const auto unit1_bfs = known_obstacle_distances(input.my_units[1], direct_obstacle_status_, inferred_obstacle_status_);
    std::array<float, CELL_COUNT> static2_mask{};
    static2_mask.fill(0.0F);
    for (int flat = 0; flat < CELL_COUNT; ++flat) {
        if (observed_high_outer_gold_mask_[flat] > 0.5F) {
            static2_mask[flat] = 1.0F;
        }
    }
    const auto static2_bfs = known_obstacle_distances_to_targets(
        static2_mask,
        direct_obstacle_status_,
        inferred_obstacle_status_);

    for (int row = 0; row < GRID_SIZE; ++row) {
        for (int col = 0; col < GRID_SIZE; ++col) {
            const int flat = index(row, col);
            const int obstacle_status = effective_obstacle_status(direct_obstacle_status_, inferred_obstacle_status_, flat);
            output.planes[plane_index(20, row, col)] = obstacle_status == UNKNOWN_OBSTACLE_STATUS ? 0.0F : 1.0F;
            output.planes[plane_index(21, row, col)] = obstacle_status == KNOWN_OBSTACLE ? 1.0F : 0.0F;
            output.planes[plane_index(22, row, col)] = scaled_clip(static_cast<float>(visible_npc_counts[flat]), 3.0F, 0.0F, 1.0F);
            output.planes[plane_index(23, row, col)] = visible_npc_counts[flat] >= 3 ? 1.0F : 0.0F;

            output.planes[plane_index(26, row, col)] =
                static_cast<float>(std::abs(row - input.my_units[0].row) + std::abs(col - input.my_units[0].col)) / 32.0F;
            output.planes[plane_index(27, row, col)] =
                static_cast<float>(std::abs(row - input.my_units[1].row) + std::abs(col - input.my_units[1].col)) / 32.0F;
            output.planes[plane_index(28, row, col)] = unit0_bfs[flat];
            output.planes[plane_index(29, row, col)] = unit1_bfs[flat];

            const int region = region_index(row, col);
            output.planes[plane_index(30 + region, row, col)] = 1.0F;
            output.planes[plane_index(35, row, col)] = last_snapshot_gold_remaining_[region];
            output.planes[plane_index(36, row, col)] = prev_snapshot_gold_remaining_[region];
            output.planes[plane_index(37, row, col)] = last_snapshot_occupants_[region];
            output.planes[plane_index(38, row, col)] = last_snapshot_gold_generated_[region];

            const bool center_cell = 4 <= row && row <= 12 && 4 <= col && col <= 12;
            if (center_cell && !known_obstacle(direct_obstacle_status_, inferred_obstacle_status_, flat)) {
                const float dr = static_cast<float>(row - 8);
                const float dc = static_cast<float>(col - 8);
                output.planes[plane_index(39, row, col)] = std::exp(-CENTER_GOLD_B * (dr * dr + dc * dc));
            }
            output.planes[plane_index(40, row, col)] = bomb_belief_[flat];
            output.planes[plane_index(41, row, col)] = static2_mask[flat];
            output.planes[plane_index(42, row, col)] = static2_bfs[flat];
        }
    }

    for (int unit_id = 0; unit_id < 2; ++unit_id) {
        const Position pos = input.my_units[unit_id];
        output.planes[plane_index(24 + unit_id, pos.row, pos.col)] = 1.0F;
    }

    const float game_phase = -PI / 2.0F + PI * static_cast<float>(input.round) / 499.0F;
    const int own_gold = input.my_units_gold[0] + input.my_units_gold[1];
    const float outer_phase = 2.0F * PI * static_cast<float>(input.round % 20) / 20.0F;
    const float snapshot_phase = 2.0F * PI * static_cast<float>(input.round % 5) / 5.0F;

    int scalar = 0;
    output.scalars[scalar++] = std::sin(game_phase);
    output.scalars[scalar++] = std::cos(game_phase);
    output.scalars[scalar++] = scaled_clip(static_cast<float>(own_gold), 2000.0F, 0.0F, 3.0F);
    output.scalars[scalar++] = scaled_clip(static_cast<float>(input.gold_opp), 2000.0F, 0.0F, 3.0F);
    output.scalars[scalar++] = scaled_clip(static_cast<float>(own_gold - input.gold_opp), 500.0F, -3.0F, 3.0F);
    output.scalars[scalar++] = std::sin(outer_phase);
    output.scalars[scalar++] = std::cos(outer_phase);
    output.scalars[scalar++] = std::sin(snapshot_phase);
    output.scalars[scalar++] = std::cos(snapshot_phase);
    output.scalars[scalar++] = last_snapshot_valid_ ? 1.0F : 0.0F;

    return output;
}

void FeatureExtractor::commit_action(const GameOutput& output) {
    if (!initialized_) {
        throw std::logic_error("FeatureExtractor is not initialized");
    }
    last_action_ = output;
}

std::string feature_schema() {
    return FEATURE_SCHEMA;
}

std::array<std::string, FEATURE_CHANNELS> channel_names() {
    return {
        "visible_mask_t0",
        "visible_mask_t1",
        "visible_mask_t2",
        "visible_mask_t3",
        "visible_mask_t4",
        "gold_count_t0",
        "gold_count_t1",
        "gold_count_t2",
        "gold_count_t3",
        "gold_count_t4",
        "bomb_mask_t0",
        "bomb_mask_t1",
        "bomb_mask_t2",
        "bomb_mask_t3",
        "bomb_mask_t4",
        "visible_enemy_count_t0",
        "visible_enemy_count_t1",
        "visible_enemy_count_t2",
        "visible_enemy_count_t3",
        "visible_enemy_count_t4",
        "obstacle_known_mask",
        "obstacle_mask",
        "visible_npc_count",
        "npc_crowded_mask",
        "own_unit0_mask",
        "own_unit1_mask",
        "unit0_manhattan_distance",
        "unit1_manhattan_distance",
        "unit0_known_obstacle_distance",
        "unit1_known_obstacle_distance",
        "region_1_center_mask",
        "region_2_up_mask",
        "region_3_left_mask",
        "region_4_down_mask",
        "region_5_right_mask",
        "last_snapshot_gold_remaining_map",
        "prev_snapshot_gold_remaining_map",
        "last_snapshot_occupants_map",
        "last_snapshot_gold_generated_map",
        "center_gold_spawn_prob",
        "bomb_belief_mask",
        "static_2_mask",
        "to_static2_distance",
    };
}

std::array<std::string, FEATURE_SCALARS> scalar_names() {
    return {
        "game_phase_sin",
        "game_phase_cos",
        "own_gold_scaled",
        "opp_gold_scaled",
        "gold_margin_scaled",
        "outer_gold_phase_sin",
        "outer_gold_phase_cos",
        "snapshot_sin",
        "snapshot_cos",
        "last_snapshot_valid",
    };
}

}  // namespace policy_runtime
