#include "policy_runtime/feature_extractor.h"

#include <algorithm>
#include <stdexcept>

namespace policy_runtime {
namespace {

constexpr int GRID_FOG = -5;
constexpr int GRID_BOMB = -3;
constexpr int GRID_OBSTACLE = -1;

int index(int row, int col) {
    return row * GRID_SIZE + col;
}

int plane_index(int channel, int row, int col) {
    return (channel * GRID_SIZE + row) * GRID_SIZE + col;
}

bool in_bounds(Position pos) {
    return 0 <= pos.row && pos.row < GRID_SIZE && 0 <= pos.col && pos.col < GRID_SIZE;
}

float clamp_scale(int value, float denominator) {
    return static_cast<float>(value) / denominator;
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
    has_last_action_ = false;
    last_action_ = GameOutput{};
    explored_.fill(0);
    last_seen_round_.fill(-1);
    last_visible_grid_.fill(GRID_FOG);
}

FeatureOutput FeatureExtractor::observe(const GameInput& input) {
    if (!initialized_) {
        throw std::logic_error("FeatureExtractor is not initialized");
    }
    if (input.round < last_round_) {
        throw std::logic_error("round regression detected; call reset() before a new episode");
    }
    last_round_ = input.round;

    FeatureOutput output;
    output.planes.assign(FEATURE_CHANNELS * GRID_SIZE * GRID_SIZE, 0.0F);
    output.scalars.assign(FEATURE_SCALARS, 0.0F);

    for (int row = 0; row < GRID_SIZE; ++row) {
        for (int col = 0; col < GRID_SIZE; ++col) {
            const int cell = input.grid[row][col];
            const bool visible = cell != GRID_FOG;
            const int flat = index(row, col);
            if (visible) {
                explored_[flat] = 1;
                last_seen_round_[flat] = input.round;
                last_visible_grid_[flat] = cell;
            }

            output.planes[plane_index(0, row, col)] = visible ? 1.0F : 0.0F;
            output.planes[plane_index(1, row, col)] = explored_[flat] ? 1.0F : 0.0F;
            output.planes[plane_index(2, row, col)] = visible && cell == GRID_OBSTACLE ? 1.0F : 0.0F;
            output.planes[plane_index(3, row, col)] = visible && cell == GRID_BOMB ? 1.0F : 0.0F;
            output.planes[plane_index(4, row, col)] = visible && cell == 0 ? 1.0F : 0.0F;
            output.planes[plane_index(5, row, col)] = visible && cell > 0 ? clamp_scale(cell, 100.0F) : 0.0F;
            if (last_seen_round_[flat] >= 0) {
                const int age = std::max(0, input.round - last_seen_round_[flat]);
                output.planes[plane_index(10, row, col)] = clamp_scale(std::min(age, 500), 500.0F);
            }
        }
    }

    for (int unit_id = 0; unit_id < 2; ++unit_id) {
        const Position pos = input.my_units[unit_id];
        if (in_bounds(pos)) {
            output.planes[plane_index(6 + unit_id, pos.row, pos.col)] = 1.0F;
        }
    }

    for (int enemy_id = 0; enemy_id < 2; ++enemy_id) {
        const Position pos = input.visible_enemies[enemy_id];
        if (in_bounds(pos)) {
            output.planes[plane_index(8, pos.row, pos.col)] = 1.0F;
        }
    }

    const int npc_count = std::max(0, std::min(input.num_visible_npcs, MAX_NPCS));
    for (int npc_index = 0; npc_index < npc_count; ++npc_index) {
        const Position pos = input.visible_npcs[npc_index].pos;
        if (in_bounds(pos)) {
            output.planes[plane_index(9, pos.row, pos.col)] = 1.0F;
        }
    }

    int scalar = 0;
    output.scalars[scalar++] = clamp_scale(input.round, 500.0F);
    output.scalars[scalar++] = clamp_scale(input.my_units_gold[0], 100.0F);
    output.scalars[scalar++] = clamp_scale(input.my_units_gold[1], 100.0F);
    output.scalars[scalar++] = clamp_scale(input.gold_opp, 100.0F);
    output.scalars[scalar++] = has_last_action_ ? 1.0F : 0.0F;
    output.scalars[scalar++] = has_last_action_ ? clamp_scale(last_action_.k, static_cast<float>(S)) : 0.0F;
    output.scalars[scalar++] = has_last_action_ ? static_cast<float>(last_action_.order) : 0.0F;
    output.scalars[scalar++] = has_last_action_ ? clamp_scale(last_action_.vp, 2.0F) : 0.0F;
    for (int action_index = 0; action_index < S; ++action_index) {
        output.scalars[scalar++] = has_last_action_ ? clamp_scale(last_action_.actions[action_index], 4.0F) : 0.0F;
    }
    return output;
}

void FeatureExtractor::commit_action(const GameOutput& output) {
    if (!initialized_) {
        throw std::logic_error("FeatureExtractor is not initialized");
    }
    last_action_ = output;
    has_last_action_ = true;
}

std::array<std::string, FEATURE_CHANNELS> channel_names() {
    return {
        "visible_mask",
        "explored_mask",
        "obstacle",
        "bomb",
        "empty",
        "positive_gold_amount_scaled",
        "own_unit_0",
        "own_unit_1",
        "visible_enemy",
        "visible_npc",
        "last_seen_age_scaled",
    };
}

std::array<std::string, FEATURE_SCALARS> scalar_names() {
    return {
        "round_scaled",
        "my_unit0_gold_scaled",
        "my_unit1_gold_scaled",
        "gold_opp_scaled",
        "has_last_action",
        "last_k_scaled",
        "last_order",
        "last_vp_scaled",
        "last_action_0_scaled",
        "last_action_1_scaled",
        "last_action_2_scaled",
        "last_action_3_scaled",
        "last_action_4_scaled",
        "last_action_5_scaled",
    };
}

}  // namespace policy_runtime
