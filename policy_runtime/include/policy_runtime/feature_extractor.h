#pragma once

#include <array>
#include <string>
#include <vector>

#include "official_sdk/code/game_api.h"

namespace policy_runtime {

constexpr const char* FEATURE_SCHEMA = "goldrush2_feature_v2";
constexpr int FEATURE_CHANNELS = 43;
constexpr int FEATURE_SCALARS = 10;
constexpr int TEMPORAL_KEYS = 4;
constexpr int TEMPORAL_WINDOW = 5;

struct FeatureOutput {
    std::vector<float> planes;
    std::vector<float> scalars;
    int channels = FEATURE_CHANNELS;
    int height = GRID_SIZE;
    int width = GRID_SIZE;
};

class FeatureExtractor {
public:
    explicit FeatureExtractor(int player_id = 1);

    void reset(int player_id);
    FeatureOutput observe(const GameInput& input);
    void commit_action(const GameOutput& output);

private:
    int player_id_ = 1;
    bool initialized_ = false;
    int last_round_ = -1;
    GameOutput last_action_{};
    std::array<std::array<float, GRID_SIZE * GRID_SIZE>, TEMPORAL_KEYS * TEMPORAL_WINDOW> temporal_{};
    std::array<int, GRID_SIZE * GRID_SIZE> direct_obstacle_status_{};
    std::array<int, GRID_SIZE * GRID_SIZE> inferred_obstacle_status_{};
    std::array<float, REGION_COUNT> last_snapshot_gold_remaining_{};
    std::array<float, REGION_COUNT> prev_snapshot_gold_remaining_{};
    std::array<float, REGION_COUNT> last_snapshot_occupants_{};
    std::array<float, REGION_COUNT> last_snapshot_gold_generated_{};
    std::array<float, GRID_SIZE * GRID_SIZE> bomb_belief_{};
    std::array<float, GRID_SIZE * GRID_SIZE> observed_high_outer_gold_mask_{};
    unsigned int possible_symmetry_axes_ = 0;
    bool last_snapshot_valid_ = false;
};

std::string feature_schema();
std::array<std::string, FEATURE_CHANNELS> channel_names();
std::array<std::string, FEATURE_SCALARS> scalar_names();

}  // namespace policy_runtime
