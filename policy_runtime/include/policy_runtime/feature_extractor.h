#pragma once

#include <array>
#include <string>
#include <vector>

#include "official_sdk/code/game_api.h"

namespace policy_runtime {

constexpr int FEATURE_CHANNELS = 11;
constexpr int FEATURE_SCALARS = 14;

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
    bool has_last_action_ = false;
    GameOutput last_action_{};
    std::array<int, GRID_SIZE * GRID_SIZE> explored_{};
    std::array<int, GRID_SIZE * GRID_SIZE> last_seen_round_{};
    std::array<int, GRID_SIZE * GRID_SIZE> last_visible_grid_{};
};

std::array<std::string, FEATURE_CHANNELS> channel_names();
std::array<std::string, FEATURE_SCALARS> scalar_names();

}  // namespace policy_runtime
