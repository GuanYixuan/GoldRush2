#pragma once

#include <array>
#include <cstdint>

#include "official_sdk/code/game_api.h"
#include "policy_runtime/feature_extractor.h"

#ifndef POLICY_RUNTIME_FAST_DEBUG
#error "POLICY_RUNTIME_FAST_DEBUG must be explicitly set to 0 or 1"
#endif

#if POLICY_RUNTIME_FAST_DEBUG != 0 && POLICY_RUNTIME_FAST_DEBUG != 1
#error "POLICY_RUNTIME_FAST_DEBUG must be 0 or 1"
#endif

#ifndef POLICY_RUNTIME_FAST_ONE_TU
#define POLICY_RUNTIME_FAST_ONE_TU 0
#endif

#if POLICY_RUNTIME_FAST_ONE_TU != 0 && POLICY_RUNTIME_FAST_ONE_TU != 1
#error "POLICY_RUNTIME_FAST_ONE_TU must be 0 or 1"
#endif

namespace policy_runtime {

constexpr int FAST_SCALAR_FEATURES = 2;
constexpr float FAST_INITIAL_ALPHA = 4.0F;
constexpr float FAST_INITIAL_BETA = 1.0F;
constexpr float FAST_CONFIDENCE_CAP = 20.0F;
constexpr int FAST_THRESHOLD_LOW = 4;
constexpr int FAST_THRESHOLD_HIGH = 30;

enum class FastStatus {
    Success = 0,
    MissNoTarget = 1,
    PathFail = 2,
};

struct FastDiagnostics {
    int fast_success = 0;
    int fast_miss_no_target = 0;
    int fast_path_fail = 0;
    int neural_fallback = 0;
    int fast_nonstay = 0;
    int fast_effective_updates = 0;
    float fast_effective_score_sum = 0.0F;
    float fast_expected_gain_sum = 0.0F;
    float fast_actual_delta_sum = 0.0F;
    float one_step_fast_delta_sum = 0.0F;
    int fast_effective_positive_updates = 0;
    int fast_effective_negative_updates = 0;
    int fast_effective_skipped_success_no_new_npc = 0;
};

struct FastTryResult {
    FastStatus status = FastStatus::MissNoTarget;
    GameOutput output{};
    int role = -1;
    Position target{0, 0};
    int action_count = 0;
};

struct NeuralPrepareResult {
    FeatureOutput actor_features;
    std::array<float, FAST_SCALAR_FEATURES> fast_scalars{};
    FastDiagnostics diagnostics{};
};

class FastRuntimeState {
public:
    static constexpr int PAD2 = 2;
    static constexpr int PAD2_STRIDE = GRID_SIZE + 2 * PAD2;
    static constexpr int PAD2_COUNT = PAD2_STRIDE * PAD2_STRIDE;

    explicit FastRuntimeState(int player_id = 1);

    void reset(int player_id);
    std::array<float, FAST_SCALAR_FEATURES> fast_scalars() const;
    FastDiagnostics diagnostics() const;
    int threshold_int() const;
    bool pending_valid() const;

    void set_next_threshold(int threshold_int);
    FastStatus try_fast_output(const GameInput& input, GameOutput* output);
#if !POLICY_RUNTIME_FAST_DEBUG && POLICY_RUNTIME_FAST_ONE_TU
    FastStatus try_fast_output_release(const GameInput& input, GameOutput* output);
#endif
    FastTryResult try_fast(const GameInput& input);
    NeuralPrepareResult prepare_neural(const GameInput& input);
    void commit_neural(const GameOutput& output);

private:
    struct PendingSnapshotRegion {
        int id = 0;
        int gold_generated = 0;
        int gold_remaining = 0;
        int occupants = 0;
    };

    struct PendingFastBackfill {
        bool valid = false;
        int round = 0;
        std::int8_t grid[PAD2_COUNT]{};
        Position my_units[2]{};
        int my_units_gold[2]{};
        int gold_opp = 0;
        Position visible_enemies[2]{};
        int num_visible_npcs = 0;
        NpcInfo visible_npcs[MAX_NPCS]{};
        int snapshot_valid = 0;
        PendingSnapshotRegion snapshot_regions[REGION_COUNT]{};
        int threshold_int = 12;
        GameOutput output{};
    };

    FeatureExtractor extractor_;
    PendingFastBackfill pending_{};
    FastDiagnostics diagnostics_{};
    float fast_alpha_ = FAST_INITIAL_ALPHA;
    float fast_beta_ = FAST_INITIAL_BETA;
    int threshold_int_ = 12;
    int player_id_ = 1;

    void clear_pending();
    void init_pending_grid();
    void pack_grid(const GameInput& input);
    void store_pending_meta(const GameInput& input, const GameOutput& output);
    GameInput restore_pending_input() const;
    void backfill_pending(const GameInput& input_now);
};

bool try_fast_gold_grab(const GameInput& input, int threshold_int, GameOutput* output, FastTryResult* result = nullptr);
int simulate_known_gold_pickups(const GameInput& input, const GameOutput& output, int role);
int infer_fast_role(const GameInput& input, const GameOutput& output);

}  // namespace policy_runtime
