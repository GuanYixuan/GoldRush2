#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ORT_HEADER_SOURCE = ROOT / "policy_runtime" / "onnxruntime_c_api"
FAST_OPTION_SOURCE = ROOT / "policy_runtime" / "src" / "fast_option.cpp"
FAST_OPTION_HEADER_SOURCE = ROOT / "policy_runtime" / "include" / "policy_runtime" / "fast_option.h"
SAFE_SO_BYTES = 15_500_000
EXPECTED_ONNX_SCHEMA = "goldrush2_stochastic_actor_onnx_export_v1"
EXPECTED_INPUTS = {
    "actor_planes": [1, 43, 17, 17],
    "actor_scalars": [1, 10],
    "fast_scalars": [1, 2],
    "rand_ko": [1, 14],
    "rand_vp": [1, 3],
    "rand_action": [1, 6, 5],
}
EXPECTED_OUTPUTS = {
    "actions": [1, 6],
    "k": [1],
    "order": [1],
    "vp": [1],
    "threshold_mu_raw": [1],
    "threshold_log_std": [1],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and build a C++ ONNX Runtime GoldRush submission policy.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--module-name", default="Player0810S")
    parser.add_argument("--fast-runtime-mode", choices=("debug", "release"), required=True)
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()

    onnx_path = args.onnx.resolve()
    output_dir = args.output_dir.resolve()
    module_name = _module_name(args.module_name)
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)
    onnx_metadata = _load_onnx_metadata(onnx_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_ort_headers(output_dir / "ort_include")
    fast_core_one_tu = args.fast_runtime_mode == "release"
    if fast_core_one_tu:
        _copy_release_fast_core(output_dir)
    _write(output_dir / "model_bytes.h", _model_bytes_header(onnx_path.read_bytes()))
    _write(output_dir / "player.cpp", _player_cpp(module_name, fast_core_one_tu=fast_core_one_tu))
    fast_debug = 1 if args.fast_runtime_mode == "debug" else 0
    _write(output_dir / "Makefile", _makefile(module_name, fast_debug=fast_debug, fast_core_one_tu=fast_core_one_tu))

    so_path = output_dir / f"{module_name}.so"
    if not args.no_build:
        subprocess.run(["make", "-j2"], cwd=output_dir, check=True)
        if not so_path.exists():
            raise RuntimeError(f"build did not produce {so_path}")
        if so_path.stat().st_size >= SAFE_SO_BYTES:
            raise RuntimeError(f"{so_path} is too large for safe submission: {so_path.stat().st_size} bytes")

    metadata = {
        "schema": "goldrush2_cpp_policy_assembly_v1",
        "onnx": str(onnx_path),
        "onnx_metadata": str(onnx_path.with_suffix(".metadata.json")),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_schema": onnx_metadata["schema"],
        "onnx_checkpoint_update": onnx_metadata.get("checkpoint_update"),
        "onnx_action_head_schema": onnx_metadata.get("action_head_schema"),
        "output_dir": str(output_dir),
        "module_name": module_name,
        "so": None if not so_path.exists() else str(so_path),
        "so_bytes": None if not so_path.exists() else so_path.stat().st_size,
        "fp32": True,
        "stochastic": True,
        "critic_exported": False,
        "fast_runtime_mode": args.fast_runtime_mode,
        "policy_runtime_fast_debug": fast_debug,
        "fast_core_one_tu": fast_core_one_tu,
    }
    _write(output_dir / "assembly_metadata.json", json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))


def _module_name(raw: str) -> str:
    if not raw or not raw[0].isalpha() or not raw.replace("_", "").isalnum():
        raise ValueError(f"module name must start with a letter and contain only letters, digits or underscore: {raw!r}")
    return raw


def _copy_ort_headers(dst: Path) -> None:
    if not ORT_HEADER_SOURCE.exists():
        raise FileNotFoundError(f"missing ORT headers: {ORT_HEADER_SOURCE}")
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("onnxruntime_c_api.h", "onnxruntime_error_code.h", "onnxruntime_ep_c_api.h"):
        shutil.copy2(ORT_HEADER_SOURCE / name, dst / name)


def _copy_release_fast_core(output_dir: Path) -> None:
    if not FAST_OPTION_SOURCE.exists():
        raise FileNotFoundError(f"missing fast option source: {FAST_OPTION_SOURCE}")
    if not FAST_OPTION_HEADER_SOURCE.exists():
        raise FileNotFoundError(f"missing fast option header: {FAST_OPTION_HEADER_SOURCE}")
    shutil.copy2(FAST_OPTION_SOURCE, output_dir / "fast_option_one_tu.cpp")
    header_dir = output_dir / "policy_runtime"
    header_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(FAST_OPTION_HEADER_SOURCE, header_dir / "fast_option.h")


def _load_onnx_metadata(onnx_path: Path) -> dict[str, Any]:
    metadata_path = onnx_path.with_suffix(".metadata.json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing actor ONNX metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != EXPECTED_ONNX_SCHEMA:
        raise ValueError(f"ONNX metadata schema must be {EXPECTED_ONNX_SCHEMA!r}, got {metadata.get('schema')!r}")
    if metadata.get("stochastic") is not True:
        raise ValueError("ONNX metadata must mark stochastic=true")
    if metadata.get("critic_exported") is not False:
        raise ValueError("ONNX metadata must mark critic_exported=false")
    if metadata.get("inputs") != EXPECTED_INPUTS:
        raise ValueError(f"ONNX input schema mismatch: {metadata.get('inputs')!r}")
    if metadata.get("outputs") != EXPECTED_OUTPUTS:
        raise ValueError(f"ONNX output schema mismatch: {metadata.get('outputs')!r}")
    return metadata


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _model_bytes_header(data: bytes) -> str:
    lines = [
        "#pragma once",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "",
        "static const unsigned char MODEL_BYTES[] = {",
    ]
    for start in range(0, len(data), 16):
        chunk = data[start : start + 16]
        lines.append("    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",")
    lines.extend(
        [
            "};",
            "",
            "static constexpr std::size_t MODEL_BYTES_SIZE = sizeof(MODEL_BYTES);",
            "",
        ]
    )
    return "\n".join(lines)


def _makefile(module_name: str, *, fast_debug: int, fast_core_one_tu: bool) -> str:
    include_flags = "-I. -I$(REPO_ROOT) -I$(REPO_ROOT)/policy_runtime/include" if fast_core_one_tu else "-I$(REPO_ROOT) -I$(REPO_ROOT)/policy_runtime/include -I."
    src = "player.cpp $(REPO_ROOT)/policy_runtime/src/feature_extractor.cpp" if fast_core_one_tu else "player.cpp $(REPO_ROOT)/policy_runtime/src/feature_extractor.cpp $(REPO_ROOT)/policy_runtime/src/fast_option.cpp"
    extra_deps = " fast_option_one_tu.cpp policy_runtime/fast_option.h" if fast_core_one_tu else ""
    one_tu_define = " -DPOLICY_RUNTIME_FAST_ONE_TU=1" if fast_core_one_tu else ""
    return f"""CXX ?= g++
REPO_ROOT := {ROOT}
CXXFLAGS ?= -std=c++17 -O3 -fPIC -Wall -Wextra -DPOLICY_RUNTIME_FAST_DEBUG={fast_debug}{one_tu_define} {include_flags}
LDFLAGS ?= -shared
LDLIBS ?= -ldl
TARGET = {module_name}.so
SRC = {src}

all: $(TARGET)

$(TARGET): $(SRC) model_bytes.h{extra_deps}
\t$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(SRC) $(LDLIBS)

clean:
\trm -f $(TARGET)
"""


def _player_cpp(module_name: str, *, fast_core_one_tu: bool) -> str:
    fast_core_include = '\n#include "fast_option_one_tu.cpp"\n' if fast_core_one_tu else ""
    return f"""#include "official_sdk/code/game_api.h"
#include "policy_runtime/fast_option.h"
#include "ort_include/onnxruntime_c_api.h"
#include "model_bytes.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <random>
{fast_core_include}

namespace {{

constexpr int kFeaturePlaneCount = policy_runtime::FEATURE_CHANNELS * GRID_SIZE * GRID_SIZE;
constexpr int kFeatureScalarCount = policy_runtime::FEATURE_SCALARS;
constexpr int kFastScalarCount = 2;
constexpr int kKoCount = 14;
constexpr int kVpCount = 3;
constexpr int kActionCount = 5;
constexpr int kMoveBudget = S;
constexpr int kOutputCount = 6;
constexpr int kThresholdLow = policy_runtime::FAST_THRESHOLD_LOW;
constexpr int kThresholdHigh = policy_runtime::FAST_THRESHOLD_HIGH;

GameOutput safe_output() {{
    GameOutput output = {{}};
    for (int i = 0; i < S; ++i) {{
        output.actions[i] = 4;
    }}
    output.k = 3;
    output.order = 0;
    output.vp = 0;
    return output;
}}

bool release_status(const OrtApi* api, OrtStatus* status) {{
    if (status == nullptr) {{
        return true;
    }}
    api->ReleaseStatus(status);
    return false;
}}

class SubmissionRuntime {{
public:
    SubmissionRuntime() : runtime_(1), rng_(seed_rng()) {{
        init_ort();
    }}

    ~SubmissionRuntime() {{
        release_outputs();
        if (actor_planes_) api_->ReleaseValue(actor_planes_);
        if (actor_scalars_) api_->ReleaseValue(actor_scalars_);
        if (fast_scalars_) api_->ReleaseValue(fast_scalars_);
        if (rand_ko_) api_->ReleaseValue(rand_ko_);
        if (rand_vp_) api_->ReleaseValue(rand_vp_);
        if (rand_action_) api_->ReleaseValue(rand_action_);
        if (memory_info_) api_->ReleaseMemoryInfo(memory_info_);
        if (session_) api_->ReleaseSession(session_);
        if (session_options_) api_->ReleaseSessionOptions(session_options_);
        if (env_) api_->ReleaseEnv(env_);
        if (ort_handle_) dlclose(ort_handle_);
    }}

    GameOutput decide(const GameInput* input) {{
        if (input == nullptr || !ready_) {{
            return safe_output();
        }}
        try {{
            maybe_reset_episode(*input);
            if (fast_armed_) {{
                GameOutput fast_output = {{}};
#if POLICY_RUNTIME_FAST_DEBUG || !POLICY_RUNTIME_FAST_ONE_TU
                const policy_runtime::FastStatus fast_status = runtime_.try_fast_output(*input, &fast_output);
#else
                const policy_runtime::FastStatus fast_status = runtime_.try_fast_output_release(*input, &fast_output);
#endif
                fast_armed_ = false;
                if (fast_status == policy_runtime::FastStatus::Success) {{
                    return fast_output;
                }}
            }}

            const policy_runtime::NeuralPrepareResult prepared = runtime_.prepare_neural(*input);
            const policy_runtime::FeatureOutput& features = prepared.actor_features;
            if (features.planes.size() != static_cast<std::size_t>(kFeaturePlaneCount) ||
                features.scalars.size() != static_cast<std::size_t>(kFeatureScalarCount)) {{
                return safe_output();
            }}
            std::copy(features.planes.begin(), features.planes.end(), actor_planes_data_);
            std::copy(features.scalars.begin(), features.scalars.end(), actor_scalars_data_);
            fast_scalars_data_[0] = prepared.fast_scalars[0];
            fast_scalars_data_[1] = prepared.fast_scalars[1];
            fill_random_inputs();

            release_outputs();
            const OrtValue* inputs[] = {{actor_planes_, actor_scalars_, fast_scalars_, rand_ko_, rand_vp_, rand_action_}};
            OrtStatus* status = api_->Run(
                session_,
                nullptr,
                input_names_,
                inputs,
                6,
                output_names_,
                kOutputCount,
                outputs_);
            if (!release_status(api_, status)) {{
                ready_ = false;
                return safe_output();
            }}
            const NeuralDecision decision = read_decision();
            if (!decision.valid) {{
                return safe_output();
            }}
            runtime_.commit_neural(decision.output);
            runtime_.set_next_threshold(decision.threshold_int);
            fast_armed_ = true;
            return decision.output;
        }} catch (const std::exception&) {{
            return safe_output();
        }} catch (...) {{
            return safe_output();
        }}
    }}

    bool ready() const {{
        return ready_;
    }}

private:
    static std::uint64_t seed_rng() {{
        const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::uint64_t seed = static_cast<std::uint64_t>(now);
        seed ^= static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&seed)) + 0x9e3779b97f4a7c15ULL;
        return seed;
    }}

    void maybe_reset_episode(const GameInput& input) {{
        if (!seen_round_ || input.round == 0 || input.round < last_round_) {{
            runtime_.reset(1);
            fast_armed_ = false;
        }}
        seen_round_ = true;
        last_round_ = input.round;
    }}

    void init_ort() {{
        for (const char* library : ort_libraries_) {{
            ort_handle_ = dlopen(library, RTLD_LAZY | RTLD_LOCAL);
            if (ort_handle_ != nullptr) {{
                break;
            }}
        }}
        if (ort_handle_ == nullptr) {{
            return;
        }}
        auto get_base = reinterpret_cast<const OrtApiBase* (*)()>(dlsym(ort_handle_, "OrtGetApiBase"));
        if (get_base == nullptr) {{
            return;
        }}
        const OrtApiBase* base = get_base();
        api_ = base != nullptr && base->GetApi != nullptr ? base->GetApi(8) : nullptr;
        if (api_ == nullptr) {{
            return;
        }}

        bool ok = release_status(api_, api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "{module_name}", &env_));
        if (ok) ok = release_status(api_, api_->CreateSessionOptions(&session_options_));
        if (ok) ok = release_status(api_, api_->SetIntraOpNumThreads(session_options_, 1));
        if (ok) ok = release_status(api_, api_->SetInterOpNumThreads(session_options_, 1));
        if (ok) ok = release_status(api_, api_->SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_EXTENDED));
        if (ok) {{
            ok = release_status(
                api_,
                api_->CreateSessionFromArray(env_, MODEL_BYTES, MODEL_BYTES_SIZE, session_options_, &session_));
        }}
        if (ok) {{
            ok = release_status(api_, api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_));
        }}
        if (ok) {{
            ok = create_tensor(actor_planes_data_, sizeof(actor_planes_data_), actor_planes_dims_, 4,
                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &actor_planes_);
        }}
        if (ok) {{
            ok = create_tensor(actor_scalars_data_, sizeof(actor_scalars_data_), actor_scalars_dims_, 2,
                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &actor_scalars_);
        }}
        if (ok) {{
            ok = create_tensor(fast_scalars_data_, sizeof(fast_scalars_data_), fast_scalars_dims_, 2,
                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &fast_scalars_);
        }}
        if (ok) {{
            ok = create_tensor(rand_ko_data_, sizeof(rand_ko_data_), rand_ko_dims_, 2,
                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &rand_ko_);
        }}
        if (ok) {{
            ok = create_tensor(rand_vp_data_, sizeof(rand_vp_data_), rand_vp_dims_, 2,
                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &rand_vp_);
        }}
        if (ok) {{
            ok = create_tensor(rand_action_data_, sizeof(rand_action_data_), rand_action_dims_, 3,
                               ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &rand_action_);
        }}
        ready_ = ok;
    }}

    bool create_tensor(
        void* data,
        std::size_t bytes,
        const int64_t* dims,
        std::size_t dim_count,
        ONNXTensorElementDataType dtype,
        OrtValue** value) {{
        return release_status(
            api_,
            api_->CreateTensorWithDataAsOrtValue(memory_info_, data, bytes, dims, dim_count, dtype, value));
    }}

    void fill_random_inputs() {{
        std::uniform_real_distribution<float> dist(1.0e-6F, 1.0F - 1.0e-6F);
        for (float& value : rand_ko_data_) {{
            value = dist(rng_);
        }}
        for (float& value : rand_vp_data_) {{
            value = dist(rng_);
        }}
        for (float& value : rand_action_data_) {{
            value = dist(rng_);
        }}
    }}

    void release_outputs() {{
        if (api_ == nullptr) {{
            return;
        }}
        for (OrtValue*& output : outputs_) {{
            if (output != nullptr) {{
                api_->ReleaseValue(output);
                output = nullptr;
            }}
        }}
    }}

    struct NeuralDecision {{
        GameOutput output{{}};
        int threshold_int = 12;
        bool valid = false;
    }};

    NeuralDecision read_decision() {{
        int64_t* actions = nullptr;
        int64_t* k = nullptr;
        int64_t* order = nullptr;
        int64_t* vp = nullptr;
        float* threshold_mu_raw = nullptr;
        float* threshold_log_std = nullptr;
        if (!get_output(0, &actions) || !get_output(1, &k) || !get_output(2, &order) || !get_output(3, &vp) ||
            !get_output(4, &threshold_mu_raw) || !get_output(5, &threshold_log_std)) {{
            ready_ = false;
            return NeuralDecision{{}};
        }}

        GameOutput output = {{}};
        for (int i = 0; i < S; ++i) {{
            const int value = static_cast<int>(actions[i]);
            output.actions[i] = (0 <= value && value <= 4) ? value : 4;
        }}
        output.k = (0 <= k[0] && k[0] <= 6) ? static_cast<int>(k[0]) : 3;
        output.order = (0 <= order[0] && order[0] <= 1) ? static_cast<int>(order[0]) : 0;
        output.vp = (0 <= vp[0] && vp[0] <= 2) ? static_cast<int>(vp[0]) : 0;
        NeuralDecision decision;
        decision.output = output;
        decision.threshold_int = sample_threshold_int(threshold_mu_raw[0], threshold_log_std[0]);
        decision.valid = true;
        return decision;
    }}

    bool get_output(int index, int64_t** data) {{
        if (outputs_[index] == nullptr) {{
            return false;
        }}
        OrtStatus* status = api_->GetTensorMutableData(outputs_[index], reinterpret_cast<void**>(data));
        return release_status(api_, status) && *data != nullptr;
    }}

    bool get_output(int index, float** data) {{
        if (outputs_[index] == nullptr) {{
            return false;
        }}
        OrtStatus* status = api_->GetTensorMutableData(outputs_[index], reinterpret_cast<void**>(data));
        return release_status(api_, status) && *data != nullptr;
    }}

    int sample_threshold_int(float mu_raw, float log_std) {{
        std::normal_distribution<float> dist(0.0F, 1.0F);
        const float raw = mu_raw + std::exp(log_std) * dist(rng_);
        const float exp_value = std::exp(raw >= 0.0F ? -raw : raw);
        const float sigmoid = raw >= 0.0F ? 1.0F / (1.0F + exp_value) : exp_value / (1.0F + exp_value);
        const float threshold = static_cast<float>(kThresholdLow) +
            static_cast<float>(kThresholdHigh - kThresholdLow) * sigmoid;
        int threshold_int = static_cast<int>(std::floor(threshold + 0.5F));
        if (threshold_int < kThresholdLow) {{
            threshold_int = kThresholdLow;
        }}
        if (threshold_int > kThresholdHigh) {{
            threshold_int = kThresholdHigh;
        }}
        return threshold_int;
    }}

    policy_runtime::FastRuntimeState runtime_;
    std::mt19937_64 rng_;
    bool seen_round_ = false;
    int last_round_ = -1;
    bool fast_armed_ = false;
    bool ready_ = false;

    void* ort_handle_ = nullptr;
    const OrtApi* api_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtSessionOptions* session_options_ = nullptr;
    OrtSession* session_ = nullptr;
    OrtMemoryInfo* memory_info_ = nullptr;
    OrtValue* actor_planes_ = nullptr;
    OrtValue* actor_scalars_ = nullptr;
    OrtValue* fast_scalars_ = nullptr;
    OrtValue* rand_ko_ = nullptr;
    OrtValue* rand_vp_ = nullptr;
    OrtValue* rand_action_ = nullptr;
    OrtValue* outputs_[kOutputCount] = {{}};

    float actor_planes_data_[kFeaturePlaneCount] = {{}};
    float actor_scalars_data_[kFeatureScalarCount] = {{}};
    float fast_scalars_data_[kFastScalarCount] = {{0.8F, 0.25F}};
    float rand_ko_data_[kKoCount] = {{}};
    float rand_vp_data_[kVpCount] = {{}};
    float rand_action_data_[kMoveBudget * kActionCount] = {{}};
    int64_t actor_planes_dims_[4] = {{1, policy_runtime::FEATURE_CHANNELS, GRID_SIZE, GRID_SIZE}};
    int64_t actor_scalars_dims_[2] = {{1, policy_runtime::FEATURE_SCALARS}};
    int64_t fast_scalars_dims_[2] = {{1, kFastScalarCount}};
    int64_t rand_ko_dims_[2] = {{1, kKoCount}};
    int64_t rand_vp_dims_[2] = {{1, kVpCount}};
    int64_t rand_action_dims_[3] = {{1, kMoveBudget, kActionCount}};
    const char* input_names_[6] = {{"actor_planes", "actor_scalars", "fast_scalars", "rand_ko", "rand_vp", "rand_action"}};
    const char* output_names_[kOutputCount] = {{"actions", "k", "order", "vp", "threshold_mu_raw", "threshold_log_std"}};
    const char* ort_libraries_[3] = {{"libonnxruntime.so", "libonnxruntime.so.1", "libonnxruntime.so.1.20.1"}};
}};

SubmissionRuntime& runtime() {{
    static SubmissionRuntime instance;
    return instance;
}}

}}  // namespace

extern "C" GameOutput moveDecision(const GameInput* input) {{
    return runtime().decide(input);
}}

extern "C" int policyRuntimeReady() {{
    return runtime().ready() ? 1 : 0;
}}
"""


if __name__ == "__main__":
    main()
