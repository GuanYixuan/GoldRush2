#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "policy_runtime/fast_option.h"
#include "policy_runtime/feature_extractor.h"

namespace py = pybind11;

namespace {

int int_attr(const py::object& object, const char* name) {
    return py::cast<int>(object.attr(name));
}

bool has_attr(const py::object& object, const char* name) {
    return py::hasattr(object, name);
}

Position position_from_object(const py::handle& object) {
    Position pos{-1, -1};
    if (py::hasattr(object, "row") && py::hasattr(object, "col")) {
        pos.row = py::cast<int>(object.attr("row"));
        pos.col = py::cast<int>(object.attr("col"));
        return pos;
    }
    py::sequence seq = py::reinterpret_borrow<py::sequence>(object);
    if (seq.size() != 2) {
        throw std::invalid_argument("position must have exactly 2 values");
    }
    pos.row = py::cast<int>(seq[0]);
    pos.col = py::cast<int>(seq[1]);
    return pos;
}

void fill_snapshot(Snapshot& dst, const py::object& src, bool valid) {
    dst.window_begin = valid ? 0 : -1;
    dst.window_end = valid ? 0 : -1;
    for (int i = 0; i < REGION_COUNT; ++i) {
        dst.regions[i] = RegionStat{};
    }
    if (!valid) {
        return;
    }
    if (src.is_none()) {
        throw std::invalid_argument("snapshot_valid is true but snapshot is None");
    }
    dst.window_begin = int_attr(src, "window_begin");
    dst.window_end = int_attr(src, "window_end");
    py::sequence regions = py::reinterpret_borrow<py::sequence>(src.attr("regions"));
    if (regions.size() != REGION_COUNT) {
        throw std::invalid_argument("snapshot.regions length must match REGION_COUNT");
    }
    for (int i = 0; i < REGION_COUNT; ++i) {
        py::object region = py::reinterpret_borrow<py::object>(regions[i]);
        dst.regions[i].id = int_attr(region, "id");
        dst.regions[i].enter = int_attr(region, "enter");
        dst.regions[i].leave = int_attr(region, "leave");
        dst.regions[i].gold_generated = int_attr(region, "gold_generated");
        dst.regions[i].gold_collected = int_attr(region, "gold_collected");
        dst.regions[i].gold_remaining = int_attr(region, "gold_remaining");
        dst.regions[i].occupants = int_attr(region, "occupants");
    }
}

GameInput game_input_from_python(const py::object& src) {
    GameInput dst{};
    dst.round = int_attr(src, "round");

    py::sequence grid = py::reinterpret_borrow<py::sequence>(src.attr("grid"));
    if (grid.size() != GRID_SIZE) {
        throw std::invalid_argument("grid must have GRID_SIZE rows");
    }
    for (int row = 0; row < GRID_SIZE; ++row) {
        py::sequence row_values = py::reinterpret_borrow<py::sequence>(grid[row]);
        if (row_values.size() != GRID_SIZE) {
            throw std::invalid_argument("grid row must have GRID_SIZE columns");
        }
        for (int col = 0; col < GRID_SIZE; ++col) {
            dst.grid[row][col] = py::cast<int>(row_values[col]);
        }
    }

    py::sequence my_units = py::reinterpret_borrow<py::sequence>(src.attr("my_units"));
    if (my_units.size() != 2) {
        throw std::invalid_argument("my_units length must be 2");
    }
    for (int i = 0; i < 2; ++i) {
        dst.my_units[i] = position_from_object(my_units[i]);
    }

    py::sequence my_units_gold = py::reinterpret_borrow<py::sequence>(src.attr("my_units_gold"));
    if (my_units_gold.size() != 2) {
        throw std::invalid_argument("my_units_gold length must be 2");
    }
    for (int i = 0; i < 2; ++i) {
        dst.my_units_gold[i] = py::cast<int>(my_units_gold[i]);
    }

    dst.gold_opp = int_attr(src, "gold_opp");

    py::sequence visible_enemies = py::reinterpret_borrow<py::sequence>(src.attr("visible_enemies"));
    if (visible_enemies.size() != 2) {
        throw std::invalid_argument("visible_enemies length must be 2");
    }
    for (int i = 0; i < 2; ++i) {
        dst.visible_enemies[i] = position_from_object(visible_enemies[i]);
    }

    py::sequence visible_npcs = py::reinterpret_borrow<py::sequence>(src.attr("visible_npcs"));
    if (visible_npcs.size() > MAX_NPCS) {
        throw std::invalid_argument("visible_npcs length exceeds MAX_NPCS");
    }
    dst.num_visible_npcs = static_cast<int>(visible_npcs.size());
    for (int i = 0; i < MAX_NPCS; ++i) {
        dst.visible_npcs[i].id = 0;
        dst.visible_npcs[i].pos = Position{-1, -1};
    }
    for (int i = 0; i < dst.num_visible_npcs; ++i) {
        py::object npc = py::reinterpret_borrow<py::object>(visible_npcs[i]);
        dst.visible_npcs[i].id = int_attr(npc, "id");
        if (has_attr(npc, "pos")) {
            dst.visible_npcs[i].pos = position_from_object(npc.attr("pos"));
        } else {
            dst.visible_npcs[i].pos = Position{int_attr(npc, "row"), int_attr(npc, "col")};
        }
    }

    dst.snapshot_valid = py::cast<bool>(src.attr("snapshot_valid")) ? 1 : 0;
    fill_snapshot(dst.snapshot, py::reinterpret_borrow<py::object>(src.attr("snapshot")), dst.snapshot_valid != 0);
    return dst;
}

GameOutput game_output_from_python(const py::object& src) {
    GameOutput dst{};
    if (py::hasattr(src, "actions")) {
        py::sequence actions = py::reinterpret_borrow<py::sequence>(src.attr("actions"));
        if (actions.size() != S) {
            throw std::invalid_argument("actions length must be S");
        }
        for (int i = 0; i < S; ++i) {
            dst.actions[i] = py::cast<int>(actions[i]);
        }
        dst.k = int_attr(src, "k");
        dst.order = int_attr(src, "order");
        dst.vp = int_attr(src, "vp");
        return dst;
    }

    py::sequence values = py::reinterpret_borrow<py::sequence>(src);
    if (values.size() != S + 3) {
        throw std::invalid_argument("GameOutput sequence must have 9 values");
    }
    for (int i = 0; i < S; ++i) {
        dst.actions[i] = py::cast<int>(values[i]);
    }
    dst.k = py::cast<int>(values[S]);
    dst.order = py::cast<int>(values[S + 1]);
    dst.vp = py::cast<int>(values[S + 2]);
    return dst;
}

py::dict feature_output_to_python(const policy_runtime::FeatureOutput& output) {
    py::array_t<float> planes({output.channels, output.height, output.width});
    py::array_t<float> scalars({static_cast<py::ssize_t>(output.scalars.size())});

    auto planes_view = planes.mutable_unchecked<3>();
    for (int channel = 0; channel < output.channels; ++channel) {
        for (int row = 0; row < output.height; ++row) {
            for (int col = 0; col < output.width; ++col) {
                const int flat = (channel * output.height + row) * output.width + col;
                planes_view(channel, row, col) = output.planes[flat];
            }
        }
    }

    auto scalars_view = scalars.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(output.scalars.size()); ++i) {
        scalars_view(i) = output.scalars[static_cast<std::size_t>(i)];
    }

    py::dict result;
    result["feature_schema"] = policy_runtime::feature_schema();
    result["planes"] = planes;
    result["scalars"] = scalars;
    result["channel_names"] = policy_runtime::channel_names();
    result["scalar_names"] = policy_runtime::scalar_names();
    return result;
}

py::dict game_output_to_python(const GameOutput& output) {
    py::tuple actions(S);
    for (int i = 0; i < S; ++i) {
        actions[i] = output.actions[i];
    }
    py::dict result;
    result["actions"] = actions;
    result["k"] = output.k;
    result["order"] = output.order;
    result["vp"] = output.vp;
    return result;
}

py::dict fast_diagnostics_to_python(const policy_runtime::FastDiagnostics& diagnostics) {
    py::dict result;
    result["fast_success"] = diagnostics.fast_success;
    result["fast_miss_no_target"] = diagnostics.fast_miss_no_target;
    result["fast_path_fail"] = diagnostics.fast_path_fail;
    result["neural_fallback"] = diagnostics.neural_fallback;
    result["fast_nonstay"] = diagnostics.fast_nonstay;
    result["fast_effective_updates"] = diagnostics.fast_effective_updates;
    result["fast_effective_score_sum"] = diagnostics.fast_effective_score_sum;
    result["fast_expected_gain_sum"] = diagnostics.fast_expected_gain_sum;
    result["fast_actual_delta_sum"] = diagnostics.fast_actual_delta_sum;
    result["one_step_fast_delta_sum"] = diagnostics.one_step_fast_delta_sum;
    result["fast_effective_positive_updates"] = diagnostics.fast_effective_positive_updates;
    result["fast_effective_negative_updates"] = diagnostics.fast_effective_negative_updates;
    result["fast_effective_skipped_success_no_new_npc"] = diagnostics.fast_effective_skipped_success_no_new_npc;
    return result;
}

const char* fast_status_name(policy_runtime::FastStatus status) {
    switch (status) {
        case policy_runtime::FastStatus::Success:
            return "success";
        case policy_runtime::FastStatus::MissNoTarget:
            return "miss_no_target";
        case policy_runtime::FastStatus::PathFail:
            return "path_fail";
    }
    return "unknown";
}

py::dict fast_try_result_to_python(const policy_runtime::FastTryResult& result) {
    py::dict payload;
    payload["status"] = fast_status_name(result.status);
    payload["success"] = result.status == policy_runtime::FastStatus::Success;
    payload["output"] = game_output_to_python(result.output);
    payload["role"] = result.role;
    payload["target"] = py::make_tuple(result.target.row, result.target.col);
    payload["action_count"] = result.action_count;
    return payload;
}

py::dict neural_prepare_result_to_python(const policy_runtime::NeuralPrepareResult& result) {
    py::dict payload;
    payload["actor_features"] = feature_output_to_python(result.actor_features);
    payload["fast_scalars"] = py::make_tuple(result.fast_scalars[0], result.fast_scalars[1]);
    payload["diagnostics"] = fast_diagnostics_to_python(result.diagnostics);
    return payload;
}

}  // namespace

PYBIND11_MODULE(_runtime, module) {
    module.doc() = "GoldRush2 policy runtime C++ bindings";
    module.def("feature_schema", &policy_runtime::feature_schema);
    module.def("channel_names", &policy_runtime::channel_names);
    module.def("scalar_names", &policy_runtime::scalar_names);

    py::class_<policy_runtime::FeatureExtractor>(module, "FeatureExtractor")
        .def(py::init<int>(), py::arg("player_id") = 1)
        .def("reset", &policy_runtime::FeatureExtractor::reset, py::arg("player_id"))
        .def(
            "observe",
            [](policy_runtime::FeatureExtractor& extractor, const py::object& game_input) {
                return feature_output_to_python(extractor.observe(game_input_from_python(game_input)));
            },
            py::arg("game_input"))
        .def(
            "commit_action",
            [](policy_runtime::FeatureExtractor& extractor, const py::object& game_output) {
                extractor.commit_action(game_output_from_python(game_output));
            },
            py::arg("game_output"));

    py::class_<policy_runtime::FastRuntimeState>(module, "FastRuntimeState")
        .def(py::init<int>(), py::arg("player_id") = 1)
        .def("reset", &policy_runtime::FastRuntimeState::reset, py::arg("player_id"))
        .def("fast_scalars", &policy_runtime::FastRuntimeState::fast_scalars)
        .def("threshold_int", &policy_runtime::FastRuntimeState::threshold_int)
        .def("pending_valid", &policy_runtime::FastRuntimeState::pending_valid)
        .def(
            "diagnostics",
            [](const policy_runtime::FastRuntimeState& state) {
                return fast_diagnostics_to_python(state.diagnostics());
            })
        .def("set_next_threshold", &policy_runtime::FastRuntimeState::set_next_threshold, py::arg("threshold_int"))
        .def(
            "try_fast",
            [](policy_runtime::FastRuntimeState& state, const py::object& game_input) {
                return fast_try_result_to_python(state.try_fast(game_input_from_python(game_input)));
            },
            py::arg("game_input"))
        .def(
            "prepare_neural",
            [](policy_runtime::FastRuntimeState& state, const py::object& game_input) {
                return neural_prepare_result_to_python(state.prepare_neural(game_input_from_python(game_input)));
            },
            py::arg("game_input"))
        .def(
            "commit_neural",
            [](policy_runtime::FastRuntimeState& state, const py::object& game_output) {
                state.commit_neural(game_output_from_python(game_output));
            },
            py::arg("game_output"));

    module.def(
        "try_fast_gold_grab",
        [](const py::object& game_input, int threshold_int) {
            GameOutput output{};
            policy_runtime::FastTryResult result{};
            policy_runtime::try_fast_gold_grab(game_input_from_python(game_input), threshold_int, &output, &result);
            return fast_try_result_to_python(result);
        },
        py::arg("game_input"),
        py::arg("threshold_int"));
    module.def(
        "simulate_known_gold_pickups",
        [](const py::object& game_input, const py::object& game_output, int role) {
            return policy_runtime::simulate_known_gold_pickups(
                game_input_from_python(game_input),
                game_output_from_python(game_output),
                role);
        },
        py::arg("game_input"),
        py::arg("game_output"),
        py::arg("role"));
    module.def(
        "infer_fast_role",
        [](const py::object& game_input, const py::object& game_output) {
            return policy_runtime::infer_fast_role(game_input_from_python(game_input), game_output_from_python(game_output));
        },
        py::arg("game_input"),
        py::arg("game_output"));
}
