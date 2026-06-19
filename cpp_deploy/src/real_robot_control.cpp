// 真实机器人闭环控制程序
// 默认只做控制 dry-run；只有显式 --execute 1 --confirm RUN_ROBOT 才向 UR 发送 servoL 指令。
#include "mp1_deploy/policy_runtime.hpp"
#include "mp1_deploy/safety_filter.hpp"

#ifdef MP1_WITH_UR_RTDE
#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#endif

#include <c10/core/ScalarType.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct WorkspaceBounds {
    std::array<std::optional<double>, 3> min{};
    std::array<std::optional<double>, 3> max{};
};

struct ControlOptions {
    std::string model_path;
    std::filesystem::path input_dir;
    std::filesystem::path config_path;
    std::string device_name = "cuda";
    std::string robot_ip;
    int steps = 10;
    int warmup_steps = 3;
    int poll_ms = 50;
    bool require_update = true;
    bool execute = false;
    double control_hz = 2.0;
    double max_translation_m = 0.0005;
    double max_rotation_rad = 0.003;
    double servo_speed = 0.05;
    double servo_acceleration = 0.10;
    double servo_lookahead = 0.10;
    double servo_gain = 100.0;
    WorkspaceBounds workspace;
};

std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key.rfind("--", 0) != 0) {
            continue;
        }
        if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0) {
            args[key.substr(2)] = "1";
        } else {
            args[key.substr(2)] = argv[++i];
        }
    }
    return args;
}

std::string get_arg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    const std::string& fallback = "") {
    const auto it = args.find(key);
    return it == args.end() ? fallback : it->second;
}

int get_int(const std::unordered_map<std::string, std::string>& args, const std::string& key, int fallback) {
    const auto value = get_arg(args, key);
    return value.empty() ? fallback : std::stoi(value);
}

double get_double(const std::unordered_map<std::string, std::string>& args, const std::string& key, double fallback) {
    const auto value = get_arg(args, key);
    return value.empty() ? fallback : std::stod(value);
}

bool get_bool(const std::unordered_map<std::string, std::string>& args, const std::string& key, bool fallback) {
    return get_int(args, key, fallback ? 1 : 0) != 0;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::string extract_json_string(const std::string& text, const std::string& key) {
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return "";
    }
    return match[1].str();
}

std::vector<std::string> split_array_items(const std::string& array_body) {
    std::vector<std::string> items;
    std::stringstream stream(array_body);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const auto first = item.find_first_not_of(" \n\r\t");
        if (first == std::string::npos) {
            items.push_back("");
            continue;
        }
        item.erase(0, first);
        item.erase(item.find_last_not_of(" \n\r\t") + 1);
        items.push_back(item);
    }
    return items;
}

std::array<std::optional<double>, 3> extract_json_optional_vec3(const std::string& text, const std::string& key) {
    std::array<std::optional<double>, 3> values{};
    const std::regex pattern("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
        return values;
    }

    const auto items = split_array_items(match[1].str());
    for (size_t i = 0; i < items.size() && i < values.size(); ++i) {
        if (items[i] == "null" || items[i].empty()) {
            continue;
        }
        values[i] = std::stod(items[i]);
    }
    return values;
}

std::array<std::optional<double>, 3> parse_optional_vec3_arg(const std::string& text) {
    std::array<std::optional<double>, 3> values{};
    const auto items = split_array_items(text);
    for (size_t i = 0; i < items.size() && i < values.size(); ++i) {
        if (items[i] == "nan" || items[i] == "null" || items[i].empty()) {
            continue;
        }
        values[i] = std::stod(items[i]);
    }
    return values;
}

torch::Tensor load_tensor(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Missing tensor file: " + path.string());
    }
    torch::jit::script::Module module = torch::jit::load(path.string(), torch::kCPU);
    module.eval();
    return module.forward({}).toTensor();
}

std::filesystem::path resolve_committed_frame_dir(const std::filesystem::path& root) {
    const auto manifest = root / "current_frame.txt";
    if (!std::filesystem::exists(manifest)) {
        return root;
    }

    std::ifstream file(manifest);
    std::string relative_dir;
    std::getline(file, relative_dir);
    if (relative_dir.empty()) {
        throw std::runtime_error("current_frame.txt is empty: " + manifest.string());
    }

    const auto frame_dir = root / relative_dir;
    if (!std::filesystem::is_directory(frame_dir)) {
        throw std::runtime_error("Committed frame directory is missing: " + frame_dir.string());
    }
    return frame_dir;
}

mp1_deploy::PolicyInputs load_real_input_dir(const std::filesystem::path& dir) {
    return {
        load_tensor(dir / "global_image.pt"),
        load_tensor(dir / "wrist_image.pt"),
        load_tensor(dir / "point_cloud.pt"),
        load_tensor(dir / "agent_pos.pt"),
        load_tensor(dir / "initial_noise.pt"),
    };
}

void validate_inputs(const mp1_deploy::PolicyInputs& inputs) {
    const auto expect_dim = [](const torch::Tensor& tensor, int64_t dim, const std::string& name) {
        if (tensor.dim() != dim) {
            throw std::runtime_error(name + " dim mismatch");
        }
    };
    expect_dim(inputs.global_image, 5, "global_image");
    expect_dim(inputs.wrist_image, 5, "wrist_image");
    expect_dim(inputs.point_cloud, 4, "point_cloud");
    expect_dim(inputs.agent_pos, 3, "agent_pos");
    expect_dim(inputs.initial_noise, 3, "initial_noise");

    if (inputs.agent_pos.size(-1) != 10) {
        throw std::runtime_error("agent_pos must be 10D for pole pickoff tasks");
    }
    if (inputs.initial_noise.size(-1) != 7) {
        throw std::runtime_error("initial_noise must be 7D for delta_tcp_pose_gripper policy");
    }
}

std::filesystem::file_time_type newest_input_mtime(const std::filesystem::path& dir) {
    const auto manifest = dir / "current_frame.txt";
    if (std::filesystem::exists(manifest)) {
        return std::filesystem::last_write_time(manifest);
    }

    std::filesystem::file_time_type newest{};
    for (const auto& name : {"global_image.pt", "wrist_image.pt", "point_cloud.pt", "agent_pos.pt", "initial_noise.pt"}) {
        const auto path = dir / name;
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("Missing input tensor: " + path.string());
        }
        newest = std::max(newest, std::filesystem::last_write_time(path));
    }
    return newest;
}

int default_warmup_steps(const std::string& device_name) {
    return device_name.find("cuda") != std::string::npos ? 3 : 0;
}

std::vector<double> tensor_to_delta6(const torch::Tensor& action) {
    const torch::Tensor cpu = action.detach().to(torch::kCPU).to(torch::kFloat64).flatten();
    if (cpu.size(0) < 6) {
        throw std::runtime_error("filtered action must have at least 6 values");
    }

    std::vector<double> delta(6, 0.0);
    for (int64_t i = 0; i < 6; ++i) {
        delta[static_cast<size_t>(i)] = cpu[i].item<double>();
    }
    return delta;
}

std::vector<double> make_target_pose(const std::vector<double>& current_tcp, const std::vector<double>& delta_tcp) {
    if (current_tcp.size() != 6 || delta_tcp.size() != 6) {
        throw std::runtime_error("TCP pose and delta must both be 6D");
    }
    std::vector<double> target(6, 0.0);
    for (size_t i = 0; i < 6; ++i) {
        // 小限幅阶段使用 base frame 下的加法近似；放大动作前应改为严格 SE(3) 组合。
        target[i] = current_tcp[i] + delta_tcp[i];
    }
    return target;
}

bool clamp_workspace(std::vector<double>& target_tcp, const WorkspaceBounds& workspace) {
    bool clamped = false;
    for (size_t i = 0; i < 3; ++i) {
        if (workspace.min[i].has_value() && target_tcp[i] < workspace.min[i].value()) {
            target_tcp[i] = workspace.min[i].value();
            clamped = true;
        }
        if (workspace.max[i].has_value() && target_tcp[i] > workspace.max[i].value()) {
            target_tcp[i] = workspace.max[i].value();
            clamped = true;
        }
    }
    return clamped;
}

void print_vec6(const std::string& name, const std::vector<double>& values) {
    std::cout << "  " << name << ": [";
    for (size_t i = 0; i < values.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << values[i];
        if (i + 1 != values.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

mp1_deploy::SafetyFilterOptions make_safety_options(const ControlOptions& options) {
    mp1_deploy::SafetyFilterOptions filter_options;
    filter_options.max_translation_m = options.max_translation_m;
    filter_options.max_rotation_rad = options.max_rotation_rad;
    filter_options.ignore_gripper_action = true;
    return filter_options;
}

ControlOptions parse_options(int argc, char** argv) {
    const auto args = parse_args(argc, argv);
    ControlOptions options;
    options.model_path = get_arg(args, "model");
    options.input_dir = get_arg(args, "input-dir");
    options.config_path = get_arg(args, "config", "../cpp_deploy/configs/pole_pickoff_real_robot.json");
    options.device_name = get_arg(args, "device", "cuda");
    options.steps = get_int(args, "steps", 10);
    options.warmup_steps = get_int(args, "warmup-steps", default_warmup_steps(options.device_name));
    options.poll_ms = get_int(args, "poll-ms", 50);
    options.require_update = get_bool(args, "require-update", true);
    options.execute = get_bool(args, "execute", false);
    options.control_hz = get_double(args, "control-hz", 2.0);
    options.max_translation_m = get_double(args, "max-translation", 0.0005);
    options.max_rotation_rad = get_double(args, "max-rotation", 0.003);
    options.servo_speed = get_double(args, "servo-speed", options.servo_speed);
    options.servo_acceleration = get_double(args, "servo-acceleration", options.servo_acceleration);
    options.servo_lookahead = get_double(args, "servo-lookahead", options.servo_lookahead);
    options.servo_gain = get_double(args, "servo-gain", options.servo_gain);

    if (options.model_path.empty() || options.input_dir.empty()) {
        throw std::runtime_error("--model and --input-dir are required");
    }
    if (options.steps < 0 || options.warmup_steps < 0 || options.poll_ms < 0) {
        throw std::runtime_error("--steps, --warmup-steps and --poll-ms must be >= 0");
    }
    if (options.control_hz <= 0.0) {
        throw std::runtime_error("--control-hz must be > 0");
    }

    const std::string config_text = std::filesystem::exists(options.config_path)
        ? read_text_file(options.config_path)
        : "";
    options.robot_ip = get_arg(args, "robot-ip", extract_json_string(config_text, "ip"));
    options.workspace.min = extract_json_optional_vec3(config_text, "workspace_min");
    options.workspace.max = extract_json_optional_vec3(config_text, "workspace_max");
    const auto workspace_min_arg = get_arg(args, "workspace-min");
    const auto workspace_max_arg = get_arg(args, "workspace-max");
    if (!workspace_min_arg.empty()) {
        options.workspace.min = parse_optional_vec3_arg(workspace_min_arg);
    }
    if (!workspace_max_arg.empty()) {
        options.workspace.max = parse_optional_vec3_arg(workspace_max_arg);
    }

    if (options.execute) {
        if (get_arg(args, "confirm") != "RUN_ROBOT") {
            throw std::runtime_error("Real robot execution requires --confirm RUN_ROBOT");
        }
        if (options.robot_ip.empty()) {
            throw std::runtime_error("Real robot execution requires robot.ip in config or --robot-ip");
        }
    }
    return options;
}

void print_usage() {
    std::cout
        << "Usage: mp1_real_robot_control --model policy_infer.pt --input-dir real_input_tensors\n"
        << "       [--config pole_pickoff_real_robot.json] [--device cuda] [--steps 10]\n"
        << "       [--warmup-steps 3] [--control-hz 2] [--require-update 1]\n"
        << "       [--max-translation 0.0005] [--max-rotation 0.003]\n"
        << "       [--execute 0] [--confirm RUN_ROBOT]\n";
}

#ifdef MP1_WITH_UR_RTDE
class UrRtdeClient {
public:
    explicit UrRtdeClient(const std::string& robot_ip)
        : control_(robot_ip), receive_(robot_ip) {}

    ~UrRtdeClient() {
        try {
            stop();
        } catch (...) {
            // 析构阶段不能继续抛异常；显式 stop() 的错误会在主流程里暴露。
        }
    }

    std::vector<double> current_tcp() {
        return receive_.getActualTCPPose();
    }

    void servo_l(
        const std::vector<double>& target_tcp,
        double speed,
        double acceleration,
        double dt,
        double lookahead,
        double gain) {
        control_.servoL(target_tcp, speed, acceleration, dt, lookahead, gain);
    }

    void stop() {
        if (stopped_) {
            return;
        }
        control_.servoStop();
        control_.stopScript();
        stopped_ = true;
    }

private:
    ur_rtde::RTDEControlInterface control_;
    ur_rtde::RTDEReceiveInterface receive_;
    bool stopped_ = false;
};
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        const ControlOptions options = parse_options(argc, argv);
        if (!std::filesystem::is_directory(options.input_dir)) {
            throw std::runtime_error("--input-dir is not a directory: " + options.input_dir.string());
        }

        if (options.execute) {
#ifndef MP1_WITH_UR_RTDE
            throw std::runtime_error("This binary was built without UR RTDE support. Reconfigure with -DMP1_ENABLE_UR_RTDE=ON.");
#endif
        }

        mp1_deploy::TorchScriptRuntime runtime(options.model_path, torch::Device(options.device_name));
        mp1_deploy::SafetyFilter safety_filter(make_safety_options(options));

        std::cout << "real robot control start\n";
        std::cout << "  execute=" << (options.execute ? "1" : "0")
                  << ", device=" << options.device_name
                  << ", steps=" << options.steps
                  << ", control_hz=" << options.control_hz
                  << ", warmup_steps=" << options.warmup_steps << "\n";
        std::cout << "  max_translation=" << options.max_translation_m
                  << ", max_rotation=" << options.max_rotation_rad << "\n";
        if (!options.execute) {
            std::cout << "  dry-run mode: no robot command will be sent.\n";
        }

        if (options.warmup_steps > 0) {
            const auto warmup_frame_dir = resolve_committed_frame_dir(options.input_dir);
            const mp1_deploy::PolicyInputs warmup_inputs = load_real_input_dir(warmup_frame_dir);
            validate_inputs(warmup_inputs);
            std::cout << "warmup start: frame_dir=" << warmup_frame_dir.string() << "\n";
            for (int i = 0; i < options.warmup_steps; ++i) {
                const auto begin = Clock::now();
                auto [action, action_pred] = runtime.infer(warmup_inputs);
                (void)action;
                (void)action_pred;
                const auto end = Clock::now();
                const double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - begin).count();
                std::cout << "  warmup " << i << " inference_ms: " << std::fixed << std::setprecision(3) << ms << "\n";
            }
            std::cout << "warmup finished, official control steps start now.\n";
        }

#ifdef MP1_WITH_UR_RTDE
        std::optional<UrRtdeClient> robot;
        if (options.execute) {
            robot.emplace(options.robot_ip);
        }
#endif

        const auto control_period = std::chrono::duration<double>(1.0 / options.control_hz);
        std::filesystem::file_time_type last_mtime{};
        int step = 0;
        while (options.steps == 0 || step < options.steps) {
            const auto loop_begin = Clock::now();
            const auto current_mtime = newest_input_mtime(options.input_dir);
            if (options.require_update && step > 0 && current_mtime <= last_mtime) {
                std::this_thread::sleep_for(std::chrono::milliseconds(options.poll_ms));
                continue;
            }
            last_mtime = current_mtime;

            const auto frame_dir = resolve_committed_frame_dir(options.input_dir);
            const auto load_begin = Clock::now();
            const mp1_deploy::PolicyInputs inputs = load_real_input_dir(frame_dir);
            validate_inputs(inputs);
            const auto infer_begin = Clock::now();
            auto [action, action_pred] = runtime.infer(inputs);
            const auto infer_end = Clock::now();

            const torch::Tensor first_action = action.select(0, 0).select(0, 0).to(torch::kFloat32);
            const torch::Tensor filtered_action = safety_filter.apply(first_action);
            const std::vector<double> delta_tcp = tensor_to_delta6(filtered_action);

            std::vector<double> current_tcp(6, 0.0);
            std::vector<double> target_tcp(6, 0.0);
            bool workspace_clamped = false;

#ifdef MP1_WITH_UR_RTDE
            if (options.execute) {
                current_tcp = robot->current_tcp();
                target_tcp = make_target_pose(current_tcp, delta_tcp);
                workspace_clamped = clamp_workspace(target_tcp, options.workspace);
                robot->servo_l(
                    target_tcp,
                    options.servo_speed,
                    options.servo_acceleration,
                    1.0 / options.control_hz,
                    options.servo_lookahead,
                    options.servo_gain);
            }
#endif

            const auto command_end = Clock::now();
            const double load_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(infer_begin - load_begin).count();
            const double infer_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(infer_end - infer_begin).count();
            const double command_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(command_end - infer_end).count();

            std::cout << "\nstep " << step << "\n";
            std::cout << "  frame_dir: " << frame_dir.string() << "\n";
            std::cout << "  load_ms: " << std::fixed << std::setprecision(3) << load_ms
                      << ", inference_ms: " << infer_ms
                      << ", command_ms: " << command_ms << "\n";
            print_vec6("delta_tcp", delta_tcp);
            if (options.execute) {
                print_vec6("current_tcp", current_tcp);
                print_vec6("target_tcp", target_tcp);
                std::cout << "  workspace_clamped: " << (workspace_clamped ? "1" : "0") << "\n";
            }

            ++step;
            const auto elapsed = Clock::now() - loop_begin;
            if (elapsed < control_period) {
                std::this_thread::sleep_for(control_period - elapsed);
            }
        }

#ifdef MP1_WITH_UR_RTDE
        if (options.execute) {
            robot->stop();
        }
#endif
        std::cout << "\nreal robot control finished.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "mp1_real_robot_control failed: " << error.what() << "\n";
        print_usage();
        return EXIT_FAILURE;
    }
}
