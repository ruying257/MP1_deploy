// 策略推理干运行工具 - 用于验证模型推理流程，不向真实机器人发送指令
// 添加安全过滤器

#include "mp1_deploy/policy_runtime.hpp"
#include "mp1_deploy/safety_filter.hpp"

#include <c10/core/ScalarType.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

/// 解析命令行参数
/// 支持格式: --key value
std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        // 检查是否以 -- 开头
        if (key.rfind("--", 0) != 0) {
            throw std::runtime_error("Unexpected positional argument: " + key);
        }
        // 检查是否有对应的值
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + key);
        }
        // 存储参数（去除 -- 前缀）
        args[key.substr(2)] = argv[++i];
    }
    return args;
}

/// 获取字符串类型的命令行参数
std::string get_arg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    const std::string& default_value = "") {
    const auto iter = args.find(key);
    return iter == args.end() ? default_value : iter->second;
}

/// 获取整数类型的命令行参数
int get_int(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    int default_value) {
    const auto iter = args.find(key);
    return iter == args.end() ? default_value : std::stoi(iter->second);
}

/// 获取双精度浮点类型的命令行参数
double get_double(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    double default_value) {
    const auto iter = args.find(key);
    return iter == args.end() ? default_value : std::stod(iter->second);
}

/// 从文件加载 PyTorch 张量
/// 假设张量以 TorchScript Module 格式保存
torch::Tensor load_tensor(const std::string& path) {
    torch::jit::script::Module module = torch::jit::load(path, torch::kCPU);
    return module.forward({}).toTensor();
}

/// 创建零值输入数据（用于无真实数据时的测试）
mp1_deploy::PolicyInputs make_zero_inputs() {
    return {
        torch::zeros({1, 2, 3, 128, 128}, torch::kUInt8),  // 全局图像 [batch, views, channels, H, W]
        torch::zeros({1, 2, 3, 96, 96}, torch::kUInt8),    // 腕部图像 [batch, views, channels, H, W]
        torch::zeros({1, 2, 512, 3}, torch::kFloat32),     // 点云 [batch, views, points, dims]
        torch::zeros({1, 2, 10}, torch::kFloat32),         // 智能体位置 [batch, views, features]
        torch::zeros({1, 4, 7}, torch::kFloat32),          // 初始噪声 [batch, samples, dims]
    };
}

/// 从目录加载输入张量数据
mp1_deploy::PolicyInputs load_tensor_dir(const std::string& dir) {
    return {
        load_tensor(dir + "/global_image.pt"),
        load_tensor(dir + "/wrist_image.pt"),
        load_tensor(dir + "/point_cloud.pt"),
        load_tensor(dir + "/agent_pos.pt"),
        load_tensor(dir + "/initial_noise.pt"),
    };
}

/// 格式化张量形状为字符串
std::string format_shape(const std::vector<int64_t>& shape) {
    std::ostringstream stream;
    stream << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            stream << ", ";
        }
        stream << shape[i];
    }
    stream << "]";
    return stream.str();
}

/// 获取数据类型名称
std::string dtype_name(c10::ScalarType dtype) {
    return std::string(c10::toString(dtype));
}

/// 检查张量形状是否符合预期
void require_shape(const torch::Tensor& tensor, const std::string& name, const std::vector<int64_t>& expected) {
    if (tensor.sizes().vec() != expected) {
        std::ostringstream message;
        message << name << " shape mismatch, expected " << format_shape(expected) << ", got " << tensor.sizes();
        throw std::runtime_error(message.str());
    }
}

/// 检查张量数据类型是否符合预期
void require_dtype(const torch::Tensor& tensor, const std::string& name, c10::ScalarType expected) {
    if (tensor.scalar_type() != expected) {
        std::ostringstream message;
        message << name << " dtype mismatch, expected " << dtype_name(expected)
                << ", got " << dtype_name(tensor.scalar_type());
        throw std::runtime_error(message.str());
    }
}

/// 验证输入数据的形状和类型是否符合模型契约
/// 这里检查的是模型契约：真实相机/机器人链路最终也必须组出这些形状和类型
void validate_inputs(const mp1_deploy::PolicyInputs& inputs) {
    // 验证形状
    require_shape(inputs.global_image, "global_image", {1, 2, 3, 128, 128});
    require_shape(inputs.wrist_image, "wrist_image", {1, 2, 3, 96, 96});
    require_shape(inputs.point_cloud, "point_cloud", {1, 2, 512, 3});
    require_shape(inputs.agent_pos, "agent_pos", {1, 2, 10});
    require_shape(inputs.initial_noise, "initial_noise", {1, 4, 7});

    // 验证数据类型
    require_dtype(inputs.global_image, "global_image", torch::kUInt8);
    require_dtype(inputs.wrist_image, "wrist_image", torch::kUInt8);
    require_dtype(inputs.point_cloud, "point_cloud", torch::kFloat32);
    require_dtype(inputs.agent_pos, "agent_pos", torch::kFloat32);
    require_dtype(inputs.initial_noise, "initial_noise", torch::kFloat32);
}

/// 打印输入数据摘要信息
void print_input_summary(const mp1_deploy::PolicyInputs& inputs) {
    std::cout << "input summary\n";
    std::cout << "  global_image: " << inputs.global_image.sizes() << ", " << dtype_name(inputs.global_image.scalar_type()) << "\n";
    std::cout << "  wrist_image:  " << inputs.wrist_image.sizes() << ", " << dtype_name(inputs.wrist_image.scalar_type()) << "\n";
    std::cout << "  point_cloud:  " << inputs.point_cloud.sizes() << ", " << dtype_name(inputs.point_cloud.scalar_type()) << "\n";
    std::cout << "  agent_pos:    " << inputs.agent_pos.sizes() << ", " << dtype_name(inputs.agent_pos.scalar_type()) << "\n";
    std::cout << "  initial_noise:" << inputs.initial_noise.sizes() << ", " << dtype_name(inputs.initial_noise.scalar_type()) << "\n";
}

/// 根据命令行参数创建安全滤波器选项
mp1_deploy::SafetyFilterOptions make_safety_options(const std::unordered_map<std::string, std::string>& args) {
    mp1_deploy::SafetyFilterOptions options;
    options.max_translation_m = get_double(args, "max-translation", 0.002);    // 最大平移量（米）
    options.max_rotation_rad = get_double(args, "max-rotation", 0.01);        // 最大旋转量（弧度）
    options.translation_deadband_m = get_double(args, "translation-deadband", options.translation_deadband_m);  // 平移死区
    options.rotation_deadband_rad = get_double(args, "rotation-deadband", options.rotation_deadband_rad);      // 旋转死区
    options.gripper_deadband = get_double(args, "gripper-deadband", options.gripper_deadband);                // 夹爪死区
    options.ignore_gripper_action = get_int(args, "ignore-gripper-action", 1) != 0;                           // 默认不控制夹爪
    return options;
}

/// 打印命令行使用说明
void print_usage() {
    std::cout
        << "Usage: mp1_dry_run --model policy_infer.pt [--tensor-dir sample_tensors] [--device cpu] [--steps 5]\n"
        << "       [--warmup-steps 3] [--max-translation 0.002] [--max-rotation 0.01] [--ignore-gripper-action 1]\n";
}

/// CUDA 默认预热几次，避免把上下文初始化和 kernel 选择耗时算进正式 step
int default_warmup_steps(const std::string& device_name) {
    return device_name.find("cuda") != std::string::npos ? 3 : 0;
}

}  // namespace

/// 主函数：执行策略推理干运行
int main(int argc, char** argv) {
    try {
        // 解析命令行参数
        const auto args = parse_args(argc, argv);
        
        // 获取必需参数：模型路径
        const std::string model_path = get_arg(args, "model");
        if (model_path.empty()) {
            print_usage();
            throw std::runtime_error("--model is required");
        }

        // 获取可选参数
        const std::string tensor_dir = get_arg(args, "tensor-dir");   // 张量数据目录（可选）
        const std::string device_name = get_arg(args, "device", "cpu"); // 推理设备（默认CPU）
        const int steps = get_int(args, "steps", 5);                  // 推理步数（默认5步）
        const int warmup_steps = get_int(args, "warmup-steps", default_warmup_steps(device_name)); // 正式计步前的预热次数
        
        // 验证步数参数
        if (steps <= 0) {
            throw std::runtime_error("--steps must be > 0");
        }
        if (warmup_steps < 0) {
            throw std::runtime_error("--warmup-steps must be >= 0");
        }

        // 加载或生成输入数据
        const mp1_deploy::PolicyInputs inputs = tensor_dir.empty() ? make_zero_inputs() : load_tensor_dir(tensor_dir);
        
        // 验证输入数据格式
        validate_inputs(inputs);
        print_input_summary(inputs);

        // 初始化运行时组件
        // dry-run 只做推理和限幅打印，绝不向 UR 或夹爪发送命令
        mp1_deploy::TorchScriptRuntime runtime(model_path, torch::Device(device_name));
        mp1_deploy::SafetyFilter safety_filter(make_safety_options(args));

        // CUDA 预热：只跑模型 forward，不打印动作，也不更新安全滤波器状态。
        if (warmup_steps > 0) {
            std::cout << "warmup start: warmup_steps=" << warmup_steps << "\n";
            for (int warmup = 0; warmup < warmup_steps; ++warmup) {
                const auto begin = Clock::now();
                auto [warmup_action, warmup_action_pred] = runtime.infer(inputs);
                (void)warmup_action;
                (void)warmup_action_pred;
                const auto end = Clock::now();
                const double warmup_ms =
                    std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - begin).count();
                std::cout << "  warmup " << warmup
                          << " inference_ms: " << std::fixed << std::setprecision(3) << warmup_ms << "\n";
            }
            std::cout << "warmup finished, official steps start now.\n";
        }

        // 执行干运行循环
        std::cout << "dry-run start: device=" << device_name
                  << ", steps=" << steps
                  << ", warmup_steps=" << warmup_steps << "\n";
        for (int step = 0; step < steps; ++step) {
            // 记录推理开始时间
            const auto begin = Clock::now();
            
            // 执行策略推理
            auto [action, action_pred] = runtime.infer(inputs);
            
            // 记录推理结束时间
            const auto end = Clock::now();

            // 提取第一个样本的第一个视角的动作
            const torch::Tensor first_action = action.select(0, 0).select(0, 0).to(torch::kFloat32);
            
            // 应用安全限幅
            const torch::Tensor filtered_action = safety_filter.apply(first_action);
            
            // 计算推理耗时
            const double elapsed_ms =
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - begin).count();

            // 打印步骤结果
            std::cout << "\nstep " << step << "\n";
            std::cout << "  inference_ms: " << std::fixed << std::setprecision(3) << elapsed_ms << "\n";
            std::cout << "  action shape: " << action.sizes() << ", action_pred shape: " << action_pred.sizes() << "\n";
            std::cout << "  raw_action: " << first_action << "\n";
            std::cout << "  filtered_action: " << filtered_action << "\n";
        }

        // 完成提示
        std::cout << "\ndry-run finished, no robot command was sent.\n";
        return EXIT_SUCCESS;
        
    } catch (const std::exception& error) {
        // 错误处理
        std::cerr << "mp1_dry_run failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
