/** 程序功能
 * 1. 加载模型
 * 2. 加载张量数据
 * 3. 执行推理
 * 4. 对比python版expected_action，验证C++ LibTorch 是否复现了 Python 的原始模型输出
 */
 // first_action 的 max_abs_diff = 3.57628e-07(极小) 证明链路基本对齐
 // CPU 推理
#include "mp1_deploy/policy_runtime.hpp"
#include "mp1_deploy/safety_filter.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

/**
 * @brief 解析命令行参数
 * 
 * 支持格式：--key1 value1 --key2 value2
 * 
 * @param argc 参数数量
 * @param argv 参数数组
 * @return 参数键值对
 * @throw std::runtime_error 如果遇到位置参数或缺少值
 */
std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        // 检查是否为选项参数
        if (key.rfind("--", 0) != 0) {
            throw std::runtime_error("Unexpected positional argument: " + key);
        }
        // 检查是否有对应的值
        if (i + 1 >= argc) {
            throw std::runtime_error("Missing value for " + key);
        }
        args[key.substr(2)] = argv[++i];
    }
    return args;
}

/**
 * @brief 获取字符串类型参数
 * 
 * @param args 参数映射
 * @param key 参数名
 * @param default_value 默认值
 * @return 参数值
 */
std::string get_arg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    const std::string& default_value = "") {
    auto iter = args.find(key);
    return iter == args.end() ? default_value : iter->second;
}

/**
 * @brief 获取浮点类型参数
 * 
 * @param args 参数映射
 * @param key 参数名
 * @param default_value 默认值
 * @return 参数值
 */
double get_double(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    double default_value) {
    auto iter = args.find(key);
    return iter == args.end() ? default_value : std::stod(iter->second);
}

/**
 * @brief 加载张量数据
 * 
 * sample_tensors 由 Python 导出为 TorchScript 常量模块，避免跨语言 torch.save 兼容问题
 * 
 * @param path 张量文件路径
 * @return 加载的张量
 */
torch::Tensor load_tensor(const std::string& path) {
    torch::jit::script::Module module = torch::jit::load(path, torch::kCPU);
    return module.forward({}).toTensor();
}

/**
 * @brief 创建零输入数据
 * @return 零初始化的策略输入
 */
mp1_deploy::PolicyInputs make_zero_inputs() {
    return {
        torch::zeros({1, 2, 3, 128, 128}, torch::kUInt8),  // global_image [B, T, C, H, W]
        torch::zeros({1, 2, 3, 96, 96}, torch::kUInt8),    // wrist_image [B, T, C, H, W]
        torch::zeros({1, 2, 512, 3}, torch::kFloat32),     // point_cloud [B, T, N, 3]
        torch::zeros({1, 2, 10}, torch::kFloat32),         // agent_pos [B, T, 10]
        torch::zeros({1, 4, 7}, torch::kFloat32),          // initial_noise [B, N, 7]
    };
}

/**
 * @brief 从目录加载输入张量
 * 
 * @param dir 张量目录路径
 * @return 策略输入数据
 */
mp1_deploy::PolicyInputs load_tensor_dir(const std::string& dir) {
    return {
        load_tensor(dir + "/global_image.pt"),
        load_tensor(dir + "/wrist_image.pt"),
        load_tensor(dir + "/point_cloud.pt"),
        load_tensor(dir + "/agent_pos.pt"),
        load_tensor(dir + "/initial_noise.pt"),
    };
}

/**
 * @brief 打印张量信息
 * 
 * @param label 张量标签
 * @param tensor 张量数据
 */
void print_tensor(const std::string& label, const torch::Tensor& tensor) {
    std::cout << label << ": " << tensor << "\n";
}

}  // namespace

/**
 * @brief 主函数：离线推理工具入口
 * 
 * 加载策略模型，执行推理，并应用安全滤波
 * 
 * 命令行参数：
 * --model: 策略模型路径（必需）
 * --device: 推理设备（默认 cpu）
 * --tensor-dir: 输入张量目录（可选，否则使用零输入）
 * --max-translation: 最大平移距离（米）
 * --max-rotation: 最大旋转角度（弧度）
 * 
 * @param argc 参数数量
 * @param argv 参数数组
 * @return 退出码
 */
int main(int argc, char** argv) {
    try {
        // 解析命令行参数
        const auto args = parse_args(argc, argv);
        
        // 获取模型路径（必需）
        const std::string backend = get_arg(args, "backend", "torchscript");
        const std::string model_path = get_arg(args, "model");
        if (backend == "torchscript" && model_path.empty()) {
            throw std::runtime_error("Usage: mp1_offline_infer --backend torchscript --model policy_infer.pt [--tensor-dir sample_tensors]");
        }

        // 获取设备和输入数据
        const std::string device_name = get_arg(args, "device", "cpu");
        const torch::Device device(device_name);
        const std::string tensor_dir = get_arg(args, "tensor-dir");
        const mp1_deploy::PolicyInputs inputs = tensor_dir.empty() ? make_zero_inputs() : load_tensor_dir(tensor_dir);

        // 创建运行时并执行推理
        const mp1_deploy::TrtRuntimeOptions trt_options{
            get_arg(args, "obs-engine"), get_arg(args, "unet-engine"), get_arg(args, "trt-meta")};
        auto runtime = mp1_deploy::create_policy_runtime(backend, model_path, device, trt_options);
        auto [action, action_pred] = runtime->infer(inputs);
        std::cout << "backend: " << backend << "\n";
        
        // 提取第一帧动作
        const torch::Tensor first_action = action.select(0, 0).select(0, 0).to(torch::kFloat32);

        // 应用安全滤波
        mp1_deploy::SafetyFilterOptions safety_options;
        safety_options.max_translation_m = get_double(args, "max-translation", safety_options.max_translation_m);
        safety_options.max_rotation_rad = get_double(args, "max-rotation", safety_options.max_rotation_rad);
        mp1_deploy::SafetyFilter safety_filter(safety_options);
        const torch::Tensor clipped_action = safety_filter.apply(first_action);

        // 输出结果
        print_tensor("action[0,0]", first_action);
        print_tensor("clipped_action", clipped_action);
        std::cout << "action shape: " << action.sizes() << "\n";
        std::cout << "action_pred shape: " << action_pred.sizes() << "\n";
        
        // 如果提供了参考结果目录，计算与期望值的差异
        if (!tensor_dir.empty()) {
            const std::string expected_path = tensor_dir + "/expected_action.pt";
            if (std::filesystem::exists(expected_path)) {
                const torch::Tensor expected = load_tensor(expected_path).to(torch::kFloat32);
                const torch::Tensor expected_first = expected.select(0, 0).select(0, 0);
                const double max_abs_diff = torch::max(torch::abs(first_action - expected_first)).item<double>();
                std::cout << "expected max_abs_diff: " << max_abs_diff << "\n";
                if (action.sizes() != expected.sizes()) {
                    throw std::runtime_error("expected_action shape does not match runtime action output");
                }
                const double full_max_abs_diff = torch::max(
                    torch::abs(action.to(torch::kFloat32) - expected)).item<double>();
                std::cout << "expected full_action max_abs_diff: " << full_max_abs_diff << "\n";
            }
        }
        
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "mp1_offline_infer failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
