// 真实输入干运行工具
// 功能：每一步从指定目录读取最新的观测张量文件，执行策略推理并打印动作输出
// 特点：不向真实机器人发送指令，仅用于验证推理流程和输出结果
#include "mp1_deploy/policy_runtime.hpp"   // 策略运行时接口
#include "mp1_deploy/safety_filter.hpp"     // 安全过滤器接口

#include <c10/core/ScalarType.h>            // PyTorch 数据类型定义

#include <chrono>                           // 时间测量
#include <cstdlib>                          // 标准库函数（EXIT_SUCCESS/FAILURE）
#include <filesystem>                       // 文件系统操作
#include <fstream>                          // 读取帧提交文件
#include <iomanip>                          // 输出格式化
#include <iostream>                         // 标准输入输出
#include <sstream>                          // 字符串流
#include <stdexcept>                        // 异常处理
#include <string>                           // 字符串处理
#include <thread>                           // 线程休眠
#include <unordered_map>                    // 哈希映射
#include <vector>                           // 向量容器

namespace {

// 使用稳定时钟进行时间测量
using Clock = std::chrono::steady_clock;

/**
 * @brief 解析命令行参数
 * @param argc 参数数量（包含程序名）
 * @param argv 参数数组
 * @return 解析后的键值对映射（去除 -- 前缀）
 * @throw std::runtime_error 遇到位置参数或缺少值时抛出异常
 */
std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        // 检查参数是否以 -- 开头
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

/**
 * @brief 获取字符串类型的命令行参数
 * @param args 已解析的参数映射
 * @param key 参数名
 * @param default_value 默认值（可选，默认为空字符串）
 * @return 参数值或默认值
 */
std::string get_arg(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    const std::string& default_value = "") {
    const auto iter = args.find(key);
    return iter == args.end() ? default_value : iter->second;
}

/**
 * @brief 获取整数类型的命令行参数
 * @param args 已解析的参数映射
 * @param key 参数名
 * @param default_value 默认值
 * @return 参数值（转换为int）或默认值
 */
int get_int(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    int default_value) {
    const auto iter = args.find(key);
    return iter == args.end() ? default_value : std::stoi(iter->second);
}

/**
 * @brief 获取双精度浮点类型的命令行参数
 * @param args 已解析的参数映射
 * @param key 参数名
 * @param default_value 默认值
 * @return 参数值（转换为double）或默认值
 */
double get_double(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& key,
    double default_value) {
    const auto iter = args.find(key);
    return iter == args.end() ? default_value : std::stod(iter->second);
}

/**
 * @brief 格式化张量形状为字符串
 * @param shape 形状向量
 * @return 格式化后的字符串，如 "[1, 2, 3]"
 */
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

/**
 * @brief 获取数据类型的名称字符串
 * @param dtype PyTorch数据类型
 * @return 类型名称字符串
 */
std::string dtype_name(c10::ScalarType dtype) {
    return std::string(c10::toString(dtype));
}

/**
 * @brief 检查文件是否存在
 * @param path 文件路径
 * @throw std::runtime_error 文件不存在时抛出异常
 */
void require_file(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Missing input tensor file: " + path.string());
    }
}

/**
 * @brief 从文件加载张量
 * @param path 张量文件路径（.pt格式）
 * @return 加载的PyTorch张量
 */
torch::Tensor load_tensor(const std::filesystem::path& path) {
    require_file(path);
    // 使用CPU设备加载模型，然后提取张量
    torch::jit::script::Module module = torch::jit::load(path.string(), torch::kCPU);
    return module.forward({}).toTensor();
}

/**
 * @brief 读取采集端提交的最新完整帧目录
 * @param root 输入根目录
 * @return 如果存在 current_frame.txt，返回其指向的帧目录；否则返回 root 兼容旧格式
 */
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

/**
 * @brief 从输入目录加载所有真实输入张量
 * @param dir 输入目录路径
 * @return PolicyInputs 结构体，包含所有必需的输入张量
 * @note 目录由真实输入采集程序生成；本工具只消费张量，不直接绑定 RealSense/RTDE
 */
mp1_deploy::PolicyInputs load_real_input_dir(const std::filesystem::path& dir) {
    return {
        load_tensor(dir / "global_image.pt"),   // 全局图像 [1, 2, 3, 128, 128]
        load_tensor(dir / "wrist_image.pt"),    // 腕部相机图像 [1, 2, 3, 96, 96]
        load_tensor(dir / "point_cloud.pt"),    // 点云数据 [1, 2, 512, 3]
        load_tensor(dir / "agent_pos.pt"),      // 智能体状态 [1, 2, 10]
        load_tensor(dir / "initial_noise.pt"),  // 初始噪声 [1, 4, 7]
    };
}

/**
 * @brief 验证张量形状是否符合预期
 * @param tensor 待验证的张量
 * @param name 张量名称（用于错误消息）
 * @param expected 期望的形状
 * @throw std::runtime_error 形状不匹配时抛出异常
 */
void require_shape(const torch::Tensor& tensor, const std::string& name, const std::vector<int64_t>& expected) {
    if (tensor.sizes().vec() != expected) {
        std::ostringstream message;
        message << name << " shape mismatch, expected " << format_shape(expected) << ", got " << tensor.sizes();
        throw std::runtime_error(message.str());
    }
}

/**
 * @brief 验证张量数据类型是否符合预期
 * @param tensor 待验证的张量
 * @param name 张量名称（用于错误消息）
 * @param expected 期望的数据类型
 * @throw std::runtime_error 数据类型不匹配时抛出异常
 */
void require_dtype(const torch::Tensor& tensor, const std::string& name, c10::ScalarType expected) {
    if (tensor.scalar_type() != expected) {
        std::ostringstream message;
        message << name << " dtype mismatch, expected " << dtype_name(expected)
                << ", got " << dtype_name(tensor.scalar_type());
        throw std::runtime_error(message.str());
    }
}

/**
 * @brief 验证所有输入张量的形状和数据类型
 * @param inputs 待验证的输入结构体
 * @note 这是模型输入契约；真实相机、点云、机器人状态最终都必须落到这些张量格式
 */
void validate_inputs(const mp1_deploy::PolicyInputs& inputs) {
    // 验证各输入张量的形状
    require_shape(inputs.global_image, "global_image", {1, 2, 3, 128, 128});
    require_shape(inputs.wrist_image, "wrist_image", {1, 2, 3, 96, 96});
    require_shape(inputs.point_cloud, "point_cloud", {1, 2, 512, 3});
    require_shape(inputs.agent_pos, "agent_pos", {1, 2, 10});
    require_shape(inputs.initial_noise, "initial_noise", {1, 4, 7});

    // 验证各输入张量的数据类型
    require_dtype(inputs.global_image, "global_image", torch::kUInt8);
    require_dtype(inputs.wrist_image, "wrist_image", torch::kUInt8);
    require_dtype(inputs.point_cloud, "point_cloud", torch::kFloat32);
    require_dtype(inputs.agent_pos, "agent_pos", torch::kFloat32);
    require_dtype(inputs.initial_noise, "initial_noise", torch::kFloat32);
}

/**
 * @brief 打印输入张量的摘要信息
 * @param inputs 输入结构体
 */
void print_input_summary(const mp1_deploy::PolicyInputs& inputs) {
    std::cout << "input summary\n";
    std::cout << "  global_image: " << inputs.global_image.sizes() << ", " << dtype_name(inputs.global_image.scalar_type()) << "\n";
    std::cout << "  wrist_image:  " << inputs.wrist_image.sizes() << ", " << dtype_name(inputs.wrist_image.scalar_type()) << "\n";
    std::cout << "  point_cloud:  " << inputs.point_cloud.sizes() << ", " << dtype_name(inputs.point_cloud.scalar_type()) << "\n";
    std::cout << "  agent_pos:    " << inputs.agent_pos.sizes() << ", " << dtype_name(inputs.agent_pos.scalar_type()) << "\n";
    std::cout << "  initial_noise:" << inputs.initial_noise.sizes() << ", " << dtype_name(inputs.initial_noise.scalar_type()) << "\n";
}

/**
 * @brief 获取输入目录中最新文件的修改时间
 * @param dir 输入目录路径
 * @return 最新文件的修改时间
 */
std::filesystem::file_time_type newest_input_mtime(const std::filesystem::path& dir) {
    // 新协议：采集端先写完整帧目录，最后原子更新 current_frame.txt。
    const auto manifest = dir / "current_frame.txt";
    if (std::filesystem::exists(manifest)) {
        return std::filesystem::last_write_time(manifest);
    }

    // 旧协议兼容：目录下直接放 5 个固定 .pt 文件。
    std::filesystem::file_time_type newest{};
    // 检查所有必需的张量文件
    for (const auto& name : {"global_image.pt", "wrist_image.pt", "point_cloud.pt", "agent_pos.pt", "initial_noise.pt"}) {
        const auto path = dir / name;
        require_file(path);
        const auto mtime = std::filesystem::last_write_time(path);
        if (mtime > newest) {
            newest = mtime;
        }
    }
    return newest;
}

/**
 * @brief 根据命令行参数构建安全过滤器选项
 * @param args 命令行参数映射
 * @return 配置好的安全过滤器选项
 */
mp1_deploy::SafetyFilterOptions make_safety_options(const std::unordered_map<std::string, std::string>& args) {
    mp1_deploy::SafetyFilterOptions options;
    options.max_translation_m = get_double(args, "max-translation", 0.002);     // 最大平移距离（米）
    options.max_rotation_rad = get_double(args, "max-rotation", 0.01);         // 最大旋转角度（弧度）
    options.translation_deadband_m = get_double(args, "translation-deadband", options.translation_deadband_m);
    options.rotation_deadband_rad = get_double(args, "rotation-deadband", options.rotation_deadband_rad);
    options.gripper_deadband = get_double(args, "gripper-deadband", options.gripper_deadband);
    options.ignore_gripper_action = get_int(args, "ignore-gripper-action", 1) != 0;  // 默认不控制夹爪
    return options;
}

/**
 * @brief 打印命令行使用说明
 */
void print_usage() {
    std::cout
        << "Usage: mp1_real_input_dry_run [--backend torchscript|tensorrt] [--model policy_infer.pt] --input-dir real_input_tensors [--device cpu]\n"
        << "       TensorRT: --obs-engine obs_encoder_fp16.engine --unet-engine unet_step_fp16.engine --trt-meta trt_runtime_meta.json\n"
        << "       [--steps 0] [--warmup-steps 3] [--poll-ms 200] [--require-update 1]\n"
        << "       [--max-translation 0.002] [--max-rotation 0.01] [--ignore-gripper-action 1]\n";
}

/**
 * @brief 根据设备选择默认预热次数
 *
 * CUDA 第一次推理通常会初始化上下文、选择 kernel、分配缓存；这些耗时不应该算进正式控制步。
 */
int default_warmup_steps(const std::string& device_name) {
    return device_name.find("cuda") != std::string::npos ? 3 : 0;
}

}  // namespace

/**
 * @brief 主函数 - 真实输入干运行入口
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return EXIT_SUCCESS 或 EXIT_FAILURE
 */
int main(int argc, char** argv) {
    try {
        // 1. 解析命令行参数
        const auto args = parse_args(argc, argv);
        
        // 2. 获取必需参数
        const std::string backend = get_arg(args, "backend", "torchscript");
        const std::string model_path = get_arg(args, "model");
        const std::string input_dir_text = get_arg(args, "input-dir");
        if ((backend == "torchscript" && model_path.empty()) || input_dir_text.empty()) {
            print_usage();
            throw std::runtime_error("--model and --input-dir are required");
        }

        // 3. 验证输入目录
        const std::filesystem::path input_dir(input_dir_text);
        if (!std::filesystem::is_directory(input_dir)) {
            throw std::runtime_error("--input-dir is not a directory: " + input_dir.string());
        }

        // 4. 获取可选参数
        const std::string device_name = get_arg(args, "device", "cpu");        // 运行设备（cpu/cuda）
        const int steps = get_int(args, "steps", 0);                          // 推理步数（0=无限循环）
        const int warmup_steps = get_int(args, "warmup-steps", default_warmup_steps(device_name)); // 正式计步前的预热次数
        const int poll_ms = get_int(args, "poll-ms", 200);                    // 轮询间隔（毫秒）
        const bool require_update = get_int(args, "require-update", 1) != 0;   // 是否等待文件更新
        
        // 参数校验
        if (steps < 0) {
            throw std::runtime_error("--steps must be >= 0");
        }
        if (poll_ms < 0) {
            throw std::runtime_error("--poll-ms must be >= 0");
        }
        if (warmup_steps < 0) {
            throw std::runtime_error("--warmup-steps must be >= 0");
        }

        // 5. 初始化运行时和安全过滤器
        const mp1_deploy::TrtRuntimeOptions trt_options{
            get_arg(args, "obs-engine"), get_arg(args, "unet-engine"), get_arg(args, "trt-meta")};
        auto runtime = mp1_deploy::create_policy_runtime(
            backend, model_path, torch::Device(device_name), trt_options);
        mp1_deploy::SafetyFilter safety_filter(make_safety_options(args));

        // 6. 初始化循环变量
        std::filesystem::file_time_type last_mtime{};  // 上一次处理的文件修改时间
        int step = 0;                                  // 当前推理步数
        
        // 打印启动信息
        std::cout << "real-input dry-run start: backend=" << backend << ", device=" << device_name
                  << ", input_dir=" << input_dir.string()
                  << ", steps=" << steps
                  << ", warmup_steps=" << warmup_steps << "\n";
        std::cout << "no robot command will be sent.\n";

        // CUDA 预热：重复使用当前最新完整帧，不要求采集端产生新帧，也不计入正式 step。
        if (warmup_steps > 0) {
            const auto warmup_frame_dir = resolve_committed_frame_dir(input_dir);
            const mp1_deploy::PolicyInputs warmup_inputs = load_real_input_dir(warmup_frame_dir);
            validate_inputs(warmup_inputs);
            std::cout << "warmup start: frame_dir=" << warmup_frame_dir.string() << "\n";
            for (int warmup = 0; warmup < warmup_steps; ++warmup) {
                const auto begin = Clock::now();
                auto [warmup_action, warmup_action_pred] = runtime->infer(warmup_inputs);
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

        // 7. 主推理循环
        while (steps == 0 || step < steps) {
            // 获取最新文件修改时间
            const auto current_mtime = newest_input_mtime(input_dir);
            
            // 如果需要等待更新且文件未变化，则休眠等待
            if (require_update && step > 0 && current_mtime <= last_mtime) {
                std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
                continue;
            }
            last_mtime = current_mtime;

            // 加载输入张量并验证
            const auto load_begin = Clock::now();
            const auto frame_dir = resolve_committed_frame_dir(input_dir);
            const mp1_deploy::PolicyInputs inputs = load_real_input_dir(frame_dir);
            validate_inputs(inputs);
            
            // 执行推理
            const auto infer_begin = Clock::now();
            auto [action, action_pred] = runtime->infer(inputs);
            const auto infer_end = Clock::now();

            // 提取第一个动作并应用安全过滤
            const torch::Tensor first_action = action.select(0, 0).select(0, 0).to(torch::kFloat32);
            const torch::Tensor filtered_action = safety_filter.apply(first_action);

            // 计算耗时
            const double load_ms =
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(infer_begin - load_begin).count();
            const double infer_ms =
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(infer_end - infer_begin).count();

            // 打印推理结果
            std::cout << "\nstep " << step << "\n";
            if (step == 0) {
                print_input_summary(inputs);  // 第一步打印输入摘要
            }
            std::cout << "  frame_dir: " << frame_dir.string() << "\n";
            std::cout << "  load_ms: " << std::fixed << std::setprecision(3) << load_ms << "\n";
            std::cout << "  inference_ms: " << std::fixed << std::setprecision(3) << infer_ms << "\n";
            std::cout << "  action shape: " << action.sizes() << ", action_pred shape: " << action_pred.sizes() << "\n";
            std::cout << "  raw_action: " << first_action << "\n";
            std::cout << "  filtered_action: " << filtered_action << "\n";

            ++step;
            
            // 如果不需要等待更新且设置了轮询间隔，则休眠
            if (!require_update && poll_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
            }
        }

        // 8. 完成
        std::cout << "\nreal-input dry-run finished, no robot command was sent.\n";
        return EXIT_SUCCESS;
        
    } catch (const std::exception& error) {
        // 异常处理
        std::cerr << "mp1_real_input_dry_run failed: " << error.what() << "\n";
        return EXIT_FAILURE;
    }
}
