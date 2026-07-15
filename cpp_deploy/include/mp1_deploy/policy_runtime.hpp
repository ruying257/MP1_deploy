#pragma once

#include <memory>
#include <string>
#include <utility>

#include <torch/script.h>
#include <torch/torch.h>

namespace mp1_deploy {

/**
 * @brief 策略输入数据结构
 */
struct PolicyInputs {
    torch::Tensor global_image;    ///< 全局相机图像 [B, T, C, H, W]
    torch::Tensor wrist_image;     ///< 腕部相机图像 [B, T, C, H, W]
    torch::Tensor point_cloud;     ///< 点云数据 [B, T, N, 3]
    torch::Tensor agent_pos;       ///< 机器人状态 [B, T, 10]
    torch::Tensor initial_noise;   ///< 初始噪声 [B, N, 7]
};

struct TrtRuntimeOptions {
    std::string obs_engine_path;
    std::string unet_engine_path;
    std::string metadata_path;
};

/**
 * @brief 策略运行时基类（抽象接口）
 * 
 * 定义策略推理的统一接口，支持不同后端实现
 */
class PolicyRuntime {
public:
    virtual ~PolicyRuntime() = default;
    
    /**
     * @brief 执行策略推理
     * @param inputs 策略输入数据
     * @return 推理结果，包含 (action, action_pred)
     */
    virtual std::pair<torch::Tensor, torch::Tensor> infer(const PolicyInputs& inputs) = 0;
};

/**
 * @brief TorchScript 策略运行时实现
 * 
 * 使用 PyTorch 的 TorchScript 格式加载并执行策略模型
 */
class TorchScriptRuntime final : public PolicyRuntime {
public:
    /**
     * @brief 构造函数
     * @param model_path TorchScript 模型文件路径
     * @param device 推理设备（CPU/GPU）
     */
    TorchScriptRuntime(const std::string& model_path, const torch::Device& device);
    
    /**
     * @brief 执行推理
     * @param inputs 策略输入数据
     * @return (action, action_pred) 元组
     */
    std::pair<torch::Tensor, torch::Tensor> infer(const PolicyInputs& inputs) override;

private:
    torch::jit::script::Module module_;  ///< TorchScript 模块
    torch::Device device_;               ///< 推理设备
};

std::unique_ptr<PolicyRuntime> create_policy_runtime(
    const std::string& backend,
    const std::string& model_path,
    const torch::Device& device,
    const TrtRuntimeOptions& trt_options = {});

}  // namespace mp1_deploy
