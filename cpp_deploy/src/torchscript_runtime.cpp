#include "mp1_deploy/policy_runtime.hpp"

#include <stdexcept>
#include <vector>

namespace mp1_deploy {

/**
 * @brief 构造函数，加载 TorchScript 模型
 * 
 * 从指定路径加载模型并设置为评估模式
 * 
 * @param model_path TorchScript 模型文件路径
 * @param device 推理设备（CPU/GPU）
 */
TorchScriptRuntime::TorchScriptRuntime(const std::string& model_path, const torch::Device& device)
    : module_(torch::jit::load(model_path, device)), device_(device) {
    // 设置模型为评估模式（禁用 dropout、batch norm 等训练行为）
    module_.eval();
}

/**
 * @brief 执行策略推理
 * 
 * 将输入数据转换为 TorchScript 格式，调用模型前向传播，返回推理结果
 * 
 * @param inputs 策略输入数据
 * @return (action, action_pred) 元组，均已转移到 CPU
 * @throw std::runtime_error 如果模型输出不是二元组
 */
std::pair<torch::Tensor, torch::Tensor> TorchScriptRuntime::infer(const PolicyInputs& inputs) {
    // 禁用梯度计算，提高推理性能
    torch::NoGradGuard no_grad;
    
    // 准备输入数据
    std::vector<torch::jit::IValue> jit_inputs;
    jit_inputs.reserve(5);
    jit_inputs.emplace_back(inputs.global_image.to(device_));
    jit_inputs.emplace_back(inputs.wrist_image.to(device_));
    jit_inputs.emplace_back(inputs.point_cloud.to(device_));
    jit_inputs.emplace_back(inputs.agent_pos.to(device_));
    jit_inputs.emplace_back(inputs.initial_noise.to(device_));

    // 执行前向传播
    const auto output = module_.forward(jit_inputs);
    
    // 验证输出格式
    if (!output.isTuple()) {
        throw std::runtime_error("TorchScript policy must return (action, action_pred)");
    }
    const auto elements = output.toTuple()->elements();
    if (elements.size() != 2) {
        throw std::runtime_error("TorchScript policy returned an unexpected tuple size");
    }
    
    // 将结果转移到 CPU 并返回
    return {elements[0].toTensor().cpu(), elements[1].toTensor().cpu()};
}

}  // namespace mp1_deploy
