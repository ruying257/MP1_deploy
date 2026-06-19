#include "mp1_deploy/safety_filter.hpp"

#include <algorithm>

namespace mp1_deploy {

/**
 * @brief 构造函数，初始化安全滤波器
 * @param options 滤波器配置选项
 */
SafetyFilter::SafetyFilter(SafetyFilterOptions options) : options_(options) {}

/**
 * @brief 对张量进行范数裁剪
 * 
 * 如果张量的范数超过最大允许值，则按比例缩放到最大范数
 * 
 * @param value 输入张量
 * @param max_norm 最大范数
 * @return 裁剪后的张量（克隆副本）
 */
torch::Tensor SafetyFilter::clip_norm(const torch::Tensor& value, double max_norm) const {
    // 如果最大范数小于等于0，直接返回原始值
    if (max_norm <= 0.0) {
        return value.clone();
    }
    // 计算张量的L2范数
    const auto norm = value.norm().item<double>();
    // 如果范数在允许范围内或接近零，直接返回
    if (norm <= max_norm || norm <= 1.0e-12) {
        return value.clone();
    }
    // 按比例缩放到最大范数
    return value * (max_norm / norm);
}

/**
 * @brief 指数移动平均计算
 * 
 * EMA(current, previous) = current * alpha + previous * (1 - alpha)
 * 
 * @param current 当前值
 * @param previous 前一帧值
 * @param alpha 平滑系数（0-1），越大越接近当前值
 * @return EMA 后的值
 */
torch::Tensor SafetyFilter::ema(const torch::Tensor& current, const torch::Tensor& previous, double alpha) const {
    // 限制 alpha 在 [0, 1] 范围内
    const double a = std::clamp(alpha, 0.0, 1.0);
    // 如果 alpha >= 1，直接使用当前值（无平滑）
    if (a >= 1.0) {
        return current.clone();
    }
    // 计算指数移动平均
    return current * a + previous * (1.0 - a);
}

/**
 * @brief 应用安全滤波到动作向量
 * 
 * 对动作向量的各个分量依次进行处理：
 * 1. 平移分量（前3维）：EMA平滑 + 死区处理 + 范数裁剪
 * 2. 旋转分量（中间3维）：EMA平滑 + 死区处理 + 范数裁剪
 * 3. 夹爪分量（最后1维）：默认忽略；需要时才启用 EMA平滑 + 死区处理
 * 
 * 支持的动作向量维度：[3]（仅平移）、[4]（平移+夹爪）、[6]（平移+旋转）、[7]（完整动作）
 * 
 * @param action 原始动作向量
 * @return 滤波后的安全动作向量
 */
torch::Tensor SafetyFilter::apply(const torch::Tensor& action) {
    // 分离梯度并转换为浮点型，展平为1D张量
    torch::Tensor filtered = action.detach().to(torch::kFloat32).flatten().clone();
    // 检查是否有上一帧动作且维度匹配
    const bool has_previous = previous_action_.has_value()
        && previous_action_->sizes().vec() == filtered.sizes().vec();

    // 处理平移分量（前3维）
    if (filtered.size(0) >= 3) {
        const torch::Tensor current = filtered.slice(0, 0, 3);
        const torch::Tensor previous = has_previous ? previous_action_->slice(0, 0, 3) : current;
        // 应用EMA平滑
        torch::Tensor xyz = ema(current, previous, options_.translation_ema_alpha);
        // 死区处理：如果范数小于死区阈值，置零
        if (xyz.norm().item<double>() < options_.translation_deadband_m) {
            xyz.zero_();
        }
        // 范数裁剪
        xyz = clip_norm(xyz, options_.max_translation_m);
        // 将处理后的值写回
        filtered.slice(0, 0, 3).copy_(xyz);
    }

    // 处理旋转分量（中间3维）
    if (filtered.size(0) >= 6) {
        const torch::Tensor current = filtered.slice(0, 3, 6);
        const torch::Tensor previous = has_previous ? previous_action_->slice(0, 3, 6) : current;
        // 应用EMA平滑
        torch::Tensor rot = ema(current, previous, options_.rotation_ema_alpha);
        // 死区处理
        if (rot.norm().item<double>() < options_.rotation_deadband_rad) {
            rot.zero_();
        }
        // 范数裁剪
        rot = clip_norm(rot, options_.max_rotation_rad);
        filtered.slice(0, 3, 6).copy_(rot);
    }

    // 处理夹爪分量（最后1维）
    if (filtered.size(0) == 4 || filtered.size(0) == 7) {
        const int64_t idx = filtered.size(0) - 1;
        if (options_.ignore_gripper_action) {
            // 当前挂杆/取杆阶段暂时不控制夹爪；保留7维动作契约，但把夹爪命令置为中性0。
            filtered.slice(0, idx, idx + 1).zero_();
            previous_action_ = filtered.clone();
            return filtered;
        }

        const torch::Tensor current = filtered.slice(0, idx, idx + 1);
        const torch::Tensor previous = has_previous ? previous_action_->slice(0, idx, idx + 1) : current;
        // 应用EMA平滑
        torch::Tensor gripper = ema(current, previous, options_.gripper_ema_alpha);
        // 死区处理：如果变化量小于死区阈值，保持上一帧值
        if (has_previous && torch::abs(gripper - previous).item<double>() < options_.gripper_deadband) {
            gripper = previous.clone();
        }
        filtered.slice(0, idx, idx + 1).copy_(gripper);
    }

    // 保存限幅后的动作，下一帧的 EMA 以实际执行值为参考
    previous_action_ = filtered.clone();
    return filtered;
}

}  // namespace mp1_deploy
