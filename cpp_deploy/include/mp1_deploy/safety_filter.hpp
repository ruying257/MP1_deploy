#pragma once

#include <optional>

#include <torch/torch.h>

namespace mp1_deploy {

/**
 * @brief 安全滤波器配置选项
 * 
 * 用于限制机器人动作的幅度，确保操作安全
 */
struct SafetyFilterOptions {
    double max_translation_m = 0.015;      ///< 最大平移距离（米）
    double max_rotation_rad = 0.08;        ///< 最大旋转角度（弧度）
    double translation_deadband_m = 0.0;    ///< 平移死区（米），小于此值的动作将被置零
    double rotation_deadband_rad = 0.0;     ///< 旋转死区（弧度），小于此值的动作将被置零
    double translation_ema_alpha = 1.0;     ///< 平移指数移动平均系数（0-1）
    double rotation_ema_alpha = 1.0;        ///< 旋转指数移动平均系数（0-1）
    double gripper_ema_alpha = 1.0;         ///< 夹爪指数移动平均系数（0-1）
    double gripper_deadband = 0.0;          ///< 夹爪死区，小于此值时保持上一帧值
    bool ignore_gripper_action = true;      ///< 暂时忽略模型输出的夹爪动作，保持7维shape但不控制夹爪
};

/**
 * @brief 安全滤波器类
 * 
 * 对机器人动作进行安全限幅，包括：
 * 1. 平移向量的范数限制
 * 2. 旋转向量的范数限制
 * 3. 指数移动平均平滑（EMA）
 * 4. 死区处理
 */
class SafetyFilter {
public:
    /**
     * @brief 构造函数
     * @param options 滤波器配置选项
     */
    explicit SafetyFilter(SafetyFilterOptions options);
    
    /**
     * @brief 应用安全滤波到动作向量
     * @param action 原始动作向量，支持 [3]、[4]、[6]、[7] 维
     * @return 滤波后的安全动作向量
     */
    torch::Tensor apply(const torch::Tensor& action);

private:
    /**
     * @brief 对张量进行范数裁剪
     * @param value 输入张量
     * @param max_norm 最大范数
     * @return 裁剪后的张量
     */
    torch::Tensor clip_norm(const torch::Tensor& value, double max_norm) const;
    
    /**
     * @brief 指数移动平均计算
     * @param current 当前值
     * @param previous 前一帧值
     * @param alpha 平滑系数（0-1）
     * @return EMA 后的值
     */
    torch::Tensor ema(const torch::Tensor& current, const torch::Tensor& previous, double alpha) const;

    SafetyFilterOptions options_;           ///< 滤波器配置
    std::optional<torch::Tensor> previous_action_;  ///< 上一帧的动作值（用于EMA）
};

}  // namespace mp1_deploy
