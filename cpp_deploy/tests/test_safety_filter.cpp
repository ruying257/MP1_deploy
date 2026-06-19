#include "mp1_deploy/safety_filter.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

/**
 * @brief 安全滤波器测试主函数
 * 
 * 测试安全滤波器的基本功能：
 * 1. 平移分量的范数裁剪
 * 2. 旋转分量的范数裁剪
 * 3. 夹爪动作默认忽略，防止误发夹爪控制
 */
int main() {
    // 配置滤波器参数
    mp1_deploy::SafetyFilterOptions options;
    options.max_translation_m = 0.005;    // 最大平移 0.005 米
    options.max_rotation_rad = 0.03;      // 最大旋转 0.03 弧度
    options.translation_deadband_m = 0.001; // 平移死区 0.001 米
    options.rotation_deadband_rad = 0.001;  // 旋转死区 0.001 弧度

    // 创建滤波器实例
    mp1_deploy::SafetyFilter filter(options);
    
    // 构造测试动作向量：平移 0.1m（超限制），旋转 0.2rad（超限制），夹爪 -2.0
    const torch::Tensor action = torch::tensor({0.1F, 0.0F, 0.0F, 0.0F, 0.2F, 0.0F, -2.0F});
    
    // 应用安全滤波
    const torch::Tensor filtered = filter.apply(action);

    // 验证平移范数被裁剪到最大值
    const double translation_norm = filtered.slice(0, 0, 3).norm().item<double>();
    assert(std::abs(translation_norm - 0.005) < 1.0e-6);
    
    // 验证旋转范数被裁剪到最大值
    const double rotation_norm = filtered.slice(0, 3, 6).norm().item<double>();
    assert(std::abs(rotation_norm - 0.03) < 1.0e-6);
    
    // 验证夹爪动作被置零；当前阶段只允许 TCP 前6维进入控制链路
    assert(std::abs(filtered[6].item<double>()) < 1.0e-6);

    std::cout << "SafetyFilter test passed\n";
    return 0;
}
