#pragma once

#include <array>

#include <torch/torch.h>

namespace mp1_deploy {

/**
 * @brief 点云处理器配置选项
 */
struct PointCloudProcessorOptions {
    std::array<float, 3> crop_min{0.28F, -0.20F, 0.0F};  ///< 裁剪区域最小值 [x, y, z]
    std::array<float, 3> crop_max{0.62F, 0.20F, 0.35F};  ///< 裁剪区域最大值 [x, y, z]
    int num_points = 512;                                 ///< 输出点云点数
};

/**
 * @brief 点云处理器基类（抽象接口）
 * 
 * 负责对点云数据进行预处理，包括裁剪、下采样等操作
 */
class PointCloudProcessor {
public:
    virtual ~PointCloudProcessor() = default;

    /**
     * @brief 处理原始点云
     * @param raw_points_xyz 原始点云数据 [N, 3]
     * @return 处理后的点云 [num_points, 3]
     * 
     * 输入点云需要保持和 Python policy_dump 同一坐标系
     */
    virtual torch::Tensor process(const torch::Tensor& raw_points_xyz) = 0;
};

}  // namespace mp1_deploy
