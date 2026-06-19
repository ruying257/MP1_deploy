#pragma once

#include <string>
#include <unordered_map>

#include <torch/torch.h>

namespace mp1_deploy {

/**
 * @brief 相机帧数据结构
 * 
 * 存储单个相机的一帧数据
 */
struct CameraFrame {
    torch::Tensor color_bgr_hwc;      ///< RGB彩色图像（BGR格式）[H, W, 3]
    torch::Tensor depth_m;            ///< 深度图像（单位：米）[H, W]
    torch::Tensor point_cloud_xyz;    ///< 点云数据 [N, 3]
    double host_capture_unix = 0.0;   ///< 主机捕获时间戳（Unix时间）
};

/**
 * @brief RealSense相机管理器基类（抽象接口）
 * 
 * 管理多个RealSense相机的捕获操作
 */
class RealSenseManager {
public:
    virtual ~RealSenseManager() = default;

    /**
     * @brief 捕获所有相机的帧数据
     * @return 相机名称到帧数据的映射
     * 
     * 相机命名需要与 deploy_meta.json 中的配置一致，例如：global_d405, wrist_d405
     */
    virtual std::unordered_map<std::string, CameraFrame> capture() = 0;
};

}  // namespace mp1_deploy
