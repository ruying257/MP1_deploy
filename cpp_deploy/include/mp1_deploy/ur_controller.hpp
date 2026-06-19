#pragma once

#include <torch/torch.h>

namespace mp1_deploy {

/**
 * @brief UR机器人控制器基类（抽象接口）
 * 
 * 定义与UR系列机器人交互的统一接口
 */
class URController {
public:
    virtual ~URController() = default;
    
    /**
     * @brief 移动机器人到Home位置
     */
    virtual void move_home() = 0;
    
    /**
     * @brief 执行增量动作
     * @param action 增量动作向量 [dx, dy, dz, drx, dry, drz, gripper]
     *
     * 当前阶段暂时不控制夹爪；控制实现只应消费前6维 TCP 增量，第7维保留为模型契约占位。
     */
    virtual void execute_delta_action(const torch::Tensor& action) = 0;
    
    /**
     * @brief 安全停止机器人运动
     */
    virtual void safe_stop() = 0;
};

}  // namespace mp1_deploy
