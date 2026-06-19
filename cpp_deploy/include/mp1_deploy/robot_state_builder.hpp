#pragma once

#include <torch/torch.h>

namespace mp1_deploy {

/**
 * @brief 机器人状态快照
 * 
 * 包含机器人当前时刻的关键状态信息
 */
struct RobotSnapshot {
    torch::Tensor tcp_pose;         ///< TCP位姿 [6]，格式为 xyz + rotvec（旋转向量）
    torch::Tensor joint_positions;  ///< 关节角度 [6]
    double gripper_fraction = 1.0;  ///< 夹爪开合比例（0.0-1.0）
};

/**
 * @brief 机器人状态构建器基类（抽象接口）
 * 
 * 将机器人状态转换为策略模型所需的输入格式
 */
class RobotStateBuilder {
public:
    virtual ~RobotStateBuilder() = default;

    /**
     * @brief 构建策略输入的 agent_pos
     * @param snapshot 机器人状态快照
     * @return agent_pos 张量 [10]
     * 
     * 输出格式需要与 Python 的 tcp_xyz_rot6d + gripper 完全一致
     */
    virtual torch::Tensor build_agent_pos(const RobotSnapshot& snapshot) = 0;
};

}  // namespace mp1_deploy
