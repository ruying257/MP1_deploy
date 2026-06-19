#pragma once

#include <string>

#include <torch/torch.h>

namespace mp1_deploy {

/**
 * @brief 单步日志记录结构
 * 
 * 记录策略部署执行过程中的关键数据，用于调试和性能分析
 */
struct StepLogRecord {
    int step = 0;                    ///< 当前步数
    double timestamp_s = 0.0;        ///< 当前时间戳（秒）
    double inference_time_s = 0.0;   ///< 推理耗时（秒）
    double control_time_s = 0.0;     ///< 控制耗时（秒）
    torch::Tensor raw_action;        ///< 原始策略输出动作
    torch::Tensor filtered_action;   ///< 经过安全滤波后的动作
    std::string exception;           ///< 异常信息（如有）
};

/**
 * @brief 部署日志器基类（抽象接口）
 * 
 * 定义日志记录的统一接口，支持不同的日志输出方式
 */
class DeployLogger {
public:
    virtual ~DeployLogger() = default;
    
    /**
     * @brief 写入单步日志
     * @param record 日志记录数据
     */
    virtual void write_step(const StepLogRecord& record) = 0;
};

}  // namespace mp1_deploy
