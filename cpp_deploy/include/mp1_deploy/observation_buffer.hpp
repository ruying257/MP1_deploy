#pragma once

#include <deque>
#include <string>
#include <unordered_map>

#include <torch/torch.h>

namespace mp1_deploy {

/**
 * @brief 观测数据类型，键值对形式存储多个张量
 */
using Observation = std::unordered_map<std::string, torch::Tensor>;

/**
 * @brief 观测缓冲区类
 * 
 * 用于累积多帧观测数据，支持按时间维度堆叠，为策略模型提供时序输入
 */
class ObservationBuffer {
public:
    /**
     * @brief 构造函数
     * @param max_size 缓冲区最大容量（帧数）
     */
    explicit ObservationBuffer(std::size_t max_size);

    /**
     * @brief 推入一帧观测数据
     * @param observation 观测数据
     */
    void push(const Observation& observation);
    
    /**
     * @brief 检查缓冲区是否已满
     * @return true 表示缓冲区已满，可以进行堆叠
     */
    bool ready() const;
    
    /**
     * @brief 获取当前缓冲区中的帧数
     * @return 当前帧数
     */
    std::size_t size() const;
    
    /**
     * @brief 获取缓冲区最大容量
     * @return 最大帧数
     */
    std::size_t max_size() const;
    
    /**
     * @brief 将缓冲区内的所有帧按时间维度堆叠
     * @return 堆叠后的观测数据，每个张量增加时间维度
     * @throw std::runtime_error 如果缓冲区未满
     */
    Observation stack() const;

private:
    std::size_t max_size_;           ///< 缓冲区最大容量
    std::deque<Observation> frames_; ///< 存储观测帧的双端队列
};

}  // namespace mp1_deploy
