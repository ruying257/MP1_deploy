#include "mp1_deploy/observation_buffer.hpp"

#include <stdexcept>
#include <vector>

namespace mp1_deploy {

/**
 * @brief 构造函数，初始化观测缓冲区
 * @param max_size 缓冲区最大容量（帧数）
 * @throw std::invalid_argument 如果 max_size <= 0
 */
ObservationBuffer::ObservationBuffer(std::size_t max_size) : max_size_(max_size) {
    if (max_size_ == 0) {
        throw std::invalid_argument("ObservationBuffer max_size must be > 0");
    }
}

/**
 * @brief 推入一帧观测数据
 * 
 * 如果缓冲区已满，先弹出最早的一帧再推入新帧（FIFO策略）
 * 
 * @param observation 观测数据
 */
void ObservationBuffer::push(const Observation& observation) {
    if (frames_.size() == max_size_) {
        frames_.pop_front();
    }
    frames_.push_back(observation);
}

/**
 * @brief 检查缓冲区是否已满
 * @return true 表示缓冲区已满，可以进行堆叠
 */
bool ObservationBuffer::ready() const {
    return frames_.size() == max_size_;
}

/**
 * @brief 获取当前缓冲区中的帧数
 * @return 当前帧数
 */
std::size_t ObservationBuffer::size() const {
    return frames_.size();
}

/**
 * @brief 获取缓冲区最大容量
 * @return 最大帧数
 */
std::size_t ObservationBuffer::max_size() const {
    return max_size_;
}

/**
 * @brief 将缓冲区内的所有帧按时间维度堆叠
 * 
 * 遍历第一帧的所有键，将所有帧中同名的张量按时间维度（dim=0）堆叠
 * 
 * @return 堆叠后的观测数据，每个张量增加时间维度
 * @throw std::runtime_error 如果缓冲区未满
 * @throw std::runtime_error 如果某帧缺少某个键
 */
Observation ObservationBuffer::stack() const {
    if (!ready()) {
        throw std::runtime_error("ObservationBuffer is not full");
    }

    Observation result;
    // 遍历第一帧的所有键
    for (const auto& [key, first_tensor] : frames_.front()) {
        std::vector<torch::Tensor> tensors;
        tensors.reserve(frames_.size());
        // 收集所有帧中该键对应的张量
        for (const auto& frame : frames_) {
            auto iter = frame.find(key);
            if (iter == frame.end()) {
                throw std::runtime_error("Observation key missing in buffer frame: " + key);
            }
            tensors.push_back(iter->second);
        }
        // 按时间维拼接，得到 [T, ...]，外层 batch 维由调用方添加
        result[key] = torch::stack(tensors, 0);
    }
    return result;
}

}  // namespace mp1_deploy
