#pragma once

namespace mp1_deploy {

/**
 * @brief 远程夹爪客户端基类（抽象接口）
 * 
 * 定义与远程夹爪通信的统一接口
 */
class RemoteGripperClient {
public:
    virtual ~RemoteGripperClient() = default;
    
    /**
     * @brief 设置夹爪开合比例
     * @param fraction 开合比例（0.0 表示完全闭合，1.0 表示完全打开）
     */
    virtual void set_fraction(double fraction) = 0;
    
    /**
     * @brief 获取上次设置的夹爪开合比例
     * @return 开合比例
     */
    virtual double last_fraction() const = 0;
};

}  // namespace mp1_deploy
