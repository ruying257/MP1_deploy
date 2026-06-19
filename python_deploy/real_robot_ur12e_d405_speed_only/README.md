# UR12E + D405 Speed-Only 采集工具

这是一个**独立目录**。只要把整个 `real_robot_ur12e_d405_speed_only` 文件夹拷走，并在目标电脑安装对应依赖，就可以单独采集数据，不依赖 `MP1` 其他训练代码。

## 目录说明

- `scripts/collect_real_ur_speed_only.py`
  - 主采集脚本
- `scripts/real_robot_utils.py`
  - 相机、机器人、raw/zarr 写盘、公用几何与格式工具
- `scripts/gripper_gpio_proxy_server.py`
  - Jetson 侧夹爪 GPIO 代理服务
- `configs/speed_only_config_template.json`
  - speed-only 模板配置
- `configs/collect_vibration_grasp_laptop_remote.json`
  - 笔记本采集、Jetson 远程控制夹爪示例
- `configs/collect_vibration_grasp_jetson_local.json`
  - Jetson 本机采集与本机 GPIO 夹爪示例
- `configs/gripper_proxy_jetson.json`
  - Jetson GPIO 代理配置
- `docs/SPEED_ONLY_COLLECTION_GUIDE.md`
  - 详细使用说明
- `data/README.md`
  - 数据落盘结构说明

## 这个目录和旧版采集工具的区别

这个版本是**纯 speed-only 控制**：

- 键盘只负责更新当前目标速度
- 高速控制线程持续发送 `speedL / speedJ`
- 低速采样线程固定频率记录相机和机器人状态
- 数据集中的 `action` 不直接写“命令速度”，而是由相邻两帧机器人状态差分重建

所以它更接近真实遥操作采集，而不是“按一次键走一个离散步长”。

当前示例配置默认按：

- 相机 `30 FPS`
- 数据采样 `30 Hz`
- 速度控制 `100 Hz`

来设置，但这只是默认值。是否能稳定跑满，仍然取决于：

- 两台相机分辨率
- USB 带宽
- 主机性能
- 点云处理与写盘负载

## 最常用启动命令

笔记本采集 + Jetson 远程夹爪：

```powershell
python scripts/gripper_gpio_proxy_server.py --config configs/gripper_proxy_jetson.json
```

上面这条先在 Jetson 上运行。

然后在笔记本上运行：

```powershell
python scripts/collect_real_ur_speed_only.py --config configs/collect_vibration_grasp_laptop_remote.json --move-home-on-start
```

Jetson 本机采集：

```powershell
python scripts/collect_real_ur_speed_only.py --config configs/collect_vibration_grasp_jetson_local.json --move-home-on-start
```

流程演练：

```powershell
python scripts/collect_real_ur_speed_only.py --config configs/speed_only_config_template.json --dry-run
```

## 依赖

至少需要：

- `numpy`
- `zarr`
- `pyrealsense2`
- `opencv-python`
- `Pillow`
- `imageio`
- `ur-rtde`

如果使用 Robotiq 夹爪，还需要：

- `robotiq_gripper`

如果使用 Jetson 本地 GPIO 夹爪，还需要：

- `Jetson.GPIO`

## 建议

- 第一次一定先跑 `--dry-run`
- 真机第一次一定空载，远离障碍物
- speed-only 模式下，建议先把线速度、角速度、关节速度设小，再逐步放开
- 真正开始录制前，先验证 `Space` 是否能正常停下
- 当前版本已经额外保存每台相机的：
  - `depth/color frame timestamp`
  - `depth/color frame number`
  - `host capture time`

详细步骤见 [docs/SPEED_ONLY_COLLECTION_GUIDE.md](./docs/SPEED_ONLY_COLLECTION_GUIDE.md)。
