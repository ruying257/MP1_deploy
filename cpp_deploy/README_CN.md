# cpp\_deploy：MP1 Jetson C++ 使用说明

## 常用命令

```bash
cd /mnt/nvme/MP1_model
source /mnt/nvme/libtorch/venv/torch_env/bin/activate

cmake -S cpp_deploy -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/mnt/nvme/libtorch/venv/torch_env/lib/python3.8/site-packages/torch \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CUDA_STANDARD=17 \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build build -j
```

如果当前 Jetson 还没有 C++ `ur_rtde` 或 RealSense SDK，先用第 3 节里的 `-DMP1_ENABLE_UR_RTDE=OFF -DMP1_ENABLE_REALTIME_PIPELINE=OFF` 编译离线/dry-run 工具。

## 结论

这个目录只放 C++/Jetson 侧的编译、离线对齐、dry-run 和真实输入采集说明。整个仓库的总体介绍见仓库根目录的 `README.md`。

当前只针对 `取杆 / pole_pickoff` 这类 7 维含夹爪任务，主线是：

1. 在 Ubuntu 主机冻结 Python 行为并导出 `policy_infer.pt`。
2. 在 Jetson 上用 LibTorch 编译 C++ 工程。
3. 先做离线对齐，再做 dry-run，再采集真实输入做 real-input dry-run。
4. 所有 dry-run 阶段都只打印动作，不发送机器人控制命令。

当前已经包含这些工具：

| 工具                               | 作用                                       | 是否发机器人 |
| -------------------------------- | ---------------------------------------- | ------ |
| `mp1_offline_infer`              | 加载 TorchScript，做黄金样本离线对齐                 | 否      |
| `mp1_dry_run`                    | 用黄金样本或零输入重复推理，检查安全限幅                     | 否      |
| `capture_real_inputs.py`         | 采集 RealSense/RTDE 状态并按完整帧写出真实输入张量        | 否      |
| `mp1_real_input_dry_run`         | 循环读取最新完整帧，推理并打印限幅动作                      | 否      |
| `mp1_real_robot_control`         | 读取真实输入并执行小限幅 TCP 闭环控制，默认 dry-run         | 默认否    |
| `mp1_realtime_robot_control`     | 单进程 C++ 采集、ring buffer、推理和小限幅控制          | 默认否    |
| `test_safety_filter`             | 测试平移/旋转限幅和夹爪动作屏蔽逻辑                       | 否      |
| `tools/dump_trt_case.py`         | 冻结 ONNX/TensorRT 对齐 case，输出中间张量          | 否      |
| `tools/export_mp1_onnx_parts.py` | 导出 `obs_encoder.onnx` 和 `unet_step.onnx` | 否      |
| `tools/check_onnx_parity.py`     | 检查 ONNX Runtime 对齐并生成 TensorRT 证据报告      | 否      |
| `tools/check_trt_case_parity.py` | 不加载训练代码，轻量检查已冻结 TensorRT case           | 否      |

## 部署原则

从第一性原理看，真机部署分成三层：

1. 模型层：`policy_infer.pt` 是否能被 LibTorch 正确加载并输出和 Python 一致的动作。
2. 输入层：真实相机、点云、机器人状态是否被整理成模型训练时看到的输入格式。
3. 控制层：动作是否经过安全限幅，再发送给机器人。

所以顺序必须是：

```text
离线对齐 -> dry-run -> 真实输入 dry-run -> 小限幅闭环 -> 正常闭环
```

不要在离线对齐和真实输入检查通过前直接进入真机闭环。

## 目录结构

```text
cpp_deploy/
  CMakeLists.txt
  configs/
    pole_pickoff_jetson.json
    pole_pickoff_real_robot.json
  include/mp1_deploy/
  src/
    offline_infer.cpp
    dry_run.cpp
    real_input_dry_run.cpp
    real_robot_control.cpp
    realtime_robot_control.cpp
  tests/
    test_safety_filter.cpp
  tools/
    capture_real_inputs.py
```

仓库根目录还有 TensorRT/ONNX 证据原型工具：

```text
tools/
  dump_trt_case.py
  export_mp1_onnx_parts.py
  check_onnx_parity.py
  check_trt_case_parity.py
  mp1_trt_utils.py
```

`deploy_artifacts/` 通常放在仓库根目录下，包含：

```text
deploy_artifacts/
  policy_infer.pt
  deploy_meta.json
  sample_tensors/
    global_image.pt
    wrist_image.pt
    point_cloud.pt
    agent_pos.pt
    initial_noise.pt
    expected_action.pt
```

## 1. Ubuntu 主机：冻结 Python 行为

在写 C++ 真机闭环前，先固定当前 Python checkpoint、config 和 `policy_dump` 样本：

```bash
python3 tools/freeze_python_behavior.py --max-samples 20
```

输出：

```text
deploy_artifacts/python_behavior_manifest.json
```

这一步的作用是保留 Python 黄金标准。后续 C++ 输出必须和这些样本对齐。

## 2. Ubuntu 主机：导出 TorchScript

在仓库根目录运行：

注意：当前部署包本身不包含完整 `python_deploy/` 训练仓库。下面命令里的 checkpoint、sample npz 和 Python 源码路径需要在 Ubuntu/Jetson 上按实际师兄工程补齐。

```bash
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 tools/export_mp1_policy.py \
  --checkpoint python_deploy/checkpoints/latest.ckpt \
  --config cpp_deploy/configs/pole_pickoff_real_robot.json \
  --sample-npz python_deploy/deploy_results/quganzi/policy_trace_20260423_233050/trial_000/policy_dumps/step_00010.npz \
  --output-dir deploy_artifacts \
  --device cpu
```

输出：

```text
deploy_artifacts/policy_infer.pt
deploy_artifacts/deploy_meta.json
deploy_artifacts/sample_tensors/*.pt
```

`initial_noise.pt` 由导出脚本固定随机种子生成，用于 Python/C++ 离线一致性对比。不要在 Jetson 运行时重新随机生成。

## 3. Jetson：编译 C++ 工程

需要 Jetson 上安装 LibTorch，并让 CMake 能找到它：

```bash
cd /mnt/nvme/MP1_model

cmake -S cpp_deploy -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build -j
ctest --test-dir build --output-on-failure
```

如果已经配置过 build 目录，后续只需要：

```bash
cmake --build build -j
```

当前 CMake targets 包括下面这些。`mp1_realtime_robot_control` 只有在 `MP1_ENABLE_REALTIME_PIPELINE=ON` 且 RealSense/ur_rtde C++ 依赖可被 CMake 找到时才会生成。

```text
build/mp1_offline_infer
build/mp1_dry_run
build/mp1_real_input_dry_run
build/mp1_real_robot_control
build/mp1_realtime_robot_control
build/test_safety_filter
```

当前 `CMakeLists.txt` 里 `MP1_ENABLE_UR_RTDE` 和 `MP1_ENABLE_REALTIME_PIPELINE` 默认是 `ON`。这意味着配置阶段会查找 C++ 版 `ur_rtde`，实时版本还会查找 C++ RealSense SDK。

如果 Jetson 已经装好 C++ `ur_rtde`，按下面方式显式开启真实机器人控制：

```bash
cmake -S cpp_deploy -B build \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DMP1_ENABLE_UR_RTDE=ON
cmake --build build -j
```

注意：Python 的 `pip install ur-rtde` 只保证 Python 采集脚本可用；C++ 真机控制还需要 C++ 版 `ur_rtde` 头文件和库，并且 CMake 能 `find_package(ur_rtde)`。如果配置时报找不到 `ur_rtde`，先安装/编译 C++ ur\_rtde，再把它的安装路径加入 `CMAKE_PREFIX_PATH`。

如果当前只想编译离线推理和 dry-run，或者机器上还没有 C++ `ur_rtde` / RealSense SDK，可以临时关闭硬件目标：

```bash
cmake -S cpp_deploy -B build \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch \
  -DMP1_ENABLE_UR_RTDE=OFF \
  -DMP1_ENABLE_REALTIME_PIPELINE=OFF
cmake --build build -j
```

如果要编译单进程实时版本，需要同时有 C++ RealSense SDK 和 C++ `ur_rtde`：

```bash
cmake -S cpp_deploy -B build \
  -DCMAKE_PREFIX_PATH=/mnt/nvme/libtorch/venv/torch_env/lib/python3.8/site-packages/torch \
  -DMP1_ENABLE_UR_RTDE=ON \
  -DMP1_ENABLE_REALTIME_PIPELINE=ON
cmake --build build -j
```

启用后会额外生成：

```text
build/mp1_realtime_robot_control
```

## 4. Jetson：离线对齐

先跑 CPU，不要先跑 CUDA：

```bash
cd /mnt/nvme/MP1_model/build

./mp1_offline_infer \
  --model ../deploy_artifacts/policy_infer.pt \
  --tensor-dir ../deploy_artifacts/sample_tensors \
  --device cpu
```

重点看：

```text
expected max_abs_diff: ...
```

判断标准：

```text
<= 1e-5      很好，可以认为 C++ 和 Python 离线对齐
1e-4 ~ 1e-3 需要谨慎，看动作量级
> 1e-3       不要进真机，先排查输入、导出或后处理
```

当前已知 Jetson CPU 对齐结果：

```text
expected max_abs_diff: 3.57628e-07
```

这说明 TorchScript CPU 路径和 Python 黄金样本已经对齐。

## 5. Jetson：基础 dry-run

`mp1_dry_run` 用于重复推理并检查安全限幅。它启动时加载一次黄金样本或零输入，不读取真实相机，也不读取 RTDE。

推荐命令：

```bash
cd /mnt/nvme/MP1_model/build

./mp1_dry_run \
  --model ../deploy_artifacts/policy_infer.pt \
  --tensor-dir ../deploy_artifacts/sample_tensors \
  --device cuda \
  --steps 5 \
  --warmup-steps 3 \
  --max-translation 0.002 \
  --max-rotation 0.01
```

如果没有 `sample_tensors`，可以用零输入检查程序是否能跑通：

```bash
./mp1_dry_run \
  --model ../deploy_artifacts/policy_infer.pt \
  --device cpu \
  --steps 5
```

输出会包含：

```text
input summary
inference_ms
action shape
action_pred shape
raw_action
filtered_action
```

`raw_action` 是模型原始动作，`filtered_action` 是安全限幅后的动作。早期只观察 `filtered_action`，不要直接发机器人。

## 6. 真实输入契约

真实输入最终必须写成 5 个 TorchScript 张量文件，并由 `current_frame.txt` 指向最新完整帧：

```text
real_input_tensors/
  current_frame.txt          # 内容示例：frames/000123
  frames/
    000123/
      global_image.pt
      wrist_image.pt
      point_cloud.pt
      agent_pos.pt
      initial_noise.pt
```

shape 和 dtype 必须是：

```text
global_image   [1, 2, 3, 128, 128] uint8
wrist_image    [1, 2, 3, 96, 96]   uint8
point_cloud    [1, 2, 512, 3]      float32
agent_pos      [1, 2, 10]          float32
initial_noise  [1, 4, 7]           float32
```

这里的 `2` 来自 `deploy_meta.json` 里的：

```text
n_obs_steps = 2
```

时间顺序建议保持：

```text
[旧帧, 新帧]
```

取杆任务的 `agent_pos` 单帧格式是：

```text
tcp_xyz + rot6d + gripper = 3 + 6 + 1 = 10
```

## 7. 采集真实输入

真实输入采集程序是：

```text
cpp_deploy/tools/capture_real_inputs.py
```

它负责：

```text
RealSense 全局图像
RealSense 腕部图像
RealSense 点云
UR RTDE TCP 位姿
夹爪比例
固定 initial_noise
```

它不推理、不控制 UR、不控制夹爪。

采集端会先写完整帧目录，再原子更新 `current_frame.txt`。`mp1_real_input_dry_run` 只读取 `current_frame.txt` 指向的帧目录，避免图像、点云和机器人状态来自不同时间步。

### 7.1 安装依赖

在 Jetson 的 `torch_env` 中确认有这些 Python 包：

```bash
pip install opencv-python
pip install pyrealsense2
pip install ur-rtde
```

如果 Jetson 已经通过 apt 或源码安装 OpenCV/RealSense Python 绑定，可以不重复安装。

### 7.2 查看 RealSense 序列号

```bash
rs-enumerate-devices | grep "Serial Number"
```

如果使用师兄部署 JSON，脚本会优先从 `cameras[*].serial` 自动读取序列号。你也可以手动覆盖：

```text
--global-serial
--wrist-serial
```

### 7.3 验证相机链路

如果暂时不读 RTDE，可以先用固定 TCP 位姿和固定夹爪比例：

```bash
cd /mnt/nvme/MP1_model

python3 cpp_deploy/tools/capture_real_inputs.py \
  --output-dir deploy_artifacts/real_input_tensors \
  --config cpp_deploy/configs/pole_pickoff_real_robot.json \
  --initial-noise deploy_artifacts/sample_tensors/initial_noise.pt \
  --global-serial GLOBAL_REALSENSE_SERIAL_PLACEHOLDER \
  --wrist-serial WRIST_REALSENSE_SERIAL_PLACEHOLDER \
  --no-rtde \
  --gripper-fraction 1.0 \
  --steps 0
```

这一步用于确认图像、点云、shape、dtype 和 `.pt` 写出流程。

### 7.4 读取 UR RTDE 状态

接入机器人状态时，脚本默认从部署 JSON 的 `robot.ip` 读取机器人 IP，不需要在命令行重复写：

```bash
python3 cpp_deploy/tools/capture_real_inputs.py \
  --output-dir deploy_artifacts/real_input_tensors \
  --config cpp_deploy/configs/pole_pickoff_real_robot.json \
  --initial-noise deploy_artifacts/sample_tensors/initial_noise.pt \
  --global-serial GLOBAL_REALSENSE_SERIAL_PLACEHOLDER \
  --wrist-serial WRIST_REALSENSE_SERIAL_PLACEHOLDER \
  --gripper-fraction 1.0 \
  --image-order rgb \
  --rot6d-mode cols \
  --steps 0
```

如果临时调试需要覆盖 JSON 里的 IP，再显式传：

```text
--robot-ip <ur_robot_ip>
```

`--steps 0` 表示持续采集，手动 `Ctrl+C` 停止。

夹爪状态目前有两种方式：

```text
--gripper-fraction 1.0
--gripper-file /path/to/gripper_fraction.txt
```

`--gripper-file` 读取文本文件里的一个 `0.0~1.0` 数值，方便后续接入你自己的夹爪状态进程。

### 7.5 点云坐标系

默认点云使用部署 JSON 里的 `collection.primary_point_cloud_camera`。师兄取杆配置通常是：

```text
primary_point_cloud_camera = wrist_d405
```

也可以手动覆盖：

```text
--pointcloud-camera wrist
--pointcloud-camera global
```

并按配置裁剪：

```json
"crop_min": [0.28, -0.2, 0.0]
"crop_max": [0.62, 0.2, 0.35]
```

如果训练时点云不是相机坐标系，需要传入 4x4 变换矩阵：

```bash
--pointcloud-transform r00,r01,r02,tx,r10,r11,r12,ty,r20,r21,r22,tz,0,0,0,1
```

这是最容易出错的地方。模型只认识训练时的坐标系，真实点云必须先变换到同一个坐标系。

取杆任务如果训练/采集时使用 base 坐标点云，就必须传入 camera->base 外参。当前脚本没有自动读取师兄部署 JSON 里的 `base_T_camera`，所以不要在外参未确认时把真实输入动作当作有效结论。

### 7.6 rot6d 格式

取杆任务默认与师兄 Python 部署代码一致：

```text
--rot6d-mode cols
```

它对应 Python 版：

```text
rotation[:, :2].T.reshape(-1)
```

如果以后确认某个旧模型使用其它展开方式，才显式改参数。这个必须和 Python 的 `tcp_xyz_rot6d + gripper` 构造方式一致。

## 8. 真实输入 dry-run

采集程序运行后，另开一个终端运行：

```bash
cd /mnt/nvme/MP1_model/build

./mp1_real_input_dry_run \
  --model ../deploy_artifacts/policy_infer.pt \
  --input-dir ../deploy_artifacts/real_input_tensors \
  --device cuda \
  --steps 20 \
  --warmup-steps 3 \
  --poll-ms 200 \
  --require-update 1 \
  --max-translation 0.002 \
  --max-rotation 0.01
```

参数说明：

| 参数                  | 作用                           |
| ------------------- | ---------------------------- |
| `--model`           | TorchScript 模型路径             |
| `--input-dir`       | 真实输入张量目录                     |
| `--device`          | 推理设备，CUDA 跑通后建议 `cuda`       |
| `--steps`           | 推理次数，`0` 表示持续循环              |
| `--warmup-steps`    | 正式 step 前的预热推理次数，CUDA 默认 `3` |
| `--poll-ms`         | 等待新输入的轮询间隔                   |
| `--require-update`  | `1` 表示只在输入文件更新后推理            |
| `--max-translation` | 单步最大平移，单位米                   |
| `--max-rotation`    | 单步最大旋转，单位弧度                  |

判断重点：

```text
frame_dir 是否持续更新
shape/dtype 是否正确
load_ms 是否稳定
warmup 后 inference_ms 是否稳定
raw_action 是否异常跳变
filtered_action 是否被安全限幅
动作方向和尺度是否符合预期
```

## 9. 真实机器人小限幅闭环

真实机器人控制程序是：

```text
build/mp1_real_robot_control
```

它和 `mp1_real_input_dry_run` 使用同一套真实输入目录、TorchScript 模型、CUDA warmup 和 `SafetyFilter`。区别是：当显式开启执行时，它会读取 UR 当前 TCP 位姿，把 `filtered_action` 前 6 维作为 TCP 增量，并默认转成 UR RTDE `speedL` 速度控制指令。程序会先检查 `current_tcp` 和预测 `target_tcp` 是否在 workspace 内；超出边界时跳过命令，不再把目标强行 clamp 到边界。

速度控制参数写在 `cpp_deploy/configs/pole_pickoff_real_robot.json` 的 `policy_speed_control` 中：

```json
"policy_speed_control": {
  "control_mode": "speedl",
  "control_hz": 2.0,
  "speed_slider_fraction": 0.05,
  "max_linear_speed_mps": 0.015,
  "max_angular_speed_rps": 0.05,
  "cartesian_acceleration": 0.05,
  "speed_command_time_s": 0.5,
  "speed_stop_acceleration": 0.2
}
```

其中 `speed_slider_fraction` 对应 UR 示教器上的速度控制条；`max_linear_speed_mps` 和 `max_angular_speed_rps` 是本程序对模型动作再做的一层速度上限。

实现上会同时使用两个 C++ RTDE 接口：

```text
RTDEIOInterface::setSpeedSlider     设置示教器等价速度条
RTDEControlInterface::speedL        发送 TCP 笛卡尔速度
RTDEControlInterface::speedStop     退出时减速停止
```

默认不发机器人命令：

```bash
cd /mnt/nvme/MP1_model/build

./mp1_real_robot_control \
  --model ../deploy_artifacts/policy_infer.pt \
  --input-dir ../deploy_artifacts/real_input_tensors \
  --config ../cpp_deploy/configs/pole_pickoff_real_robot.json \
  --device cuda \
  --steps 10 \
  --warmup-steps 3 \
  --control-hz 2 \
  --require-update 1 \
  --max-translation 0.0005 \
  --max-rotation 0.003 \
  --execute 0
```

确认 `delta_tcp` 方向和尺度合理后，才允许真发命令：

```bash
./mp1_real_robot_control \
  --model ../deploy_artifacts/policy_infer.pt \
  --input-dir ../deploy_artifacts/real_input_tensors \
  --config ../cpp_deploy/configs/pole_pickoff_real_robot.json \
  --device cuda \
  --steps 10 \
  --warmup-steps 3 \
  --control-hz 2 \
  --require-update 1 \
  --max-translation 0.0005 \
  --max-rotation 0.003 \
  --execute 1 \
  --confirm RUN_ROBOT
```

安全边界：

```text
默认 control_hz = 2
默认 max_translation = 0.0005 m
默认 max_rotation = 0.003 rad
默认 control_mode = speedl
默认 speed_slider_fraction = 0.05
默认 max_linear_speed_mps = 0.015
默认 max_angular_speed_rps = 0.05
默认忽略第7维夹爪动作
必须显式 --execute 1 --confirm RUN_ROBOT 才发命令
```

第一版闭环只做 10 步。若方向正确、无抖动、workspace 没有频繁跳过命令，再逐步放宽到：

```text
control_hz = 5
max_translation = 0.002
max_rotation = 0.01
speed_slider_fraction = 0.10
max_linear_speed_mps = 0.03
```

注意：当前小限幅阶段把旋转向量增量按 base frame 加到当前 TCP pose 上。这个近似只适合极小动作验证；后续扩大动作前，需要改成严格的 SE(3) 位姿组合。

## 10. 单进程实时 C++ 闭环

`mp1_realtime_robot_control` 用 C++ 在一个进程内完成：

```text
RealSense 双相机采集
UR RTDE 当前 TCP 读取
内存 ring buffer 保存最近观测
TorchScript CUDA 推理
SafetyFilter 小限幅
默认 speedL 速度控制
```

它不再依赖 `capture_real_inputs.py` 写 `.pt` 文件，也不读取 `real_input_tensors/current_frame.txt`。这个程序用于降低文件 I/O 和跨进程轮询带来的延迟，并避免图像、点云、机器人状态混帧。

默认不发机器人命令：

```bash
cd /mnt/nvme/MP1_model/build

./mp1_realtime_robot_control \
  --model ../deploy_artifacts/policy_infer.pt \
  --config ../cpp_deploy/configs/pole_pickoff_real_robot.json \
  --initial-noise ../deploy_artifacts/sample_tensors/initial_noise.pt \
  --device cuda \
  --steps 10 \
  --warmup-steps 3 \
  --capture-hz 15 \
  --control-hz 2 \
  --ring-size 8 \
  --max-translation 0.0005 \
  --max-rotation 0.003 \
  --execute 0
```

确认 `delta_tcp`、`capture_ms`、`inference_ms` 和 `command_sent=0` 正常后，才允许真发：

```bash
./mp1_realtime_robot_control \
  --model ../deploy_artifacts/policy_infer.pt \
  --config ../cpp_deploy/configs/pole_pickoff_real_robot.json \
  --initial-noise ../deploy_artifacts/sample_tensors/initial_noise.pt \
  --device cuda \
  --steps 10 \
  --warmup-steps 3 \
  --capture-hz 15 \
  --control-hz 2 \
  --ring-size 8 \
  --max-translation 0.0005 \
  --max-rotation 0.003 \
  --execute 1 \
  --confirm RUN_ROBOT
```

实时版本的安全策略：

```text
只取 ring buffer 里最新两帧组成 [B=1,T=2,...]
默认第7维夹爪动作置零
默认使用 policy_speed_control.control_mode = speedl
启动 execute 后先设置 UR speed slider
delta_tcp 会转换为 speed_tcp，并按线速度/角速度模长限幅
current_tcp 超出 workspace 时跳过命令
target_tcp 超出 workspace 时跳过命令
不会把 target 强行 clamp 到 workspace 边界
```

如果只需要排查模型输出或文件帧链路，继续用 `mp1_real_input_dry_run` 和 `mp1_real_robot_control`。如果要验证真实闭环时序，再切到 `mp1_realtime_robot_control`。

## 11. 当前已知限制

### CUDA warmup

CUDA 第一次推理会初始化上下文、选择 kernel、分配缓存，所以前几次 `inference_ms` 不能代表稳定速度。当前程序默认规则是：

```text
--device cuda -> 默认 --warmup-steps 3
--device cpu  -> 默认 --warmup-steps 0
```

warmup 只做模型 forward，不计入正式 step，不发送机器人命令，也不作为控制输出。

如果再次遇到旧版模型的 CUDA device 错误：

```text
Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

先用 `--device cpu` 保持正确性验证，再回到导出脚本修 normalizer 常量 device。

### 推理频率

Jetson CPU dry-run 已观察到推理耗时大约是数百毫秒量级；CUDA warmup 后已能到几十毫秒量级。正式闭环前以 warmup 后的 `inference_ms` 为准，建议先按 5Hz 小限幅验证，不要一开始追更高控制频率。

### 夹爪状态

`capture_real_inputs.py` 当前不直接连接远程夹爪。夹爪比例可以先固定，或由外部进程写入文本文件再通过 `--gripper-file` 读取。

## 12. TensorRT 证据原型

结论：TensorRT 当前只做离线证据原型，不直接接入 `mp1_real_robot_control`。真机闭环仍以 `policy_infer.pt` 的 TorchScript 路径为主路径。

从第一性原理看，TensorRT 加速要先证明三件事：

1. `obs_encoder.onnx` 输出的 `global_cond` 和 PyTorch 一致。
2. `unet_step.onnx` 每一步输出的 `v_pred` 和 PyTorch 一致。
3. 外层积分 `x_current = x_current + v_pred * dt` 后得到的 `final_action` 误差小于安全限幅预算。

推荐 Jetson 执行顺序：

```bash
cd /mnt/nvme/MP1_model

python3 tools/dump_trt_case.py \
  --checkpoint python_deploy/checkpoints/latest.ckpt \
  --tensor-dir deploy_artifacts/sample_tensors \
  --output-dir deploy_artifacts/trt_cases/case_000 \
  --device cpu

python3 tools/export_mp1_onnx_parts.py \
  --checkpoint python_deploy/checkpoints/latest.ckpt \
  --tensor-dir deploy_artifacts/sample_tensors \
  --output-dir deploy_artifacts/onnx \
  --device cpu

python3 tools/check_onnx_parity.py \
  --checkpoint python_deploy/checkpoints/latest.ckpt \
  --tensor-dir deploy_artifacts/sample_tensors \
  --onnx-dir deploy_artifacts/onnx \
  --torchscript-model deploy_artifacts/policy_infer.pt \
  --device cuda \
  --warmup 20 \
  --repeats 200 \
  --report deploy_artifacts/TRT_ACCEL_REPORT.md
```

如果 Jetson TensorRT 验证环境不想安装完整训练依赖，可以使用轻量 case 检查脚本。它只消费 `dump_trt_case.py` 生成的 `case_000` 和 ONNX 文件，不加载 checkpoint、Hydra、dill 或训练代码：

```bash
python3 tools/check_trt_case_parity.py \
  --case-dir deploy_artifacts/trt_cases/case_000 \
  --onnx-dir deploy_artifacts/onnx \
  --trt-output-dir deploy_artifacts/trt_engines \
  --report deploy_artifacts/TRT_ACCEL_REPORT_LITE.md
```

生成 TensorRT engine：

```bash
mkdir -p deploy_artifacts/trt_engines

trtexec --onnx=deploy_artifacts/onnx/obs_encoder.onnx \
  --saveEngine=deploy_artifacts/trt_engines/obs_encoder_fp16.engine \
  --fp16 --warmUp=500 --duration=20

trtexec --onnx=deploy_artifacts/onnx/unet_step.onnx \
  --saveEngine=deploy_artifacts/trt_engines/unet_step_fp16.engine \
  --fp16 --warmUp=500 --duration=20
```

`deploy_artifacts/TRT_ACCEL_REPORT.md` 必须记录：

```text
TorchScript CUDA p50 / p95 / p99
TensorRT FP16 p50 / p95 / p99
global_cond 误差
v_pred 误差
final_action 误差
Jetson 型号、JetPack、CUDA、TensorRT 版本
```

如果 TensorRT runner 已经能导出 `.npy`，约定放在：

```text
deploy_artifacts/trt_engines/global_cond_trt.npy
deploy_artifacts/trt_engines/v_pred_trt_000.npy
deploy_artifacts/trt_engines/action_trt.npy
```

`tools/check_onnx_parity.py` 会自动读取这些文件并计算 PyTorch vs TensorRT FP16 误差。

验收边界：

```text
PyTorch vs ONNX Runtime FP32: max_abs_diff <= 1e-4 为目标
PyTorch vs TensorRT FP16: final_action 误差 < 单步安全限幅的 5%
TensorRT 任一子图误差超阈值时，不进入真机链路
```

## 13. 下一步

当 `mp1_real_input_dry_run` 稳定后，再进入机器人小限幅闭环：

1. 确认真实输入 shape/dtype 全部正确。
2. 确认点云坐标系和训练时一致。
3. 确认 `agent_pos = tcp_xyz + rot6d + gripper` 和 Python 训练代码一致。
4. 确认 `filtered_action` 方向和尺度合理。
5. 用更保守限幅进入 dry-run 控制程序。
6. 最后才进入正常闭环。

建议初始安全限幅：

```bash
--max-translation 0.002 --max-rotation 0.01
```

