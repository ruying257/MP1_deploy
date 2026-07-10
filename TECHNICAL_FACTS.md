# MP1 Jetson 部署项目技术事实盘点

本文档只整理仓库中可验证的信息，用于后续简历和面试素材筛选。未在仓库中找到直接证据的内容不作为已验证事实。

## A. 项目目标与部署对象

- 项目任务：将 MP1 Python 真机部署链迁移到 Jetson C++/LibTorch 运行链路，当前任务为 `pole_pickoff / 取杆`。
- 部署平台：Jetson aarch64。CTest XML 记录主机 `NVIDIA`、`OSRelease=5.10.120-tegra`、`OSPlatform=aarch64`、12 逻辑 CPU、约 62.8GB 物理内存。
- 模型：MP1 策略模型，导出为 TorchScript `deploy_artifacts/policy_infer.pt`，并拆出 ONNX 子图 `obs_encoder.onnx`、`unet_step.onnx` 用于 TensorRT 证据原型。
- 输入：双 RealSense D405，全局相机 + 腕部相机；输入包含全局 RGB、腕部 RGB、点云、UR TCP 位姿 rot6d、夹爪比例、固定 `initial_noise`。
- 控制对象：UR12E 机械臂，配置 IP `192.168.31.100`；控制链路包含 RTDE 读取 TCP 和可选 `servoL` 发命令。
- 整体链路：Python checkpoint/config -> 冻结 Python 行为 -> TorchScript 导出 -> C++/LibTorch 离线对齐 -> fixed-input dry-run -> real-input dry-run -> 小限幅真机控制入口 -> ONNX/TensorRT 离线证据原型。

## B. 我完成的核心工作清单

### 1. 模型迁移到 TorchScript

- 背景/问题：原始策略依赖 Python 训练/部署代码，不能直接在 Jetson C++ 真机链路中稳定使用。
- 动作：实现 `tools/export_mp1_policy.py`，把 normalizer、encoder、U-Net 采样循环封装进 `MP1TorchScriptWrapper`，导出 `policy_infer.pt`。
- 技术细节：TorchScript 输入为 `global_image, wrist_image, point_cloud, agent_pos, initial_noise`，输出为 `(action, action_pred)`。
- 结果/指标：TorchScript 文件约 1004MB；checkpoint 约 4.0GB；运行时 action 输出为 `[1, 3, 7]`，完整 `action_pred` 为 `[1, 4, 7]`。
- 证据：`tools/export_mp1_policy.py`、`deploy_artifacts/deploy_meta.json`。

### 2. 修复 TorchScript CUDA 设备固化风险

- 背景/问题：README 记录旧版模型在 CUDA 推理时可能出现 CPU/CUDA 混用错误。
- 动作：导出脚本中不直接调用原 normalizer 的 `normalize`，而是手写 `_normalize_field`。
- 技术细节：注释明确说明 `torch.jit.trace` 会把导出时 device 写死，Jetson CUDA 会出现部分算子 CPU、部分算子 GPU。
- 结果/意义：为同一 TorchScript 模型支持 CPU 正确性验证和 Jetson CUDA 推理提供基础。
- 证据：`tools/export_mp1_policy.py`、`cpp_deploy/README_CN.md`。

### 3. Python 行为冻结

- 背景/问题：C++ 迁移需要 Python 黄金标准，防止 C++ 只跑通 shape 但语义错误。
- 动作：实现/使用 `tools/freeze_python_behavior.py` 生成 `python_behavior_manifest.json`。
- 技术细节：冻结 20 个 policy dump 样本，记录输入 shape/dtype/statistics 和 raw/executed action。
- 结果/指标：`sample_count=20`；输入契约包括 `global_image [1,2,3,128,128]`、`wrist_image [1,2,3,96,96]`、`point_cloud [1,2,512,3]`、`agent_pos [1,2,10]`。
- 证据：`deploy_artifacts/python_behavior_manifest.json`。

### 4. C++/LibTorch 推理运行时

- 动作：实现 `TorchScriptRuntime`，加载 JIT module、`eval()`、`NoGradGuard`、把 5 路输入搬到目标 device。
- 技术细节：返回 action/action_pred，并统一 `.cpu()` 回传，便于后处理和日志打印。
- 结果/意义：支撑 `mp1_offline_infer`、`mp1_dry_run`、`mp1_real_input_dry_run`、机器人控制入口复用。
- 证据：`cpp_deploy/src/torchscript_runtime.cpp`。

### 5. 离线数值一致性验证

- 动作：实现 `mp1_offline_infer`，加载样本张量和 expected action，计算 `expected max_abs_diff`。
- 结果/指标：README 记录 Jetson CPU 对齐结果 `expected max_abs_diff: 3.57628e-07`。
- 意义：证明 TorchScript CPU 路径和 Python 黄金样本在单样本上数值一致。
- 证据：`cpp_deploy/src/offline_infer.cpp`、`cpp_deploy/README_CN.md`。

### 6. Fixed-Input Dry-Run

- 动作：实现 `mp1_dry_run`，支持 sample tensors 或 zero input，支持 CPU/CUDA，支持 warmup。
- 技术细节：CUDA 默认 warmup 3 次，CPU 默认 0 次；打印 `inference_ms`、action shape、raw/filtered action。
- 意义：把“模型可跑”和“动作经过安全限幅”拆出来验证，不接触机器人。
- 证据：`cpp_deploy/src/dry_run.cpp`。

### 7. Real-Input Dry-Run

- 动作：实现 `mp1_real_input_dry_run`，读取 `current_frame.txt` 指向的完整帧目录，加载 5 个 TorchScript tensor。
- 技术细节：验证 shape 和 dtype；打印 `load_ms`、`inference_ms`、action shape、raw/filtered action；默认不发命令。
- 结果/指标：仓库中存在 2444 个真实输入帧目录、12220 个 `.pt` 文件，`current_frame.txt` 指向 `frames/002443`。
- 证据：`cpp_deploy/src/real_input_dry_run.cpp`、`deploy_artifacts/real_input_tensors/frames/`。

### 8. 真实输入原子提交机制

- 背景/问题：采集端持续写文件时，C++ 轮询可能读到半写入或混帧数据。
- 动作：`capture_real_inputs.py` 用临时文件 + `os.replace` 原子写 tensor，并最后原子更新 `current_frame.txt`。
- 技术细节：每帧目录包含 `global_image.pt, wrist_image.pt, point_cloud.pt, agent_pos.pt, initial_noise.pt`。
- 意义：降低 real-input dry-run 读到半帧/混帧的风险。
- 证据：`cpp_deploy/tools/capture_real_inputs.py`。

### 9. 多模态输入预处理

- 动作：实现图像 resize、BGR/RGB 通道转换、HWC->CHW、点云裁剪、FPS 下采样、TCP rotvec->rot6d、gripper fraction 拼接。
- 技术细节：global image 128x128，wrist image 96x96，点云 512 点，agent_pos 10 维。
- 结果/意义：把双相机、点云、UR TCP、夹爪状态统一成模型输入契约。
- 证据：`cpp_deploy/tools/capture_real_inputs.py`。

### 10. 安全限幅与夹爪屏蔽

- 动作：实现 `SafetyFilter`，支持平移/旋转 L2 范数裁剪、deadband、EMA、夹爪动作默认置零。
- 技术细节：默认最大平移 0.015m、最大旋转 0.08rad；小限幅配置 0.005m / 0.03rad；real robot 入口默认 0.0005m / 0.003rad。
- 结果/意义：把模型输出和机器人执行之间加了一层可配置安全边界。
- 证据：`cpp_deploy/src/safety_filter.cpp`、`cpp_deploy/tests/test_safety_filter.cpp`。

### 11. 安全过滤单元测试

- 动作：写 `test_safety_filter.cpp`，构造平移 0.1m、旋转 0.2rad、夹爪 -2.0 的动作。
- 结果/指标：断言平移范数裁剪到 0.005，旋转范数裁剪到 0.03，夹爪维置零。
- 注意：当前 CTest XML 的 TestList 为空，不能从日志宣称测试已运行通过；只能说测试源码和二进制存在。
- 证据：`cpp_deploy/tests/test_safety_filter.cpp`。

### 12. UR RTDE 控制入口

- 动作：实现 `mp1_real_robot_control`，读取真实输入、推理、滤波、转换为 delta TCP，并在 execute 模式下通过 UR RTDE `servoL` 发送。
- 技术细节：默认 `execute=false`；真机执行必须 `--execute 1 --confirm RUN_ROBOT`；支持 workspace clamp。
- 风险：当前工作区该文件含 Git 冲突标记，不能作为当前干净可构建源码事实。
- 证据：`cpp_deploy/src/real_robot_control.cpp`。

### 13. 单进程实时管线

- 背景/问题：Python 采集 + 文件轮询会增加延迟和混帧风险。
- 动作：实现 `mp1_realtime_robot_control`，用 C++ 单进程集成 RealSense、UR RTDE、ring buffer、TorchScript 推理和可选 servoL 控制。
- 技术细节：默认 camera 640x480@15fps，capture_hz 15，control_hz 2，ring_size 8，点云 512 点。
- 意义：为降低闭环延迟、减少文件 IO 依赖提供工程路径。
- 证据：`cpp_deploy/src/realtime_robot_control.cpp`。

### 14. ONNX 子图拆分

- 动作：实现 `export_mp1_onnx_parts.py`，导出 `obs_encoder.onnx` 和 `unet_step.onnx`。
- 技术细节：obs encoder 输入多模态观测输出 `global_cond`；unet step 输入 `x_current, timestep, global_cond, r` 输出 `v_pred`。
- 结果/指标：`obs_encoder.onnx` 约 2.2MB，`unet_step.onnx` 约 1002MB。
- 证据：`tools/export_mp1_onnx_parts.py`、`deploy_artifacts/onnx/`。

### 15. TensorRT 证据原型

- 动作：实现 `dump_trt_case.py`、`run_trt_case.py`、`check_trt_case_parity.py`。
- 技术细节：case 不加载 checkpoint；只消费冻结的 `.npy` 和 ONNX/engine；TensorRT runner 使用 torch CUDA tensor 作为 device buffer，避免 pycuda 依赖。
- 结果/指标：已有 `obs_encoder_fp16.engine` 约 2.3MB，`unet_step_fp16.engine` 约 504MB。
- 证据：`tools/run_trt_case.py`、`deploy_artifacts/trt_engines/`。

### 16. ONNX/TensorRT 数值误差统计

- 结果/指标：
  - ONNX `global_cond` max_abs_diff: 0.000384748
  - ONNX `v_pred_step_000` max_abs_diff: 0.000145674
  - ONNX `final_action` max_abs_diff: 9.52482e-05
  - TensorRT `global_cond` max_abs_diff: 0.00334206
  - TensorRT `v_pred_step_000` max_abs_diff: 0.00225765
  - TensorRT `final_action` max_abs_diff: 0.000688314
- 意义：TensorRT 有离线误差证据，但 final action 误差相对小限幅 0.005m 的 5% 阈值是否合格需要谨慎判断。
- 证据：`deploy_artifacts/TRT_ACCEL_REPORT_LITE.md`。

### 17. TensorRT Profiling 数据

- 结果/指标：profile JSON 中 obs_encoder 非零层 103 个，sum averageMs 约 1.056ms；unet_step 非零层 154 个，sum averageMs 约 4.930ms。
- 注意：这是层级 profile 加总，不等价于完整端到端闭环延迟。
- 证据：`deploy_artifacts/trt_engines/obs_encoder_profile.json`、`deploy_artifacts/trt_engines/unet_step_profile.json`。

## C. 可量化结果

| 模块/环节 | 指标名称 | 数值 | 优化前 | 优化后 | 备注/证据位置 |
| --- | --- | --- | --- | --- | --- |
| Python 行为冻结 | 样本数 | 20 | 无 | 20 | `python_behavior_manifest.json` |
| 输入契约 | global_image | `[1,2,3,128,128] uint8` | Python dump | C++ 校验 | `real_input_dry_run.cpp` |
| 输入契约 | wrist_image | `[1,2,3,96,96] uint8` | Python dump | C++ 校验 | `real_input_dry_run.cpp` |
| 输入契约 | point_cloud | `[1,2,512,3] float32` | 原始深度点云 | 裁剪+FPS 512点 | `capture_real_inputs.py` |
| 输入契约 | agent_pos | `[1,2,10] float32` | UR TCP pose | xyz + rot6d + gripper | `capture_real_inputs.py` |
| 输入契约 | initial_noise | `[1,4,7] float32` | 随机 | 固定复用 | `deploy_meta.json` |
| 动作输出 | action_dim | 7 | Python 模型 | C++ 保持 7 维契约 | `deploy_meta.json` |
| 动作输出 | runtime action shape | `[1,3,7]` | action_pred `[1,4,7]` | 执行动作窗口 `[1,3,7]` | `deploy_meta.json` |
| 控制频率 | deploy meta control_hz | 5Hz | 未统一 | 5Hz | `deploy_meta.json` |
| 离线一致性 | C++ vs expected action | `3.57628e-07` | 未验证 | Jetson CPU 对齐 | `README_CN.md` |
| 真实输入 | 完整帧目录数 | 2444 | 无 | 2444 | `real_input_tensors/frames` |
| 真实输入 | tensor 文件数 | 12220 | 无 | 2444 * 5 | `real_input_tensors/frames` |
| 模型大小 | checkpoint | 4.0GB | 原始 | 原始 | `du -h` |
| 模型大小 | TorchScript | 1004MB | checkpoint 4.0GB | 1004MB | `policy_infer.pt` |
| ONNX 大小 | obs_encoder | 2.2MB | TorchScript 1004MB | 子图 2.2MB | `deploy_artifacts/onnx` |
| ONNX 大小 | unet_step | 1002MB | TorchScript 1004MB | 子图 1002MB | `deploy_artifacts/onnx` |
| TensorRT engine | obs_encoder FP16 | 2.3MB | ONNX 2.2MB | engine 2.3MB | `trt_engines` |
| TensorRT engine | unet_step FP16 | 504MB | ONNX 1002MB | engine 504MB | `trt_engines` |
| ONNX 误差 | global_cond max_abs_diff | 0.000384748 | 无 | ONNX Runtime | `TRT_ACCEL_REPORT_LITE.md` |
| ONNX 误差 | v_pred step0 max_abs_diff | 0.000145674 | 无 | ONNX Runtime | 同上 |
| ONNX 误差 | final_action max_abs_diff | 9.52482e-05 | 无 | ONNX Runtime | 同上 |
| TensorRT 误差 | global_cond max_abs_diff | 0.00334206 | 无 | TensorRT FP16 | 同上 |
| TensorRT 误差 | v_pred step0 max_abs_diff | 0.00225765 | 无 | TensorRT FP16 | 同上 |
| TensorRT 误差 | final_action max_abs_diff | 0.000688314 | 无 | TensorRT FP16 | 同上 |
| ONNX Runtime 延迟 | p50/p95/p99/mean | 32699ms | 无 | 报告值 | 只有报告值，可信度需复测 |
| TensorRT profile | obs_encoder sum averageMs | 约 1.056ms | 无 | TensorRT profile | 层级 profile 加总 |
| TensorRT profile | unet_step sum averageMs | 约 4.930ms | 无 | TensorRT profile | 层级 profile 加总 |
| TensorRT profile | obs profile count | 2231 | 无 | trtexec profile | `obs_encoder_profile.json` |
| TensorRT profile | unet profile count | 614 | 无 | trtexec profile | `unet_step_profile.json` |
| 安全限幅 | 默认平移 | 0.015m | 无 | SafetyFilter 默认 | `safety_filter.hpp` |
| 安全限幅 | 默认旋转 | 0.08rad | 无 | SafetyFilter 默认 | 同上 |
| 小限幅配置 | 平移/旋转 | 0.005m / 0.03rad | normal 0.015/0.08 | 小限幅 | `pole_pickoff_jetson.json` |
| real robot 默认 | 平移/旋转 | 0.0005m / 0.003rad | 小限幅 0.005/0.03 | 更保守 | `real_robot_control.cpp` |
| RealSense | 采集分辨率/FPS | 640x480@15fps | 未统一 | 配置化 | `pole_pickoff_real_robot.json` |
| 点云裁剪 | crop_min/crop_max | `[0.28,-0.2,0.0]` / `[0.62,0.2,0.35]` | 原始点云 | ROI 点云 | config |
| Jetson 构建 | 架构/系统 | aarch64, 5.10.120-tegra | x86 开发 | Jetson | CTest XML |
| C++ 构建产物 | 可执行文件 | 6 个 | 无 | build 目录已有 | `ls build/mp1_*` |

## D. 关键技术难点与解决方法

### TorchScript 导出后 CPU/CUDA Device 混用

- 现象：README 记录 CUDA 推理可能报 `Expected all tensors to be on the same device... cuda:0 and cpu`。
- 原因：trace 时如果 normalizer 内部把输入搬到 scale.device，会把导出设备固化进图。
- 办法：在 `MP1TorchScriptWrapper` 中手写 `_normalize_field`，只匹配 dtype，不把输入强行搬到 normalizer 参数 device。
- 结果：导出模型能够作为 CPU 校验和 CUDA 推理的统一入口；仓库记录 CPU 对齐误差 `3.57628e-07`。

### 真实输入文件轮询存在混帧风险

- 现象：采集端持续写 `.pt`，推理端可能读到部分新帧、部分旧帧。
- 原因：多个 tensor 文件不是天然事务提交。
- 办法：每帧写入独立 `frames/xxxxxx/` 目录，tensor 用临时文件 + `os.replace` 原子替换，最后写 `current_frame.txt` 提交。
- 结果：C++ 只读取已提交完整帧；仓库已有 2444 个完整帧目录。

### Python 多模态输入到 C++ 模型输入契约复杂

- 现象：模型输入涉及双图像、点云、TCP pose、rot6d、夹爪和 initial noise。
- 原因：训练/部署代码里的预处理隐含了 shape、dtype、通道顺序、点云采样和坐标约定。
- 办法：显式实现 shape/dtype 校验和采集转换：RGB/CHW、固定图像尺寸、点云裁剪+FPS、TCP rotvec->rot6d、2 帧堆叠。
- 结果：真实输入 dry-run 能在读取阶段强校验 `[1,2,3,128,128]`、`[1,2,512,3]` 等契约。

### 模型输出 7 维但当前不控制夹爪

- 现象：模型动作包含夹爪，但当前 C++ 控制链路只应执行 TCP 6 维。
- 原因：训练模型契约是 `delta_tcp_pose_gripper`，但真实夹爪控制暂未接入/不应误发。
- 办法：SafetyFilter 保持 7 维 shape，但 `ignore_gripper_action=true` 时把最后一维置零。
- 结果：既保持模型输出契约，又避免夹爪误动作。

### 真机执行风险

- 现象：推理输出不能直接发给 UR。
- 原因：动作可能超限，输入可能语义错，工作空间可能越界。
- 办法：默认 dry-run；必须 `--execute 1 --confirm RUN_ROBOT`；执行前经过限幅、workspace 检查/裁剪。
- 结果：仓库中所有真机入口都有显式执行门槛。

### TensorRT 不能直接承诺可用于真机

- 现象：TensorRT FP16 会带来数值误差。
- 原因：ONNX/TensorRT 子图、FP16、外层积分和反归一化都可能放大误差。
- 办法：先做冻结 case，分别比较 `global_cond`、`v_pred_step_000`、`final_action`。
- 结果：已有误差报告，但 TensorRT 仍标注为离线证据原型，不直接进入真机链路。

### 单进程实时化

- 现象：Python 采集 + 文件 IO + C++ 轮询会增加 `load_ms` 和同步成本。
- 办法：新增 C++ 实时入口，用 ring buffer 保存最新两帧，采集线程和控制线程分离。
- 结果：源码层面打通了 RealSense + RTDE + LibTorch 的单进程架构；缺少完整运行 benchmark。

## E. 可直接写进简历的高价值事实素材

1. 事实：将 MP1 Python 策略导出为包含 normalizer、encoder 和 U-Net 采样循环的 TorchScript 模型。证据：`tools/export_mp1_policy.py`、`policy_infer.pt`。可量化信息：TorchScript 约 1004MB，checkpoint 约 4.0GB。能力标签：模型迁移。
2. 事实：建立 Python 行为冻结机制，为 C++ 迁移提供黄金样本。证据：`python_behavior_manifest.json`。可量化信息：冻结 20 个样本。能力标签：数值一致性验证。
3. 事实：实现 C++/LibTorch TorchScript 推理运行时。证据：`torchscript_runtime.cpp`。可量化信息：5 路输入，2 路输出，支持 CPU/CUDA device。能力标签：C++工程化。
4. 事实：实现 C++ 离线对齐工具，比较 C++ 输出和 expected action。证据：`offline_infer.cpp`、`README_CN.md`。可量化信息：Jetson CPU `expected max_abs_diff=3.57628e-07`。能力标签：数值一致性验证。
5. 事实：设计多模态输入契约校验。证据：`real_input_dry_run.cpp`。可量化信息：global `[1,2,3,128,128]`，wrist `[1,2,3,96,96]`，point cloud `[1,2,512,3]`，agent_pos `[1,2,10]`。能力标签：多模态输入处理。
6. 事实：实现真实输入采集到 TorchScript tensor 的原子提交机制。证据：`capture_real_inputs.py`。可量化信息：仓库中已有 2444 个完整帧目录，12220 个 tensor 文件。能力标签：机器人系统集成。
7. 事实：实现 RealSense 双相机输入预处理。证据：`capture_real_inputs.py`。可量化信息：640x480@15fps 采集，输出 128x128 和 96x96 CHW 图像。能力标签：多模态输入处理。
8. 事实：实现点云 ROI 裁剪和 FPS 下采样。证据：`capture_real_inputs.py`、`pole_pickoff_real_robot.json`。可量化信息：裁剪框 `[0.28,-0.2,0.0]` 到 `[0.62,0.2,0.35]`，输出 512 点。能力标签：多模态输入处理。
9. 事实：实现 TCP rotvec 到 rot6d 的状态构造。证据：`capture_real_inputs.py`。可量化信息：agent_pos 10 维 = xyz 3 + rot6d 6 + gripper 1。能力标签：机器人系统集成。
10. 事实：实现 C++ SafetyFilter，限制模型动作进入机器人。证据：`safety_filter.cpp`。可量化信息：默认 0.015m / 0.08rad；小限幅 0.005m / 0.03rad；real robot 默认 0.0005m / 0.003rad。能力标签：安全控制。
11. 事实：保留 7 维模型契约但默认屏蔽夹爪动作。证据：`safety_filter.cpp`。可量化信息：7 维动作最后一维置零。能力标签：安全控制。
12. 事实：实现 fixed-input dry-run 和 real-input dry-run。证据：`dry_run.cpp`、`real_input_dry_run.cpp`。可量化信息：打印 warmup、load_ms、inference_ms、raw_action、filtered_action。能力标签：端侧部署。
13. 事实：加入 CUDA warmup 机制，避免首次 CUDA 初始化耗时污染正式 step。证据：`dry_run.cpp`、`real_input_dry_run.cpp`、`README_CN.md`。可量化信息：CUDA 默认 warmup 3 次，CPU 默认 0 次。能力标签：CUDA性能优化。
14. 事实：实现 UR RTDE 真机控制入口的执行保护。证据：`real_robot_control.cpp`。可量化信息：执行必须 `--execute 1 --confirm RUN_ROBOT`。能力标签：机器人系统集成。
15. 事实：实现单进程实时采集-推理-控制原型。证据：`realtime_robot_control.cpp`。可量化信息：ring_size 默认 8，capture_hz 15，control_hz 2，point_count 512。能力标签：端侧部署。
16. 事实：拆分 ONNX 子图用于 TensorRT 证据验证。证据：`export_mp1_onnx_parts.py`、`deploy_artifacts/onnx`。可量化信息：obs_encoder ONNX 2.2MB，unet_step ONNX 1002MB。能力标签：推理优化。
17. 事实：构建 TensorRT FP16 engine 并保存离线输出。证据：`deploy_artifacts/trt_engines`。可量化信息：obs engine 2.3MB，unet engine 504MB。能力标签：推理优化。
18. 事实：实现不加载训练代码的轻量 TensorRT case 检查。证据：`tools/check_trt_case_parity.py`。可量化信息：只消费 case、ONNX、engine，不依赖 checkpoint。能力标签：端侧部署。
19. 事实：完成 ONNX Runtime 数值误差报告。证据：`TRT_ACCEL_REPORT_LITE.md`。可量化信息：final_action max_abs_diff `9.52482e-05`。能力标签：数值一致性验证。
20. 事实：完成 TensorRT FP16 数值误差报告。证据：`TRT_ACCEL_REPORT_LITE.md`。可量化信息：TensorRT final_action max_abs_diff `0.000688314`。能力标签：推理优化。
21. 事实：保留 TorchScript fallback，TensorRT 只作为通过误差阈值后的 dry-run 候选。证据：`check_onnx_parity.py`、`TRT_ACCEL_REPORT_LITE.md`。可量化信息：报告明确“任一误差超阈值不进入真机链路”。能力标签：算法-系统联合优化。
22. 事实：Jetson 上已有 C++ 构建产物。证据：`build/mp1_offline_infer` 等二进制。可量化信息：6 个二进制，目标平台 aarch64。能力标签：C++工程化。

## F. 风险项或不能夸大的地方

- 不能写“TensorRT 已接入真机闭环”。仓库证据显示 TensorRT 是离线证据原型，真机主路径仍是 TorchScript。
- 不能写“完整 TensorRT 加速效果已验证”。有 profile 和轻量报告，但缺少严格端到端 p50/p95/p99 benchmark；ONNX Runtime 延迟报告 32699ms 也需要复测确认。
- 不能写“真机闭环成功率”。仓库没有成功率、运行时长、失败次数、任务完成率等日志。
- 不能写“CTest 全部通过”。CTest XML 里 `TestList` 为空；只能说测试源码和二进制存在。
- 不能写“当前源码干净可构建”。`cpp_deploy/src/real_robot_control.cpp` 当前含 Git 冲突标记。
- 不能写“做了量化/剪枝/蒸馏”。仓库只有 TensorRT FP16 engine 和 ONNX/TensorRT 验证，没有量化训练、剪枝、蒸馏证据。
- 不能写“实现高频闭环”。配置和代码里有 2Hz、5Hz、15Hz、100Hz 不同层级，但没有正式闭环稳定频率 benchmark。
- 不能写“显存/内存优化完成”。仓库没有显存峰值、内存曲线、CUDA memory profile。
- 不能写“多线程实时管线已实机长时间稳定运行”。有单进程源码，但缺少运行日志。
- 不能写“安全机制已覆盖所有风险”。已有限幅、dry-run、confirm、workspace，但没有碰撞检测、力控、急停链路日志。
- 不能写“完全主导项目”，除非能在面试中说明哪些部分是本人完成；仓库只能证明代码存在和提交记录，不证明个人贡献边界。

## G. 建议后续补充采集的指标

1. TorchScript CPU/CUDA 端到端延迟 p50/p95/p99。重要性：端侧部署岗位最看重稳定延迟。采集方式：`mp1_dry_run --steps 200 --warmup-steps 10 --device cpu/cuda`，保存每步 `inference_ms` 后统计。
2. Real-input dry-run 的 `load_ms` vs `inference_ms`。重要性：区分模型耗时和文件 IO/加载耗时。采集方式：`mp1_real_input_dry_run --steps 200 --require-update 0`，解析日志。
3. 单进程实时管线的 `capture_ms / inference_ms / loop_ms`。重要性：证明 ring buffer 替代文件轮询是否降低闭环延迟。采集方式：`mp1_realtime_robot_control --execute 0 --steps 300`，统计日志。
4. TorchScript vs TensorRT 完整 loop 延迟。重要性：证明 TensorRT 是否有实际加速收益。采集方式：完善 `run_trt_case.py --warmup 20 --repeats 200` 输出，并与 TorchScript 同输入同设备对比。
5. CUDA 显存峰值。重要性：Jetson 端部署受显存限制。采集方式：推理前后调用 `tegrastats` 或 `torch.cuda.max_memory_allocated()`，记录 TorchScript 和 TensorRT。
6. 功耗和温度。重要性：端侧长期运行会受热降频影响。采集方式：推理 benchmark 同时运行 `tegrastats --interval 1000`，记录 GPU/CPU 温度、功耗、频率。
7. 真实输入采集丢帧率。重要性：多相机 + RTDE 同步质量会影响策略稳定性。采集方式：在 `capture_real_inputs.py` 中记录每步实际周期、超时次数、RealSense frame number/timestamp。
8. 数值误差按动作维度拆分。重要性：final_action 总 max_abs_diff 不知道误差落在哪个轴。采集方式：比较 PyTorch/ONNX/TensorRT action，输出 7 维 per-dim max/mean abs diff。
9. 安全限幅触发率。重要性：频繁触发限幅说明模型输出和控制约束不匹配。采集方式：在 `SafetyFilter` 或上层日志记录 raw norm、filtered norm、是否 clipped，统计 200-1000 步。
10. 真机 dry-run 和 execute 运行时长。重要性：简历里“实机验证”需要时长和步数支撑。采集方式：dry-run 先跑 10min，execute 小限幅跑固定步数；记录 steps、异常、workspace_clamped、command_sent、急停次数。

