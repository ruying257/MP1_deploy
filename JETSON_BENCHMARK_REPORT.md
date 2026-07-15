# MP1 Jetson 端侧部署补实验报告

## 1. 实验环境
- 设备：Jetson aarch64 主机，hostname `NVIDIA`
- 系统：`Linux NVIDIA 5.10.120-tegra #1 SMP PREEMPT Tue Aug 1 12:32:50 PDT 2023 aarch64`
- GPU：PyTorch CUDA probe 已通过，`torch.cuda.is_available() = True`，`device_count = 1`，设备名 `Orin`
- 关键依赖：Python 3.8.10，PyTorch `2.1.0a0+41361538.nv23.06`，CUDA toolkit 11.4，ONNX Runtime 1.15.1，NumPy 1.24.4，TensorRT Python `8.5.2.2`，`trtexec` 路径 `/usr/src/tensorrt/bin/trtexec`，版本 `TensorRT v8502`
- 可执行文件/脚本：
  - `build/mp1_dry_run`
  - `build/mp1_real_input_dry_run`
  - `build/mp1_realtime_robot_control`
  - `tools/run_trt_case.py`
  - `tools/check_trt_case_parity.py`
  - `cpp_deploy/tools/capture_real_inputs.py`
- 配置文件：
  - `cpp_deploy/configs/pole_pickoff_jetson.json`
  - `cpp_deploy/configs/pole_pickoff_real_robot.json`
  - `deploy_artifacts/sample_tensors`
  - `deploy_artifacts/real_input_tensors`
  - `deploy_artifacts/trt_cases/case_000`
  - `deploy_artifacts/trt_engines`

## 2. TorchScript CPU/CUDA 延迟
- 实验命令：
  - CPU：`./build/mp1_dry_run --model deploy_artifacts/policy_infer.pt --tensor-dir deploy_artifacts/sample_tensors --device cpu --steps 200 --warmup-steps 5`
  - CUDA probe：`./build/mp1_dry_run --model deploy_artifacts/policy_infer.pt --tensor-dir deploy_artifacts/sample_tensors --device cuda --steps 1 --warmup-steps 0`
- 实验设置：
  - 固定输入：`deploy_artifacts/sample_tensors`
  - warmup：5 次，正式样本：200 次
  - CPU 统计口径：`TorchScriptRuntime::infer()` 全段，包含输入 `.to(device_)` 和输出 `.cpu()`；CPU 下数据搬运基本为 CPU tensor 转换/拷贝，不包含 SafetyFilter
  - CUDA 统计口径：同样为 `TorchScriptRuntime::infer()` 全段，包含输入 `.to(cuda)`、模型 forward、输出 `.cpu()`；不包含 SafetyFilter
- 原始结果摘要：
  - CPU 原始日志：`deploy_artifacts/jetson_benchmark_logs/torchscript_cpu_fixed_200.log`
  - CUDA 原始日志：`deploy_artifacts/jetson_benchmark_logs/torchscript_cuda_fixed_200.log`
- p50/p95/p99/mean/min/max：

| 路径 | 样本数 | p50 ms | p95 ms | p99 ms | mean ms | min ms | max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TorchScript CPU fixed-input | 200 | 331.012 | 409.941 | 422.202 | 331.678 | 238.554 | 456.724 |
| TorchScript CUDA fixed-input | 200 | 56.908 | 62.730 | 64.233 | 57.344 | 51.465 | 64.685 |

- 结论：
  - 当前环境可复现的 TorchScript CPU fixed-input p50 为 `331.012 ms`，mean 为 `331.678 ms`。
  - TorchScript CUDA fixed-input p50 为 `56.908 ms`，相对 CPU fixed-input p50 约 `5.82x` 加速。
  - CUDA warmup 前两次包含上下文/缓存初始化，分别为 `5134.436 ms` 和 `8192.950 ms`；正式统计只使用 warmup 后 200 次样本。

## 3. Real-input Load/Inference 拆分
- 实验命令：
  - 当前帧重复：`./build/mp1_real_input_dry_run --model deploy_artifacts/policy_infer.pt --input-dir deploy_artifacts/real_input_tensors --device cpu --steps 200 --warmup-steps 5 --require-update 0 --poll-ms 0`
  - 已采集帧顺序扫描：`./build/mp1_real_input_dry_run --model deploy_artifacts/policy_infer.pt --input-dir deploy_artifacts/real_input_tensors --device cpu --steps 200 --warmup-steps 5 --scan-frames 1 --require-update 0 --poll-ms 0`
- 实验设置：
  - 输入目录：`deploy_artifacts/real_input_tensors`
  - 已采集帧数：2444，扫描实验使用 `frames/000000` 到 `frames/000199`
  - warmup：5 次，正式样本：200 次
  - `load_ms` 包含解析 committed frame 目录、加载 5 个 `.pt` 张量、输入 shape/dtype 校验
  - `inference_ms` 包含 `TorchScriptRuntime::infer()`，不包含 SafetyFilter 打印和 sleep
- 结果表格：

| 模式 | 指标 | 样本数 | p50 ms | p95 ms | p99 ms | mean ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| current_frame 重复 | load_ms | 200 | 4.759 | 7.845 | 10.418 | 5.185 |
| current_frame 重复 | inference_ms | 200 | 332.173 | 387.075 | 400.656 | 332.534 |
| scan_frames 顺序扫描 | load_ms | 200 | 12.163 | 28.972 | 34.819 | 14.249 |
| scan_frames 顺序扫描 | inference_ms | 200 | 486.009 | 650.084 | 684.592 | 497.102 |

- 结论：
  - 单帧重复主要测稳定推理链路；顺序扫描更接近离线回放真实帧，load p50 从 `4.759 ms` 增至 `12.163 ms`。
  - 顺序扫描时 CPU inference p50 为 `486.009 ms`，受并行 CPU benchmark 和真实帧差异影响，建议后续单独复测。

## 4. TorchScript vs TensorRT 延迟对比
- 对比对象：
  - TorchScript：`build/mp1_dry_run` / `build/mp1_real_input_dry_run`
  - ONNX Runtime：`tools/check_trt_case_parity.py`
  - TensorRT：`tools/run_trt_case.py`，已有 engine 和 `.npy` 输出
- 实验命令：
  - TorchScript fixed loop：见第 2 节
  - TensorRT runner：`PYTHONPATH=/usr/lib/python3.8/dist-packages:$PYTHONPATH python3 tools/run_trt_case.py --case-dir deploy_artifacts/trt_cases/case_000 --engine-dir deploy_artifacts/trt_engines --output-dir deploy_artifacts/trt_engines --warmup 5 --repeats 200`
  - TensorRT 数值检查：`python3 tools/check_trt_case_parity.py --case-dir deploy_artifacts/trt_cases/case_000 --onnx-dir deploy_artifacts/onnx --trt-output-dir deploy_artifacts/trt_engines --report deploy_artifacts/TRT_ACCEL_REPORT_LITE.md --warmup 0 --repeats 1`
- 结果表格：

| 对象 | 完整 loop 延迟 p50/p95/p99/mean | 当前状态 |
| --- | --- | --- |
| TorchScript CPU fixed-input | 331.012 / 409.941 / 422.202 / 331.678 ms | 已完成，200 样本 |
| TorchScript CUDA fixed-input | 56.908 / 62.730 / 64.233 / 57.344 ms | 已完成，200 样本 |
| ONNX Runtime CUDA case | 21999.260 / 21999.260 / 21999.260 / 21999.260 ms | 已完成，`--repeats 1`，仅作可运行性参考 |
| TensorRT FP16 case full loop | 14.686 / 19.255 / 21.261 / 15.260 ms | 已完成，200 样本 |

- 说明 benchmark 边界：
  - TorchScript fixed-input 包含输入搬运到 runtime device、engine forward、输出 `.cpu()`，不包含真实输入 load 和 SafetyFilter。
  - TensorRT FP16 benchmark 基于离线 evidence prototype：`obs_encoder_fp16.engine` 执行一次，`unet_step_fp16.engine` 按 frozen case 的扩散步数执行完整积分，并包含 `run_case()` 内部 `.npy` 输入读取、host/device buffer 拷贝、engine execute、输出回 CPU、最终 action 反归一化。
  - 已完成 TensorRT 数值支撑：PyTorch case vs TensorRT `global_cond` max_abs_diff `0.00334206`，`v_pred_step_000` `0.00225765`，`final_action` `0.000688314`。
- 结论：
  - 当前环境下 TorchScript CUDA fixed-input p50 为 `56.908 ms`，TensorRT FP16 case full-loop p50 为 `14.686 ms`，按 p50 约 `3.87x` 加速。
  - TensorRT 是否进入真机闭环仍取决于数值误差，尤其 `final_action` max_abs_diff `0.000688314` 是否明显小于单步安全限幅；当前报告只证明离线 case latency 和数值误差。

## 5. CUDA 显存峰值
- 统计方法：
  - 尝试使用 `torch.cuda.max_memory_allocated()` / `torch.cuda.max_memory_reserved()`。
  - 原始日志：`deploy_artifacts/jetson_benchmark_logs/cuda_memory_current.log`、`deploy_artifacts/jetson_benchmark_logs/trt_memory_current.log`
- 结果表格：

| 路径 | allocated peak | reserved peak | 状态 |
| --- | ---: | ---: | --- |
| TorchScript CUDA | 1,197,010,944 B / 1141.558 MiB | 1,388,314,624 B / 1324.000 MiB | PyTorch `torch.cuda.max_memory_*`，20 次正式 forward 后峰值 |
| TensorRT FP16 runner | 629,248 B / 0.600 MiB | 2,097,152 B / 2.000 MiB | PyTorch allocator 口径，只覆盖 runner 中 torch CUDA buffer，不覆盖 TensorRT runtime 内部非 PyTorch 分配 |

- 结论：
  - TorchScript 路径峰值主要发生在模型加载、CUDA 输入搬运和 warmup 后的 forward 缓存建立阶段。
  - TensorRT runner 的 PyTorch allocator 峰值很低，因为 engine workspace 和 TensorRT runtime 内部分配不完全进入 `torch.cuda.max_memory_*` 口径；后续若要报告系统级显存，应同时采集 `tegrastats`。

## 6. SafetyFilter 限幅触发率
- 统计方法：
  - 最小修改 `SafetyFilter`：新增 `last_result()`，记录 clip 前后范数和触发标志，不改变 `apply()` 输出语义。
  - 使用 real-input 顺序扫描 200 帧：`--scan-frames 1 --steps 200`
  - 原始 CSV：`deploy_artifacts/jetson_benchmark_logs/safety_filter_scan_frames_200.csv`
- 结果表格：

| 指标 | 数值 |
| --- | ---: |
| 总步数 | 200 |
| 平移限幅触发次数 | 200 |
| 旋转限幅触发次数 | 0 |
| 任一限幅触发次数 | 200 |
| 夹爪屏蔽次数 | 200 |
| 平移限幅触发率 | 100.0% |
| 旋转限幅触发率 | 0.0% |
| 任一限幅触发率 | 100.0% |
| 夹爪屏蔽率 | 100.0% |
| raw_translation_norm p50 / p95 / p99 | 0.005281 / 0.005289 / 0.005295 |
| raw_rotation_norm p50 / p95 / p99 | 0.004525 / 0.004534 / 0.004538 |

- 示例样本：

| step | frame | trans_clip | rot_clip | gripper_mask | raw_trans | filt_trans | raw_rot | filt_rot |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `frames/000000` | 1 | 0 | 1 | 0.005299 | 0.002000 | 0.004555 | 0.004555 |
| 1 | `frames/000001` | 1 | 0 | 1 | 0.005295 | 0.002000 | 0.004537 | 0.004537 |
| 2 | `frames/000002` | 1 | 0 | 1 | 0.005300 | 0.002000 | 0.004530 | 0.004530 |
| 198 | `frames/000198` | 1 | 0 | 1 | 0.005286 | 0.002000 | 0.004521 | 0.004521 |
| 199 | `frames/000199` | 1 | 0 | 1 | 0.005280 | 0.002000 | 0.004523 | 0.004523 |

- 结论：
  - 在 `max_translation=0.002 m`、`max_rotation=0.01 rad`、`ignore_gripper_action=1` 的 dry-run 设置下，200 帧真实输入全部触发平移限幅，旋转未触发限幅，夹爪全部被屏蔽。
