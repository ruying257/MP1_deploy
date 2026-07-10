# 工具说明

`tools/` 提供 MP1 从 Python 策略到 TorchScript、ONNX 和 Jetson TensorRT 的离线部署与验证工具。所有命令均应在仓库根目录执行。工具链不直接发送机器人控制指令。

## 目录与职责

- `freeze_python_behavior.py`：冻结 Python 侧部署行为摘要，输出 `deploy_artifacts/python_behavior_manifest.json`。
- `export_mp1_policy.py`：从训练 checkpoint 导出 TorchScript `policy_infer.pt`、部署元数据和黄金样本张量。
- `test_npz_output.py`：检查策略 dump 的 `.npz` 内容。
- `tensorrt/`：ONNX/TensorRT 离线加速证据链。

`tensorrt/` 内的脚本按职责分层：

| 脚本 | 作用 | 是否执行 TensorRT engine |
| --- | --- | --- |
| `dump_trt_case.py` | 冻结固定输入、逐步 PyTorch 参考输出和元数据 | 否 |
| `export_mp1_onnx_parts.py` | 导出 `obs_encoder.onnx`、`unet_step.onnx` | 否 |
| `check_onnx_parity.py` | 在完整 Python 环境中对齐 PyTorch 与 ONNX Runtime，并汇总可选 TensorRT 结果 | 否 |
| `run_trt_case.py` | 加载 TensorRT `.engine`，执行完整采样循环并保存结果 | 是 |
| `check_trt_case_parity.py` | 在轻量 Jetson 环境中校验冻结 case、ONNX 和 TensorRT 输出 | 否 |
| `mp1_trt_utils.py` | 上述脚本共享的模型加载、采样、归一化和统计函数 | 否 |

## 推荐流程

### 1. 导出 TorchScript 与黄金样本

```bash
python3 tools/freeze_python_behavior.py --max-samples 20

python3 tools/export_mp1_policy.py \
  --checkpoint python_deploy/checkpoints/latest.ckpt \
  --config cpp_deploy/configs/pole_pickoff_real_robot.json \
  --output-dir deploy_artifacts \
  --device cpu
```

### 2. 冻结 TensorRT 对齐 case 并导出 ONNX 子图

```bash
python3 -m tools.tensorrt.dump_trt_case \
  --checkpoint python_deploy/checkpoints/latest.ckpt \
  --tensor-dir deploy_artifacts/sample_tensors \
  --output-dir deploy_artifacts/trt_cases/case_000 \
  --device cpu

python3 -m tools.tensorrt.export_mp1_onnx_parts \
  --checkpoint python_deploy/checkpoints/latest.ckpt \
  --tensor-dir deploy_artifacts/sample_tensors \
  --output-dir deploy_artifacts/onnx \
  --device cpu
```

`case_000` 是可复现测试夹具，保存输入、`global_cond`、每步 `x_current/timestep/v_pred`、最终动作和反归一化参数。它用于跨机器复现和定位误差，不等同于统计性验证。

### 3. 在完整 Python 环境校验 PyTorch 与 ONNX Runtime

```bash
python3 -m tools.tensorrt.check_onnx_parity \
  --checkpoint python_deploy/checkpoints/latest.ckpt \
  --tensor-dir deploy_artifacts/sample_tensors \
  --onnx-dir deploy_artifacts/onnx \
  --device cuda \
  --report deploy_artifacts/TRT_ACCEL_REPORT.md
```

该脚本只使用 ONNX Runtime 的 CUDA/CPU provider，不加载 TensorRT engine。它检查 `global_cond`、首步 `v_pred` 和最终 `action` 的 `max_abs_diff`，并统计 TorchScript 与 ONNX Runtime 延迟。

### 4. 在 Jetson 构建并运行 TensorRT FP16 engine

在 Jetson 本机执行，不能将 Windows 结果作为 TensorRT 性能结论：

```bash
mkdir -p deploy_artifacts/trt_engines

trtexec --onnx=deploy_artifacts/onnx/obs_encoder.onnx \
  --saveEngine=deploy_artifacts/trt_engines/obs_encoder_fp16.engine \
  --fp16 --warmUp=500 --duration=20

trtexec --onnx=deploy_artifacts/onnx/unet_step.onnx \
  --saveEngine=deploy_artifacts/trt_engines/unet_step_fp16.engine \
  --fp16 --warmUp=500 --duration=20

python3 -m tools.tensorrt.run_trt_case \
  --case-dir deploy_artifacts/trt_cases/case_000 \
  --engine-dir deploy_artifacts/trt_engines \
  --output-dir deploy_artifacts/trt_engines \
  --warmup 20 --repeats 200
```

`run_trt_case.py` 使用 TensorRT Python API 反序列化 engine，并以 CUDA tensor 作为绑定缓冲区。它先运行一次观测编码器，再在图外循环运行 U-Net：`x_next = x_current + v_pred * dt`。

### 5. 校验 TensorRT 结果

```bash
python3 -m tools.tensorrt.check_trt_case_parity \
  --case-dir deploy_artifacts/trt_cases/case_000 \
  --onnx-dir deploy_artifacts/onnx \
  --trt-output-dir deploy_artifacts/trt_engines \
  --report deploy_artifacts/TRT_ACCEL_REPORT_LITE.md \
  --warmup 20 --repeats 200
```

该轻量脚本不加载 checkpoint、Hydra、dill 或训练代码。它消费 `case_000`、ONNX 文件和 TensorRT runner 输出，比较以下文件：

```text
deploy_artifacts/trt_engines/
├── global_cond_trt.npy
├── v_pred_trt_000.npy
├── final_x_trt.npy
└── action_trt.npy
```

## 结果与验收边界

主要产物位于 `deploy_artifacts/`：

```text
policy_infer.pt                         # TorchScript 策略
sample_tensors/                         # Python/C++ 黄金样本
trt_cases/case_000/                     # 冻结 TensorRT 对齐用例
onnx/obs_encoder.onnx                   # 观测编码子图
onnx/unet_step.onnx                     # 单步 U-Net 子图
trt_engines/*.engine                    # 仅在对应 Jetson/TensorRT 环境可用
TRT_ACCEL_REPORT.md                     # 完整环境报告
TRT_ACCEL_REPORT_LITE.md                # 轻量 case 报告
```

TensorRT 对齐通过的最低条件是：ONNX FP32 先通过数值对齐，随后 TensorRT 输出在 `global_cond`、`v_pred` 和 `final_action` 上满足预设误差与动作安全限幅要求。单个冻结 case 仅证明固定输入上的对齐；需要多场景、多噪声 case 和 p50/p95/p99 延迟统计，才能形成更可信的部署证据。

在 TensorRT 链路完成完整离线验证前，真实机器人控制应继续保留 TorchScript 路径和安全过滤作为 fallback。
