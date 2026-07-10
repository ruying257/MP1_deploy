"""运行 MP1 TensorRT engine，并生成冻结 case 的推理输出。

本脚本仅消费 ``dump_trt_case.py`` 生成的固定 case，以及在同一 Jetson
环境中构建的两个 TensorRT engine：

1. ``obs_encoder_fp16.engine``：多模态观测 -> ``global_cond``。
2. ``unet_step_fp16.engine``：``x_current, timestep, global_cond, r`` -> ``v_pred``。

脚本在图外复现完整的 U-Net 采样循环，并保存标准化输出，供
``check_trt_case_parity.py`` 进行数值对齐和离线验证。它不加载训练
checkpoint，也不进入真机控制链路。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch

# TensorRT 8.5 的 Python 绑定仍会访问 np.bool；NumPy 1.24 已删除该别名。
# 在 JetPack 5.1.2 + Python 3.8 的 venv 中先补回别名，避免 trt.nptype 报错。
if not hasattr(np, "bool"):
    np.bool = np.bool_


def parse_args() -> argparse.Namespace:
    """解析冻结 case 的 TensorRT engine 执行参数。

    Returns:
        包含 case、engine、输出目录及延迟基准参数的命名空间。
    """
    parser = argparse.ArgumentParser(description="Run MP1 TensorRT engines on a frozen case and save .npy outputs.")
    parser.add_argument("--case-dir", default="deploy_artifacts/trt_cases/case_000")
    parser.add_argument("--engine-dir", default="deploy_artifacts/trt_engines")
    parser.add_argument("--output-dir", default="deploy_artifacts/trt_engines")
    parser.add_argument("--obs-engine", default="obs_encoder_fp16.engine")
    parser.add_argument("--unet-engine", default="unet_step_fp16.engine")
    parser.add_argument("--warmup", type=int, default=0, help="额外测速预热次数；默认只生成输出，不做耗时预热。")
    parser.add_argument("--repeats", type=int, default=1, help="额外测速重复次数；默认 1 次。")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    """读取 UTF-8 编码的 JSON 元数据。

    Args:
        path: JSON 文件路径。

    Returns:
        解析后的字典。
    """
    return json.loads(path.read_text(encoding="utf-8"))


def percentile_stats(values_ms) -> Dict[str, float]:
    """计算毫秒级延迟样本的 p50、p95、p99 和均值。

    Args:
        values_ms: 延迟样本序列，单位为毫秒。

    Returns:
        延迟统计字典。
    """
    values = np.asarray(list(values_ms), dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(np.mean(values)),
    }


def torch_dtype_from_numpy(dtype: np.dtype) -> torch.dtype:
    """将 TensorRT binding 的 NumPy dtype 映射为 PyTorch dtype。

    Args:
        dtype: TensorRT binding 对应的 NumPy 数据类型。

    Returns:
        可分配 CUDA 缓冲区的 PyTorch 数据类型。

    Raises:
        TypeError: dtype 不受当前 runner 支持时抛出。
    """
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return torch.float32
    if dtype == np.float16:
        return torch.float16
    if dtype == np.int32:
        return torch.int32
    if dtype == np.int64:
        return torch.int64
    if dtype == np.bool_:
        return torch.bool
    raise TypeError(f"unsupported TensorRT binding dtype for torch buffer: {dtype}")


def action_from_final_x(final_x: np.ndarray, meta: Mapping[str, Any]) -> np.ndarray:
    """将 TensorRT 积分后的归一化状态还原为最终可执行动作。

    Args:
        final_x: 外部采样循环结束时的归一化轨迹状态。
        meta: 冻结 case 的元数据，需包含动作 normalizer 和时间步配置。

    Returns:
        策略当前周期实际消费的动作段。

    Raises:
        RuntimeError: case 元数据缺少动作反归一化参数时抛出。
    """
    normalizer = meta.get("action_normalizer")
    if not normalizer:
        raise RuntimeError("case meta 缺少 action_normalizer；请先用新版 tools/dump_trt_case.py 重新生成 case。")

    action_dim = int(meta["action_dim"])
    n_obs_steps = int(meta["n_obs_steps"])
    n_action_steps = int(meta["n_action_steps"])
    scale = np.asarray(normalizer["scale"], dtype=np.float32)
    offset = np.asarray(normalizer["offset"], dtype=np.float32)
    normalized_action = np.asarray(final_x[..., :action_dim], dtype=np.float32)
    src_shape = normalized_action.shape
    flat = normalized_action.reshape(-1, scale.shape[0])
    action_pred = ((flat - offset) / scale).reshape(src_shape)
    start = n_obs_steps - 1
    return action_pred[:, start : start + n_action_steps]


class TrtEngineRunner:
    """用 torch CUDA tensor 作为 TensorRT 输入/输出缓冲区。

    这样可以避免额外依赖 pycuda；TensorRT 只需要拿到 CUDA device pointer。
    """

    def __init__(self, engine_path: Path):
        """反序列化 TensorRT engine 并创建执行上下文。

        Args:
            engine_path: TensorRT ``.engine`` 文件路径。

        Raises:
            RuntimeError: TensorRT Python 绑定缺失或 engine 无法反序列化时抛出。
        """
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError("缺少 tensorrt Python 绑定；请确认 Jetson 已安装 TensorRT Python 包。") from exc

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        # 某些 engine 会包含 TensorRT plugin 层。trtexec 构建时会自动注册，
        # 但 Python 反序列化 engine 前必须手动初始化 plugin registry。
        trt.init_libnvinfer_plugins(self.logger, "")
        with engine_path.open("rb") as file:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(file.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.binding_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]

    def _binding_index(self, name: str) -> int:
        """查询 binding 名称对应的 engine 索引。

        Args:
            name: ONNX 导出时定义的输入或输出名称。

        Returns:
            TensorRT engine binding 索引。

        Raises:
            KeyError: engine 中不存在该 binding 时抛出。
        """
        if name not in self.binding_names:
            raise KeyError(f"binding {name!r} not found; available bindings: {self.binding_names}")
        return self.binding_names.index(name)

    def _binding_np_dtype(self, index: int) -> np.dtype:
        """返回指定 binding 的 NumPy 数据类型。

        Args:
            index: TensorRT engine binding 索引。

        Returns:
            与 binding 对应的 NumPy dtype。
        """
        return np.dtype(self.trt.nptype(self.engine.get_binding_dtype(index)))

    def infer(self, inputs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """执行一次 TensorRT engine 推理并将输出复制回 CPU。

        Args:
            inputs: 按 ONNX binding 名组织的连续 NumPy 输入数组。

        Returns:
            按输出 binding 名组织的 NumPy 输出数组。

        Raises:
            RuntimeError: binding 类型、shape 或 TensorRT 执行失败时抛出。
        """
        # bindings 保存 CUDA device pointer；缓冲区生命周期必须覆盖 execute_v2。
        bindings = [0] * self.engine.num_bindings
        device_buffers: Dict[str, torch.Tensor] = {}

        # 先绑定输入，并为动态 shape engine 设置实际输入形状。
        for name, value in inputs.items():
            index = self._binding_index(name)
            if not self.engine.binding_is_input(index):
                raise RuntimeError(f"binding {name} is not an input")
            np_dtype = self._binding_np_dtype(index)
            host_value = np.ascontiguousarray(value.astype(np_dtype, copy=False))
            tensor = torch.from_numpy(host_value).to(device="cuda")
            device_buffers[name] = tensor
            bindings[index] = int(tensor.data_ptr())
            shape = tuple(int(v) for v in tensor.shape)
            if any(dim < 0 for dim in tuple(self.engine.get_binding_shape(index))):
                self.context.set_binding_shape(index, shape)

        # 再按实际输出形状分配输出缓冲区。
        for index in range(self.engine.num_bindings):
            if self.engine.binding_is_input(index):
                continue
            name = self.engine.get_binding_name(index)
            shape = tuple(int(v) for v in self.context.get_binding_shape(index))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"output binding {name} still has dynamic shape: {shape}")
            dtype = torch_dtype_from_numpy(self._binding_np_dtype(index))
            tensor = torch.empty(shape, dtype=dtype, device="cuda")
            device_buffers[name] = tensor
            bindings[index] = int(tensor.data_ptr())

        # execute_v2 是同步接口；随后显式 synchronize，保证输出可拷回 CPU。
        ok = self.context.execute_v2(bindings)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 returned false")
        torch.cuda.synchronize()

        outputs: Dict[str, np.ndarray] = {}
        for index in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(index):
                name = self.engine.get_binding_name(index)
                outputs[name] = device_buffers[name].detach().cpu().numpy()
        return outputs


def load_case_inputs(case_dir: Path) -> Dict[str, np.ndarray]:
    """加载观测编码器所需的冻结 case 输入。

    Args:
        case_dir: ``dump_trt_case.py`` 生成的 case 目录。

    Returns:
        全局/腕部图像、点云和机器人状态输入数组。
    """
    return {
        "global_image": np.load(case_dir / "global_image.npy"),
        "wrist_image": np.load(case_dir / "wrist_image.npy"),
        "point_cloud": np.load(case_dir / "point_cloud.npy").astype(np.float32),
        "agent_pos": np.load(case_dir / "agent_pos.npy").astype(np.float32),
    }


def run_case(obs_runner: TrtEngineRunner, unet_runner: TrtEngineRunner, case_dir: Path, meta: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """执行完整 TensorRT case：编码一次观测，循环积分多步 U-Net。

    Args:
        obs_runner: 观测编码器 TensorRT engine runner。
        unet_runner: 单步 U-Net TensorRT engine runner。
        case_dir: 冻结 case 目录。
        meta: 包含采样步数、步长和动作 normalizer 的 case 元数据。

    Returns:
        ``global_cond``、首步 ``v_pred``、最终归一化状态和最终动作。
    """
    # 观测编码器仅运行一次；其条件特征在所有 U-Net 采样步之间复用。
    obs_outputs = obs_runner.infer(load_case_inputs(case_dir))
    if "global_cond" not in obs_outputs:
        raise RuntimeError(f"obs engine did not produce global_cond; outputs={list(obs_outputs.keys())}")
    global_cond = obs_outputs["global_cond"].astype(np.float32)

    x_current = np.load(case_dir / "initial_noise.npy").astype(np.float32)
    r = np.load(case_dir / "r.npy").astype(np.float32)
    steps = int(meta["num_inference_steps"])
    dt = float(meta["dt"])
    first_v_pred = None
    # ONNX/TensorRT 只承载 U-Net 单步，数值积分循环保持在图外。
    for index in range(steps):
        timestep = (r + float(index) / float(steps)).astype(np.float32)
        # TensorRT engine 的输入名来自 ONNX 导出：x_current, timestep, global_cond, r。
        unet_outputs = unet_runner.infer({
            "x_current": np.ascontiguousarray(x_current),
            "timestep": np.ascontiguousarray(timestep),
            "global_cond": np.ascontiguousarray(global_cond),
            "r": np.ascontiguousarray(r),
        })
        if "v_pred" not in unet_outputs:
            raise RuntimeError(f"unet engine did not produce v_pred; outputs={list(unet_outputs.keys())}")
        v_pred = unet_outputs["v_pred"].astype(np.float32)
        if index == 0:
            first_v_pred = v_pred.copy()
        x_current = x_current + v_pred * dt

    if first_v_pred is None:
        raise RuntimeError("num_inference_steps must be > 0")
    action = action_from_final_x(x_current, meta)
    return {
        "global_cond": global_cond,
        "v_pred_000": first_v_pred,
        "final_x": x_current,
        "action": action,
    }


def main() -> None:
    """运行 TensorRT engines，保存对齐输出并可选统计完整 case 延迟。"""
    args = parse_args()
    case_dir = Path(args.case_dir).resolve()
    engine_dir = Path(args.engine_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # runner 只消费冻结 case 和序列化 engines，不加载训练 checkpoint。
    meta = load_json(case_dir / "meta.json")
    obs_runner = TrtEngineRunner(engine_dir / args.obs_engine)
    unet_runner = TrtEngineRunner(engine_dir / args.unet_engine)

    # 保存标准化命名的输出，供 check_trt_case_parity.py 自动发现并对比。
    result = run_case(obs_runner, unet_runner, case_dir, meta)
    np.save(output_dir / "global_cond_trt.npy", result["global_cond"])
    np.save(output_dir / "v_pred_trt_000.npy", result["v_pred_000"])
    np.save(output_dir / "final_x_trt.npy", result["final_x"])
    np.save(output_dir / "action_trt.npy", result["action"])

    # 统计完整 case 延迟，包含两张 engine 的执行、GPU 同步和图外积分开销。
    samples = []
    for _ in range(max(0, args.warmup)):
        run_case(obs_runner, unet_runner, case_dir, meta)
    for _ in range(max(0, args.repeats)):
        begin = time.perf_counter()
        run_case(obs_runner, unet_runner, case_dir, meta)
        samples.append((time.perf_counter() - begin) * 1000.0)

    payload: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "saved": [
            "global_cond_trt.npy",
            "v_pred_trt_000.npy",
            "final_x_trt.npy",
            "action_trt.npy",
        ],
    }
    if samples:
        payload["latency_ms"] = percentile_stats(samples)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
