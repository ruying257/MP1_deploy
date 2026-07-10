"""MP1 ONNX/TensorRT evidence-prototype helpers.

这些工具只服务离线加速证据链：固定样本、导出子图、比较误差和统计延迟。
真实机器人闭环仍以 TorchScript 路径作为安全 fallback。
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
PY_DEPLOY_ROOT = REPO_ROOT / "python_deploy"
MP1_ROOT = PY_DEPLOY_ROOT / "MP1"
ROBOT_SCRIPT_DIR = PY_DEPLOY_ROOT / "real_robot_ur12e_d405_speed_only" / "scripts"
for _path in [PY_DEPLOY_ROOT, MP1_ROOT, ROBOT_SCRIPT_DIR]:
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

INPUT_NAMES = ["global_image", "wrist_image", "point_cloud", "agent_pos", "initial_noise"]


def resolve_repo_path(path: str | Path) -> Path:
    """将相对路径解析为仓库根目录下的绝对路径。

    Args:
        path: 用户传入的相对或绝对路径。

    Returns:
        可用于文件读写的路径对象。
    """
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def load_policy(checkpoint: str | Path, device: torch.device):
    """加载用于离线对齐的 EMA 策略。

    Args:
        checkpoint: 完整训练 checkpoint 的路径。
        device: 策略运行所在的 PyTorch 设备。

    Returns:
        训练配置和处于 eval 模式的 EMA 策略。
    """
    # 真正加载 checkpoint 时才导入部署栈，避免 --help 被 dill/torch 依赖挡住。
    from deploy_real_policy import load_workspace_policy

    cfg, policy = load_workspace_policy(resolve_repo_path(checkpoint), device=device, use_ema=True)
    policy.eval()
    return cfg, policy


def load_tensor_module(path: str | Path, device: torch.device) -> torch.Tensor:
    """从 TorchScript 常量模块读取一个张量。

    Args:
        path: 张量模块文件路径。
        device: 目标设备。

    Returns:
        迁移到目标设备的张量。
    """
    module = torch.jit.load(str(resolve_repo_path(path)), map_location=device)
    module.eval()
    return module.forward().to(device)


def load_sample_tensors(tensor_dir: str | Path, device: torch.device) -> Dict[str, torch.Tensor]:
    """加载并规范化黄金样本的输入 dtype。

    Args:
        tensor_dir: ``sample_tensors`` 目录。
        device: 推理设备。

    Returns:
        按模型输入名组织的黄金张量字典。
    """
    root = resolve_repo_path(tensor_dir)
    tensors = {name: load_tensor_module(root / f"{name}.pt", device) for name in INPUT_NAMES}
    tensors["global_image"] = tensors["global_image"].to(torch.uint8)
    tensors["wrist_image"] = tensors["wrist_image"].to(torch.uint8)
    tensors["point_cloud"] = tensors["point_cloud"].to(torch.float32)
    tensors["agent_pos"] = tensors["agent_pos"].to(torch.float32)
    tensors["initial_noise"] = tensors["initial_noise"].to(torch.float32)
    return tensors


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    """将脱离计算图的 CPU 张量转换为 NumPy 数组。

    Args:
        value: 任意设备上的 PyTorch 张量。

    Returns:
        对应的 CPU NumPy 数组。
    """
    return value.detach().cpu().numpy()


def save_numpy(path: str | Path, value: torch.Tensor | np.ndarray) -> None:
    """将张量或数组保存为 ``.npy`` 文件。

    Args:
        path: 输出文件路径。
        value: 需要保存的张量或 NumPy 数组。
    """
    array = tensor_to_numpy(value) if torch.is_tensor(value) else np.asarray(value)
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """以 UTF-8 和稳定键排序写出 JSON 元数据。

    Args:
        path: 输出 JSON 路径。
        payload: 可 JSON 序列化的数据映射。
    """
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个数组的最大逐元素绝对误差。

    Args:
        a: 第一个待比较数组。
        b: 第二个待比较数组。

    Returns:
        转为 ``float32`` 后的最大绝对误差。
    """
    return float(np.max(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def summarize_array(value: np.ndarray) -> Dict[str, Any]:
    """提取数组的 shape、dtype 和基础数值统计量。

    Args:
        value: 需要摘要的数组。

    Returns:
        包含 shape、dtype、最值、均值和标准差的字典。
    """
    array = np.asarray(value)
    array_f = array.astype(np.float32, copy=False)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(np.min(array_f)),
        "max": float(np.max(array_f)),
        "mean": float(np.mean(array_f)),
        "std": float(np.std(array_f)),
    }


def sync_if_cuda(device: torch.device) -> None:
    """在 CUDA 设备上同步，保证延迟统计覆盖实际 GPU 执行时间。

    Args:
        device: 当前执行设备。
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile_stats(values_ms: Iterable[float]) -> Dict[str, float]:
    """汇总毫秒级样本的 p50、p95、p99 和均值。

    Args:
        values_ms: 延迟样本序列，单位为毫秒。

    Returns:
        延迟分位数和均值；空序列返回 NaN。
    """
    values = np.asarray(list(values_ms), dtype=np.float64)
    if values.size == 0:
        return {"p50_ms": float("nan"), "p95_ms": float("nan"), "p99_ms": float("nan"), "mean_ms": float("nan")}
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(np.mean(values)),
    }


def command_output(command: list[str]) -> Optional[str]:
    """尽力读取外部版本命令的首行输出，失败时不阻断验证。

    Args:
        command: 待执行的命令及参数。

    Returns:
        首行标准输出/错误输出；执行失败时返回 ``None``。
    """
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0] if text else None


def platform_report() -> Dict[str, Any]:
    """收集复现实验所需的 Python、CUDA、GPU 和 TensorRT 环境信息。

    Returns:
        平台和工具版本信息字典。
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "trtexec": command_output(["trtexec", "--version"]),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "jetson_release": command_output(["bash", "-lc", "test -f /etc/nv_tegra_release && head -n 1 /etc/nv_tegra_release"]),
    }


class MP1ObsEncoderPart(nn.Module):
    """导出观测编码器子图：多模态观测 -> ``global_cond``。

    Args:
        policy: 已加载且包含 normalizer 的 MP1 策略。
    """

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def _normalize_field(self, value: torch.Tensor, key: str, forward: bool) -> torch.Tensor:
        """按策略 normalizer 对一个观测字段做正向或反向变换。

        Args:
            value: 保持原始 batch/时间维度的输入张量。
            key: normalizer 中对应字段名。
            forward: 为 ``True`` 时执行正向归一化。

        Returns:
            恢复原始 shape 的变换后张量。
        """
        params = self.policy.normalizer.params_dict[key]
        scale = params["scale"]
        offset = params["offset"]
        value = value.to(dtype=scale.dtype)
        src_shape = value.shape
        value = value.reshape(-1, scale.shape[0])
        if forward:
            value = value * scale + offset
        else:
            value = (value - offset) / scale
        return value.reshape(src_shape)

    def forward(
        self,
        global_image: torch.Tensor,
        wrist_image: torch.Tensor,
        point_cloud: torch.Tensor,
        agent_pos: torch.Tensor,
    ) -> torch.Tensor:
        """将四类观测编码为全局条件特征。

        Args:
            global_image: 全局相机图像序列。
            wrist_image: 腕部相机图像序列。
            point_cloud: 点云观测序列。
            agent_pos: 机器人状态序列。

        Returns:
            供 U-Net 条件输入使用的 ``global_cond``。
        """
        nobs = {
            "global_image": self._normalize_field(global_image, "global_image", True),
            "wrist_image": self._normalize_field(wrist_image, "wrist_image", True),
            "point_cloud": self._normalize_field(point_cloud, "point_cloud", True),
            "agent_pos": self._normalize_field(agent_pos, "agent_pos", True),
        }
        if not self.policy.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]
        batch_size = global_image.shape[0]
        _, _, global_cond = self.policy._encode_obs(nobs, batch_size)
        if global_cond is None:
            raise RuntimeError("obs_as_global_cond=False is not supported by the TensorRT prototype")
        return global_cond


class MP1UnetStepPart(nn.Module):
    """导出单步动作生成子图：``x_current, t, global_cond, r -> v_pred``。

    Args:
        policy: 已加载的 MP1 策略。
    """

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(
        self,
        x_current: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """运行一次条件 U-Net 速度预测。

        Args:
            x_current: 当前归一化动作轨迹状态。
            timestep: 当前 MeanFlow 时间步。
            global_cond: 观测编码器输出的条件特征。
            r: MeanFlow 的参考时间。

        Returns:
            对应当前状态的速度预测 ``v_pred``。
        """
        model_output = self.policy.model(
            sample=x_current,
            timestep=timestep,
            local_cond=None,
            global_cond=global_cond,
            r=r,
            training=False,
        )
        return model_output[0] if isinstance(model_output, tuple) else model_output


def unnormalize_action(policy: nn.Module, normalized_action: torch.Tensor) -> torch.Tensor:
    """将归一化动作恢复到机器人动作表示。

    Args:
        policy: 含动作 normalizer 的策略。
        normalized_action: 归一化动作轨迹。

    Returns:
        反归一化后的动作轨迹。
    """
    params = policy.normalizer.params_dict["action"]
    scale = params["scale"]
    offset = params["offset"]
    value = normalized_action.to(dtype=scale.dtype)
    src_shape = value.shape
    value = value.reshape(-1, scale.shape[0])
    value = (value - offset) / scale
    return value.reshape(src_shape)


@torch.no_grad()
def run_reference_parts(
    policy: nn.Module,
    tensors: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """执行 PyTorch 参考子图和完整的外部采样循环。

    Args:
        policy: 处于 eval 模式的 MP1 策略。
        tensors: 固定的多模态观测和初始噪声。

    Returns:
        包含 ``global_cond``、逐步 U-Net 记录、完整动作轨迹和可执行动作段的字典。
    """
    obs_part = MP1ObsEncoderPart(policy).eval()
    unet_part = MP1UnetStepPart(policy).eval()
    global_cond = obs_part(
        tensors["global_image"],
        tensors["wrist_image"],
        tensors["point_cloud"],
        tensors["agent_pos"],
    )

    # 外部循环显式保存每一步状态，使 ONNX/TensorRT 可逐阶段定位误差。
    x_current = tensors["initial_noise"].to(dtype=policy.dtype)
    steps = int(policy.num_inference_steps if policy.num_inference_steps is not None else 10)
    dt = 1.0 / float(steps)
    r = x_current[:, 0, 0] * 0.0
    step_records: list[Dict[str, torch.Tensor]] = []
    for index in range(steps):
        timestep = r + float(index) / float(steps)
        v_pred = unet_part(x_current, timestep, global_cond, r)
        step_records.append({
            "x_current": x_current.detach().clone(),
            "timestep": timestep.detach().clone(),
            "v_pred": v_pred.detach().clone(),
        })
        x_current = x_current + v_pred * dt

    action_pred = unnormalize_action(policy, x_current[..., : int(policy.action_dim)])
    start = tensors["global_image"].shape[1] - 1
    action = action_pred[:, start : start + int(policy.n_action_steps)]
    return {
        "global_cond": global_cond,
        "final_x": x_current,
        "action_pred": action_pred,
        "action": action,
        "r": r,
        "steps": step_records,  # type: ignore[dict-item]
    }


def case_input_numpy(tensors: Mapping[str, torch.Tensor], image_as_float: bool = True) -> Dict[str, np.ndarray]:
    """转换并整理 ONNX/TensorRT 对齐 case 的输入数组。

    Args:
        tensors: PyTorch 黄金输入张量。
        image_as_float: 是否将图像转换为取值仍为 ``0..255`` 的 ``float32``。

    Returns:
        按 ONNX 输入名组织的 NumPy 数组。
    """
    result: Dict[str, np.ndarray] = {}
    for name in INPUT_NAMES:
        array = tensor_to_numpy(tensors[name])
        if image_as_float and name in {"global_image", "wrist_image"}:
            array = array.astype(np.float32)
        result[name] = array
    return result


def action_from_normalized_numpy(policy: nn.Module, normalized_action: np.ndarray, n_obs_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """反归一化 NumPy 动作轨迹，并截取策略实际执行的动作段。

    Args:
        policy: 含动作 normalizer 和动作步数配置的策略。
        normalized_action: U-Net 采样后的归一化动作轨迹。
        n_obs_steps: 本次观测序列的时间步数。

    Returns:
        可执行动作段和完整反归一化动作预测。
    """
    params = policy.normalizer.params_dict["action"]
    scale = tensor_to_numpy(params["scale"]).astype(np.float32)
    offset = tensor_to_numpy(params["offset"]).astype(np.float32)
    src_shape = normalized_action.shape
    flat = normalized_action.reshape(-1, scale.shape[0]).astype(np.float32)
    action_pred = ((flat - offset) / scale).reshape(src_shape)
    start = int(n_obs_steps) - 1
    action = action_pred[:, start : start + int(policy.n_action_steps)]
    return action, action_pred
