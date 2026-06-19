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


REPO_ROOT = Path(__file__).resolve().parents[1]
PY_DEPLOY_ROOT = REPO_ROOT / "python_deploy"
MP1_ROOT = PY_DEPLOY_ROOT / "MP1"
ROBOT_SCRIPT_DIR = PY_DEPLOY_ROOT / "real_robot_ur12e_d405_speed_only" / "scripts"
for _path in [PY_DEPLOY_ROOT, MP1_ROOT, ROBOT_SCRIPT_DIR]:
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

INPUT_NAMES = ["global_image", "wrist_image", "point_cloud", "agent_pos", "initial_noise"]


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def load_policy(checkpoint: str | Path, device: torch.device):
    # 真正加载 checkpoint 时才导入部署栈，避免 --help 被 dill/torch 依赖挡住。
    from deploy_real_policy import load_workspace_policy

    cfg, policy = load_workspace_policy(resolve_repo_path(checkpoint), device=device, use_ema=True)
    policy.eval()
    return cfg, policy


def load_tensor_module(path: str | Path, device: torch.device) -> torch.Tensor:
    module = torch.jit.load(str(resolve_repo_path(path)), map_location=device)
    module.eval()
    return module.forward().to(device)


def load_sample_tensors(tensor_dir: str | Path, device: torch.device) -> Dict[str, torch.Tensor]:
    root = resolve_repo_path(tensor_dir)
    tensors = {name: load_tensor_module(root / f"{name}.pt", device) for name in INPUT_NAMES}
    tensors["global_image"] = tensors["global_image"].to(torch.uint8)
    tensors["wrist_image"] = tensors["wrist_image"].to(torch.uint8)
    tensors["point_cloud"] = tensors["point_cloud"].to(torch.float32)
    tensors["agent_pos"] = tensors["agent_pos"].to(torch.float32)
    tensors["initial_noise"] = tensors["initial_noise"].to(torch.float32)
    return tensors


def tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def save_numpy(path: str | Path, value: torch.Tensor | np.ndarray) -> None:
    array = tensor_to_numpy(value) if torch.is_tensor(value) else np.asarray(value)
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def summarize_array(value: np.ndarray) -> Dict[str, Any]:
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
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile_stats(values_ms: Iterable[float]) -> Dict[str, float]:
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
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    text = (result.stdout or result.stderr).strip()
    return text.splitlines()[0] if text else None


def platform_report() -> Dict[str, Any]:
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
    """导出 obs_encoder 子图：多模态观测 -> global_cond。"""

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def _normalize_field(self, value: torch.Tensor, key: str, forward: bool) -> torch.Tensor:
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
    """导出单步动作生成子图：x_current,t,global_cond,r -> v_pred。"""

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
    obs_part = MP1ObsEncoderPart(policy).eval()
    unet_part = MP1UnetStepPart(policy).eval()
    global_cond = obs_part(
        tensors["global_image"],
        tensors["wrist_image"],
        tensors["point_cloud"],
        tensors["agent_pos"],
    )

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
    result: Dict[str, np.ndarray] = {}
    for name in INPUT_NAMES:
        array = tensor_to_numpy(tensors[name])
        if image_as_float and name in {"global_image", "wrist_image"}:
            array = array.astype(np.float32)
        result[name] = array
    return result


def action_from_normalized_numpy(policy: nn.Module, normalized_action: np.ndarray, n_obs_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    params = policy.normalizer.params_dict["action"]
    scale = tensor_to_numpy(params["scale"]).astype(np.float32)
    offset = tensor_to_numpy(params["offset"]).astype(np.float32)
    src_shape = normalized_action.shape
    flat = normalized_action.reshape(-1, scale.shape[0]).astype(np.float32)
    action_pred = ((flat - offset) / scale).reshape(src_shape)
    start = int(n_obs_steps) - 1
    action = action_pred[:, start : start + int(policy.n_action_steps)]
    return action, action_pred
