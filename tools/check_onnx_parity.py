"""检查 MP1 ONNX/TensorRT 证据原型的数值误差和延迟。

第一性原理：TensorRT 只能证明“同一组输入、同一段子图”的输出足够接近。
所以这个脚本固定 sample_tensors，分别跑 PyTorch 子图、ONNX Runtime 子图，
并把 Jetson TensorRT 结果汇总到同一份报告。真实机器人闭环仍不走这里。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch

from mp1_trt_utils import (
    action_from_normalized_numpy,
    case_input_numpy,
    load_policy,
    load_sample_tensors,
    max_abs_diff,
    percentile_stats,
    platform_report,
    resolve_repo_path,
    run_reference_parts,
    sync_if_cuda,
    tensor_to_numpy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check MP1 ONNX Runtime parity and write TensorRT evidence report.")
    parser.add_argument("--checkpoint", default="python_deploy/checkpoints/latest.ckpt")
    parser.add_argument("--tensor-dir", default="deploy_artifacts/sample_tensors")
    parser.add_argument("--onnx-dir", default="deploy_artifacts/onnx")
    parser.add_argument("--torchscript-model", default="deploy_artifacts/policy_infer.pt")
    parser.add_argument("--report", default="deploy_artifacts/TRT_ACCEL_REPORT.md")
    parser.add_argument("--metrics-json", default="deploy_artifacts/trt_engines/trt_metrics.json")
    parser.add_argument("--trt-output-dir", default="deploy_artifacts/trt_engines", help="Optional TensorRT .npy outputs to compare.")
    parser.add_argument("--device", default="cuda", help="Benchmark device. Falls back to cpu when CUDA is unavailable.")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--skip-ort", action="store_true", help="Only generate the report shell without ONNX Runtime.")
    parser.add_argument("--image-input-float", action="store_true", default=True, help="Use float32 0..255 image inputs for ONNX.")
    parser.add_argument("--uint8-images", dest="image_input_float", action="store_false", help="Use uint8 image inputs for ONNX.")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def load_ort_session(path: Path, requested_device: torch.device):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("缺少 onnxruntime；请先安装 onnxruntime 或 onnxruntime-gpu 后再做 ONNX 对齐。") from exc

    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"] if requested_device.type == "cuda" else ["CPUExecutionProvider"]
    providers = [provider for provider in preferred if provider in available]
    if not providers:
        providers = available
    return ort.InferenceSession(str(path), providers=providers), providers


def run_ort_with_sessions(
    policy,
    tensors: Mapping[str, torch.Tensor],
    obs_session,
    unet_session,
    image_input_float: bool,
) -> Dict[str, Any]:
    inputs = case_input_numpy(tensors, image_as_float=image_input_float)
    obs_inputs = {
        "global_image": np.ascontiguousarray(inputs["global_image"]),
        "wrist_image": np.ascontiguousarray(inputs["wrist_image"]),
        "point_cloud": np.ascontiguousarray(inputs["point_cloud"].astype(np.float32)),
        "agent_pos": np.ascontiguousarray(inputs["agent_pos"].astype(np.float32)),
    }
    global_cond = obs_session.run(["global_cond"], obs_inputs)[0]

    x_current = np.ascontiguousarray(inputs["initial_noise"].astype(np.float32))
    steps = int(policy.num_inference_steps if policy.num_inference_steps is not None else 10)
    dt = 1.0 / float(steps)
    r = (x_current[:, 0, 0] * 0.0).astype(np.float32)
    step_records = []
    for index in range(steps):
        timestep = (r + float(index) / float(steps)).astype(np.float32)
        v_pred = unet_session.run(
            ["v_pred"],
            {
                "x_current": np.ascontiguousarray(x_current),
                "timestep": np.ascontiguousarray(timestep),
                "global_cond": np.ascontiguousarray(global_cond.astype(np.float32)),
                "r": np.ascontiguousarray(r),
            },
        )[0]
        step_records.append({"timestep": timestep, "v_pred": v_pred, "x_current": x_current.copy()})
        x_current = x_current + v_pred.astype(np.float32) * dt

    action, action_pred = action_from_normalized_numpy(policy, x_current[..., : int(policy.action_dim)], tensors["global_image"].shape[1])
    return {
        "global_cond": global_cond,
        "final_x": x_current,
        "action_pred": action_pred,
        "action": action,
        "steps": step_records,
    }


def run_ort_parts(
    policy,
    tensors: Mapping[str, torch.Tensor],
    onnx_dir: Path,
    device: torch.device,
    image_input_float: bool,
) -> Dict[str, Any]:
    obs_session, obs_providers = load_ort_session(onnx_dir / "obs_encoder.onnx", device)
    unet_session, unet_providers = load_ort_session(onnx_dir / "unet_step.onnx", device)
    result = run_ort_with_sessions(policy, tensors, obs_session, unet_session, image_input_float)
    result["providers"] = {"obs_encoder": obs_providers, "unet_step": unet_providers}
    return result


def timed_loop(callable_obj, warmup: int, repeats: int, device: torch.device) -> Dict[str, float]:
    for _ in range(max(0, warmup)):
        callable_obj()
    sync_if_cuda(device)

    samples_ms = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        callable_obj()
        sync_if_cuda(device)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return percentile_stats(samples_ms)


def benchmark_torchscript(model_path: Path, tensors: Mapping[str, torch.Tensor], device: torch.device, warmup: int, repeats: int) -> Optional[Dict[str, float]]:
    if not model_path.exists():
        return None
    module = torch.jit.load(str(model_path), map_location=device).eval()
    inputs = [
        tensors["global_image"].to(device),
        tensors["wrist_image"].to(device),
        tensors["point_cloud"].to(device),
        tensors["agent_pos"].to(device),
        tensors["initial_noise"].to(device),
    ]

    @torch.no_grad()
    def call_once():
        return module(*inputs)

    return timed_loop(call_once, warmup=warmup, repeats=repeats, device=device)


def benchmark_ort(
    policy,
    tensors: Mapping[str, torch.Tensor],
    onnx_dir: Path,
    device: torch.device,
    warmup: int,
    repeats: int,
    image_input_float: bool,
) -> Optional[Dict[str, float]]:
    if not (onnx_dir / "obs_encoder.onnx").exists() or not (onnx_dir / "unet_step.onnx").exists():
        return None

    obs_session, _ = load_ort_session(onnx_dir / "obs_encoder.onnx", device)
    unet_session, _ = load_ort_session(onnx_dir / "unet_step.onnx", device)

    def call_once():
        return run_ort_with_sessions(policy, tensors, obs_session, unet_session, image_input_float)

    return timed_loop(call_once, warmup=warmup, repeats=repeats, device=torch.device("cpu"))


def read_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def compare_optional_trt_outputs(trt_output_dir: Path, reference: Mapping[str, Any]) -> Dict[str, float]:
    """读取 TensorRT runner 可选输出，直接和 PyTorch 参考输出比较。

    约定文件名：
    - global_cond_trt.npy
    - v_pred_trt_000.npy
    - action_trt.npy 或 final_action_trt.npy
    """
    comparisons: Dict[str, float] = {}
    global_cond_path = trt_output_dir / "global_cond_trt.npy"
    v_pred_path = trt_output_dir / "v_pred_trt_000.npy"
    action_path = trt_output_dir / "action_trt.npy"
    final_action_path = trt_output_dir / "final_action_trt.npy"

    if global_cond_path.exists():
        comparisons["global_cond_max_abs_diff"] = max_abs_diff(tensor_to_numpy(reference["global_cond"]), np.load(global_cond_path))
    if v_pred_path.exists():
        comparisons["v_pred_max_abs_diff"] = max_abs_diff(tensor_to_numpy(reference["steps"][0]["v_pred"]), np.load(v_pred_path))
    if action_path.exists():
        comparisons["final_action_max_abs_diff"] = max_abs_diff(tensor_to_numpy(reference["action"]), np.load(action_path))
    elif final_action_path.exists():
        comparisons["final_action_max_abs_diff"] = max_abs_diff(tensor_to_numpy(reference["action"]), np.load(final_action_path))
    return comparisons


def fmt_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "N/A"
    return f"{float(value):.{digits}g}"


def latency_row(name: str, stats: Optional[Mapping[str, Any]]) -> str:
    if not stats:
        return f"| {name} | N/A | N/A | N/A | N/A |"
    return (
        f"| {name} | {fmt_float(stats.get('p50_ms'), 4)} | {fmt_float(stats.get('p95_ms'), 4)} | "
        f"{fmt_float(stats.get('p99_ms'), 4)} | {fmt_float(stats.get('mean_ms'), 4)} |"
    )


def write_report(
    report_path: Path,
    payload: Mapping[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    platform_info = payload["platform"]
    diffs = payload.get("diffs", {})
    latencies = payload.get("latencies", {})
    trt_metrics = payload.get("trt_metrics")
    providers = payload.get("ort_providers", {})

    lines = [
        "# MP1 TensorRT 加速证据报告",
        "",
        "## 结论",
        "",
        "- v1 目标是离线证据链，不直接替换真机闭环 runtime。",
        "- TensorRT 子图只允许在 `global_cond`、`v_pred`、`final_action` 误差通过阈值后进入 dry-run。",
        "- TorchScript `policy_infer.pt` 仍是主闭环 fallback。",
        "",
        "## 平台信息",
        "",
        "| 字段 | 值 |",
        "| --- | --- |",
    ]
    for key in ["platform", "machine", "python", "torch", "torch_cuda", "cuda_available", "gpu_name", "trtexec", "nvidia_smi", "jetson_release"]:
        lines.append(f"| {key} | {platform_info.get(key)} |")

    lines.extend(
        [
            "",
            "## ONNX Runtime Provider",
            "",
            f"- obs_encoder: `{providers.get('obs_encoder', 'N/A')}`",
            f"- unet_step: `{providers.get('unet_step', 'N/A')}`",
            "",
            "## 数值对齐",
            "",
            "| 对比项 | max_abs_diff | 目标 |",
            "| --- | ---: | --- |",
            f"| PyTorch vs ONNX `global_cond` | {fmt_float(diffs.get('global_cond'))} | <= 1e-4 |",
            f"| PyTorch vs ONNX `v_pred_step_000` | {fmt_float(diffs.get('v_pred_step_000'))} | <= 1e-4 |",
            f"| PyTorch vs ONNX `final_action` | {fmt_float(diffs.get('final_action'))} | <= 1e-4 优先 |",
            f"| PyTorch vs TensorRT FP16 `global_cond` | {fmt_float((trt_metrics or {}).get('global_cond_max_abs_diff'))} | 需结合动作安全限幅判断 |",
            f"| PyTorch vs TensorRT FP16 `v_pred` | {fmt_float((trt_metrics or {}).get('v_pred_max_abs_diff'))} | 需结合动作安全限幅判断 |",
            f"| PyTorch vs TensorRT FP16 `final_action` | {fmt_float((trt_metrics or {}).get('final_action_max_abs_diff'))} | < 单步安全限幅 5% |",
            "",
            "## 延迟统计",
            "",
            f"warmup={payload.get('warmup')}，repeats={payload.get('repeats')}。warmup 不计入统计。",
            "",
            "| Runtime | p50 ms | p95 ms | p99 ms | mean ms |",
            "| --- | ---: | ---: | ---: | ---: |",
            latency_row("TorchScript CUDA/selected device", latencies.get("torchscript")),
            latency_row("ONNX Runtime", latencies.get("onnxruntime")),
            latency_row("TensorRT FP16 obs_encoder", (trt_metrics or {}).get("obs_encoder_latency")),
            latency_row("TensorRT FP16 unet_step", (trt_metrics or {}).get("unet_step_latency")),
            latency_row("TensorRT FP16 full loop", (trt_metrics or {}).get("full_loop_latency")),
            "",
            "## Jetson TensorRT Engine 生成命令",
            "",
            "在 Jetson 本机执行，不要在 Windows 上承诺 TensorRT 性能：",
            "",
            "```bash",
            "mkdir -p deploy_artifacts/trt_engines",
            "trtexec --onnx=deploy_artifacts/onnx/obs_encoder.onnx \\",
            "  --saveEngine=deploy_artifacts/trt_engines/obs_encoder_fp16.engine \\",
            "  --fp16 --warmUp=500 --duration=20",
            "",
            "trtexec --onnx=deploy_artifacts/onnx/unet_step.onnx \\",
            "  --saveEngine=deploy_artifacts/trt_engines/unet_step_fp16.engine \\",
            "  --fp16 --warmUp=500 --duration=20",
            "```",
            "",
            "## TensorRT 指标录入格式",
            "",
            "`tools/check_onnx_parity.py` 会自动读取 `deploy_artifacts/trt_engines/trt_metrics.json`，也会读取同目录下可选的 `global_cond_trt.npy`、`v_pred_trt_000.npy`、`action_trt.npy` 做直接误差比较。格式示例：",
            "",
            "```json",
            json.dumps(
                {
                    "global_cond_max_abs_diff": 0.0,
                    "v_pred_max_abs_diff": 0.0,
                    "final_action_max_abs_diff": 0.0,
                    "obs_encoder_latency": {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0},
                    "unet_step_latency": {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0},
                    "full_loop_latency": {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0},
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
            "## Gate",
            "",
            "- ONNX FP32 未对齐时，不生成 TensorRT 结论。",
            "- TensorRT FP16 `final_action` 超过单步安全限幅 5% 时，不进入真机链路。",
            "- TensorRT 加速只作为 dry-run 候选；TorchScript fallback 必须保持可用。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    requested_device = torch.device(args.device)
    device = select_device(args.device)
    checkpoint = resolve_repo_path(args.checkpoint)
    tensor_dir = resolve_repo_path(args.tensor_dir)
    onnx_dir = resolve_repo_path(args.onnx_dir)
    report_path = resolve_repo_path(args.report)
    torchscript_model = resolve_repo_path(args.torchscript_model)
    metrics_json = resolve_repo_path(args.metrics_json)
    trt_output_dir = resolve_repo_path(args.trt_output_dir)

    _, policy = load_policy(checkpoint, device=device)
    tensors = load_sample_tensors(tensor_dir, device=device)
    reference = run_reference_parts(policy, tensors)

    diffs: Dict[str, float] = {}
    ort_result: Optional[Dict[str, Any]] = None
    ort_providers: Dict[str, Any] = {}
    if not args.skip_ort:
        ort_result = run_ort_parts(policy, tensors, onnx_dir, device, bool(args.image_input_float))
        ort_providers = ort_result["providers"]
        diffs = {
            "global_cond": max_abs_diff(tensor_to_numpy(reference["global_cond"]), ort_result["global_cond"]),
            "v_pred_step_000": max_abs_diff(tensor_to_numpy(reference["steps"][0]["v_pred"]), ort_result["steps"][0]["v_pred"]),
            "final_action": max_abs_diff(tensor_to_numpy(reference["action"]), ort_result["action"]),
        }

    latencies = {
        "torchscript": benchmark_torchscript(torchscript_model, tensors, device=device, warmup=args.warmup, repeats=args.repeats),
        "onnxruntime": None if args.skip_ort else benchmark_ort(policy, tensors, onnx_dir, device, args.warmup, args.repeats, bool(args.image_input_float)),
    }
    trt_metrics = read_optional_json(metrics_json) or {}
    trt_metrics.update(compare_optional_trt_outputs(trt_output_dir, reference))
    if not trt_metrics:
        trt_metrics = None

    payload = {
        "checkpoint": str(checkpoint),
        "tensor_dir": str(tensor_dir),
        "onnx_dir": str(onnx_dir),
        "trt_output_dir": str(trt_output_dir),
        "requested_device": str(requested_device),
        "selected_device": str(device),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "platform": platform_report(),
        "diffs": diffs,
        "ort_providers": ort_providers,
        "latencies": latencies,
        "trt_metrics": trt_metrics,
    }
    write_report(report_path, payload)
    print(json.dumps({"report": str(report_path), "diffs": diffs, "latencies": latencies}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
