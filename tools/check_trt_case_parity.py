"""轻量检查 ONNX/TensorRT case，不加载 MP1 checkpoint。

这个脚本用于 Jetson 侧：只消费导出机生成的 ONNX 和 case_000，不导入
train_real / hydra / dill / MP1 训练代码。它适合在依赖很干净的 TensorRT
验证环境里运行。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check exported MP1 ONNX/TensorRT case without loading training code.")
    parser.add_argument("--case-dir", default="deploy_artifacts/trt_cases/case_000")
    parser.add_argument("--onnx-dir", default="deploy_artifacts/onnx")
    parser.add_argument("--trt-output-dir", default="deploy_artifacts/trt_engines")
    parser.add_argument("--report", default="deploy_artifacts/TRT_ACCEL_REPORT_LITE.md")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--skip-ort", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def percentile_stats(values_ms) -> Dict[str, float]:
    values = np.asarray(list(values_ms), dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(np.mean(values)),
    }


def load_ort_session(path: Path):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("缺少 onnxruntime；轻量 Jetson 检查只需要 onnxruntime/onnxruntime-gpu，不需要训练依赖。") from exc

    available = ort.get_available_providers()
    preferred = [provider for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider in available]
    providers = preferred if preferred else available
    return ort.InferenceSession(str(path), providers=providers), providers


def action_from_final_x(final_x: np.ndarray, meta: Mapping[str, Any]) -> np.ndarray:
    normalizer = meta.get("action_normalizer")
    if not normalizer:
        raise RuntimeError("case meta 缺少 action_normalizer；请用新版 tools/dump_trt_case.py 重新生成 case。")
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


def run_ort_case(case_dir: Path, onnx_dir: Path, meta: Mapping[str, Any]) -> Dict[str, Any]:
    obs_session, obs_providers = load_ort_session(onnx_dir / "obs_encoder.onnx")
    unet_session, unet_providers = load_ort_session(onnx_dir / "unet_step.onnx")
    global_cond = obs_session.run(
        ["global_cond"],
        {
            "global_image": np.load(case_dir / "global_image.npy"),
            "wrist_image": np.load(case_dir / "wrist_image.npy"),
            "point_cloud": np.load(case_dir / "point_cloud.npy").astype(np.float32),
            "agent_pos": np.load(case_dir / "agent_pos.npy").astype(np.float32),
        },
    )[0]

    x_current = np.load(case_dir / "initial_noise.npy").astype(np.float32)
    r = np.load(case_dir / "r.npy").astype(np.float32)
    steps = int(meta["num_inference_steps"])
    dt = float(meta["dt"])
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
        step_records.append({"v_pred": v_pred, "x_current": x_current.copy()})
        x_current = x_current + v_pred.astype(np.float32) * dt
    return {
        "providers": {"obs_encoder": obs_providers, "unet_step": unet_providers},
        "global_cond": global_cond,
        "final_x": x_current,
        "action": action_from_final_x(x_current, meta),
        "steps": step_records,
    }


def benchmark(call_once, warmup: int, repeats: int) -> Dict[str, float]:
    for _ in range(max(0, warmup)):
        call_once()
    samples = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        call_once()
        samples.append((time.perf_counter() - start) * 1000.0)
    return percentile_stats(samples)


def optional_trt_diffs(case_dir: Path, trt_output_dir: Path) -> Dict[str, float]:
    diffs: Dict[str, float] = {}
    refs = {
        "global_cond": case_dir / "global_cond_ref.npy",
        "v_pred": case_dir / "v_pred_ref_000.npy",
        "final_action": case_dir / "action_ref.npy",
    }
    outs = {
        "global_cond": trt_output_dir / "global_cond_trt.npy",
        "v_pred": trt_output_dir / "v_pred_trt_000.npy",
        "final_action": trt_output_dir / "action_trt.npy",
    }
    for key, out_path in outs.items():
        if out_path.exists():
            diffs[key] = max_abs_diff(np.load(refs[key]), np.load(out_path))
    return diffs


def fmt(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{float(value):.6g}"


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ort_diffs = payload.get("ort_diffs", {})
    trt_diffs = payload.get("trt_diffs", {})
    latency = payload.get("onnxruntime_latency") or {}
    lines = [
        "# MP1 TensorRT 轻量 Case 对齐报告",
        "",
        "## 结论",
        "",
        "- 本报告不加载 MP1 checkpoint，也不依赖训练代码。",
        "- Jetson 只消费导出好的 `case_000`、`obs_encoder.onnx`、`unet_step.onnx`。",
        "- TensorRT 任一误差超阈值时，不进入真机链路。",
        "",
        "## ONNX Runtime Provider",
        "",
        f"- obs_encoder: `{payload.get('providers', {}).get('obs_encoder', 'N/A')}`",
        f"- unet_step: `{payload.get('providers', {}).get('unet_step', 'N/A')}`",
        "",
        "## 数值对齐",
        "",
        "| 对比项 | max_abs_diff |",
        "| --- | ---: |",
        f"| PyTorch case vs ONNX `global_cond` | {fmt(ort_diffs.get('global_cond'))} |",
        f"| PyTorch case vs ONNX `v_pred_step_000` | {fmt(ort_diffs.get('v_pred_step_000'))} |",
        f"| PyTorch case vs ONNX `final_action` | {fmt(ort_diffs.get('final_action'))} |",
        f"| PyTorch case vs TensorRT `global_cond` | {fmt(trt_diffs.get('global_cond'))} |",
        f"| PyTorch case vs TensorRT `v_pred_step_000` | {fmt(trt_diffs.get('v_pred'))} |",
        f"| PyTorch case vs TensorRT `final_action` | {fmt(trt_diffs.get('final_action'))} |",
        "",
        "## ONNX Runtime 延迟",
        "",
        "| p50 ms | p95 ms | p99 ms | mean ms |",
        "| ---: | ---: | ---: | ---: |",
        f"| {fmt(latency.get('p50_ms'))} | {fmt(latency.get('p95_ms'))} | {fmt(latency.get('p99_ms'))} | {fmt(latency.get('mean_ms'))} |",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    case_dir = Path(args.case_dir).resolve()
    onnx_dir = Path(args.onnx_dir).resolve()
    trt_output_dir = Path(args.trt_output_dir).resolve()
    report = Path(args.report).resolve()
    meta = load_json(case_dir / "meta.json")

    ort_diffs: Dict[str, float] = {}
    providers: Dict[str, Any] = {}
    latency = None
    if not args.skip_ort:
        ort_result = run_ort_case(case_dir, onnx_dir, meta)
        providers = ort_result["providers"]
        ort_diffs = {
            "global_cond": max_abs_diff(np.load(case_dir / "global_cond_ref.npy"), ort_result["global_cond"]),
            "v_pred_step_000": max_abs_diff(np.load(case_dir / "v_pred_ref_000.npy"), ort_result["steps"][0]["v_pred"]),
            "final_action": max_abs_diff(np.load(case_dir / "action_ref.npy"), ort_result["action"]),
        }
        latency = benchmark(lambda: run_ort_case(case_dir, onnx_dir, meta), warmup=args.warmup, repeats=args.repeats)

    payload = {
        "case_dir": str(case_dir),
        "onnx_dir": str(onnx_dir),
        "providers": providers,
        "ort_diffs": ort_diffs,
        "trt_diffs": optional_trt_diffs(case_dir, trt_output_dir),
        "onnxruntime_latency": latency,
    }
    write_report(report, payload)
    print(json.dumps({"report": str(report), **payload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
