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


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    """将相对路径解析到仓库根目录。

    Args:
        path: 相对或绝对路径。

    Returns:
        对应的路径对象。
    """
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def parse_args() -> argparse.Namespace:
    """解析轻量 ONNX/TensorRT case 校验参数。

    Returns:
        包含 case、ONNX、TensorRT 输出、报告和基准参数的命名空间。
    """
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
    """读取 UTF-8 JSON 文件。

    Args:
        path: JSON 文件路径。

    Returns:
        解析后的字典。
    """
    return json.loads(path.read_text(encoding="utf-8"))


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个数组间的最大逐元素绝对误差。

    Args:
        a: 参考数组。
        b: 待比较数组。

    Returns:
        基于 ``float32`` 的最大绝对误差。
    """
    return float(np.max(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))))


def percentile_stats(values_ms) -> Dict[str, float]:
    """计算延迟样本的分位数与均值。

    Args:
        values_ms: 单位为毫秒的延迟样本。

    Returns:
        p50、p95、p99 和均值。
    """
    values = np.asarray(list(values_ms), dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "mean_ms": float(np.mean(values)),
    }


def load_ort_session(path: Path):
    """按可用 provider 创建 ONNX Runtime 会话。

    Args:
        path: ONNX 模型文件路径。

    Returns:
        ONNX Runtime 会话和实际启用的 provider 列表。

    Raises:
        RuntimeError: 未安装 ``onnxruntime`` 或 ``onnxruntime-gpu`` 时抛出。
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("缺少 onnxruntime；轻量 Jetson 检查只需要 onnxruntime/onnxruntime-gpu，不需要训练依赖。") from exc

    available = ort.get_available_providers()
    preferred = [provider for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider in available]
    providers = preferred if preferred else available
    return ort.InferenceSession(str(path), providers=providers), providers


def action_from_final_x(final_x: np.ndarray, meta: Mapping[str, Any]) -> np.ndarray:
    """从最终归一化状态还原实际执行的动作段。

    Args:
        final_x: 外部采样循环结束后的归一化状态。
        meta: 冻结 case 的元数据，必须包含动作 normalizer。

    Returns:
        与 ``action_ref.npy`` 对应的可执行动作段。

    Raises:
        RuntimeError: 元数据缺少动作反归一化参数时抛出。
    """
    normalizer = meta.get("action_normalizer")
    if not normalizer:
        raise RuntimeError("case meta 缺少 action_normalizer；请用 python3 -m tools.tensorrt.dump_trt_case 重新生成 case。")
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
    """使用冻结 case 在 ONNX Runtime 中复现完整采样过程。

    Args:
        case_dir: 包含输入和 PyTorch 参考输出的冻结 case 目录。
        onnx_dir: ``obs_encoder.onnx`` 与 ``unet_step.onnx`` 所在目录。
        meta: case 元数据。

    Returns:
        编码器输出、逐步 U-Net 输出、最终状态、动作和 provider 信息。
    """
    obs_session, obs_providers = load_ort_session(onnx_dir / "obs_encoder.onnx")
    unet_session, unet_providers = load_ort_session(onnx_dir / "unet_step.onnx")
    # 先运行观测编码器，再由外部循环重复调用单步 U-Net 子图。
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
    """对一次完整 case 推理执行预热和延迟采样。

    Args:
        call_once: 单次待测调用。
        warmup: 不计入统计的预热次数。
        repeats: 计入统计的重复次数。

    Returns:
        毫秒级 p50、p95、p99 和均值。
    """
    for _ in range(max(0, warmup)):
        call_once()
    samples = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        call_once()
        samples.append((time.perf_counter() - start) * 1000.0)
    return percentile_stats(samples)


def optional_trt_diffs(case_dir: Path, trt_output_dir: Path) -> Dict[str, float]:
    """比较可选的 TensorRT runner 输出与冻结 PyTorch 参考输出。

    Args:
        case_dir: 冻结 case 目录。
        trt_output_dir: 可选 ``.npy`` TensorRT 输出目录。

    Returns:
        已存在输出文件的误差字典；缺失文件不会视为错误。
    """
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
    """将可选浮点数格式化为报告表格单元格。

    Args:
        value: 待显示数值。

    Returns:
        数值字符串；缺失时返回 ``N/A``。
    """
    return "N/A" if value is None else f"{float(value):.6g}"


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    """生成不依赖训练代码的 Markdown 对齐报告。

    Args:
        path: Markdown 报告输出路径。
        payload: ONNX Runtime、TensorRT 误差和延迟统计结果。
    """
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
    """执行轻量 case 对齐、可选 TensorRT 误差比较并写出报告。"""
    args = parse_args()
    case_dir = resolve_repo_path(args.case_dir)
    onnx_dir = resolve_repo_path(args.onnx_dir)
    trt_output_dir = resolve_repo_path(args.trt_output_dir)
    report = resolve_repo_path(args.report)
    meta = load_json(case_dir / "meta.json")

    ort_diffs: Dict[str, float] = {}
    providers: Dict[str, Any] = {}
    latency = None
    # ONNX Runtime 可选；即使 Jetson 仅保留 TensorRT 产物，也能生成报告。
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
