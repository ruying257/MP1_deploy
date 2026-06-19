"""冻结 MP1 TensorRT 对齐用例。

输出 case_000 目录，里面包含 ONNX/TensorRT 子图所需输入和 PyTorch 参考输出。
这个脚本不生成 engine，也不接触真实机器人。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from mp1_trt_utils import (
    case_input_numpy,
    load_policy,
    load_sample_tensors,
    platform_report,
    resolve_repo_path,
    run_reference_parts,
    save_numpy,
    summarize_array,
    tensor_to_numpy,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump a frozen MP1 TensorRT parity case.")
    parser.add_argument("--checkpoint", default="python_deploy/checkpoints/latest.ckpt")
    parser.add_argument("--tensor-dir", default="deploy_artifacts/sample_tensors")
    parser.add_argument("--output-dir", default="deploy_artifacts/trt_cases/case_000")
    parser.add_argument("--device", default="cpu", help="Use cpu for portable dumps, cuda on Jetson for speed.")
    parser.add_argument("--image-input-float", action="store_true", default=True, help="Save ONNX image inputs as float32 0..255.")
    parser.add_argument("--uint8-images", dest="image_input_float", action="store_false", help="Save image inputs as uint8.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, policy = load_policy(args.checkpoint, device=device)
    tensors = load_sample_tensors(args.tensor_dir, device=device)
    reference = run_reference_parts(policy, tensors)

    inputs = case_input_numpy(tensors, image_as_float=bool(args.image_input_float))
    for name, value in inputs.items():
        save_numpy(output_dir / f"{name}.npy", value)

    save_numpy(output_dir / "global_cond_ref.npy", reference["global_cond"])
    save_numpy(output_dir / "final_x_ref.npy", reference["final_x"])
    save_numpy(output_dir / "action_pred_ref.npy", reference["action_pred"])
    save_numpy(output_dir / "action_ref.npy", reference["action"])
    save_numpy(output_dir / "r.npy", reference["r"])

    step_meta = []
    for index, item in enumerate(reference["steps"]):
        save_numpy(output_dir / f"x_current_{index:03d}.npy", item["x_current"])
        save_numpy(output_dir / f"timestep_{index:03d}.npy", item["timestep"])
        save_numpy(output_dir / f"v_pred_ref_{index:03d}.npy", item["v_pred"])
        step_meta.append({
            "index": index,
            "x_current": f"x_current_{index:03d}.npy",
            "timestep": f"timestep_{index:03d}.npy",
            "v_pred_ref": f"v_pred_ref_{index:03d}.npy",
        })

    input_summary: Dict[str, Any] = {name: summarize_array(value) for name, value in inputs.items()}
    action_params = policy.normalizer.params_dict["action"]
    meta = {
        "format_version": 1,
        "purpose": "MP1 ONNX/TensorRT parity case",
        "checkpoint": str(resolve_repo_path(args.checkpoint)),
        "tensor_dir": str(resolve_repo_path(args.tensor_dir)),
        "device": str(device),
        "image_input_dtype": "float32" if args.image_input_float else "uint8",
        "image_value_range": "0..255",
        "n_obs_steps": int(cfg.n_obs_steps),
        "horizon": int(policy.horizon),
        "action_dim": int(policy.action_dim),
        "n_action_steps": int(policy.n_action_steps),
        "num_inference_steps": int(policy.num_inference_steps if policy.num_inference_steps is not None else 10),
        "dt": 1.0 / float(policy.num_inference_steps if policy.num_inference_steps is not None else 10),
        "action_normalizer": {
            "scale": tensor_to_numpy(action_params["scale"]).astype("float32").tolist(),
            "offset": tensor_to_numpy(action_params["offset"]).astype("float32").tolist(),
        },
        "inputs": input_summary,
        "steps": step_meta,
        "outputs": {
            "global_cond_ref": "global_cond_ref.npy",
            "final_x_ref": "final_x_ref.npy",
            "action_pred_ref": "action_pred_ref.npy",
            "action_ref": "action_ref.npy",
        },
        "platform": platform_report(),
    }
    write_json(output_dir / "meta.json", meta)
    print(json.dumps({"case_dir": str(output_dir), "steps": len(step_meta)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
