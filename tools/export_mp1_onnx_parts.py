"""导出 MP1 TensorRT 证据原型所需 ONNX 子图。

导出两个固定 shape 子图：
1. obs_encoder.onnx: 多模态观测 -> global_cond
2. unet_step.onnx: x_current,timestep,global_cond,r -> v_pred
"""

from __future__ import annotations

import argparse
import json

import torch

from mp1_trt_utils import (
    MP1ObsEncoderPart,
    MP1UnetStepPart,
    load_policy,
    load_sample_tensors,
    platform_report,
    resolve_repo_path,
    run_reference_parts,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MP1 obs_encoder/unet_step ONNX parts.")
    parser.add_argument("--checkpoint", default="python_deploy/checkpoints/latest.ckpt")
    parser.add_argument("--tensor-dir", default="deploy_artifacts/sample_tensors")
    parser.add_argument("--output-dir", default="deploy_artifacts/onnx")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--image-input-float", action="store_true", default=True, help="Export image inputs as float32 0..255.")
    parser.add_argument("--uint8-images", dest="image_input_float", action="store_false", help="Export image inputs as uint8.")
    return parser.parse_args()


def maybe_float_images(tensors):
    return {
        "global_image": tensors["global_image"].float(),
        "wrist_image": tensors["wrist_image"].float(),
        "point_cloud": tensors["point_cloud"].float(),
        "agent_pos": tensors["agent_pos"].float(),
        "initial_noise": tensors["initial_noise"].float(),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, policy = load_policy(args.checkpoint, device=device)
    tensors = load_sample_tensors(args.tensor_dir, device=device)
    if args.image_input_float:
        tensors = maybe_float_images(tensors)

    obs_part = MP1ObsEncoderPart(policy).eval().to(device)
    unet_part = MP1UnetStepPart(policy).eval().to(device)
    reference = run_reference_parts(policy, tensors)

    obs_path = output_dir / "obs_encoder.onnx"
    unet_path = output_dir / "unet_step.onnx"

    with torch.no_grad():
        torch.onnx.export(
            obs_part,
            (
                tensors["global_image"],
                tensors["wrist_image"],
                tensors["point_cloud"],
                tensors["agent_pos"],
            ),
            str(obs_path),
            input_names=["global_image", "wrist_image", "point_cloud", "agent_pos"],
            output_names=["global_cond"],
            opset_version=int(args.opset),
            do_constant_folding=True,
        )

        first_step = reference["steps"][0]
        torch.onnx.export(
            unet_part,
            (
                first_step["x_current"],
                first_step["timestep"],
                reference["global_cond"],
                reference["r"],
            ),
            str(unet_path),
            input_names=["x_current", "timestep", "global_cond", "r"],
            output_names=["v_pred"],
            opset_version=int(args.opset),
            do_constant_folding=True,
        )

    meta = {
        "format_version": 1,
        "checkpoint": str(resolve_repo_path(args.checkpoint)),
        "tensor_dir": str(resolve_repo_path(args.tensor_dir)),
        "opset": int(args.opset),
        "image_input_dtype": "float32" if args.image_input_float else "uint8",
        "image_value_range": "0..255",
        "obs_encoder": str(obs_path),
        "unet_step": str(unet_path),
        "n_obs_steps": int(cfg.n_obs_steps),
        "horizon": int(policy.horizon),
        "action_dim": int(policy.action_dim),
        "num_inference_steps": int(policy.num_inference_steps if policy.num_inference_steps is not None else 10),
        "platform": platform_report(),
    }
    write_json(output_dir / "onnx_export_meta.json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

