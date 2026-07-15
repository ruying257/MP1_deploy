"""导出用于 TensorRT 数值对齐的 MP1 策略 ONNX 子图。

本脚本不导出完整的迭代采样循环，而是将策略拆分为两个固定 shape 的子图：

1. ``obs_encoder.onnx``：多模态观测 -> ``global_cond``。
2. ``unet_step.onnx``：``x_current, timestep, global_cond, r`` -> ``v_pred``。

导出的模型仅供离线 ONNX Runtime/TensorRT 对齐工具使用，不进入真机控制链路。
"""

from __future__ import annotations

import argparse
import json

import torch

from .mp1_trt_utils import (
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
    """解析 ONNX 子图导出的命令行参数。

    Returns:
        导出参数，包括源 checkpoint、黄金样本张量、输出目录、ONNX opset
        和图像输入 dtype。
    """
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
    """将导出输入转换为 ONNX TensorRT 原型所需的 dtype。

    图像张量从 ``uint8`` 转为 ``float32``，但保留 ``0..255`` 的数值范围。
    点云、机器人状态和初始噪声也会显式转换为 ``float32``，使 ONNX 输入
    契约保持统一。

    Args:
        tensors: 从 ``sample_tensors`` 加载的黄金输入张量。

    Returns:
        所有数值均为 ``float32`` 的新张量映射。
    """
    return {
        "global_image": tensors["global_image"].float(),
        "wrist_image": tensors["wrist_image"].float(),
        "point_cloud": tensors["point_cloud"].float(),
        "agent_pos": tensors["agent_pos"].float(),
        "initial_noise": tensors["initial_noise"].float(),
    }


def main() -> None:
    """导出 ONNX 子图，并写出可复现实验元数据。

    函数加载 EMA 策略和固定黄金样本，执行一次 PyTorch 参考推理，并导出：

    * 以原始多模态观测为输入的观测编码器；
    * 以第一个参考采样步为输入的 U-Net 速度预测单步子图。

    首步张量为 ONNX trace 提供具体 shape 和 dtype。``onnx_export_meta.json``
    会记录模型契约与本机平台信息，供后续解读对齐报告使用。
    """
    args = parse_args()
    device = torch.device(args.device)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 EMA 权重，确保 ONNX 和 PyTorch 参考路径使用相同参数。
    cfg, policy = load_policy(args.checkpoint, device=device)
    tensors = load_sample_tensors(args.tensor_dir, device=device)
    if args.image_input_float:
        # TensorRT 部署通常使用 FP32 图像输入，数值范围仍保持为 0..255。
        tensors = maybe_float_images(tensors)

    obs_part = MP1ObsEncoderPart(policy).eval().to(device)
    unet_part = MP1UnetStepPart(policy).eval().to(device)
    # 记录 U-Net 第一个采样步所需的中间参考张量。
    reference = run_reference_parts(policy, tensors)

    obs_path = output_dir / "obs_encoder.onnx"
    unet_path = output_dir / "unet_step.onnx"

    with torch.no_grad():
        # 将多模态观测编码与迭代动作采样拆分为两个可独立验证的子图。
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

        # U-Net 导出为可复用的单步子图；外部运行时负责循环：
        # x_next = x_current + v_pred * dt。
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

    # 在模型旁保存静态 ONNX 契约和环境信息。
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
    action_params = policy.normalizer.params_dict["action"]
    trt_runtime_meta = {
        "format_version": 1,
        "image_input_dtype": meta["image_input_dtype"],
        "n_obs_steps": int(cfg.n_obs_steps),
        "horizon": int(policy.horizon),
        "action_dim": int(policy.action_dim),
        "n_action_steps": int(policy.n_action_steps),
        "num_inference_steps": int(policy.num_inference_steps if policy.num_inference_steps is not None else 10),
        "dt": 1.0 / float(policy.num_inference_steps if policy.num_inference_steps is not None else 10),
        "action_normalizer": {
            "scale": action_params["scale"].detach().cpu().to(torch.float32).tolist(),
            "offset": action_params["offset"].detach().cpu().to(torch.float32).tolist(),
        },
    }
    meta["trt_runtime_meta"] = str(output_dir / "trt_runtime_meta.json")
    write_json(output_dir / "onnx_export_meta.json", meta)
    write_json(output_dir / "trt_runtime_meta.json", trt_runtime_meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
