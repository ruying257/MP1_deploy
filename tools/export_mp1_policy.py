"""
MP1 策略导出工具

将训练好的 MP1 (Motion Planning Transformer) 策略模型导出为 TorchScript 格式，
用于 C++ 部署。主要输出包括：
1. policy_infer.pt - 冻结的 TorchScript 模型（包含 normalizer、encoder、采样循环）
2. deploy_meta.json - 部署所需的元数据配置（动作维度、观测形状、归一化参数等）
3. sample_tensors/ - 可选的测试样本张量（用于离线测试和 C++ 推理对齐）
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf


# 路径配置：添加项目依赖路径到 sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
PY_DEPLOY_ROOT = REPO_ROOT / "python_deploy"
MP1_ROOT = PY_DEPLOY_ROOT / "MP1"
ROBOT_SCRIPT_DIR = PY_DEPLOY_ROOT / "real_robot_ur12e_d405_speed_only" / "scripts"
for path in [PY_DEPLOY_ROOT, MP1_ROOT, ROBOT_SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# 导入策略加载模块
from deploy_real_policy import load_workspace_policy 


class MP1TorchScriptWrapper(nn.Module):
    """
    MP1 策略的 TorchScript 包装器
    
    将完整的推理流程封装为单个 TorchScript 模块，包含：
    1. 观测归一化 (normalizer)
    2. 观测编码 (encoder)
    3. U-Net 扩散采样循环
    
    这样可以减少 C++ 侧的实现复杂度和潜在误差源。
    
    Args:
        policy: 训练好的 MP1 策略模型
    """

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def _normalize_field(self, value: torch.Tensor, key: str, forward: bool) -> torch.Tensor:
        """
        执行单个字段的线性归一化/反归一化。

        注意：这里不能按 normalizer.py（原版MP1）里的实现把输入搬到 scale.device。
        torch.jit.trace 会把导出时的设备写死；如果在 CPU 上导出，Jetson CUDA 推理时
        就会出现部分算子固定在 CPU、部分算子在 GPU 的混用错误。
        现在的逻辑：仅改变输入张量的 dtype 为 scale 的 dtype，设备由调用方决定
        """
        params = self.policy.normalizer.params_dict[key]
        scale = params["scale"]
        offset = params["offset"]
        value = value.to(dtype=scale.dtype)
        src_shape = value.shape     # 记录原始形状
        value = value.reshape(-1, scale.shape[0])   # 批次展开，方便广播
        if forward:
            value = value * scale + offset
        else:
            value = (value - offset) / scale
        return value.reshape(src_shape)     # 恢复原始形状（数据是归一化后的）

    def forward(
        self,
        global_image: torch.Tensor,
        wrist_image: torch.Tensor,
        point_cloud: torch.Tensor,
        agent_pos: torch.Tensor,
        initial_noise: torch.Tensor,
    ):
        """
        执行完整的 MP1 推理流程
        
        Args:
            global_image: 全局相机图像 [B, T, C, H, W]
            wrist_image: 腕部相机图像 [B, T, C, H, W]
            point_cloud: 点云数据 [B, T, N, 3/6]
            agent_pos: 机器人当前位姿 [B, T, D]
            initial_noise: 初始噪声 [B, horizon, action_dim]
            
        Returns:
            action: 下一步要执行的动作 [B, n_action_steps, action_dim]
            action_pred: 完整的预测动作序列 [B, horizon, action_dim]
        """
        # 1. 构造观测字典
        obs_dict = {
            "global_image": global_image,
            "wrist_image": wrist_image,
            "point_cloud": point_cloud,
            "agent_pos": agent_pos,
        }
        
        # 2. 观测归一化
        # 不直接调用 policy.normalizer.normalize，避免 trace 出固定 CPU device。
        nobs = {
            "global_image": self._normalize_field(obs_dict["global_image"], "global_image", True),
            "wrist_image": self._normalize_field(obs_dict["wrist_image"], "wrist_image", True),
            "point_cloud": self._normalize_field(obs_dict["point_cloud"], "point_cloud", True),
            "agent_pos": self._normalize_field(obs_dict["agent_pos"], "agent_pos", True),
        }
        
        # 3. 如果不使用点云颜色，只保留前3维（XYZ）
        if not self.policy.use_pc_color:
            nobs["point_cloud"] = nobs["point_cloud"][..., :3]

        # 4. 获取维度参数
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        horizon = int(self.policy.horizon)
        action_dim = int(self.policy.action_dim)
        dtype = self.policy.dtype

        # 5. 编码观测得到全局条件
        _, _, global_cond = self.policy._encode_obs(nobs, batch_size)
        
        # 6. 准备初始噪声（用于确定性推理）
        x_current = initial_noise.to(dtype=dtype)
        if x_current.shape[1] != horizon or x_current.shape[2] != action_dim:
            raise RuntimeError("initial_noise shape must be [B, horizon, action_dim]")

        # 7. 扩散采样循环（ODE Solver）
        # 使用显式 initial_noise，离线测试和 C++ 推理可以逐样本确定性对齐
        steps = self.policy.num_inference_steps if self.policy.num_inference_steps is not None else 10
        dt = 1.0 / float(steps)  # 时间步长
        # 从 x_current 派生，避免 torch.jit.trace 把 new_zeros 的导出设备写死。
        r_zeros = x_current[:, 0, 0] * 0.0  # MeanFlow 的参考时间 r；部署采样固定为 0，保证确定性和训练/推理语义一致。

        for i in range(int(steps)):
            # 当前时间步
            t_tensor = r_zeros + float(i) / float(steps)
            
            # 调用 U-Net 模型预测速度
            model_output = self.policy.model(
                sample=x_current,
                timestep=t_tensor,
                local_cond=None,
                global_cond=global_cond,
                r=r_zeros,
                training=False,
            )
            
            # 提取速度预测（模型可能返回元组）
            v_pred = model_output[0] if isinstance(model_output, tuple) else model_output
            
            # ODE 数值积分：x_{t+1} = x_t + v_pred * dt
            x_current = x_current + v_pred * dt

        # 8. 动作反归一化
        naction_pred = x_current[..., :action_dim]
        action_pred = self._normalize_field(naction_pred, "action", False)
        
        # 9. 提取当前时刻需要执行的动作片段
        start = global_image.shape[1] - 1  # 从最后一个观测时刻开始
        action = action_pred[:, start : start + int(self.policy.n_action_steps)]
        
        return action, action_pred


class ConstantTensorModule(nn.Module):
    """
    常量张量的 TorchScript 封装模块
    
    将单个 tensor 存储为 TorchScript 模块，解决 C++ 侧直接加载 Python torch.save 张量文件
    在部分版本组合下不稳定的问题。C++ 侧可以使用同一套 torch::jit::load API 读取。
    
    Args:
        value: 要存储的张量
    """

    def __init__(self, value: torch.Tensor):
        super().__init__()
        self.register_buffer("value", value)

    def forward(self):
        return self.value


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 包含以下参数的命名空间：
            --checkpoint: 模型检查点路径（默认: python_deploy/checkpoints/latest.ckpt）
            --config: 部署配置文件路径（默认: real_robot_ur12e_d405_speed_only/configs/...）
            --sample-npz: 可选的测试样本 npz 文件路径
            --output-dir: 输出目录（默认: deploy_artifacts）
            --device: 运行设备（默认: cpu）
            --seed: 随机种子（默认: 42）
    """
    parser = argparse.ArgumentParser(description="Export MP1 policy artifacts for C++ deployment.")
    parser.add_argument("--checkpoint", default="python_deploy/checkpoints/latest.ckpt")
    parser.add_argument("--config", default="python_deploy/real_robot_ur12e_d405_speed_only/configs/collect_pole_pickoff_laptop_remote.json")
    parser.add_argument("--sample-npz", default=None)
    parser.add_argument("--output-dir", default="deploy_artifacts")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def tensor_to_json(value: torch.Tensor) -> Any:
    """
    将 PyTorch 张量转换为 JSON 可序列化的 Python 对象
    
    Args:
        value: 输入张量
        
    Returns:
        Python 列表或标量值（可直接序列化为 JSON）
    """
    return value.detach().cpu().tolist()


def normalizer_to_json(policy: nn.Module) -> Dict[str, Any]:
    """
    将策略的归一化器参数转换为 JSON 格式
    
    提取 normalizer 中每个观测/动作类型的 scale、offset 和 input_stats 参数，
    用于 C++ 侧进行归一化/反归一化操作。
    
    Args:
        policy: MP1 策略模型
        
    Returns:
        包含归一化器参数的字典
    """
    result: Dict[str, Any] = {}
    for key, params in policy.normalizer.params_dict.items():
        result[key] = {
            "scale": tensor_to_json(params["scale"]),
            "offset": tensor_to_json(params["offset"]),
            "input_stats": {
                stat_key: tensor_to_json(stat_value)
                for stat_key, stat_value in params["input_stats"].items()
            },
        }
    return result


def load_optional_json(path: Optional[str]) -> Dict[str, Any]:
    """
    安全加载可选的 JSON 文件
    
    Args:
        path: JSON 文件路径（可为 None）
        
    Returns:
        JSON 内容字典，如果路径为空则返回空字典
    """
    if not path:
        return {}
    json_path = resolve_path(path)
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_path(path: str) -> Path:
    """
    解析路径，将相对路径转换为绝对路径
    
    Args:
        path: 输入路径（相对或绝对）
        
    Returns:
        绝对路径
    """
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def make_dummy_inputs(cfg: Any, policy: nn.Module, device: torch.device, seed: int):
    """
    生成 TorchScript 追踪所需的虚拟输入张量
    
    根据配置中的形状元信息生成符合要求的零张量和随机噪声张量，
    用于 torch.jit.trace 进行模型追踪。
    
    Args:
        cfg: 配置对象，包含形状元信息
        policy: MP1 策略模型
        device: 设备类型
        seed: 随机种子
        
    Returns:
        元组：(global_image, wrist_image, point_cloud, agent_pos, initial_noise)
    """
    obs_meta = dict(cfg.shape_meta.obs)
    n_obs_steps = int(cfg.n_obs_steps)
    tensors = []
    
    # 为每个观测类型生成零张量
    for key in ["global_image", "wrist_image", "point_cloud", "agent_pos"]:
        shape = list(obs_meta[key]["shape"])
        # RGB 图像使用 uint8，其他使用 float32
        dtype = torch.uint8 if obs_meta[key].get("type") == "rgb" else torch.float32
        tensors.append(torch.zeros((1, n_obs_steps, *shape), dtype=dtype, device=device))
    
    # 生成初始噪声张量（用于确定性推理）
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    tensors.append(
        torch.randn(
            (1, int(policy.horizon), int(policy.action_dim)),
            dtype=policy.dtype,
            device=device,
            generator=generator,
        )
    )
    
    return tuple(tensors)


def save_sample_tensors(sample_npz: str, tensor_dir: Path, wrapper: nn.Module, device: torch.device, policy: nn.Module, seed: int) -> None:
    """
    从测试样本 npz 文件生成用于离线测试的张量
    
    将观测数据、初始噪声和预期输出动作保存为 TorchScript 模块，
    用于 C++ 部署时的离线测试和推理结果对齐验证。
    
    Args:
        sample_npz: 测试样本 npz 文件路径
        tensor_dir: 张量输出目录
        wrapper: MP1TorchScriptWrapper 实例
        device: 设备类型
        policy: MP1 策略模型
        seed: 随机种子
    """
    # 加载 npz 数据
    data = np.load(resolve_path(sample_npz))
    tensor_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义 npz 键名到输出张量名的映射
    names = {
        "global_image": "model_obs_global_image",
        "wrist_image": "model_obs_wrist_image",
        "point_cloud": "model_obs_point_cloud",
        "agent_pos": "model_obs_agent_pos",
    }
    
    tensors: Dict[str, torch.Tensor] = {}
    
    # 将 npz 数据转换为 PyTorch 张量
    for out_name, npz_key in names.items():
        tensor = torch.from_numpy(data[npz_key]).unsqueeze(0).to(device)
        # 点云和 agent_pos 转换为 float，图像保持原类型
        tensors[out_name] = tensor.float() if out_name in {"point_cloud", "agent_pos"} else tensor

    # 生成确定性的初始噪声
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    tensors["initial_noise"] = torch.randn(
        (1, int(policy.horizon), int(policy.action_dim)),
        dtype=policy.dtype,
        device=device,
        generator=generator,
    )
    
    # 执行推理获取预期输出
    with torch.no_grad():
        action, action_pred = wrapper(
            tensors["global_image"],
            tensors["wrist_image"],
            tensors["point_cloud"],
            tensors["agent_pos"],
            tensors["initial_noise"],
        )
    
    # 保存推理结果
    tensors["expected_action"] = action.detach().cpu()
    tensors["expected_action_pred"] = action_pred.detach().cpu()
    
    # 如果 npz 中包含实际执行的动作，也一并保存
    if "action_executed" in data:
        tensors["python_action_executed"] = torch.from_numpy(data["action_executed"]).float()
     
    # 将所有张量保存为 TorchScript 常量模块
    # C++ torch::load 对 Python torch.save 的 tensor 文件在部分版本组合下不稳定；
    # 保存为 TorchScript 常量模块能让 C++ 使用同一套 jit loader
    for name, tensor in tensors.items():
        module = torch.jit.script(ConstantTensorModule(tensor.detach().cpu()))
        module.save(str(tensor_dir / f"{name}.pt"))


def build_meta(cfg: Any, policy: nn.Module, args: argparse.Namespace, deploy_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    构建部署元数据字典
    
    收集模型配置、任务配置、机器人配置等信息，生成完整的部署元数据，
    用于 C++ 侧了解模型输入输出规格、动作约束、工作空间范围等。
    
    Args:
        cfg: 模型配置对象
        policy: MP1 策略模型
        args: 命令行参数
        deploy_config: 部署配置字典
        
    Returns:
        包含所有部署元数据的字典
        action_mode
        n_obs_steps
        horizon
        runtime_action_shape
        action_dim
        image shape
        point_cloud_shape
        agent_pos_shape
        normalizer
        agent_pos_shape 
        torchscript_inputs
        torchscript_outputs

    """
    obs_meta = OmegaConf.to_container(cfg.shape_meta.obs, resolve=True)
    action_shape = list(OmegaConf.to_container(cfg.shape_meta.action.shape, resolve=True))
    task_cfg = deploy_config.get("task", {})
    robot_cfg = deploy_config.get("robot", {})
    n_obs_steps = int(cfg.n_obs_steps)
    horizon = int(cfg.horizon)
    configured_action_steps = int(cfg.n_action_steps)
    action_dim = int(action_shape[0])
    policy_action_start = n_obs_steps - 1
    # Python policy 从 To-1 开始切动作；当 horizon 不足时，实际返回步数会小于 n_action_steps。
    runtime_action_steps = max(0, min(configured_action_steps, horizon - policy_action_start))
    
    return {
        "format_version": 1,  # 元数据格式版本号
        "task": task_cfg.get("task_id", "pole_pickoff"),  # 任务 ID
        "action_mode": deploy_config.get("representation", {}).get("action_mode", "delta_tcp_pose_gripper"),
        "n_obs_steps": n_obs_steps,  # 观测时间步数
        "horizon": horizon,  # 预测视野长度
        "n_action_steps": configured_action_steps,  # 训练配置中的动作步数
        "policy_action_start": policy_action_start,  # action_pred 中开始执行的索引
        "runtime_action_steps": runtime_action_steps,  # TorchScript 实际返回的可执行动作步数
        "runtime_action_shape": [1, runtime_action_steps, action_dim],  # TorchScript action 输出形状
        "action_pred_shape": [1, horizon, action_dim],  # TorchScript action_pred 输出形状
        "action_dim": action_dim,  # 动作维度
        "point_cloud_shape": obs_meta["point_cloud"]["shape"],  # 点云形状
        "global_image_shape": obs_meta["global_image"]["shape"],  # 全局图像形状
        "wrist_image_shape": obs_meta["wrist_image"]["shape"],  # 腕部图像形状
        "agent_pos_shape": obs_meta["agent_pos"]["shape"],  # 机器人位姿形状
        "control_hz": float(deploy_config.get("collection", {}).get("sample_hz", 5.0)),  # 控制频率
        "max_translation_per_step_m": 0.015,  # 每步最大平移距离（米）
        "max_rotation_per_step_rad": 0.08,  # 每步最大旋转角度（弧度）
        "workspace_min": robot_cfg.get("workspace_min"),  # 工作空间最小值
        "workspace_max": robot_cfg.get("workspace_max"),  # 工作空间最大值
        "checkpoint": str(resolve_path(args.checkpoint)),  # 检查点路径
        "torchscript_inputs": ["global_image", "wrist_image", "point_cloud", "agent_pos", "initial_noise"],
        "torchscript_outputs": ["action", "action_pred"],
        "normalizer": normalizer_to_json(policy),  # 归一化器参数
    }


def main() -> None:
    """
    主函数：执行 MP1 策略导出流程
    
    完整流程：
    1. 解析命令行参数
    2. 加载预训练策略模型
    3. 创建 TorchScript 包装器
    4. 追踪并冻结模型
    5. 保存 TorchScript 模型
    6. 构建并保存部署元数据
    7. 可选：保存测试样本张量
    """
    # 1. 解析命令行参数
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # 2. 设置随机种子，确保确定性
    torch.manual_seed(int(args.seed))
    
    # 3. 加载预训练策略模型（使用 EMA 权重）
    cfg, policy = load_workspace_policy(resolve_path(args.checkpoint), device=device, use_ema=True)
    policy.eval()
    
    # 4. 创建 TorchScript 包装器
    wrapper = MP1TorchScriptWrapper(policy).eval().to(device)

    # 5. 追踪并冻结模型
    with torch.no_grad():
        # 生成虚拟输入用于追踪
        dummy_inputs = make_dummy_inputs(cfg, policy, device, args.seed)
        # 追踪模型
        traced = torch.jit.trace(wrapper, dummy_inputs, strict=False)
        # 冻结模型（优化并移除训练相关代码）
        traced = torch.jit.freeze(traced.eval())
        # 保存冻结后的模型
        traced.save(str(output_dir / "policy_infer.pt"))

    # 6. 加载部署配置并构建元数据
    deploy_config = load_optional_json(args.config)
    with (output_dir / "deploy_meta.json").open("w", encoding="utf-8") as file:
        json.dump(build_meta(cfg, policy, args, deploy_config), file, ensure_ascii=False, indent=2, sort_keys=True)

    # 7. 可选：保存测试样本张量用于离线验证
    if args.sample_npz:
        save_sample_tensors(args.sample_npz, output_dir / "sample_tensors", traced, device, policy, args.seed)

    # 输出完成信息
    print(f"exported {output_dir / 'policy_infer.pt'}")
    print(f"metadata {output_dir / 'deploy_meta.json'}")


if __name__ == "__main__":
    main()
