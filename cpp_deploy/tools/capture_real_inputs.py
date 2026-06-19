# -*- coding: utf-8 -*-
"""采集真实输入并写出给 mp1_real_input_dry_run 使用的 TorchScript 张量文件。

这个脚本只负责“感知与状态采集”，不做策略推理，也不向机器人发送控制命令。
输出目录会持续更新 current_frame.txt，文件内容指向最新完整帧目录：
  frames/000000/global_image.pt, wrist_image.pt, point_cloud.pt, agent_pos.pt, initial_noise.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional

import numpy as np
import torch


class TensorModule(torch.nn.Module):
    """把一个 Tensor 包成无输入 TorchScript Module，匹配 C++ 侧 torch::jit::load 读取方式。"""

    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("value", value)

    def forward(self) -> torch.Tensor:
        return self.value


def save_tensor_module_atomic(tensor: torch.Tensor, path: Path) -> None:
    """原子写出张量，降低 dry-run 读到半写入文件的概率。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    module = torch.jit.script(TensorModule(tensor.detach().cpu().contiguous()))
    module.save(str(tmp_path))
    os.replace(tmp_path, path)


def write_text_atomic(text: str, path: Path) -> None:
    """原子写出文本提交文件；dry-run 只在这个文件更新后读取一整帧输入。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_vec(text: str, expected: int, name: str) -> np.ndarray:
    values = [float(item) for item in text.split(",") if item.strip()]
    if len(values) != expected:
        raise ValueError(f"{name} expects {expected} comma-separated values, got {len(values)}")
    return np.asarray(values, dtype=np.float32)


def parse_transform(text: str) -> np.ndarray:
    values = parse_vec(text, 16, "--pointcloud-transform")
    return values.reshape(4, 4).astype(np.float32)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues 公式：UR TCP 的旋转向量 -> 3x3 旋转矩阵。"""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1.0e-12:
        return np.eye(3, dtype=np.float32)

    axis = rotvec.astype(np.float32) / theta
    x, y, z = axis
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float32,
    )
    eye = np.eye(3, dtype=np.float32)
    return eye + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def matrix_to_rot6d(matrix: np.ndarray, mode: str) -> np.ndarray:
    """把旋转矩阵转成 6D 表示；cols 与师兄 Python 部署代码保持一致。"""
    if mode == "rows":
        return matrix[:2, :].reshape(6).astype(np.float32)
    if mode == "cols":
        return matrix[:, :2].T.reshape(6).astype(np.float32)
    raise ValueError(f"Unsupported rot6d mode: {mode}")


def build_agent_pos(tcp_pose: np.ndarray, gripper_fraction: float, rot6d_mode: str) -> torch.Tensor:
    """构造模型需要的 agent_pos: tcp_xyz + rot6d + gripper = 10 维。"""
    if tcp_pose.shape != (6,):
        raise ValueError(f"tcp_pose must be [6], got {tcp_pose.shape}")
    rotation = rotvec_to_matrix(tcp_pose[3:6])
    rot6d = matrix_to_rot6d(rotation, rot6d_mode)
    gripper = np.asarray([gripper_fraction], dtype=np.float32)
    agent_pos = np.concatenate([tcp_pose[:3].astype(np.float32), rot6d, gripper], axis=0)
    return torch.from_numpy(agent_pos)


def resize_color_chw(color_bgr: np.ndarray, height: int, width: int, image_order: str) -> torch.Tensor:
    """RealSense 输出 HWC 图像；模型输入需要 CHW uint8。"""
    import cv2  # 延迟导入，方便缺依赖时给出明确报错

    resized = cv2.resize(color_bgr, (width, height), interpolation=cv2.INTER_AREA)
    if image_order == "rgb":
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    elif image_order != "bgr":
        raise ValueError(f"Unsupported image order: {image_order}")
    chw = np.ascontiguousarray(resized.transpose(2, 0, 1))
    return torch.from_numpy(chw).to(torch.uint8)


def apply_transform(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """把点云从相机坐标系转换到训练时使用的坐标系；默认 transform 是单位阵。"""
    if points_xyz.size == 0:
        return points_xyz.astype(np.float32)
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
    homogeneous = np.concatenate([points_xyz.astype(np.float32), ones], axis=1)
    transformed = homogeneous @ transform.T
    return transformed[:, :3].astype(np.float32)


def crop_points(points_xyz: np.ndarray, crop_min: np.ndarray, crop_max: np.ndarray) -> np.ndarray:
    mask = np.all(points_xyz >= crop_min[None, :], axis=1) & np.all(points_xyz <= crop_max[None, :], axis=1)
    return points_xyz[mask]


def farthest_point_sample(points_xyz: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    """简单 FPS 下采样；点数不足时用重复点补齐，保证输出恒定为 [num_points, 3]。"""
    if points_xyz.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    rng = np.random.default_rng(seed)
    if points_xyz.shape[0] > 8192:
        keep = rng.choice(points_xyz.shape[0], size=8192, replace=False)
        points_xyz = points_xyz[keep]

    if points_xyz.shape[0] <= num_points:
        pad_count = num_points - points_xyz.shape[0]
        if pad_count == 0:
            return points_xyz.astype(np.float32)
        pad_index = rng.choice(points_xyz.shape[0], size=pad_count, replace=True)
        return np.concatenate([points_xyz, points_xyz[pad_index]], axis=0).astype(np.float32)

    selected = np.empty((num_points,), dtype=np.int64)
    selected[0] = int(rng.integers(points_xyz.shape[0]))
    distances = np.full((points_xyz.shape[0],), np.inf, dtype=np.float32)
    for i in range(1, num_points):
        last = points_xyz[selected[i - 1]]
        dist = np.sum((points_xyz - last[None, :]) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        selected[i] = int(np.argmax(distances))
    return points_xyz[selected].astype(np.float32)


def read_gripper_fraction(args: argparse.Namespace) -> float:
    """读取夹爪开合比例；没有真实接口时可先用固定值做 dry-run。"""
    if args.gripper_file:
        text = Path(args.gripper_file).read_text(encoding="utf-8").strip()
        value = float(text)
    else:
        value = float(args.gripper_fraction)
    return float(np.clip(value, 0.0, 1.0))


def read_tcp_pose(args: argparse.Namespace, rtde_receive_client) -> np.ndarray:
    """读取 UR TCP 位姿；关闭 RTDE 时使用固定 TCP 方便先验证相机链路。"""
    if rtde_receive_client is not None:
        return np.asarray(rtde_receive_client.getActualTCPPose(), dtype=np.float32)
    return parse_vec(args.fixed_tcp, 6, "--fixed-tcp")


def camera_config_by_role(config: Dict, role: str) -> Optional[Dict]:
    """从师兄部署 JSON 中按 role 找相机配置。"""
    for camera_cfg in config.get("cameras", []):
        if not camera_cfg.get("enabled", True):
            continue
        if str(camera_cfg.get("role", "")).lower() == role:
            return camera_cfg
    return None


def resolve_pointcloud_camera(config: Dict, requested: str) -> str:
    """挂杆任务默认使用配置里的 primary_point_cloud_camera。"""
    if requested != "config":
        return requested
    primary = str(config.get("collection", {}).get("primary_point_cloud_camera", "")).lower()
    if "wrist" in primary:
        return "wrist"
    if "global" in primary:
        return "global"
    print("[WARN] config has no primary_point_cloud_camera, fallback to wrist", file=sys.stderr)
    return "wrist"


def apply_config_defaults(args: argparse.Namespace, config: Dict) -> None:
    """用部署 JSON 填充机器人 IP 和相机序列号；命令行显式传入时优先使用命令行。"""
    global_cfg = camera_config_by_role(config, "global")
    wrist_cfg = camera_config_by_role(config, "wrist")
    if not args.no_rtde and not args.robot_ip:
        args.robot_ip = str(config.get("robot", {}).get("ip", ""))
    if not args.global_serial and global_cfg is not None:
        args.global_serial = str(global_cfg.get("serial", ""))
    if not args.wrist_serial and wrist_cfg is not None:
        args.wrist_serial = str(wrist_cfg.get("serial", ""))
    args.pointcloud_camera = resolve_pointcloud_camera(config, args.pointcloud_camera)


def discover_realsense_serials() -> Iterable[str]:
    import pyrealsense2 as rs

    context = rs.context()
    for device in context.query_devices():
        yield device.get_info(rs.camera_info.serial_number)


class RealSensePair:
    """管理全局相机和腕部相机。"""

    def __init__(self, args: argparse.Namespace) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.pointcloud = rs.pointcloud()
        serials = list(discover_realsense_serials())
        global_serial = args.global_serial
        wrist_serial = args.wrist_serial
        if not global_serial or not wrist_serial:
            if len(serials) < 2:
                raise RuntimeError(
                    "Need two RealSense cameras, or pass --global-serial and --wrist-serial explicitly"
                )
            global_serial = global_serial or serials[0]
            wrist_serial = wrist_serial or serials[1]

        self.global_pipeline = self._start_pipeline(global_serial, args)
        self.wrist_pipeline = self._start_pipeline(wrist_serial, args)
        self.global_align = rs.align(rs.stream.color)
        self.wrist_align = rs.align(rs.stream.color)
        self.global_serial = global_serial
        self.wrist_serial = wrist_serial

    def _start_pipeline(self, serial: str, args: argparse.Namespace):
        config = self.rs.config()
        config.enable_device(serial)
        config.enable_stream(self.rs.stream.color, args.camera_width, args.camera_height, self.rs.format.bgr8, args.fps)
        config.enable_stream(self.rs.stream.depth, args.camera_width, args.camera_height, self.rs.format.z16, args.fps)
        pipeline = self.rs.pipeline()
        pipeline.start(config)
        return pipeline

    def warmup(self, frames: int) -> None:
        for _ in range(frames):
            self.global_pipeline.wait_for_frames()
            self.wrist_pipeline.wait_for_frames()

    def stop(self) -> None:
        self.global_pipeline.stop()
        self.wrist_pipeline.stop()

    def capture(self) -> Dict[str, Dict[str, object]]:
        return {
            "global": self._capture_one(self.global_pipeline, self.global_align),
            "wrist": self._capture_one(self.wrist_pipeline, self.wrist_align),
        }

    def _capture_one(self, pipeline, align) -> Dict[str, object]:
        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture color/depth frames from RealSense")

        color_bgr = np.asanyarray(color_frame.get_data())
        points = self.pointcloud.calculate(depth_frame)
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        return {"color_bgr": color_bgr, "points_xyz": vertices}


def make_initial_noise(args: argparse.Namespace) -> torch.Tensor:
    """优先复用导出的 initial_noise，避免每次随机导致动作不可复现。"""
    initial_noise_path = Path(args.initial_noise)
    if initial_noise_path.exists():
        module = torch.jit.load(str(initial_noise_path), map_location="cpu")
        tensor = module.forward()
        return tensor.to(dtype=torch.float32)
    print(f"[WARN] initial_noise not found: {initial_noise_path}, use zeros instead", file=sys.stderr)
    return torch.zeros((1, 4, 7), dtype=torch.float32)


def write_inputs(
    output_dir: Path,
    frame_id: int,
    global_buffer: Deque[torch.Tensor],
    wrist_buffer: Deque[torch.Tensor],
    point_buffer: Deque[torch.Tensor],
    agent_buffer: Deque[torch.Tensor],
    initial_noise: torch.Tensor,
) -> None:
    """把 2 帧观测堆叠成模型输入，并以整帧目录提交，避免 dry-run 读到混帧。"""
    global_image = torch.stack(list(global_buffer), dim=0).unsqueeze(0).to(torch.uint8)
    wrist_image = torch.stack(list(wrist_buffer), dim=0).unsqueeze(0).to(torch.uint8)
    point_cloud = torch.stack(list(point_buffer), dim=0).unsqueeze(0).to(torch.float32)
    agent_pos = torch.stack(list(agent_buffer), dim=0).unsqueeze(0).to(torch.float32)

    frame_name = f"{frame_id:06d}"
    frame_dir = output_dir / "frames" / frame_name
    save_tensor_module_atomic(global_image, frame_dir / "global_image.pt")
    save_tensor_module_atomic(wrist_image, frame_dir / "wrist_image.pt")
    save_tensor_module_atomic(point_cloud, frame_dir / "point_cloud.pt")
    save_tensor_module_atomic(agent_pos, frame_dir / "agent_pos.pt")
    save_tensor_module_atomic(initial_noise.to(torch.float32), frame_dir / "initial_noise.pt")

    # 最后提交 current_frame.txt；C++ dry-run 看到它更新后才读取该帧目录。
    write_text_atomic(f"frames/{frame_name}\n", output_dir / "current_frame.txt")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="采集 MP1 真实输入张量，供 mp1_real_input_dry_run 消费。")
    parser.add_argument("--output-dir", required=True, help="输出张量目录，例如 deploy_artifacts/real_input_tensors")
    parser.add_argument("--config", default="cpp_deploy/configs/pole_pickoff_real_robot.json", help="Jetson 部署配置 JSON")
    parser.add_argument("--initial-noise", default="deploy_artifacts/sample_tensors/initial_noise.pt", help="导出时固定的 initial_noise.pt")

    parser.add_argument("--global-serial", default="", help="全局 RealSense 序列号；不填时自动取第 1 个设备")
    parser.add_argument("--wrist-serial", default="", help="腕部 RealSense 序列号；不填时自动取第 2 个设备")
    parser.add_argument("--camera-width", type=int, default=640, help="RealSense 采集宽度")
    parser.add_argument("--camera-height", type=int, default=480, help="RealSense 采集高度")
    parser.add_argument("--fps", type=int, default=15, help="RealSense 采集帧率；挂杆/取杆配置默认 15Hz")
    parser.add_argument("--warmup-frames", type=int, default=20, help="相机预热帧数")
    parser.add_argument("--image-order", choices=["bgr", "rgb"], default="rgb", help="写入模型的图像通道顺序；挂杆任务默认与 Python 部署一致使用 RGB")

    parser.add_argument("--robot-ip", default="", help="可选覆盖 UR 机器人 IP；为空时读取 config.robot.ip，仍为空才使用 --fixed-tcp")
    parser.add_argument("--no-rtde", action="store_true", help="不连接 UR RTDE，强制使用 --fixed-tcp 验证相机链路")
    parser.add_argument("--fixed-tcp", default="0,0,0,0,0,0", help="无 RTDE 时使用的固定 TCP: x,y,z,rx,ry,rz")
    parser.add_argument("--gripper-fraction", type=float, default=1.0, help="固定夹爪开合比例，0=闭合，1=打开")
    parser.add_argument("--gripper-file", default="", help="可选：从文本文件读取夹爪比例")
    parser.add_argument("--rot6d-mode", choices=["rows", "cols"], default="cols", help="rot6d 展开方式；挂杆任务默认与 Python 部署一致使用前两列展开")

    parser.add_argument("--pointcloud-camera", choices=["config", "global", "wrist"], default="config", help="使用哪一路深度生成点云；默认读取配置里的 primary_point_cloud_camera")
    parser.add_argument("--pointcloud-transform", default="1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1", help="4x4 点云坐标变换，行优先 16 个数")
    parser.add_argument("--steps", type=int, default=0, help="采集步数；0 表示持续运行")
    parser.add_argument("--control-hz", type=float, default=5.0, help="写出频率")
    parser.add_argument("--seed", type=int, default=7, help="点云采样随机种子")
    return parser


def main() -> int:
    """
    主函数：采集真实输入并输出为 TorchScript 张量文件
    
    执行流程:
        1. 解析命令行参数，加载配置文件
        2. 初始化 RealSense 相机和 RTDE 客户端
        3. 预热相机，稳定图像输出
        4. 循环采集：
           - 拍摄全局/腕部图像并 resize
           - 采集深度图并转为点云，进行裁剪和下采样
           - 读取 TCP 位姿和夹爪状态
           - 将 2 帧观测堆叠成模型输入
           - 原子写入张量文件和帧目录
        5. 按控制频率同步输出，避免 C++ dry-run 读到混帧
        
    返回:
        int: 退出码（0 表示成功）
    """
    # 1. 解析命令行参数和配置文件
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    config = load_json(Path(args.config))
    apply_config_defaults(args, config)
    
    # 2. 解析点云配置参数
    point_cfg = config.get("point_cloud", {})
    crop_min = np.asarray(point_cfg.get("crop_min", [0.28, -0.2, 0.0]), dtype=np.float32)
    crop_max = np.asarray(point_cfg.get("crop_max", [0.62, 0.2, 0.35]), dtype=np.float32)
    num_points = int(point_cfg.get("num_points", 512))
    transform = parse_transform(args.pointcloud_transform)
    initial_noise = make_initial_noise(args)

    # 3. 初始化 RTDE 客户端（可选）；如果无机器人 IP 则跳过，方便只验证相机链路
    # 机器人状态读取是可选的；先支持固定 TCP，方便只接相机时也能验证图像/点云链路。
    rtde_receive_client = None
    if args.robot_ip:
        import rtde_receive

        rtde_receive_client = rtde_receive.RTDEReceiveInterface(args.robot_ip)

    # 4. 初始化相机对（全局 + 腕部）
    cameras = RealSensePair(args)
    print(f"robot ip: {args.robot_ip if args.robot_ip else '<fixed-tcp>'}")
    print(f"global camera serial: {cameras.global_serial}")
    print(f"wrist camera serial: {cameras.wrist_serial}")
    print(f"pointcloud camera: {args.pointcloud_camera}")
    print(f"output dir: {output_dir}")
    print("capture_real_inputs only writes tensors; it never sends robot commands.")
    # 检查点云变换矩阵是否为单位阵，若是则提醒用户可能需要配置外参
    if np.allclose(transform, np.eye(4, dtype=np.float32)):
        print(
            "[WARN] pointcloud transform is identity. 挂杆模型若训练时使用 base 坐标点云，"
            "必须通过 --pointcloud-transform 传入 camera->base 外参。",
            file=sys.stderr,
        )

    # 5. 初始化环形缓冲区，用于维护最近 2 帧观测（对应模型的 n_obs_steps=2）
    global_buffer: Deque[torch.Tensor] = deque(maxlen=2)
    wrist_buffer: Deque[torch.Tensor] = deque(maxlen=2)
    point_buffer: Deque[torch.Tensor] = deque(maxlen=2)
    agent_buffer: Deque[torch.Tensor] = deque(maxlen=2)

    period_s = 1.0 / float(args.control_hz)  # 控制周期（秒），由频率换算得到
    step = 0
    try:
        # 6. 相机预热，稳定输出
        cameras.warmup(args.warmup_frames)
        
        # 7. 主循环：按控制频率持续采集
        while args.steps == 0 or step < args.steps:
            begin = time.monotonic()
            
            # 8. 拍摄图像并获取点云
            frames = cameras.capture()

            # 9. 图像 resize 到模型输入尺寸（全局 480x640，腕部 480x640）
            # 注意：此处参数顺序为 (height, width)，但函数内部是 (width, height)，保持与原始代码一致
            global_image = resize_color_chw(frames["global"]["color_bgr"], 480, 640, args.image_order)
            wrist_image = resize_color_chw(frames["wrist"]["color_bgr"], 480, 640, args.image_order)

            # 10. 点云处理：坐标变换 → 裁剪 → FPS 下采样
            raw_points = frames[args.pointcloud_camera]["points_xyz"]
            points = apply_transform(raw_points, transform)
            points = crop_points(points, crop_min, crop_max)
            cropped_count = int(points.shape[0])
            points = farthest_point_sample(points, num_points, args.seed + step)
            point_cloud = torch.from_numpy(points)

            # 11. 读取机器人状态：TCP 位姿 + 夹爪开合度 → 构造 10 维 agent_pos
            tcp_pose = read_tcp_pose(args, rtde_receive_client)
            gripper_fraction = read_gripper_fraction(args)
            agent_pos = build_agent_pos(tcp_pose, gripper_fraction, args.rot6d_mode)

            # 12. 将单帧观测追加到缓冲区（自动维护最新 2 帧）
            global_buffer.append(global_image)
            wrist_buffer.append(wrist_image)
            point_buffer.append(point_cloud)
            agent_buffer.append(agent_pos)

            # 13. 当缓冲区满 2 帧时，堆叠成模型输入并写入
            if len(global_buffer) == 2:
                write_inputs(output_dir, step, global_buffer, wrist_buffer, point_buffer, agent_buffer, initial_noise)
                print(
                    f"step={step} wrote frame=frames/{step:06d}, cropped_points={cropped_count}, "
                    f"tcp_xyz={tcp_pose[:3].round(4).tolist()}, gripper={gripper_fraction:.3f}",
                    flush=True,
                )
                step += 1

            # 14. 控制频率同步：确保每帧间隔固定，避免 C++ dry-run 读到混帧
            elapsed = time.monotonic() - begin
            sleep_s = max(0.0, period_s - elapsed)
            time.sleep(sleep_s)
    finally:
        # 15. 确保相机资源被正确释放
        cameras.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
