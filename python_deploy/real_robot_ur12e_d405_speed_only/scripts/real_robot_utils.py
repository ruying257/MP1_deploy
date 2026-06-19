import json
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - 运行时依赖
    rs = None

try:
    import zarr
except ImportError:  # pragma: no cover - 运行时依赖
    zarr = None

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - 运行时依赖
    imageio = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - 运行时依赖
    Image = None

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
except ImportError:  # pragma: no cover - 运行时依赖
    RTDEControlInterface = None
    RTDEReceiveInterface = None

try:
    from robotiq_gripper import RobotiqGripper
except ImportError:  # pragma: no cover - 运行时依赖
    RobotiqGripper = None

try:
    import Jetson.GPIO as JetsonGPIO
except ImportError:  # pragma: no cover - 运行时依赖
    JetsonGPIO = None


OBS_MODE_ALIASES = {
    "tcp_xyz_rot6d": "tcp_xyz_rot6d",
    "tcp_xyz_rot6d_9d": "tcp_xyz_rot6d",
    "tcp_pose": "tcp_pose",
    "tcp_pose_6d": "tcp_pose",
    "tcp_xyz_fingertips": "tcp_xyz_fingertips",
    "tcp_xyz_fingertips_9d": "tcp_xyz_fingertips",
}

ACTION_MODE_ALIASES = {
    "delta_tcp_pose": "delta_tcp_pose",
    "delta_tcp_pose_6d": "delta_tcp_pose",
    "delta_tcp_xyz_gripper": "delta_tcp_xyz_gripper",
    "delta_tcp_xyz_gripper_4d": "delta_tcp_xyz_gripper",
    "delta_tcp_pose_gripper": "delta_tcp_pose_gripper",
    "delta_tcp_pose_gripper_7d": "delta_tcp_pose_gripper",
}


def _workspace_bound_array(values: Optional[Iterable[Any]], default: float) -> np.ndarray:
    if values is None:
        return np.full((3,), float(default), dtype=np.float64)
    values_list = list(values)
    if len(values_list) != 3:
        raise ValueError(f"workspace bound must have 3 elements, got {values_list}")
    result = []
    for value in values_list:
        if value is None:
            result.append(float(default))
        else:
            result.append(float(value))
    return np.asarray(result, dtype=np.float64)


def get_workspace_bounds(robot_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    workspace_min = _workspace_bound_array(robot_cfg.get("workspace_min"), default=-np.inf)
    workspace_max = _workspace_bound_array(robot_cfg.get("workspace_max"), default=np.inf)
    return workspace_min, workspace_max


def apply_workspace_bounds(xyz: np.ndarray, robot_cfg: Dict[str, Any]) -> np.ndarray:
    workspace_min, workspace_max = get_workspace_bounds(robot_cfg)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
    return np.clip(xyz, workspace_min, workspace_max)


class TwoPinGPIOGripper:
    _gpio_mode_name = None

    def __init__(self, gripper_cfg: Dict[str, Any], default_fraction: float):
        self.cfg = gripper_cfg
        self.open_pin = int(gripper_cfg["open_pin"])
        self.close_pin = int(gripper_cfg["close_pin"])
        self.limit_open_pin = int(gripper_cfg.get("limit_open_pin", -1))
        self.limit_close_pin = int(gripper_cfg.get("limit_close_pin", -1))
        self.gpio_mode_name = str(gripper_cfg.get("gpio_mode", "BOARD")).upper()
        self.active_high = bool(gripper_cfg.get("active_high", True))
        self.limit_trigger_low = bool(gripper_cfg.get("limit_trigger_low", True))
        self.stop_after_command = bool(gripper_cfg.get("stop_after_command", True))
        self.pulse_duration_s = max(float(gripper_cfg.get("pulse_duration_s", 2.0)), 0.0)
        self.open_threshold = float(np.clip(gripper_cfg.get("open_threshold", 0.5), 0.0, 1.0))
        self.fraction = float(np.clip(default_fraction, 0.0, 1.0))
        self.stop_deadline = None
        self.connected = False

    def _gpio_mode(self):
        if self.gpio_mode_name == "BCM":
            return JetsonGPIO.BCM
        return JetsonGPIO.BOARD

    def _active_level(self):
        return JetsonGPIO.HIGH if self.active_high else JetsonGPIO.LOW

    def _inactive_level(self):
        return JetsonGPIO.LOW if self.active_high else JetsonGPIO.HIGH

    def connect(self) -> None:
        if JetsonGPIO is None:
            raise ImportError("需要安装 Jetson.GPIO 才能使用 twopin_gpio 夹爪")
        JetsonGPIO.setwarnings(False)
        if TwoPinGPIOGripper._gpio_mode_name is None:
            JetsonGPIO.setmode(self._gpio_mode())
            TwoPinGPIOGripper._gpio_mode_name = self.gpio_mode_name
        elif TwoPinGPIOGripper._gpio_mode_name != self.gpio_mode_name:
            raise ValueError("twopin_gpio 夹爪的 GPIO 模式不一致")

        JetsonGPIO.setup(self.open_pin, JetsonGPIO.OUT, initial=self._inactive_level())
        JetsonGPIO.setup(self.close_pin, JetsonGPIO.OUT, initial=self._inactive_level())
        if self.limit_open_pin >= 0:
            JetsonGPIO.setup(self.limit_open_pin, JetsonGPIO.IN)
        if self.limit_close_pin >= 0:
            JetsonGPIO.setup(self.limit_close_pin, JetsonGPIO.IN)
        self.connected = True
        self.set_fraction(self.fraction, wait=True)

    def _set_pin(self, pin: int, active: bool) -> None:
        if not self.connected or pin < 0:
            return
        JetsonGPIO.output(pin, self._active_level() if active else self._inactive_level())

    def _read_limit(self, pin: int) -> Optional[bool]:
        if not self.connected or pin < 0:
            return None
        raw_value = JetsonGPIO.input(pin)
        trigger_level = JetsonGPIO.LOW if self.limit_trigger_low else JetsonGPIO.HIGH
        return bool(raw_value == trigger_level)

    def update(self) -> None:
        if self.stop_deadline is not None and time.time() >= self.stop_deadline:
            self.stop()
            self.stop_deadline = None
        limit_open = self._read_limit(self.limit_open_pin)
        limit_close = self._read_limit(self.limit_close_pin)
        if limit_open is True:
            self.fraction = 1.0
        elif limit_close is True:
            self.fraction = 0.0

    def _arm_stop(self, wait: bool) -> None:
        if not self.stop_after_command or self.pulse_duration_s <= 0.0:
            return
        if wait:
            time.sleep(self.pulse_duration_s)
            self.stop()
            self.stop_deadline = None
            return
        self.stop_deadline = time.time() + self.pulse_duration_s

    def open(self, wait: bool = False) -> None:
        self._set_pin(self.close_pin, False)
        self._set_pin(self.open_pin, True)
        self.fraction = 1.0
        self._arm_stop(wait=wait)

    def close_gripper(self, wait: bool = False) -> None:
        self._set_pin(self.open_pin, False)
        self._set_pin(self.close_pin, True)
        self.fraction = 0.0
        self._arm_stop(wait=wait)

    def stop(self) -> None:
        self._set_pin(self.open_pin, False)
        self._set_pin(self.close_pin, False)

    def set_fraction(self, fraction: float, wait: bool = False) -> None:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        if fraction >= self.open_threshold:
            self.open(wait=wait)
        else:
            self.close_gripper(wait=wait)

    def get_fraction(self) -> float:
        self.update()
        return self.fraction

    def disconnect(self) -> None:
        if not self.connected:
            return
        self.stop()
        for pin in [self.open_pin, self.close_pin, self.limit_open_pin, self.limit_close_pin]:
            if pin < 0:
                continue
            try:
                JetsonGPIO.cleanup(pin)
            except Exception:
                pass
        self.connected = False


class RemoteTwoPinGripper:
    def __init__(self, gripper_cfg: Dict[str, Any], default_fraction: float):
        self.host = str(gripper_cfg["host"])
        self.port = int(gripper_cfg.get("port", 8765))
        self.connect_timeout_s = float(gripper_cfg.get("connect_timeout_s", 2.0))
        self.request_timeout_s = float(gripper_cfg.get("request_timeout_s", 2.0))
        self.sync_interval_s = max(float(gripper_cfg.get("sync_interval_s", 0.5)), 0.0)
        self.apply_default_on_connect = bool(gripper_cfg.get("apply_default_on_connect", True))
        self.stop_on_disconnect = bool(gripper_cfg.get("stop_on_disconnect", False))
        self.fraction = float(np.clip(default_fraction, 0.0, 1.0))
        self.last_sync_time = 0.0

    def _send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        message = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            with socket.create_connection((self.host, self.port), timeout=self.connect_timeout_s) as sock:
                sock.settimeout(self.request_timeout_s)
                sock.sendall(message)
                chunks = []
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"\n" in chunk:
                        break
        except socket.timeout as exc:
            raise RuntimeError(
                f"远程夹爪代理 {self.host}:{self.port} 请求超时。"
                f" connect_timeout_s={self.connect_timeout_s}, request_timeout_s={self.request_timeout_s}。"
                " 请检查 Jetson 代理是否已启动、IP/端口是否正确，以及客户端超时是否短于服务端脉冲时长。"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"无法连接远程夹爪代理 {self.host}:{self.port}: {exc}") from exc
        raw = b"".join(chunks).split(b"\n", 1)[0].decode("utf-8").strip()
        if not raw:
            raise RuntimeError("远程夹爪代理没有返回有效响应")
        response = json.loads(raw)
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error", "远程夹爪代理返回失败")))
        return response

    def _sync_state(self, force: bool = False) -> float:
        now = time.time()
        if not force and (now - self.last_sync_time) < self.sync_interval_s:
            return self.fraction
        response = self._send_request({"command": "state"})
        self.fraction = float(np.clip(response.get("fraction", self.fraction), 0.0, 1.0))
        self.last_sync_time = now
        return self.fraction

    def connect(self) -> None:
        if self.apply_default_on_connect:
            # Startup should not block on the full GPIO pulse width.
            self.set_fraction(self.fraction, wait=False)
            return
        self._sync_state(force=True)

    def set_fraction(self, fraction: float, wait: bool = False) -> None:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        response = self._send_request(
            {
                "command": "set_fraction",
                "fraction": fraction,
                "wait": bool(wait),
            }
        )
        self.fraction = float(np.clip(response.get("fraction", fraction), 0.0, 1.0))
        self.last_sync_time = time.time()

    def get_fraction(self) -> float:
        try:
            return self._sync_state(force=False)
        except Exception:
            return self.fraction

    def disconnect(self) -> None:
        if self.stop_on_disconnect:
            try:
                self._send_request({"command": "stop"})
            except Exception:
                pass


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True, ensure_ascii=False)


def resolve_local_path(base_dir: Path, path_str: str) -> str:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def resolve_collection_paths(config: Dict[str, Any], config_path: str) -> Dict[str, Any]:
    config_dir = Path(config_path).resolve().parent
    dataset_cfg = config.setdefault("dataset", {})
    for key in ["zarr_path", "raw_root"]:
        value = dataset_cfg.get(key)
        if value:
            dataset_cfg[key] = resolve_local_path(config_dir, value)
    return config


def make_timestamp_string() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def ensure_array(
    value: Iterable[float],
    shape: Optional[Tuple[int, ...]] = None,
    dtype=np.float32,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ValueError(f"期望形状为 {shape}，实际得到 {array.shape}")
    return array


def clip_translation(delta_xyz: np.ndarray, max_norm: float) -> np.ndarray:
    delta_norm = float(np.linalg.norm(delta_xyz))
    if max_norm <= 0 or delta_norm <= max_norm:
        return delta_xyz
    return delta_xyz * (max_norm / max(delta_norm, 1.0e-8))


def clip_rotation(delta_rotvec: np.ndarray, max_norm: float) -> np.ndarray:
    delta_norm = float(np.linalg.norm(delta_rotvec))
    if max_norm <= 0 or delta_norm <= max_norm:
        return delta_rotvec
    return delta_rotvec * (max_norm / max(delta_norm, 1.0e-8))


def rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    cos_angle = max(min((trace - 1.0) * 0.5, 1.0), -1.0)
    angle = math.acos(cos_angle)
    if angle < 1.0e-9:
        return np.zeros(3, dtype=np.float64)
    skew = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis = skew / max(2.0 * math.sin(angle), 1.0e-8)
    return axis * angle


def align_rotvec_to_reference(rotvec: np.ndarray, reference_rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    reference_rotvec = np.asarray(reference_rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1.0e-9:
        return rotvec.copy()
    axis = rotvec / angle
    best = rotvec.copy()
    best_distance = float(np.linalg.norm(best - reference_rotvec))
    for k in range(-2, 3):
        candidate = rotvec + (2.0 * math.pi * k) * axis
        distance = float(np.linalg.norm(candidate - reference_rotvec))
        if distance < best_distance:
            best = candidate
            best_distance = distance
    return best


def rotvec_to_rotation_matrix(rotvec: Iterable[float]) -> np.ndarray:
    rx, ry, rz = np.asarray(list(rotvec), dtype=np.float64)
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    kx, ky, kz = rx / theta, ry / theta, rz / theta
    skew = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=np.float64,
    )
    identity = np.eye(3, dtype=np.float64)
    return identity + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def rotation_matrix_to_rot6d(rotation: np.ndarray) -> np.ndarray:
    return rotation[:, :2].T.reshape(-1).astype(np.float64)


def pose_vector_to_matrix(pose: Iterable[float]) -> np.ndarray:
    pose = np.asarray(list(pose), dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotvec_to_rotation_matrix(pose[3:6])
    matrix[:3, 3] = pose[:3]
    return matrix


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.astype(np.float32)
    homog = np.concatenate(
        [points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed = (transform @ homog.T).T[:, :3]
    return transformed.astype(np.float32)


def save_png_image(path: Path, bgr_image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_image = np.asarray(bgr_image, dtype=np.uint8)[..., ::-1]
    if imageio is not None:
        imageio.imwrite(path, rgb_image)
        return
    if Image is not None:
        Image.fromarray(rgb_image).save(path)
        return
    raise ImportError("保存 PNG 需要安装 imageio 或 Pillow")


def save_point_cloud_ply(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    with open(path, "w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        for point in points:
            file.write(f"{float(point[0]):.6f} {float(point[1]):.6f} {float(point[2]):.6f}\n")


def save_array_csv(path: Path, array: np.ndarray, header: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    np.savetxt(path, array, delimiter=",", header=header or "", comments="")


def chunk_shape_for_array(array: np.ndarray, time_chunk: int = 64) -> Tuple[int, ...]:
    first_dim = max(1, min(int(array.shape[0]), int(time_chunk)))
    return (first_dim,) + tuple(array.shape[1:])


def sanitize_camera_key(name: str) -> str:
    safe = []
    for char in str(name):
        if char.isalnum():
            safe.append(char.lower())
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def safe_frame_timestamp_ms(frame) -> float:
    if frame is None:
        return float("nan")
    try:
        return float(frame.get_timestamp())
    except Exception:
        return float("nan")


def safe_frame_number(frame) -> int:
    if frame is None:
        return -1
    try:
        return int(frame.get_frame_number())
    except Exception:
        return -1


def safe_frame_timestamp_domain(frame) -> str:
    if frame is None:
        return "missing"
    try:
        return str(frame.get_frame_timestamp_domain())
    except Exception:
        return "unknown"


def crop_points(points: np.ndarray, crop_min=None, crop_max=None) -> np.ndarray:
    if points.size == 0:
        return points.astype(np.float32)
    cropped = points
    if crop_min is not None:
        crop_min = ensure_array(crop_min, shape=(3,), dtype=np.float32)
        cropped = cropped[np.all(cropped[:, :3] >= crop_min[None, :], axis=1)]
    if crop_max is not None:
        crop_max = ensure_array(crop_max, shape=(3,), dtype=np.float32)
        cropped = cropped[np.all(cropped[:, :3] <= crop_max[None, :], axis=1)]
    return cropped.astype(np.float32)


def random_downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    indices = np.random.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def farthest_point_sampling(points: np.ndarray, num_points: int) -> np.ndarray:
    if num_points <= 0:
        raise ValueError("num_points 必须为正数")
    if points.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if points.shape[0] <= num_points:
        pad = np.zeros((num_points - points.shape[0], points.shape[1]), dtype=points.dtype)
        return np.concatenate([points, pad], axis=0)

    xyz = points[:, :3].astype(np.float32)
    chosen = np.zeros((num_points,), dtype=np.int64)
    distances = np.full((xyz.shape[0],), np.inf, dtype=np.float32)
    farthest = np.random.randint(0, xyz.shape[0])

    for i in range(num_points):
        chosen[i] = farthest
        centroid = xyz[farthest]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        distances = np.minimum(distances, dist)
        farthest = int(np.argmax(distances))

    return points[chosen]


def sample_point_cloud(points: np.ndarray, num_points: int, method: str = "fps") -> np.ndarray:
    if method == "fps":
        return farthest_point_sampling(points, num_points)
    if method == "random":
        points = random_downsample(points, num_points)
        if points.shape[0] < num_points:
            pad = np.zeros((num_points - points.shape[0], points.shape[1]), dtype=points.dtype)
            points = np.concatenate([points, pad], axis=0)
        return points
    raise ValueError(f"不支持的点云采样方式: {method}")


def normalize_gripper_target(raw_value: float) -> float:
    return float(np.clip(raw_value, -1.0, 1.0))


def gripper_target_to_fraction(raw_value: float) -> float:
    return 0.5 * (normalize_gripper_target(raw_value) + 1.0)


def fraction_to_gripper_target(fraction: float) -> float:
    return float(np.clip(fraction, 0.0, 1.0) * 2.0 - 1.0)


def snapshot_gripper_fraction(snapshot: Dict[str, np.ndarray], prefer_target: bool = False) -> float:
    keys = ["gripper_target_fraction", "gripper_fraction"] if prefer_target else ["gripper_fraction", "gripper_target_fraction"]
    for key in keys:
        value = snapshot.get(key)
        if value is None:
            continue
        try:
            return float(np.clip(np.asarray(value, dtype=np.float32).reshape(-1)[0], 0.0, 1.0))
        except Exception:
            continue
    return 0.0


def get_representation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    rep_cfg = dict(config.get("representation", {}))
    rep_cfg.setdefault("obs_mode", "tcp_xyz_rot6d")
    rep_cfg.setdefault("action_mode", "delta_tcp_pose")
    return rep_cfg


def get_obs_mode(config: Dict[str, Any]) -> str:
    raw_mode = str(get_representation_config(config).get("obs_mode", "tcp_xyz_rot6d"))
    if raw_mode not in OBS_MODE_ALIASES:
        raise ValueError(f"不支持的观测表示方式: {raw_mode}")
    return OBS_MODE_ALIASES[raw_mode]


def get_action_mode(config: Dict[str, Any]) -> str:
    raw_mode = str(get_representation_config(config).get("action_mode", "delta_tcp_pose"))
    if raw_mode not in ACTION_MODE_ALIASES:
        raise ValueError(f"不支持的动作表示方式: {raw_mode}")
    return ACTION_MODE_ALIASES[raw_mode]


def infer_state_dim(config: Dict[str, Any]) -> int:
    obs_mode = get_obs_mode(config)
    if obs_mode == "tcp_xyz_rot6d":
        return 9
    if obs_mode == "tcp_pose":
        return 6
    if obs_mode == "tcp_xyz_fingertips":
        return 9
    raise ValueError(f"无法推断观测维度: {obs_mode}")


def infer_action_dim(config: Dict[str, Any]) -> int:
    action_mode = get_action_mode(config)
    if action_mode == "delta_tcp_pose":
        return 6
    if action_mode == "delta_tcp_xyz_gripper":
        return 4
    if action_mode == "delta_tcp_pose_gripper":
        return 7
    raise ValueError(f"无法推断动作维度: {action_mode}")


def build_fingertip_state_vector(
    tcp_pose: np.ndarray,
    gripper_fraction: float,
    kinematics_cfg: Dict[str, Any],
) -> np.ndarray:
    tcp_transform = pose_vector_to_matrix(tcp_pose)
    fingertip_center = ensure_array(
        kinematics_cfg["finger_tip_center_in_tcp_m"],
        shape=(3,),
        dtype=np.float64,
    )
    open_axis = ensure_array(
        kinematics_cfg["finger_open_axis_in_tcp"],
        shape=(3,),
        dtype=np.float64,
    )
    open_axis = open_axis / max(np.linalg.norm(open_axis), 1.0e-8)
    max_width = float(kinematics_cfg["max_gripper_width_m"])
    opening = max_width * float(np.clip(gripper_fraction, 0.0, 1.0))
    right_local = fingertip_center - 0.5 * opening * open_axis
    left_local = fingertip_center + 0.5 * opening * open_axis
    right_world = apply_transform(right_local[None, :].astype(np.float32), tcp_transform)[0]
    left_world = apply_transform(left_local[None, :].astype(np.float32), tcp_transform)[0]
    return np.concatenate([tcp_pose[:3], right_world, left_world], axis=0).astype(np.float32)


def build_state_vector(
    tcp_pose: np.ndarray,
    gripper_fraction: float,
    config: Dict[str, Any],
) -> np.ndarray:
    obs_mode = get_obs_mode(config)
    if obs_mode == "tcp_xyz_rot6d":
        rotation = rotvec_to_rotation_matrix(tcp_pose[3:6])
        rot6d = rotation_matrix_to_rot6d(rotation)
        return np.concatenate([tcp_pose[:3], rot6d], axis=0).astype(np.float32)
    if obs_mode == "tcp_pose":
        return tcp_pose.astype(np.float32)
    if obs_mode == "tcp_xyz_fingertips":
        if "kinematics" not in config:
            raise KeyError("当前观测表示需要 kinematics 配置")
        return build_fingertip_state_vector(tcp_pose, gripper_fraction, config["kinematics"])
    raise ValueError(f"不支持的观测表示方式: {obs_mode}")


def deproject_depth_image(
    depth_m: np.ndarray,
    intrinsics: Any,
    depth_min_m: float,
    depth_max_m: float,
    stride: int = 1,
) -> np.ndarray:
    ys = np.arange(0, depth_m.shape[0], stride, dtype=np.float32)
    xs = np.arange(0, depth_m.shape[1], stride, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    depth = depth_m[::stride, ::stride]
    valid = np.isfinite(depth)
    valid &= depth > depth_min_m
    valid &= depth < depth_max_m
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)
    z = depth[valid]
    x = (grid_x[valid] - intrinsics.ppx) / intrinsics.fx * z
    y = (grid_y[valid] - intrinsics.ppy) / intrinsics.fy * z
    return np.stack([x, y, z], axis=1).astype(np.float32)


def format_action_summary(action: np.ndarray, action_mode: str) -> str:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_mode == "delta_tcp_pose":
        delta_xyz = np.round(action[:3], 5).tolist()
        delta_xyz_mm = np.round(action[:3] * 1000.0, 2).tolist()
        delta_rot = np.round(action[3:6], 5).tolist()
        return f"平移增量(m)={delta_xyz} | mm={delta_xyz_mm}，旋转增量(rad)={delta_rot}"
    if action_mode == "delta_tcp_xyz_gripper":
        delta_xyz = np.round(action[:3], 5).tolist()
        delta_xyz_mm = np.round(action[:3] * 1000.0, 2).tolist()
        return f"平移增量(m)={delta_xyz} | mm={delta_xyz_mm}，夹爪目标={float(action[3]):.3f}"
    if action_mode == "delta_tcp_pose_gripper":
        delta_xyz = np.round(action[:3], 5).tolist()
        delta_xyz_mm = np.round(action[:3] * 1000.0, 2).tolist()
        delta_rot = np.round(action[3:6], 5).tolist()
        return f"平移增量(m)={delta_xyz} | mm={delta_xyz_mm}，旋转增量(rad)={delta_rot}，夹爪目标={float(action[6]):.3f}"
    return f"动作={np.round(action, 5).tolist()}"


def resolve_default_tcp_pose(robot_cfg: Dict[str, Any]) -> np.ndarray:
    if "home_tcp_pose" not in robot_cfg:
        raise ValueError("采集配置必须提供 robot.home_tcp_pose，当前不再使用额外的默认姿态参考")
    return ensure_array(robot_cfg["home_tcp_pose"], shape=(6,), dtype=np.float64)


class KeyPoller:
    def __enter__(self) -> "KeyPoller":
        self._is_windows = os.name == "nt"
        self._file = None
        self._old_settings = None
        if not self._is_windows:
            import select
            import termios
            import tty

            self._select = select
            self._file = sys.stdin
            self._old_settings = termios.tcgetattr(self._file)
            tty.setcbreak(self._file.fileno())
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._is_windows and self._old_settings is not None:
            import termios

            termios.tcsetattr(self._file, termios.TCSADRAIN, self._old_settings)

    def poll(self) -> Optional[str]:
        if self._is_windows:
            import msvcrt

            if msvcrt.kbhit():
                return msvcrt.getwch()
            return None
        ready, _, _ = self._select.select([self._file], [], [], 0)
        if ready:
            return self._file.read(1)
        return None


class MultiRealSenseManager:
    def __init__(self, config: Dict[str, Any]):
        self.camera_cfgs = [cfg for cfg in config.get("cameras", []) if cfg.get("enabled", True)]
        self.point_cloud_cfg = config["point_cloud"]
        self.capture_cfg = config.get("collection", {})
        self._cameras: List[Dict[str, Any]] = []

    def start(self) -> None:
        if rs is None:
            raise ImportError("需要安装 pyrealsense2 才能采集 RealSense 数据")
        self._cameras = []
        for cam_cfg in self.camera_cfgs:
            pipeline = rs.pipeline()
            rs_config = rs.config()
            rs_config.enable_device(cam_cfg["serial"])
            rs_config.enable_stream(
                rs.stream.depth,
                int(cam_cfg["depth_width"]),
                int(cam_cfg["depth_height"]),
                rs.format.z16,
                int(cam_cfg["fps"]),
            )
            enable_color = bool(cam_cfg.get("enable_color", True))
            if enable_color:
                rs_config.enable_stream(
                    rs.stream.color,
                    int(cam_cfg.get("color_width", cam_cfg["depth_width"])),
                    int(cam_cfg.get("color_height", cam_cfg["depth_height"])),
                    rs.format.bgr8,
                    int(cam_cfg["fps"]),
                )
            try:
                profile = pipeline.start(rs_config)
            except Exception as exc:
                camera_name = str(cam_cfg.get("name", cam_cfg.get("serial", "unknown_camera")))
                depth_spec = (
                    f"{int(cam_cfg['depth_width'])}x{int(cam_cfg['depth_height'])}@{int(cam_cfg['fps'])}"
                )
                color_spec = (
                    f"{int(cam_cfg.get('color_width', cam_cfg['depth_width']))}"
                    f"x{int(cam_cfg.get('color_height', cam_cfg['depth_height']))}"
                    f"@{int(cam_cfg['fps'])}"
                    if enable_color
                    else "disabled"
                )
                self.stop()
                raise RuntimeError(
                    f"RealSense 相机启动失败: name={camera_name}, serial={cam_cfg['serial']}, "
                    f"depth={depth_spec}, color={color_spec}. "
                    "常见原因是当前分辨率/FPS 组合不被这台相机支持。"
                    " D405 在当前采集链里建议优先尝试 640x480@15 或 640x480@30，不建议 640x480@5。"
                ) from exc
            depth_sensor = profile.get_device().first_depth_sensor()
            if "laser_power" in cam_cfg:
                try:
                    depth_sensor.set_option(rs.option.laser_power, float(cam_cfg["laser_power"]))
                except Exception:
                    pass
            if "visual_preset" in cam_cfg:
                try:
                    depth_sensor.set_option(rs.option.visual_preset, float(cam_cfg["visual_preset"]))
                except Exception:
                    pass
            align = rs.align(rs.stream.color) if enable_color else None
            intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            self._cameras.append(
                {
                    "cfg": cam_cfg,
                    "pipeline": pipeline,
                    "align": align,
                    "depth_scale": float(depth_sensor.get_depth_scale()),
                    "intrinsics": intrinsics,
                    "base_T_camera": (
                        np.asarray(cam_cfg["base_T_camera"], dtype=np.float32)
                        if "base_T_camera" in cam_cfg
                        else None
                    ),
                }
            )
        warmup_frames = max([int(cfg.get("warmup_frames", 15)) for cfg in self.camera_cfgs] or [0])
        for _ in range(warmup_frames):
            for camera in self._cameras:
                camera["pipeline"].wait_for_frames()

    def stop(self) -> None:
        for camera in self._cameras:
            try:
                camera["pipeline"].stop()
            except Exception:
                pass
        self._cameras = []

    def capture(self) -> Dict[str, Any]:
        if not self._cameras:
            raise RuntimeError("RealSense 管理器尚未启动")
        debug_image = None
        debug_depth = None
        per_camera = {}
        raw_cameras = {}
        primary_point_cloud = None
        primary_camera_name = self.capture_cfg.get("primary_point_cloud_camera")

        point_cloud_mode = str(self.point_cloud_cfg.get("aggregation", "fused")).lower()
        fused_points: List[np.ndarray] = []

        for camera in self._cameras:
            camera_cfg = camera["cfg"]
            frames = camera["pipeline"].wait_for_frames(timeout_ms=5000)
            frame_host_timestamp = time.time()
            if camera["align"] is not None:
                frames = camera["align"].process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if depth_frame is None:
                raise RuntimeError(f"相机 {camera['cfg']['serial']} 缺少深度帧")
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * camera["depth_scale"]
            color = None
            if color_frame is not None:
                color = np.asanyarray(color_frame.get_data()).astype(np.uint8)
            points_camera = deproject_depth_image(
                depth_m=depth_m,
                intrinsics=camera["intrinsics"],
                depth_min_m=float(camera_cfg.get("depth_min_m", 0.05)),
                depth_max_m=float(camera_cfg.get("depth_max_m", 0.5)),
                stride=int(camera_cfg.get("stride", 2)),
            )
            raw_points_camera = points_camera.astype(np.float32)

            point_cloud_frame = str(
                camera_cfg.get(
                    "point_cloud_frame",
                    "base" if camera["base_T_camera"] is not None else "camera",
                )
            ).lower()
            if point_cloud_frame == "base":
                if camera["base_T_camera"] is None:
                    raise ValueError(f"相机 {camera_cfg['name']} 未配置 base_T_camera，无法输出 base 坐标点云")
                points_output = apply_transform(points_camera, camera["base_T_camera"])
            elif point_cloud_frame == "camera":
                points_output = points_camera.astype(np.float32)
            else:
                raise ValueError(f"不支持的 point_cloud_frame: {point_cloud_frame}")

            crop_min = camera_cfg.get("crop_min")
            crop_max = camera_cfg.get("crop_max")
            if crop_min is None and crop_max is None and point_cloud_frame == "base":
                crop_min = self.point_cloud_cfg.get("crop_min")
                crop_max = self.point_cloud_cfg.get("crop_max")
            points_output = crop_points(points_output, crop_min=crop_min, crop_max=crop_max)

            pre_random_sample = int(camera_cfg.get("pre_random_sample", self.point_cloud_cfg.get("pre_random_sample", 0)))
            if pre_random_sample > 0:
                points_output = random_downsample(points_output, pre_random_sample)
            sampling = camera_cfg.get("sampling", self.point_cloud_cfg.get("sampling", "fps"))
            num_points = int(camera_cfg.get("num_points", self.point_cloud_cfg["num_points"]))
            points_output = sample_point_cloud(points_output, num_points=num_points, method=sampling).astype(np.float32)

            per_camera[camera_cfg["name"]] = {
                "serial": str(camera_cfg["serial"]),
                "role": str(camera_cfg.get("role", camera_cfg["name"])),
                "point_cloud_frame": point_cloud_frame,
                "num_points": int(points_output.shape[0]),
                "depth_min_m": float(np.min(depth_m)) if depth_m.size else 0.0,
                "depth_max_m": float(np.max(depth_m)) if depth_m.size else 0.0,
                "depth_shape": list(depth_m.shape),
                "color_shape": list(color.shape) if color is not None else None,
                "host_capture_unix": float(frame_host_timestamp),
                "depth_frame_timestamp_ms": safe_frame_timestamp_ms(depth_frame),
                "color_frame_timestamp_ms": safe_frame_timestamp_ms(color_frame),
                "depth_frame_number": safe_frame_number(depth_frame),
                "color_frame_number": safe_frame_number(color_frame),
                "depth_timestamp_domain": safe_frame_timestamp_domain(depth_frame),
                "color_timestamp_domain": safe_frame_timestamp_domain(color_frame),
            }
            raw_cameras[camera_cfg["name"]] = {
                "color": color.copy() if color is not None else None,
                "depth": depth_m.astype(np.float32),
                "point_cloud": points_output,
                "point_cloud_raw": raw_points_camera,
                "point_cloud_raw_frame": "camera",
                "point_cloud_frame": point_cloud_frame,
                "role": str(camera_cfg.get("role", camera_cfg["name"])),
                "serial": str(camera_cfg["serial"]),
                "host_capture_unix": float(frame_host_timestamp),
                "depth_frame_timestamp_ms": safe_frame_timestamp_ms(depth_frame),
                "color_frame_timestamp_ms": safe_frame_timestamp_ms(color_frame),
                "depth_frame_number": safe_frame_number(depth_frame),
                "color_frame_number": safe_frame_number(color_frame),
                "depth_timestamp_domain": safe_frame_timestamp_domain(depth_frame),
                "color_timestamp_domain": safe_frame_timestamp_domain(color_frame),
                "intrinsics": {
                    "fx": float(camera["intrinsics"].fx),
                    "fy": float(camera["intrinsics"].fy),
                    "ppx": float(camera["intrinsics"].ppx),
                    "ppy": float(camera["intrinsics"].ppy),
                    "width": int(camera["intrinsics"].width),
                    "height": int(camera["intrinsics"].height),
                },
            }

            use_for_primary = bool(camera_cfg.get("use_for_primary_point_cloud", False))
            if primary_camera_name is not None:
                use_for_primary = use_for_primary or camera_cfg["name"] == primary_camera_name
            if primary_point_cloud is None or use_for_primary:
                primary_point_cloud = points_output

            if point_cloud_mode == "fused" and point_cloud_frame == "base" and points_output.size > 0:
                fused_points.append(points_output)

            if debug_image is None or camera_cfg.get("is_debug_camera", False) or camera_cfg.get("use_for_primary_image", False):
                debug_image = color
                debug_depth = depth_m

        if point_cloud_mode == "fused":
            fused = np.concatenate(fused_points, axis=0) if fused_points else np.zeros((0, 3), dtype=np.float32)
            if fused.size > 0:
                global_pre_random = int(self.point_cloud_cfg.get("pre_random_sample", 0))
                if global_pre_random > 0:
                    fused = random_downsample(fused, global_pre_random)
                fused = sample_point_cloud(
                    fused,
                    num_points=int(self.point_cloud_cfg["num_points"]),
                    method=self.point_cloud_cfg.get("sampling", "fps"),
                ).astype(np.float32)
                primary_point_cloud = fused

        if primary_point_cloud is None:
            primary_point_cloud = np.zeros((int(self.point_cloud_cfg["num_points"]), 3), dtype=np.float32)

        return {
            "point_cloud": primary_point_cloud.astype(np.float32),
            "img": debug_image,
            "depth": debug_depth,
            "camera_stats": per_camera,
            "raw_cameras": raw_cameras,
            "timestamp": time.time(),
        }


class URRTDEController:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.robot_cfg = config["robot"]
        self.action_mode = get_action_mode(config)
        self.control = None
        self.receive = None
        self.gripper = None
        self.gripper_type = "none"
        self.gripper_target = float(self.robot_cfg.get("default_gripper_fraction", 0.0))

    def connect(self) -> None:
        if RTDEControlInterface is None or RTDEReceiveInterface is None:
            raise ImportError("需要安装 ur-rtde 才能控制 UR 机械臂")
        try:
            self.control = RTDEControlInterface(self.robot_cfg["ip"])
            self.receive = RTDEReceiveInterface(self.robot_cfg["ip"])
            if "tcp_offset" in self.robot_cfg:
                self.control.setTcp(self.robot_cfg["tcp_offset"])
            if "payload_kg" in self.robot_cfg:
                cog = self.robot_cfg.get("payload_cog", [0.0, 0.0, 0.0])
                self.control.setPayload(float(self.robot_cfg["payload_kg"]), cog)
            self._connect_gripper()
        except Exception:
            self.disconnect()
            raise

    def _connect_gripper(self) -> None:
        gripper_cfg = self.robot_cfg.get("gripper", {})
        if not gripper_cfg.get("enabled", False):
            return
        gripper_type = str(gripper_cfg.get("type", "robotiq")).lower()
        if gripper_type == "robotiq":
            if RobotiqGripper is None:
                raise ImportError("需要安装 robotiq_gripper 才能控制 Robotiq 夹爪")
            self.gripper = RobotiqGripper()
            port = int(gripper_cfg.get("port", 63352))
            self.gripper.connect(self.robot_cfg["ip"], port)
            self.gripper.activate()
            self.gripper_type = "robotiq"
            self.set_gripper_fraction(self.gripper_target, wait=True)
            return
        if gripper_type in {"twopin_gpio", "two_pin_gpio"}:
            self.gripper = TwoPinGPIOGripper(gripper_cfg, default_fraction=self.gripper_target)
            self.gripper.connect()
            self.gripper_type = "twopin_gpio"
            return
        if gripper_type in {"twopin_gpio_remote", "two_pin_gpio_remote", "remote_twopin_gpio"}:
            self.gripper = RemoteTwoPinGripper(gripper_cfg, default_fraction=self.gripper_target)
            self.gripper.connect()
            self.gripper_type = "twopin_gpio_remote"
            return
        raise ValueError(f"不支持的夹爪类型: {gripper_type}")

    def disconnect(self) -> None:
        if self.gripper is not None and hasattr(self.gripper, "disconnect"):
            try:
                self.gripper.disconnect()
            except Exception:
                pass
        self.gripper = None
        if self.control is not None:
            try:
                self.control.stopScript()
            except Exception:
                pass
            try:
                self.control.disconnect()
            except Exception:
                pass
        if self.receive is not None:
            try:
                self.receive.disconnect()
            except Exception:
                pass
        self.control = None
        self.receive = None

    def move_home(self) -> None:
        home_joints = self.robot_cfg.get("home_joint_rad")
        if home_joints is not None:
            self.control.moveJ(
                home_joints,
                float(self.robot_cfg.get("movej_speed", 0.5)),
                float(self.robot_cfg.get("movej_acceleration", 0.8)),
            )
            self.refresh_receive_connection()
            return
        home_tcp_pose = self.robot_cfg.get("home_tcp_pose")
        if home_tcp_pose is None:
            raise ValueError("调用 move_home 前必须配置 robot.home_joint_rad 或 robot.home_tcp_pose")
        self.control.moveL(
            ensure_array(home_tcp_pose, shape=(6,), dtype=np.float64).tolist(),
            float(self.robot_cfg.get("movel_speed", self.robot_cfg.get("movej_speed", 0.25))),
            float(self.robot_cfg.get("movel_acceleration", self.robot_cfg.get("movej_acceleration", 0.5))),
        )
        self.refresh_receive_connection()

    def refresh_receive_connection(self) -> None:
        if RTDEReceiveInterface is None:
            return
        if self.receive is not None:
            try:
                self.receive.disconnect()
            except Exception:
                pass
        self.receive = RTDEReceiveInterface(self.robot_cfg["ip"])

    def get_gripper_fraction(self) -> float:
        if self.gripper is None:
            return self.gripper_target
        if self.gripper_type in {"twopin_gpio", "twopin_gpio_remote"}:
            try:
                return float(np.clip(self.gripper.get_fraction(), 0.0, 1.0))
            except Exception:
                return self.gripper_target
        try:
            position = float(self.gripper.get_current_position())
            max_position = float(self.robot_cfg.get("gripper", {}).get("max_position", 255.0))
            return float(np.clip(position / max(max_position, 1.0), 0.0, 1.0))
        except Exception:
            return self.gripper_target

    def set_gripper_fraction(self, fraction: float, wait: bool = False) -> None:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        self.gripper_target = fraction
        if self.gripper is None:
            return
        if self.gripper_type in {"twopin_gpio", "twopin_gpio_remote"}:
            self.gripper.set_fraction(fraction, wait=wait)
            return
        gripper_cfg = self.robot_cfg.get("gripper", {})
        max_position = float(gripper_cfg.get("max_position", 255.0))
        target_position = int(round(fraction * max_position))
        speed = int(gripper_cfg.get("speed", 128))
        force = int(gripper_cfg.get("force", 128))
        if hasattr(self.gripper, "move_and_wait_for_pos"):
            self.gripper.move_and_wait_for_pos(target_position, speed, force)
            return
        if hasattr(self.gripper, "move"):
            self.gripper.move(target_position, speed, force)
            if wait and hasattr(self.gripper, "wait_until_move_complete"):
                self.gripper.wait_until_move_complete()

    def get_robot_snapshot(self) -> Dict[str, np.ndarray]:
        last_error: Optional[Exception] = None
        for attempt in range(2):
            try:
                tcp_pose = np.asarray(self.receive.getActualTCPPose(), dtype=np.float64)
                joint_positions = np.asarray(self.receive.getActualQ(), dtype=np.float64)
                tcp_speed = np.asarray(self.receive.getActualTCPSpeed(), dtype=np.float64)
                controller_timestamp = float("nan")
                try:
                    if hasattr(self.receive, "getTimestamp"):
                        controller_timestamp = float(self.receive.getTimestamp())
                except Exception:
                    controller_timestamp = float("nan")
                gripper_fraction = self.get_gripper_fraction()
                state = build_state_vector(tcp_pose=tcp_pose, gripper_fraction=gripper_fraction, config=self.config)
                return {
                    "tcp_pose": tcp_pose.astype(np.float32),
                    "joint_positions": joint_positions.astype(np.float32),
                    "tcp_speed": tcp_speed.astype(np.float32),
                    "gripper_fraction": np.asarray([gripper_fraction], dtype=np.float32),
                    "gripper_target_fraction": np.asarray([self.gripper_target], dtype=np.float32),
                    "controller_timestamp": np.asarray([controller_timestamp], dtype=np.float64),
                    "state": state,
                }
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    self.refresh_receive_connection()
                    continue
                raise
        raise RuntimeError(f"读取机器人快照失败: {last_error}")

    def execute_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        if self.action_mode == "delta_tcp_pose":
            return self._execute_delta_tcp_pose(action)
        if self.action_mode == "delta_tcp_xyz_gripper":
            return self._execute_delta_tcp_xyz_gripper(action)
        if self.action_mode == "delta_tcp_pose_gripper":
            return self._execute_delta_tcp_pose_gripper(action)
        raise ValueError(f"不支持的动作表示方式: {self.action_mode}")

    def execute_control_command(self, command: Dict[str, Any]) -> Dict[str, np.ndarray]:
        control_mode = str(command["control_mode"]).lower()
        value = np.asarray(command["value"], dtype=np.float32)
        if control_mode == "dataset_action":
            return self.execute_action(value)
        if control_mode == "cartesian_pose":
            return self._execute_delta_tcp_pose(value)
        if control_mode == "cartesian_xyz_gripper":
            return self._execute_delta_tcp_xyz_gripper(value)
        if control_mode == "cartesian_pose_gripper":
            return self._execute_delta_tcp_pose_gripper(value)
        if control_mode == "joint_delta":
            return self._execute_delta_joint(value)
        raise ValueError(f"不支持的控制模式: {control_mode}")

    def _execute_delta_tcp_pose(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 6:
            raise ValueError(f"当前动作模式要求 6 维动作，实际得到 {action.shape}")
        current_pose = np.asarray(self.receive.getActualTCPPose(), dtype=np.float64)
        delta_xyz = clip_translation(
            action[:3] * float(self.robot_cfg.get("action_scale_xyz", 1.0)),
            max_norm=float(self.robot_cfg.get("max_abs_translation_per_step_m", 0.01)),
        )
        delta_rotvec = clip_rotation(
            action[3:6] * float(self.robot_cfg.get("action_scale_rotvec", 1.0)),
            max_norm=float(self.robot_cfg.get("max_abs_rotation_per_step_rad", 0.12)),
        )

        target_pose = current_pose.copy()
        target_pose[:3] += delta_xyz
        target_pose[:3] = apply_workspace_bounds(target_pose[:3], self.robot_cfg)

        if float(np.linalg.norm(delta_rotvec)) < 1.0e-9:
            target_pose[3:6] = current_pose[3:6]
        else:
            current_rotation = rotvec_to_rotation_matrix(current_pose[3:6])
            delta_rotation = rotvec_to_rotation_matrix(delta_rotvec)
            rotation_delta_frame = str(self.robot_cfg.get("rotation_delta_frame", "base")).lower()
            if rotation_delta_frame == "tool":
                target_rotation = current_rotation @ delta_rotation
            else:
                target_rotation = delta_rotation @ current_rotation
            target_rotvec = rotation_matrix_to_rotvec(target_rotation)
            target_pose[3:6] = align_rotvec_to_reference(target_rotvec, current_pose[3:6])

        # Terminal teleop is edge-triggered, so use blocking incremental moves
        # instead of servo/speed commands that require continuous target updates.
        self.control.moveL(
            target_pose.tolist(),
            float(self.robot_cfg.get("movel_speed", self.robot_cfg.get("servo_speed", 0.15))),
            float(self.robot_cfg.get("movel_acceleration", self.robot_cfg.get("servo_acceleration", 0.3))),
        )
        return {
            "commanded_pose": target_pose.astype(np.float32),
            "executed_action": np.concatenate([delta_xyz, delta_rotvec], axis=0).astype(np.float32),
        }

    def _execute_delta_tcp_xyz_gripper(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 4:
            raise ValueError(f"当前动作模式要求 4 维动作，实际得到 {action.shape}")
        current_pose = np.asarray(self.receive.getActualTCPPose(), dtype=np.float64)
        delta_xyz = clip_translation(
            action[:3] * float(self.robot_cfg.get("action_scale_xyz", 1.0)),
            max_norm=float(self.robot_cfg.get("max_abs_translation_per_step_m", 0.015)),
        )
        target_pose = current_pose.copy()
        target_pose[:3] += delta_xyz
        target_pose[:3] = apply_workspace_bounds(target_pose[:3], self.robot_cfg)
        target_pose[3:6] = current_pose[3:6]

        self.control.moveL(
            target_pose.tolist(),
            float(self.robot_cfg.get("movel_speed", self.robot_cfg.get("servo_speed", 0.25))),
            float(self.robot_cfg.get("movel_acceleration", self.robot_cfg.get("servo_acceleration", 0.5))),
        )

        gripper_fraction = gripper_target_to_fraction(float(action[3]))
        self.set_gripper_fraction(gripper_fraction, wait=False)
        return {
            "commanded_pose": target_pose.astype(np.float32),
            "executed_action": np.asarray(
                [delta_xyz[0], delta_xyz[1], delta_xyz[2], fraction_to_gripper_target(gripper_fraction)],
                dtype=np.float32,
            ),
        }

    def _execute_delta_tcp_pose_gripper(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"当前动作模式要求 7 维动作，实际得到 {action.shape}")
        current_pose = np.asarray(self.receive.getActualTCPPose(), dtype=np.float64)
        delta_xyz = clip_translation(
            action[:3] * float(self.robot_cfg.get("action_scale_xyz", 1.0)),
            max_norm=float(self.robot_cfg.get("max_abs_translation_per_step_m", 0.01)),
        )
        delta_rotvec = clip_rotation(
            action[3:6] * float(self.robot_cfg.get("action_scale_rotvec", 1.0)),
            max_norm=float(self.robot_cfg.get("max_abs_rotation_per_step_rad", 0.12)),
        )
        target_pose = current_pose.copy()
        target_pose[:3] += delta_xyz
        target_pose[:3] = apply_workspace_bounds(target_pose[:3], self.robot_cfg)

        if float(np.linalg.norm(delta_rotvec)) < 1.0e-9:
            target_pose[3:6] = current_pose[3:6]
        else:
            current_rotation = rotvec_to_rotation_matrix(current_pose[3:6])
            delta_rotation = rotvec_to_rotation_matrix(delta_rotvec)
            rotation_delta_frame = str(self.robot_cfg.get("rotation_delta_frame", "base")).lower()
            if rotation_delta_frame == "tool":
                target_rotation = current_rotation @ delta_rotation
            else:
                target_rotation = delta_rotation @ current_rotation
            target_rotvec = rotation_matrix_to_rotvec(target_rotation)
            target_pose[3:6] = align_rotvec_to_reference(target_rotvec, current_pose[3:6])

        self.control.moveL(
            target_pose.tolist(),
            float(self.robot_cfg.get("movel_speed", self.robot_cfg.get("servo_speed", 0.2))),
            float(self.robot_cfg.get("movel_acceleration", self.robot_cfg.get("servo_acceleration", 0.4))),
        )

        gripper_fraction = gripper_target_to_fraction(float(action[6]))
        self.set_gripper_fraction(gripper_fraction, wait=False)
        return {
            "commanded_pose": target_pose.astype(np.float32),
            "executed_action": np.asarray(
                [
                    delta_xyz[0],
                    delta_xyz[1],
                    delta_xyz[2],
                    delta_rotvec[0],
                    delta_rotvec[1],
                    delta_rotvec[2],
                    fraction_to_gripper_target(gripper_fraction),
                ],
                dtype=np.float32,
            ),
        }

    def _execute_delta_joint(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 6:
            raise ValueError(f"关节增量控制要求 6 维动作，实际得到 {action.shape}")
        current_q = np.asarray(self.receive.getActualQ(), dtype=np.float64)
        joint_scale = float(self.robot_cfg.get("action_scale_joint", 1.0))
        max_joint_delta = float(self.robot_cfg.get("max_abs_joint_delta_per_step_rad", 0.08))
        delta_q = np.clip(action.astype(np.float64) * joint_scale, -max_joint_delta, max_joint_delta)
        target_q = current_q + delta_q
        self.control.moveJ(
            target_q.tolist(),
            float(self.robot_cfg.get("movej_speed", self.robot_cfg.get("servo_speed", 0.15))),
            float(self.robot_cfg.get("movej_acceleration", self.robot_cfg.get("servo_acceleration", 0.3))),
        )
        return {
            "commanded_joints": target_q.astype(np.float32),
            "executed_action": delta_q.astype(np.float32),
        }


class DryRunRobotController:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.robot_cfg = config["robot"]
        self.action_mode = get_action_mode(config)
        home_joints = ensure_array(self.robot_cfg.get("home_joint_rad", [0.0] * 6), shape=(6,), dtype=np.float64)
        self.joint_positions = home_joints.astype(np.float32)
        self.tcp_pose = resolve_default_tcp_pose(self.robot_cfg).astype(np.float32)
        self.home_tcp_pose = self.tcp_pose.copy()
        self.tcp_speed = np.zeros((6,), dtype=np.float32)
        self.gripper_fraction = float(self.robot_cfg.get("default_gripper_fraction", 0.0))

    def connect(self) -> None:
        return

    def disconnect(self) -> None:
        return

    def move_home(self) -> None:
        home_tcp_pose = self.robot_cfg.get("home_tcp_pose")
        if home_tcp_pose is not None:
            self.tcp_pose = ensure_array(home_tcp_pose, shape=(6,), dtype=np.float32)
        else:
            self.tcp_pose = self.home_tcp_pose.copy()
        self.joint_positions = ensure_array(self.robot_cfg.get("home_joint_rad", [0.0] * 6), shape=(6,), dtype=np.float32)
        self.tcp_speed[:] = 0.0

    def get_gripper_fraction(self) -> float:
        return self.gripper_fraction

    def set_gripper_fraction(self, fraction: float, wait: bool = False) -> None:
        self.gripper_fraction = float(np.clip(fraction, 0.0, 1.0))

    def get_robot_snapshot(self) -> Dict[str, np.ndarray]:
        state = build_state_vector(self.tcp_pose, self.gripper_fraction, self.config)
        return {
            "tcp_pose": self.tcp_pose.astype(np.float32),
            "joint_positions": self.joint_positions.astype(np.float32),
            "tcp_speed": self.tcp_speed.astype(np.float32),
            "gripper_fraction": np.asarray([self.gripper_fraction], dtype=np.float32),
            "gripper_target_fraction": np.asarray([self.gripper_fraction], dtype=np.float32),
            "controller_timestamp": np.asarray([time.time()], dtype=np.float64),
            "state": state,
        }

    def execute_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        if self.action_mode == "delta_tcp_pose":
            return self._execute_delta_tcp_pose(action)
        if self.action_mode == "delta_tcp_xyz_gripper":
            return self._execute_delta_tcp_xyz_gripper(action)
        if self.action_mode == "delta_tcp_pose_gripper":
            return self._execute_delta_tcp_pose_gripper(action)
        raise ValueError(f"不支持的动作表示方式: {self.action_mode}")

    def execute_control_command(self, command: Dict[str, Any]) -> Dict[str, np.ndarray]:
        control_mode = str(command["control_mode"]).lower()
        value = np.asarray(command["value"], dtype=np.float32)
        if control_mode == "dataset_action":
            return self.execute_action(value)
        if control_mode == "cartesian_pose":
            return self._execute_delta_tcp_pose(value)
        if control_mode == "cartesian_xyz_gripper":
            return self._execute_delta_tcp_xyz_gripper(value)
        if control_mode == "cartesian_pose_gripper":
            return self._execute_delta_tcp_pose_gripper(value)
        if control_mode == "joint_delta":
            return self._execute_delta_joint(value)
        raise ValueError(f"不支持的控制模式: {control_mode}")

    def _execute_delta_tcp_pose(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 6:
            raise ValueError(f"当前动作模式要求 6 维动作，实际得到 {action.shape}")
        delta_xyz = clip_translation(
            action[:3] * float(self.robot_cfg.get("action_scale_xyz", 1.0)),
            max_norm=float(self.robot_cfg.get("max_abs_translation_per_step_m", 0.01)),
        )
        delta_rotvec = clip_rotation(
            action[3:6] * float(self.robot_cfg.get("action_scale_rotvec", 1.0)),
            max_norm=float(self.robot_cfg.get("max_abs_rotation_per_step_rad", 0.12)),
        )
        new_xyz = apply_workspace_bounds(self.tcp_pose[:3].astype(np.float64) + delta_xyz.astype(np.float64), self.robot_cfg)
        current_rotvec = self.tcp_pose[3:6].astype(np.float64)
        if float(np.linalg.norm(delta_rotvec)) < 1.0e-9:
            target_rotvec = current_rotvec
        else:
            current_rotation = rotvec_to_rotation_matrix(current_rotvec)
            delta_rotation = rotvec_to_rotation_matrix(delta_rotvec)
            rotation_delta_frame = str(self.robot_cfg.get("rotation_delta_frame", "base")).lower()
            if rotation_delta_frame == "tool":
                target_rotation = current_rotation @ delta_rotation
            else:
                target_rotation = delta_rotation @ current_rotation
            target_rotvec = align_rotvec_to_reference(rotation_matrix_to_rotvec(target_rotation), current_rotvec)
        servo_dt = max(float(self.robot_cfg.get("servo_dt", 0.1)), 1.0e-6)
        self.tcp_speed[:3] = delta_xyz.astype(np.float32) / servo_dt
        self.tcp_speed[3:6] = delta_rotvec.astype(np.float32) / servo_dt
        self.tcp_pose[:3] = new_xyz.astype(np.float32)
        self.tcp_pose[3:6] = target_rotvec.astype(np.float32)
        time.sleep(float(self.robot_cfg.get("servo_dt", 0.1)))
        return {
            "commanded_pose": self.tcp_pose.copy(),
            "executed_action": np.concatenate([delta_xyz, delta_rotvec], axis=0).astype(np.float32),
        }

    def _execute_delta_tcp_xyz_gripper(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 4:
            raise ValueError(f"当前动作模式要求 4 维动作，实际得到 {action.shape}")
        delta_xyz = clip_translation(
            action[:3] * float(self.robot_cfg.get("action_scale_xyz", 1.0)),
            max_norm=float(self.robot_cfg.get("max_abs_translation_per_step_m", 0.015)),
        )
        new_xyz = apply_workspace_bounds(self.tcp_pose[:3].astype(np.float64) + delta_xyz.astype(np.float64), self.robot_cfg)
        servo_dt = max(float(self.robot_cfg.get("servo_dt", 0.1)), 1.0e-6)
        self.tcp_speed[:3] = delta_xyz.astype(np.float32) / servo_dt
        self.tcp_speed[3:6] = 0.0
        self.tcp_pose[:3] = new_xyz.astype(np.float32)
        self.set_gripper_fraction(gripper_target_to_fraction(float(action[3])))
        time.sleep(float(self.robot_cfg.get("servo_dt", 0.1)))
        return {
            "commanded_pose": self.tcp_pose.copy(),
            "executed_action": np.asarray(
                [delta_xyz[0], delta_xyz[1], delta_xyz[2], fraction_to_gripper_target(self.gripper_fraction)],
                dtype=np.float32,
            ),
        }

    def _execute_delta_tcp_pose_gripper(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 7:
            raise ValueError(f"当前动作模式要求 7 维动作，实际得到 {action.shape}")
        delta_xyz = clip_translation(
            action[:3] * float(self.robot_cfg.get("action_scale_xyz", 1.0)),
            max_norm=float(self.robot_cfg.get("max_abs_translation_per_step_m", 0.01)),
        )
        delta_rotvec = clip_rotation(
            action[3:6] * float(self.robot_cfg.get("action_scale_rotvec", 1.0)),
            max_norm=float(self.robot_cfg.get("max_abs_rotation_per_step_rad", 0.12)),
        )
        new_xyz = apply_workspace_bounds(self.tcp_pose[:3].astype(np.float64) + delta_xyz.astype(np.float64), self.robot_cfg)
        current_rotvec = self.tcp_pose[3:6].astype(np.float64)
        if float(np.linalg.norm(delta_rotvec)) < 1.0e-9:
            target_rotvec = current_rotvec
        else:
            current_rotation = rotvec_to_rotation_matrix(current_rotvec)
            delta_rotation = rotvec_to_rotation_matrix(delta_rotvec)
            rotation_delta_frame = str(self.robot_cfg.get("rotation_delta_frame", "base")).lower()
            if rotation_delta_frame == "tool":
                target_rotation = current_rotation @ delta_rotation
            else:
                target_rotation = delta_rotation @ current_rotation
            target_rotvec = align_rotvec_to_reference(rotation_matrix_to_rotvec(target_rotation), current_rotvec)
        servo_dt = max(float(self.robot_cfg.get("servo_dt", 0.1)), 1.0e-6)
        self.tcp_speed[:3] = delta_xyz.astype(np.float32) / servo_dt
        self.tcp_speed[3:6] = delta_rotvec.astype(np.float32) / servo_dt
        self.tcp_pose[:3] = new_xyz.astype(np.float32)
        self.tcp_pose[3:6] = target_rotvec.astype(np.float32)
        self.set_gripper_fraction(gripper_target_to_fraction(float(action[6])))
        time.sleep(float(self.robot_cfg.get("servo_dt", 0.1)))
        return {
            "commanded_pose": self.tcp_pose.copy(),
            "executed_action": np.asarray(
                [
                    delta_xyz[0],
                    delta_xyz[1],
                    delta_xyz[2],
                    delta_rotvec[0],
                    delta_rotvec[1],
                    delta_rotvec[2],
                    fraction_to_gripper_target(self.gripper_fraction),
                ],
                dtype=np.float32,
            ),
        }

    def _execute_delta_joint(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 6:
            raise ValueError(f"关节增量控制要求 6 维动作，实际得到 {action.shape}")
        current_q = self.joint_positions.astype(np.float64)
        joint_scale = float(self.robot_cfg.get("action_scale_joint", 1.0))
        max_joint_delta = float(self.robot_cfg.get("max_abs_joint_delta_per_step_rad", 0.08))
        delta_q = np.clip(action.astype(np.float64) * joint_scale, -max_joint_delta, max_joint_delta)
        self.joint_positions = (current_q + delta_q).astype(np.float32)
        self.tcp_speed[:] = 0.0
        time.sleep(float(self.robot_cfg.get("servo_dt", 0.1)))
        return {
            "commanded_joints": self.joint_positions.copy(),
            "executed_action": delta_q.astype(np.float32),
        }


def make_robot_controller(config: Dict[str, Any], dry_run: bool = False):
    if dry_run:
        return DryRunRobotController(config)
    return URRTDEController(config)


class RealRobotDatasetWriter:
    def __init__(self, zarr_path: str, overwrite: bool = False):
        if zarr is None:
            raise ImportError("需要安装 zarr 才能写入数据集")
        self.zarr_path = Path(zarr_path)
        self.manifest_path = str(self.zarr_path) + ".manifest.json"
        if overwrite and self.zarr_path.exists():
            import shutil

            shutil.rmtree(self.zarr_path)
        if overwrite and Path(self.manifest_path).exists():
            Path(self.manifest_path).unlink()
        self.zarr_path.parent.mkdir(parents=True, exist_ok=True)
        self.root = zarr.open(str(self.zarr_path), mode="a")
        self.data_group = self.root.require_group("data")
        self.meta_group = self.root.require_group("meta")
        if "episode_ends" not in self.meta_group:
            self._create_zero_array(
                self.meta_group,
                name="episode_ends",
                shape=(0,),
                chunks=(1024,),
                dtype=np.int64,
            )
        self.root.attrs["dataset_format"] = "real_robot_collection_v1"
        self.manifest = {"episodes": []}
        if Path(self.manifest_path).exists():
            self.manifest = load_json(self.manifest_path)

    @staticmethod
    def _create_zero_array(group, name: str, shape: Tuple[int, ...], chunks: Tuple[int, ...], dtype) -> None:
        shape = tuple(int(x) for x in shape)
        chunks = tuple(int(x) for x in chunks)
        create_attempts = [
            ("zeros", {"name": name, "shape": shape, "chunks": chunks, "dtype": dtype}),
            (
                "create_array",
                {
                    "name": name,
                    "shape": shape,
                    "chunks": chunks,
                    "dtype": dtype,
                    "fill_value": 0,
                },
            ),
            (
                "create_dataset",
                {
                    "name": name,
                    "shape": shape,
                    "chunks": chunks,
                    "dtype": dtype,
                    "fill_value": 0,
                },
            ),
        ]
        last_error = None
        for method_name, kwargs in create_attempts:
            method = getattr(group, method_name, None)
            if method is None:
                continue
            try:
                method(**kwargs)
                return
            except TypeError as exc:
                last_error = exc
                continue
        raise TypeError(f"无法在 zarr group 中创建数组 {name}: {last_error}")

    def _episode_ends(self):
        return self.meta_group["episode_ends"]

    def _append_array(self, key: str, value: np.ndarray) -> None:
        value = np.asarray(value)
        if key not in self.data_group:
            self._create_zero_array(
                self.data_group,
                name=key,
                shape=value.shape,
                chunks=chunk_shape_for_array(value),
                dtype=value.dtype,
            )
            self.data_group[key][:] = value
            return

        arr = self.data_group[key]
        if tuple(arr.shape[1:]) != tuple(value.shape[1:]):
            raise ValueError(f"字段 {key} 的形状不一致: 已有 {arr.shape[1:]}, 新值 {value.shape[1:]}")
        old_len = int(arr.shape[0])
        new_len = old_len + int(value.shape[0])
        arr.resize((new_len,) + tuple(arr.shape[1:]))
        arr[old_len:new_len] = value

    def add_episode(self, episode: Dict[str, List[np.ndarray]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        required = ["state", "action", "point_cloud"]
        for key in required:
            if key not in episode or len(episode[key]) == 0:
                raise ValueError(f"轨迹字段 {key} 为空，无法写入")

        np_episode = {}
        for key, values in episode.items():
            if len(values) == 0:
                continue
            np_episode[key] = np.stack(values, axis=0)

        episode_length = int(np_episode["state"].shape[0])
        episode_ends = self._episode_ends()
        prev_total = int(episode_ends[-1]) if episode_ends.shape[0] > 0 else 0
        new_total = prev_total + episode_length

        for key, value in np_episode.items():
            self._append_array(key, value)

        old_count = int(episode_ends.shape[0])
        episode_ends.resize((old_count + 1,))
        episode_ends[old_count] = new_total

        metadata = dict(metadata)
        metadata["length"] = episode_length
        metadata["episode_index"] = len(self.manifest["episodes"])
        self.manifest["episodes"].append(metadata)
        save_json(self.manifest_path, self.manifest)
        return metadata


class RealRobotRawDatasetWriter:
    def __init__(self, raw_root: str, config: Dict[str, Any], session_name: Optional[str] = None):
        self.raw_root = Path(raw_root)
        dataset_cfg = config.get("dataset", {})
        prefix = session_name or dataset_cfg.get("raw_session_prefix", "real_capture")
        append_timestamp = bool(dataset_cfg.get("raw_append_timestamp_session", False))
        if append_timestamp:
            self.session_name = f"{prefix}_{make_timestamp_string()}"
            self.session_dir = self.raw_root / self.session_name
        else:
            self.session_name = str(prefix)
            self.session_dir = self.raw_root
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.session_dir / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = load_json(str(self.manifest_path))
            self.manifest.setdefault("session_name", self.session_name)
            self.manifest.setdefault("created_at_unix", time.time())
            self.manifest.setdefault("episodes", [])
        else:
            self.manifest = {
                "session_name": self.session_name,
                "created_at_unix": time.time(),
                "episodes": [],
            }
        save_json(str(self.session_dir / "config_snapshot_latest.json"), config)
        if not (self.session_dir / "config_snapshot.json").exists():
            save_json(str(self.session_dir / "config_snapshot.json"), config)
        save_json(str(self.manifest_path), self.manifest)

    def _next_episode_index(self) -> int:
        existing_indices = []
        for path in self.session_dir.glob("episode_*"):
            if not path.is_dir():
                continue
            suffix = path.name[len("episode_") :]
            try:
                existing_indices.append(int(suffix))
            except ValueError:
                continue
        manifest_count = len(self.manifest.get("episodes", []))
        if not existing_indices:
            return manifest_count
        return max(max(existing_indices) + 1, manifest_count)

    def add_episode(self, steps: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
        if len(steps) == 0:
            raise ValueError("raw episode 为空，无法写入")

        episode_index = self._next_episode_index()
        episode_dir = self.session_dir / f"episode_{episode_index:04d}"
        episode_dir.mkdir(parents=True, exist_ok=False)

        robot_dir = episode_dir / "robot"
        processed_dir = episode_dir / "processed"
        cameras_dir = episode_dir / "cameras"
        robot_dir.mkdir()
        processed_dir.mkdir()
        cameras_dir.mkdir()
        (processed_dir / "point_cloud").mkdir()
        (processed_dir / "debug_rgb").mkdir()
        (processed_dir / "debug_depth").mkdir()

        def stack_field(field: str, dtype=None):
            values = []
            for step in steps:
                value = step[field]
                values.append(np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value))
            return np.stack(values, axis=0)

        robot_arrays = {
            "agent_pos": stack_field("agent_pos", np.float32),
            "joint_positions": stack_field("joint_positions", np.float32),
            "tcp_pose": stack_field("tcp_pose", np.float32),
            "tcp_speed": stack_field("tcp_speed", np.float32),
            "gripper_fraction": stack_field("gripper_fraction", np.float32),
            "gripper_target_fraction": stack_field("gripper_target_fraction", np.float32),
            "controller_timestamp": stack_field("controller_timestamp", np.float64),
            "action": stack_field("action", np.float32),
            "timestamp": stack_field("timestamp", np.float64),
            "capture_completed_timestamp": stack_field("capture_completed_timestamp", np.float64),
            "action_source_flag": stack_field("action_source_flag", np.int8),
        }
        for name, array in robot_arrays.items():
            np.save(robot_dir / f"{name}.npy", array)
        save_array_csv(robot_dir / "agent_pos.csv", robot_arrays["agent_pos"])
        save_array_csv(robot_dir / "joint_positions.csv", robot_arrays["joint_positions"], header="q0,q1,q2,q3,q4,q5")
        save_array_csv(robot_dir / "tcp_pose.csv", robot_arrays["tcp_pose"], header="x,y,z,rx,ry,rz")
        save_array_csv(robot_dir / "tcp_speed.csv", robot_arrays["tcp_speed"], header="vx,vy,vz,wx,wy,wz")
        save_array_csv(robot_dir / "gripper_fraction.csv", robot_arrays["gripper_fraction"], header="gripper_fraction")
        save_array_csv(
            robot_dir / "gripper_target_fraction.csv",
            robot_arrays["gripper_target_fraction"],
            header="gripper_target_fraction",
        )
        save_array_csv(robot_dir / "controller_timestamp.csv", robot_arrays["controller_timestamp"], header="controller_timestamp")
        save_array_csv(robot_dir / "action.csv", robot_arrays["action"])
        save_array_csv(robot_dir / "timestamp.csv", robot_arrays["timestamp"], header="timestamp")
        save_array_csv(robot_dir / "action_source_flag.csv", robot_arrays["action_source_flag"], header="action_source_flag")
        save_array_csv(
            robot_dir / "capture_completed_timestamp.csv",
            robot_arrays["capture_completed_timestamp"],
            header="capture_completed_timestamp",
        )

        for idx, step in enumerate(steps):
            save_point_cloud_ply(processed_dir / "point_cloud" / f"{idx:06d}.ply", step["point_cloud"])
            save_png_image(processed_dir / "debug_rgb" / f"{idx:06d}.png", step["img"])
            np.save(processed_dir / "debug_depth" / f"{idx:06d}.npy", np.asarray(step["depth"], dtype=np.float32))
        save_json(
            str(processed_dir / "camera_stats.json"),
            {f"step_{idx:04d}": step.get("camera_stats", {}) for idx, step in enumerate(steps)},
        )

        camera_names = sorted(steps[0].get("raw_cameras", {}).keys())
        for camera_name in camera_names:
            camera_dir = cameras_dir / camera_name
            camera_dir.mkdir()
            (camera_dir / "rgb").mkdir()
            (camera_dir / "depth").mkdir()
            (camera_dir / "point_cloud").mkdir()
            camera_host_capture_unix = []
            depth_frame_timestamp_ms = []
            color_frame_timestamp_ms = []
            depth_frame_number = []
            color_frame_number = []
            camera_meta = None
            for idx, step in enumerate(steps):
                camera_step = step["raw_cameras"][camera_name]
                color = camera_step.get("color")
                if color is None:
                    color = np.zeros((1, 1, 3), dtype=np.uint8)
                depth = camera_step.get("depth")
                if depth is None:
                    depth = np.zeros((1, 1), dtype=np.float32)
                point_cloud = camera_step.get("point_cloud_raw")
                if point_cloud is None:
                    point_cloud = np.zeros((0, 3), dtype=np.float32)
                save_png_image(camera_dir / "rgb" / f"{idx:06d}.png", np.asarray(color, dtype=np.uint8))
                np.save(camera_dir / "depth" / f"{idx:06d}.npy", np.asarray(depth, dtype=np.float32))
                save_point_cloud_ply(camera_dir / "point_cloud" / f"{idx:06d}.ply", np.asarray(point_cloud, dtype=np.float32))
                camera_host_capture_unix.append([float(camera_step.get("host_capture_unix", np.nan))])
                depth_frame_timestamp_ms.append([float(camera_step.get("depth_frame_timestamp_ms", np.nan))])
                color_frame_timestamp_ms.append([float(camera_step.get("color_frame_timestamp_ms", np.nan))])
                depth_frame_number.append([int(camera_step.get("depth_frame_number", -1))])
                color_frame_number.append([int(camera_step.get("color_frame_number", -1))])
                if camera_meta is None:
                    camera_meta = {
                        "point_cloud_frame": camera_step.get("point_cloud_frame", "unknown"),
                        "point_cloud_raw_frame": camera_step.get("point_cloud_raw_frame", "camera"),
                        "role": camera_step.get("role", camera_name),
                        "serial": camera_step.get("serial"),
                        "depth_timestamp_domain": camera_step.get("depth_timestamp_domain", "unknown"),
                        "color_timestamp_domain": camera_step.get("color_timestamp_domain", "unknown"),
                        "intrinsics": camera_step.get("intrinsics"),
                    }
            if camera_meta is not None:
                np.save(camera_dir / "host_capture_unix.npy", np.asarray(camera_host_capture_unix, dtype=np.float64))
                np.save(camera_dir / "depth_frame_timestamp_ms.npy", np.asarray(depth_frame_timestamp_ms, dtype=np.float64))
                np.save(camera_dir / "color_frame_timestamp_ms.npy", np.asarray(color_frame_timestamp_ms, dtype=np.float64))
                np.save(camera_dir / "depth_frame_number.npy", np.asarray(depth_frame_number, dtype=np.int64))
                np.save(camera_dir / "color_frame_number.npy", np.asarray(color_frame_number, dtype=np.int64))
                save_array_csv(camera_dir / "host_capture_unix.csv", np.asarray(camera_host_capture_unix, dtype=np.float64), header="host_capture_unix")
                save_array_csv(camera_dir / "depth_frame_timestamp_ms.csv", np.asarray(depth_frame_timestamp_ms, dtype=np.float64), header="depth_frame_timestamp_ms")
                save_array_csv(camera_dir / "color_frame_timestamp_ms.csv", np.asarray(color_frame_timestamp_ms, dtype=np.float64), header="color_frame_timestamp_ms")
                save_array_csv(camera_dir / "depth_frame_number.csv", np.asarray(depth_frame_number, dtype=np.int64), header="depth_frame_number")
                save_array_csv(camera_dir / "color_frame_number.csv", np.asarray(color_frame_number, dtype=np.int64), header="color_frame_number")
                save_json(str(camera_dir / "metadata.json"), camera_meta)

        metadata = dict(metadata)
        metadata["episode_index"] = episode_index
        metadata["length"] = len(steps)
        metadata["episode_dir"] = str(episode_dir)
        self.manifest["episodes"].append(metadata)
        save_json(str(episode_dir / "metadata.json"), metadata)
        save_json(str(self.manifest_path), self.manifest)
        return metadata


def capture_robot_observation(
    robot,
    camera_manager: MultiRealSenseManager,
    robot_snapshot: Optional[Dict[str, np.ndarray]] = None,
    host_timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    vision = camera_manager.capture()
    if robot_snapshot is None:
        robot_snapshot = robot.get_robot_snapshot()
    timestamp = host_timestamp if host_timestamp is not None else float(vision["timestamp"])
    capture_completed_timestamp = float(vision["timestamp"])
    return {
        "point_cloud": vision["point_cloud"].astype(np.float32),
        "agent_pos": robot_snapshot["state"].astype(np.float32),
        "img": vision["img"],
        "depth": vision["depth"],
        "tcp_pose": robot_snapshot["tcp_pose"],
        "joint_positions": robot_snapshot["joint_positions"],
        "tcp_speed": robot_snapshot["tcp_speed"],
        "gripper_fraction": robot_snapshot["gripper_fraction"],
        "gripper_target_fraction": robot_snapshot.get("gripper_target_fraction", robot_snapshot["gripper_fraction"]),
        "controller_timestamp": robot_snapshot["controller_timestamp"],
        "timestamp": np.asarray([timestamp], dtype=np.float64),
        "capture_completed_timestamp": np.asarray([capture_completed_timestamp], dtype=np.float64),
        "camera_stats": vision["camera_stats"],
        "raw_cameras": vision["raw_cameras"],
    }


def build_dataset_step(observation: Dict[str, Any], action: np.ndarray) -> Dict[str, np.ndarray]:
    img = observation["img"]
    depth = observation["depth"]
    if img is None:
        img = np.zeros((1, 1, 3), dtype=np.uint8)
    if depth is None:
        depth = np.zeros((1, 1), dtype=np.float32)
    data = {
        "point_cloud": observation["point_cloud"].astype(np.float32),
        "state": observation["agent_pos"].astype(np.float32),
        "action": action.astype(np.float32),
        "img": img.astype(np.uint8),
        "depth": depth.astype(np.float32),
        "tcp_pose": observation["tcp_pose"].astype(np.float32),
        "joint_positions": observation["joint_positions"].astype(np.float32),
        "tcp_speed": observation["tcp_speed"].astype(np.float32),
        "gripper_fraction": observation["gripper_fraction"].astype(np.float32),
        "gripper_target_fraction": observation["gripper_target_fraction"].astype(np.float32),
        "controller_timestamp": observation["controller_timestamp"].astype(np.float64),
        "timestamp": observation["timestamp"].astype(np.float64),
        "capture_completed_timestamp": observation["capture_completed_timestamp"].astype(np.float64),
    }
    for camera_name, camera_value in observation.get("raw_cameras", {}).items():
        camera_key = sanitize_camera_key(camera_name)
        point_cloud = camera_value.get("point_cloud")
        if point_cloud is not None:
            data[f"camera_{camera_key}_point_cloud"] = np.asarray(point_cloud, dtype=np.float32)
        color = camera_value.get("color")
        if color is not None:
            data[f"camera_{camera_key}_img"] = np.asarray(color, dtype=np.uint8)
        depth_value = camera_value.get("depth")
        if depth_value is not None:
            data[f"camera_{camera_key}_depth"] = np.asarray(depth_value, dtype=np.float32)
        data[f"camera_{camera_key}_host_capture_unix"] = np.asarray(
            [float(camera_value.get("host_capture_unix", np.nan))], dtype=np.float64
        )
        data[f"camera_{camera_key}_depth_frame_timestamp_ms"] = np.asarray(
            [float(camera_value.get("depth_frame_timestamp_ms", np.nan))], dtype=np.float64
        )
        data[f"camera_{camera_key}_color_frame_timestamp_ms"] = np.asarray(
            [float(camera_value.get("color_frame_timestamp_ms", np.nan))], dtype=np.float64
        )
        data[f"camera_{camera_key}_depth_frame_number"] = np.asarray(
            [int(camera_value.get("depth_frame_number", -1))], dtype=np.int64
        )
        data[f"camera_{camera_key}_color_frame_number"] = np.asarray(
            [int(camera_value.get("color_frame_number", -1))], dtype=np.int64
        )
    return data


def build_raw_step(observation: Dict[str, Any], action: np.ndarray) -> Dict[str, Any]:
    img = observation["img"]
    depth = observation["depth"]
    if img is None:
        img = np.zeros((1, 1, 3), dtype=np.uint8)
    if depth is None:
        depth = np.zeros((1, 1), dtype=np.float32)

    raw_cameras = {}
    for camera_name, camera_value in observation.get("raw_cameras", {}).items():
        raw_cameras[camera_name] = {
            "color": None if camera_value.get("color") is None else np.asarray(camera_value["color"], dtype=np.uint8),
            "depth": None if camera_value.get("depth") is None else np.asarray(camera_value["depth"], dtype=np.float32),
            "point_cloud": None if camera_value.get("point_cloud") is None else np.asarray(camera_value["point_cloud"], dtype=np.float32),
            "point_cloud_raw": None
            if camera_value.get("point_cloud_raw") is None
            else np.asarray(camera_value["point_cloud_raw"], dtype=np.float32),
            "point_cloud_raw_frame": camera_value.get("point_cloud_raw_frame", "camera"),
            "point_cloud_frame": camera_value.get("point_cloud_frame", "unknown"),
            "role": camera_value.get("role", camera_name),
            "serial": camera_value.get("serial"),
            "host_capture_unix": float(camera_value.get("host_capture_unix", np.nan)),
            "depth_frame_timestamp_ms": float(camera_value.get("depth_frame_timestamp_ms", np.nan)),
            "color_frame_timestamp_ms": float(camera_value.get("color_frame_timestamp_ms", np.nan)),
            "depth_frame_number": int(camera_value.get("depth_frame_number", -1)),
            "color_frame_number": int(camera_value.get("color_frame_number", -1)),
            "depth_timestamp_domain": camera_value.get("depth_timestamp_domain", "unknown"),
            "color_timestamp_domain": camera_value.get("color_timestamp_domain", "unknown"),
            "intrinsics": camera_value.get("intrinsics"),
        }

    return {
        "point_cloud": observation["point_cloud"].astype(np.float32),
        "agent_pos": observation["agent_pos"].astype(np.float32),
        "img": np.asarray(img, dtype=np.uint8),
        "depth": np.asarray(depth, dtype=np.float32),
        "tcp_pose": observation["tcp_pose"].astype(np.float32),
        "joint_positions": observation["joint_positions"].astype(np.float32),
        "tcp_speed": observation["tcp_speed"].astype(np.float32),
        "gripper_fraction": observation["gripper_fraction"].astype(np.float32),
        "gripper_target_fraction": observation["gripper_target_fraction"].astype(np.float32),
        "controller_timestamp": observation["controller_timestamp"].astype(np.float64),
        "timestamp": observation["timestamp"].astype(np.float64),
        "capture_completed_timestamp": observation["capture_completed_timestamp"].astype(np.float64),
        "action": np.asarray(action, dtype=np.float32),
        "camera_stats": observation.get("camera_stats", {}),
        "raw_cameras": raw_cameras,
    }


def infer_action_from_snapshots(
    prev_snapshot: Dict[str, np.ndarray],
    curr_snapshot: Dict[str, np.ndarray],
    config: Dict[str, Any],
) -> np.ndarray:
    action_mode = get_action_mode(config)
    if action_mode == "delta_tcp_pose":
        prev_pose = np.asarray(prev_snapshot["tcp_pose"], dtype=np.float64)
        curr_pose = np.asarray(curr_snapshot["tcp_pose"], dtype=np.float64)
        delta_xyz = (curr_pose[:3] - prev_pose[:3]).astype(np.float32)
        prev_rot = rotvec_to_rotation_matrix(prev_pose[3:6])
        curr_rot = rotvec_to_rotation_matrix(curr_pose[3:6])
        rotation_delta_frame = str(config.get("robot", {}).get("rotation_delta_frame", "base")).lower()
        if rotation_delta_frame == "tool":
            delta_rot = prev_rot.T @ curr_rot
        else:
            delta_rot = curr_rot @ prev_rot.T
        delta_rotvec = rotation_matrix_to_rotvec(delta_rot).astype(np.float32)
        return np.concatenate([delta_xyz, delta_rotvec], axis=0).astype(np.float32)

    if action_mode == "delta_tcp_xyz_gripper":
        prev_pose = np.asarray(prev_snapshot["tcp_pose"], dtype=np.float64)
        curr_pose = np.asarray(curr_snapshot["tcp_pose"], dtype=np.float64)
        delta_xyz = (curr_pose[:3] - prev_pose[:3]).astype(np.float32)
        curr_gripper_fraction = snapshot_gripper_fraction(curr_snapshot, prefer_target=True)
        return np.asarray(
            [
                delta_xyz[0],
                delta_xyz[1],
                delta_xyz[2],
                fraction_to_gripper_target(curr_gripper_fraction),
            ],
            dtype=np.float32,
        )

    if action_mode == "delta_tcp_pose_gripper":
        prev_pose = np.asarray(prev_snapshot["tcp_pose"], dtype=np.float64)
        curr_pose = np.asarray(curr_snapshot["tcp_pose"], dtype=np.float64)
        delta_xyz = (curr_pose[:3] - prev_pose[:3]).astype(np.float32)
        prev_rot = rotvec_to_rotation_matrix(prev_pose[3:6])
        curr_rot = rotvec_to_rotation_matrix(curr_pose[3:6])
        rotation_delta_frame = str(config.get("robot", {}).get("rotation_delta_frame", "base")).lower()
        if rotation_delta_frame == "tool":
            delta_rot = prev_rot.T @ curr_rot
        else:
            delta_rot = curr_rot @ prev_rot.T
        delta_rotvec = rotation_matrix_to_rotvec(delta_rot).astype(np.float32)
        curr_gripper_fraction = snapshot_gripper_fraction(curr_snapshot, prefer_target=True)
        return np.asarray(
            [
                delta_xyz[0],
                delta_xyz[1],
                delta_xyz[2],
                delta_rotvec[0],
                delta_rotvec[1],
                delta_rotvec[2],
                fraction_to_gripper_target(curr_gripper_fraction),
            ],
            dtype=np.float32,
        )

    raise ValueError(f"不支持的动作表示方式: {action_mode}")
