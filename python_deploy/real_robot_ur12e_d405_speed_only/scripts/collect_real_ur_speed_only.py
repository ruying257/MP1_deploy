import argparse
import math
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from real_robot_utils import (
    KeyPoller,
    MultiRealSenseManager,
    RealRobotDatasetWriter,
    RealRobotRawDatasetWriter,
    align_rotvec_to_reference,
    build_dataset_step,
    build_raw_step,
    capture_robot_observation,
    clip_rotation,
    clip_translation,
    fraction_to_gripper_target,
    format_action_summary,
    get_action_mode,
    get_obs_mode,
    infer_action_from_snapshots,
    load_json,
    make_robot_controller,
    resolve_collection_paths,
    rotvec_to_rotation_matrix,
    rotation_matrix_to_rotvec,
    snapshot_gripper_fraction,
)


def parse_args():
    parser = argparse.ArgumentParser(description="UR12e + D405 speed-only 真机采集脚本")
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    parser.add_argument("--overwrite", action="store_true", help="若 zarr 已存在则覆盖")
    parser.add_argument("--dry-run", action="store_true", help="不连接真机，仅演练流程")
    parser.add_argument("--move-home-on-start", action="store_true", help="启动后先回 home")
    parser.add_argument("--sample-hz", type=float, help="覆盖配置中的采样频率")
    parser.add_argument("--control-hz", type=float, help="覆盖配置中的速度控制频率")
    return parser.parse_args()


def get_task_config(config: Dict[str, Any]) -> Dict[str, Any]:
    task_cfg = dict(config.get("task", {}))
    task_cfg.setdefault("task_id", "unnamed_task")
    task_cfg.setdefault("name", task_cfg["task_id"])
    task_cfg.setdefault("description", "")
    return task_cfg


def empty_episode() -> Dict[str, List[np.ndarray]]:
    return defaultdict(list)


def append_step(episode: Dict[str, List[np.ndarray]], step: Dict[str, np.ndarray]) -> None:
    for key, value in step.items():
        episode[key].append(value)


def is_effectively_zero(value: np.ndarray, eps: float = 1.0e-9) -> bool:
    return not bool(np.any(np.abs(np.asarray(value, dtype=np.float64)) > eps))


def is_motion_effectively_zero(action: np.ndarray, action_mode: str, eps: float = 1.0e-9) -> bool:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action_mode == "delta_tcp_xyz_gripper":
        action = action[:3]
    elif action_mode == "delta_tcp_pose_gripper":
        action = action[:6]
    return not bool(np.any(np.abs(action) > eps))


def has_gripper_target_changed(
    prev_snapshot: Dict[str, np.ndarray],
    curr_snapshot: Dict[str, np.ndarray],
    eps: float = 1.0e-6,
) -> bool:
    prev_fraction = snapshot_gripper_fraction(prev_snapshot, prefer_target=True)
    curr_fraction = snapshot_gripper_fraction(curr_snapshot, prefer_target=True)
    return abs(curr_fraction - prev_fraction) > eps


def extract_controller_timestamp(snapshot: Optional[Dict[str, np.ndarray]]) -> float:
    if snapshot is None:
        return float("nan")
    value = snapshot.get("controller_timestamp")
    if value is None:
        return float("nan")
    try:
        return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
    except Exception:
        return float("nan")


def format_snapshot_debug(
    prev_snapshot: Dict[str, np.ndarray],
    curr_snapshot: Dict[str, np.ndarray],
    action_source_flag: np.ndarray,
) -> str:
    prev_controller_timestamp = extract_controller_timestamp(prev_snapshot)
    curr_controller_timestamp = extract_controller_timestamp(curr_snapshot)
    if np.isfinite(prev_controller_timestamp) and np.isfinite(curr_controller_timestamp):
        controller_dt_ms = (curr_controller_timestamp - prev_controller_timestamp) * 1000.0
        controller_dt_text = f"{controller_dt_ms:.1f}ms"
    else:
        controller_dt_text = "nan"
    tcp_speed = np.asarray(curr_snapshot.get("tcp_speed", np.zeros((6,), dtype=np.float32)), dtype=np.float64).reshape(-1)
    linear_speed = float(np.linalg.norm(tcp_speed[:3])) if tcp_speed.size >= 3 else float("nan")
    angular_speed = float(np.linalg.norm(tcp_speed[3:6])) if tcp_speed.size >= 6 else float("nan")
    source = "fallback" if int(np.asarray(action_source_flag).reshape(-1)[0]) == 1 else "diff"
    return (
        f"source={source}, ctrl_dt={controller_dt_text}, "
        f"tcp_speed={linear_speed:.4f}m/s, tcp_omega={angular_speed:.4f}rad/s"
    )


def build_cartesian_fallback_action(
    cartesian_delta: np.ndarray,
    prev_snapshot: Dict[str, np.ndarray],
    curr_snapshot: Dict[str, np.ndarray],
    action_mode: str,
) -> Optional[np.ndarray]:
    cartesian_delta = np.asarray(cartesian_delta, dtype=np.float32).reshape(-1)
    if cartesian_delta.shape[0] != 6:
        return None
    if is_effectively_zero(cartesian_delta, eps=1.0e-8):
        return None
    if action_mode == "delta_tcp_pose":
        return cartesian_delta.astype(np.float32)
    if action_mode == "delta_tcp_pose_gripper":
        prev_fraction = snapshot_gripper_fraction(prev_snapshot, prefer_target=True)
        curr_fraction = snapshot_gripper_fraction(curr_snapshot, prefer_target=True)
        gripper_target = fraction_to_gripper_target(curr_fraction)
        return np.asarray(
            [
                cartesian_delta[0],
                cartesian_delta[1],
                cartesian_delta[2],
                cartesian_delta[3],
                cartesian_delta[4],
                cartesian_delta[5],
                gripper_target if abs(curr_fraction - prev_fraction) > 1.0e-6 else fraction_to_gripper_target(prev_fraction),
            ],
            dtype=np.float32,
        )
    return None


def infer_action_with_fallback(
    prev_snapshot: Dict[str, np.ndarray],
    curr_snapshot: Dict[str, np.ndarray],
    config: Dict[str, Any],
    motion_integral: Optional[Dict[str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    inferred_action = infer_action_from_snapshots(prev_snapshot, curr_snapshot, config).astype(np.float32)
    prev_controller_timestamp = extract_controller_timestamp(prev_snapshot)
    curr_controller_timestamp = extract_controller_timestamp(curr_snapshot)
    controller_timestamp_stale = (
        np.isfinite(prev_controller_timestamp)
        and np.isfinite(curr_controller_timestamp)
        and abs(curr_controller_timestamp - prev_controller_timestamp) < 1.0e-9
    )
    fallback_action = None
    if motion_integral is not None and motion_integral.get("last_mode") == "cartesian":
        fallback_action = build_cartesian_fallback_action(
            motion_integral["cartesian_delta"],
            prev_snapshot=prev_snapshot,
            curr_snapshot=curr_snapshot,
            action_mode=get_action_mode(config),
        )
    if (
        fallback_action is not None
        and is_motion_effectively_zero(inferred_action, get_action_mode(config), eps=1.0e-7)
        and controller_timestamp_stale
    ):
        return fallback_action.astype(np.float32), np.asarray([1], dtype=np.int8), True
    return inferred_action, np.asarray([0], dtype=np.int8), controller_timestamp_stale


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    if image.shape[0] == target_height:
        return image
    scale = target_height / max(int(image.shape[0]), 1)
    target_width = max(int(round(image.shape[1] * scale)), 1)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def render_preview(
    observation: Dict[str, Any],
    recording: bool,
    sample_hz: float,
    backend_lines: List[str],
    window_name: str,
) -> None:
    if cv2 is None:
        return
    camera_panels = []
    for camera_name, camera_value in observation.get("raw_cameras", {}).items():
        color = camera_value.get("color")
        if color is None:
            continue
        # RealSense color stream is configured as rs.format.bgr8, which already
        # matches OpenCV's default display format.
        panel = np.asarray(color, dtype=np.uint8)
        panel = resize_to_height(panel, 320)
        cv2.putText(panel, camera_name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        camera_panels.append(panel)
    if not camera_panels:
        return

    canvas = np.hstack(camera_panels)
    footer_lines = [
        f"REC={'ON' if recording else 'OFF'} | sample_hz={sample_hz:.2f} | backend=speed-only",
        f"timestamp={float(observation['timestamp'][0]):.3f}",
    ] + backend_lines
    footer = np.zeros((26 * len(footer_lines) + 8, canvas.shape[1], 3), dtype=np.uint8)
    for idx, line in enumerate(footer_lines):
        cv2.putText(
            footer,
            line,
            (10, 24 + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imshow(window_name, np.vstack([canvas, footer]))
    cv2.waitKey(1)


class VelocityCommandState:
    def __init__(self):
        self._lock = threading.Lock()
        self.cartesian = np.zeros((6,), dtype=np.float64)
        self.joint = np.zeros((6,), dtype=np.float64)
        self.mode = "cartesian"
        self.last_update_monotonic = -1.0
        self.paused = False

    def pause(self) -> None:
        with self._lock:
            self.paused = True
            self.cartesian[:] = 0.0
            self.joint[:] = 0.0
            self.last_update_monotonic = -1.0

    def resume(self) -> None:
        with self._lock:
            self.paused = False
            self.cartesian[:] = 0.0
            self.joint[:] = 0.0
            self.last_update_monotonic = -1.0

    def set_cartesian(self, value: np.ndarray) -> None:
        with self._lock:
            self.mode = "cartesian"
            self.cartesian = np.asarray(value, dtype=np.float64).reshape(6)
            self.joint[:] = 0.0
            self.last_update_monotonic = time.monotonic()

    def set_joint(self, value: np.ndarray) -> None:
        with self._lock:
            self.mode = "joint"
            self.joint = np.asarray(value, dtype=np.float64).reshape(6)
            self.cartesian[:] = 0.0
            self.last_update_monotonic = time.monotonic()

    def stop(self) -> None:
        with self._lock:
            self.cartesian[:] = 0.0
            self.joint[:] = 0.0
            self.last_update_monotonic = -1.0

    def snapshot(self, hold_timeout_s: float) -> Tuple[str, np.ndarray]:
        now = time.monotonic()
        with self._lock:
            if self.paused:
                return self.mode, np.zeros((6,), dtype=np.float64)
            if self.last_update_monotonic < 0.0:
                return self.mode, np.zeros((6,), dtype=np.float64)
            if (now - self.last_update_monotonic) > hold_timeout_s:
                self.cartesian[:] = 0.0
                self.joint[:] = 0.0
                self.last_update_monotonic = -1.0
                return self.mode, np.zeros((6,), dtype=np.float64)
            if self.mode == "joint":
                return "joint", self.joint.copy()
            return "cartesian", self.cartesian.copy()


class SpeedOnlyKeyboardBackend:
    def __init__(self, config: Dict[str, Any], command_state: VelocityCommandState):
        self.config = config
        self.command_state = command_state
        self.robot_cfg = config.get("robot", {})
        teleop_cfg = config.get("speed_teleop", {})
        self.linear_speed_mps = float(teleop_cfg.get("linear_speed_mps", 0.02))
        self.linear_speed_scale = float(teleop_cfg.get("linear_speed_scale", 1.25))
        self.min_linear_speed_mps = float(teleop_cfg.get("min_linear_speed_mps", 0.002))
        self.max_linear_speed_mps = float(teleop_cfg.get("max_linear_speed_mps", 0.08))
        self.angular_speed_rps = float(teleop_cfg.get("angular_speed_rps", 0.25))
        self.angular_speed_scale = float(teleop_cfg.get("angular_speed_scale", 1.25))
        self.min_angular_speed_rps = float(teleop_cfg.get("min_angular_speed_rps", 0.05))
        self.max_angular_speed_rps = float(teleop_cfg.get("max_angular_speed_rps", 1.0))
        self.joint_speed_rps = float(teleop_cfg.get("joint_speed_rps", 0.20))
        self.joint_speed_scale = float(teleop_cfg.get("joint_speed_scale", 1.25))
        self.min_joint_speed_rps = float(teleop_cfg.get("min_joint_speed_rps", 0.05))
        self.max_joint_speed_rps = float(teleop_cfg.get("max_joint_speed_rps", 0.8))
        self.gripper_step = float(teleop_cfg.get("gripper_step", 1.0))
        self.gripper_enabled = bool(self.robot_cfg.get("gripper", {}).get("enabled", False))
        self.last_motion_summary = "停止"

    def reset(self) -> None:
        self.command_state.stop()
        self.last_motion_summary = "停止"

    def stop_motion(self) -> None:
        self.command_state.stop()
        self.last_motion_summary = "停止"

    def print_help(self) -> None:
        print("===== Speed-Only 键盘控制 =====")
        print("按住键盘依靠系统自动连发维持速度，松开后超时自动归零。")
        print("TCP 线速度: w/s, a/d, r/f")
        print("TCP 角速度: i/k, j/l, u/o")
        print("关节速度: 1/2, 3/4, 5/6, 7/8, 9/0, -/=")
        if self.gripper_enabled:
            print("夹爪开合: z/c")
        print("全部停下: Space")
        print("线速度调节: [ / ]")
        print("角速度调节: ; / '")
        print("关节速度调节: , / .")

    def get_status_lines(self) -> List[str]:
        lines = [
            f"linear_speed={self.linear_speed_mps:.3f} m/s",
            f"angular_speed={self.angular_speed_rps:.3f} rad/s",
            f"joint_speed={self.joint_speed_rps:.3f} rad/s",
            f"cmd={self.last_motion_summary}",
        ]
        if self.gripper_enabled:
            lines.append(f"gripper_step={self.gripper_step:.3f}")
        return lines

    def _set_cartesian_velocity(self, vector: np.ndarray, label: str) -> None:
        self.command_state.set_cartesian(vector)
        self.last_motion_summary = label

    def _set_joint_velocity(self, vector: np.ndarray, label: str) -> None:
        self.command_state.set_joint(vector)
        self.last_motion_summary = label

    def _adjust_speed(self, key: str) -> bool:
        if key == "[":
            self.linear_speed_mps = max(self.min_linear_speed_mps, self.linear_speed_mps / self.linear_speed_scale)
            print(f"线速度已调整为 {self.linear_speed_mps:.4f} m/s")
            return True
        if key == "]":
            self.linear_speed_mps = min(self.max_linear_speed_mps, self.linear_speed_mps * self.linear_speed_scale)
            print(f"线速度已调整为 {self.linear_speed_mps:.4f} m/s")
            return True
        if key == ";":
            self.angular_speed_rps = max(self.min_angular_speed_rps, self.angular_speed_rps / self.angular_speed_scale)
            print(f"角速度已调整为 {self.angular_speed_rps:.4f} rad/s")
            return True
        if key == "'":
            self.angular_speed_rps = min(self.max_angular_speed_rps, self.angular_speed_rps * self.angular_speed_scale)
            print(f"角速度已调整为 {self.angular_speed_rps:.4f} rad/s")
            return True
        if key == ",":
            self.joint_speed_rps = max(self.min_joint_speed_rps, self.joint_speed_rps / self.joint_speed_scale)
            print(f"关节速度已调整为 {self.joint_speed_rps:.4f} rad/s")
            return True
        if key == ".":
            self.joint_speed_rps = min(self.max_joint_speed_rps, self.joint_speed_rps * self.joint_speed_scale)
            print(f"关节速度已调整为 {self.joint_speed_rps:.4f} rad/s")
            return True
        return False

    def handle_key(self, key: str, robot) -> bool:
        if self._adjust_speed(key):
            return True

        if key == " ":
            self.stop_motion()
            return True

        cartesian_map = {
            "w": (0, self.linear_speed_mps, "+X"),
            "s": (0, -self.linear_speed_mps, "-X"),
            "a": (1, self.linear_speed_mps, "+Y"),
            "d": (1, -self.linear_speed_mps, "-Y"),
            "r": (2, self.linear_speed_mps, "+Z"),
            "f": (2, -self.linear_speed_mps, "-Z"),
            "i": (3, self.angular_speed_rps, "+Rx"),
            "k": (3, -self.angular_speed_rps, "-Rx"),
            "j": (4, self.angular_speed_rps, "+Ry"),
            "l": (4, -self.angular_speed_rps, "-Ry"),
            "u": (5, self.angular_speed_rps, "+Rz"),
            "o": (5, -self.angular_speed_rps, "-Rz"),
        }
        if key in cartesian_map:
            index, value, label = cartesian_map[key]
            vector = np.zeros((6,), dtype=np.float64)
            vector[index] = value
            self._set_cartesian_velocity(vector, f"cartesian {label}")
            return True

        joint_map = {
            "1": (0, self.joint_speed_rps, "q0+"),
            "2": (0, -self.joint_speed_rps, "q0-"),
            "3": (1, self.joint_speed_rps, "q1+"),
            "4": (1, -self.joint_speed_rps, "q1-"),
            "5": (2, self.joint_speed_rps, "q2+"),
            "6": (2, -self.joint_speed_rps, "q2-"),
            "7": (3, self.joint_speed_rps, "q3+"),
            "8": (3, -self.joint_speed_rps, "q3-"),
            "9": (4, self.joint_speed_rps, "q4+"),
            "0": (4, -self.joint_speed_rps, "q4-"),
            "-": (5, self.joint_speed_rps, "q5+"),
            "=": (5, -self.joint_speed_rps, "q5-"),
        }
        if key in joint_map:
            index, value, label = joint_map[key]
            vector = np.zeros((6,), dtype=np.float64)
            vector[index] = value
            self._set_joint_velocity(vector, f"joint {label}")
            return True

        if key in {"z", "c"} and self.gripper_enabled:
            snapshot = robot.get_robot_snapshot()
            current_fraction = float(snapshot["gripper_fraction"][0])
            if key == "z":
                target_fraction = min(1.0, current_fraction + self.gripper_step)
            else:
                target_fraction = max(0.0, current_fraction - self.gripper_step)
            robot.set_gripper_fraction(target_fraction, wait=False)
            self.last_motion_summary = f"gripper={target_fraction:.2f}"
            return True

        return False


class SpeedControlLoop:
    def __init__(self, robot, config: Dict[str, Any], command_state: VelocityCommandState, dry_run: bool):
        self.robot = robot
        self.config = config
        self.command_state = command_state
        self.dry_run = dry_run
        speed_cfg = config.get("speed_control", {})
        self.control_hz = max(float(speed_cfg.get("control_hz", 100.0)), 5.0)
        self.control_dt = 1.0 / self.control_hz
        self.hold_timeout_s = float(speed_cfg.get("key_hold_timeout_s", 0.12))
        self.cartesian_acceleration = float(speed_cfg.get("cartesian_acceleration", 0.5))
        self.joint_acceleration = float(speed_cfg.get("joint_acceleration", 1.0))
        self.max_linear_speed_mps = float(speed_cfg.get("max_linear_speed_mps", 0.08))
        self.max_angular_speed_rps = float(speed_cfg.get("max_angular_speed_rps", 1.0))
        self.max_joint_speed_rps = float(speed_cfg.get("max_joint_speed_rps", 0.8))
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_nonzero_mode: Optional[str] = None
        self._last_error: Optional[str] = None
        self._lock = threading.Lock()
        self._integrated_cartesian_delta = np.zeros((6,), dtype=np.float64)
        self._integrated_joint_delta = np.zeros((6,), dtype=np.float64)
        self._last_applied_mode: Optional[str] = None
        self._last_applied_command = np.zeros((6,), dtype=np.float64)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="speed-control-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._send_stop()

    def pause(self) -> None:
        self.command_state.pause()
        self._send_stop()

    def resume(self) -> None:
        self.command_state.resume()
        self._last_nonzero_mode = None

    def get_last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def consume_motion_integral(self) -> Dict[str, np.ndarray]:
        with self._lock:
            payload = {
                "cartesian_delta": self._integrated_cartesian_delta.copy(),
                "joint_delta": self._integrated_joint_delta.copy(),
                "last_mode": self._last_applied_mode,
                "last_command": self._last_applied_command.copy(),
            }
            self._integrated_cartesian_delta[:] = 0.0
            self._integrated_joint_delta[:] = 0.0
            return payload

    def _set_error(self, message: str) -> None:
        with self._lock:
            if self._last_error is None:
                self._last_error = message

    def _send_stop(self) -> None:
        if self.dry_run:
            self.robot.tcp_speed[:] = 0.0
            return
        control = getattr(self.robot, "control", None)
        if control is None:
            return
        try:
            if hasattr(control, "speedStop"):
                control.speedStop()
        except Exception:
            pass

    def _clip_cartesian_speed(self, command: np.ndarray) -> np.ndarray:
        linear = clip_translation(command[:3], max_norm=self.max_linear_speed_mps)
        angular = clip_rotation(command[3:6], max_norm=self.max_angular_speed_rps)
        return np.concatenate([linear, angular], axis=0).astype(np.float64)

    def _clip_joint_speed(self, command: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(command, dtype=np.float64), -self.max_joint_speed_rps, self.max_joint_speed_rps)

    def _apply_dry_run_cartesian(self, command: np.ndarray) -> None:
        current_pose = self.robot.tcp_pose.astype(np.float64)
        delta_xyz = np.asarray(command[:3], dtype=np.float64) * self.control_dt
        delta_rotvec = np.asarray(command[3:6], dtype=np.float64) * self.control_dt
        workspace_min = np.asarray(self.robot.robot_cfg["workspace_min"], dtype=np.float64)
        workspace_max = np.asarray(self.robot.robot_cfg["workspace_max"], dtype=np.float64)
        next_pose = current_pose.copy()
        next_pose[:3] = np.clip(next_pose[:3] + delta_xyz, workspace_min, workspace_max)
        if float(np.linalg.norm(delta_rotvec)) >= 1.0e-12:
            current_rotation = rotvec_to_rotation_matrix(current_pose[3:6])
            delta_rotation = rotvec_to_rotation_matrix(delta_rotvec)
            rotation_delta_frame = str(self.robot.robot_cfg.get("rotation_delta_frame", "base")).lower()
            if rotation_delta_frame == "tool":
                target_rotation = current_rotation @ delta_rotation
            else:
                target_rotation = delta_rotation @ current_rotation
            next_pose[3:6] = align_rotvec_to_reference(rotation_matrix_to_rotvec(target_rotation), current_pose[3:6])
        self.robot.tcp_pose[:] = next_pose.astype(np.float32)
        self.robot.tcp_speed[:] = np.asarray(command, dtype=np.float32)

    def _apply_dry_run_joint(self, command: np.ndarray) -> None:
        delta_q = np.asarray(command, dtype=np.float64) * self.control_dt
        self.robot.joint_positions[:] = (self.robot.joint_positions.astype(np.float64) + delta_q).astype(np.float32)
        self.robot.tcp_speed[:] = 0.0

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            next_tick += self.control_dt
            try:
                mode, command = self.command_state.snapshot(self.hold_timeout_s)
                if mode == "joint":
                    command = self._clip_joint_speed(command)
                else:
                    command = self._clip_cartesian_speed(command)
                nonzero = bool(np.any(np.abs(command) > 1.0e-9))

                if self.dry_run:
                    if nonzero:
                        if mode == "joint":
                            self._apply_dry_run_joint(command)
                        else:
                            self._apply_dry_run_cartesian(command)
                        with self._lock:
                            if mode == "joint":
                                self._integrated_joint_delta += command * self.control_dt
                            else:
                                self._integrated_cartesian_delta += command * self.control_dt
                            self._last_applied_mode = mode
                            self._last_applied_command = command.copy()
                    else:
                        self.robot.tcp_speed[:] = 0.0
                        with self._lock:
                            self._last_applied_mode = None
                            self._last_applied_command[:] = 0.0
                    self._last_nonzero_mode = mode if nonzero else None
                else:
                    control = getattr(self.robot, "control", None)
                    if control is None:
                        raise RuntimeError("speed-only 控制线程未拿到 RTDEControlInterface")
                    if nonzero:
                        if self._last_nonzero_mode is not None and self._last_nonzero_mode != mode:
                            self._send_stop()
                            time.sleep(min(self.control_dt, 0.01))
                        if mode == "joint":
                            control.speedJ(command.tolist(), self.joint_acceleration, self.control_dt)
                        else:
                            control.speedL(command.tolist(), self.cartesian_acceleration, self.control_dt)
                        with self._lock:
                            if mode == "joint":
                                self._integrated_joint_delta += command * self.control_dt
                            else:
                                self._integrated_cartesian_delta += command * self.control_dt
                            self._last_applied_mode = mode
                            self._last_applied_command = command.copy()
                        self._last_nonzero_mode = mode
                    elif self._last_nonzero_mode is not None:
                        self._send_stop()
                        with self._lock:
                            self._last_applied_mode = None
                            self._last_applied_command[:] = 0.0
                        self._last_nonzero_mode = None
            except Exception as exc:  # pragma: no cover
                self._set_error(str(exc))
                self._send_stop()
                break

            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()


def print_common_help(sample_hz: float, control_hz: float, obs_mode: str, action_mode: str) -> None:
    print("")
    print("===== Speed-Only 固定频率采集 =====")
    print(f"采样频率: {sample_hz:.2f} Hz")
    print(f"速度控制频率: {control_hz:.2f} Hz")
    print(f"观测表示: {obs_mode}")
    print(f"动作表示: {action_mode}")
    print("")
    print("===== 轨迹控制 =====")
    print("b : 自动回 home 后开始录制新轨迹")
    print("v : 保存当前轨迹为成功")
    print("n : 保存当前轨迹为失败")
    print("x : 丢弃当前轨迹")
    print("h : 未录制状态下手动回 home")
    print("Space : 立即清零当前速度")
    print("q : 退出程序")
    print("")


def main():
    args = parse_args()
    config = load_json(args.config)
    config = resolve_collection_paths(config, args.config)
    action_mode = get_action_mode(config)
    obs_mode = get_obs_mode(config)
    task_cfg = get_task_config(config)

    collection_cfg = config.setdefault("collection", {})
    sample_hz = float(args.sample_hz or collection_cfg.get("sample_hz", 10.0))
    if sample_hz <= 0:
        raise ValueError("sample_hz 必须大于 0")
    sample_period = 1.0 / sample_hz

    speed_cfg = config.setdefault("speed_control", {})
    control_hz = float(args.control_hz or speed_cfg.get("control_hz", 100.0))
    if control_hz <= 0:
        raise ValueError("control_hz 必须大于 0")
    speed_cfg["control_hz"] = control_hz

    move_home_on_start = bool(
        args.move_home_on_start
        or collection_cfg.get(
            "move_home_on_start",
            config.get("robot", {}).get("home_joint_rad") is not None or config.get("robot", {}).get("home_tcp_pose") is not None,
        )
    )
    auto_home_before_episode = bool(
        collection_cfg.get(
            "auto_home_before_episode",
            config.get("robot", {}).get("home_joint_rad") is not None or config.get("robot", {}).get("home_tcp_pose") is not None,
        )
    )
    save_failures_to_zarr = bool(collection_cfg.get("save_failures_to_zarr", False))
    home_settle_s = float(collection_cfg.get("home_settle_s", 1.0))
    show_camera_preview = bool(collection_cfg.get("show_camera_preview", True))
    preview_window_name = str(collection_cfg.get("preview_window_name", "UR12E Speed-Only Collection"))
    if show_camera_preview and cv2 is None:
        print("警告: 未安装 OpenCV，无法显示相机实时画面，将继续采集但不显示预览。")
        show_camera_preview = False

    camera_manager = MultiRealSenseManager(config)
    robot = make_robot_controller(config, dry_run=args.dry_run)
    writer = RealRobotDatasetWriter(config["dataset"]["zarr_path"], overwrite=args.overwrite)
    raw_writer = None
    raw_root = config.get("dataset", {}).get("raw_root")
    if raw_root:
        raw_writer = RealRobotRawDatasetWriter(raw_root=raw_root, config=config)

    command_state = VelocityCommandState()
    backend = SpeedOnlyKeyboardBackend(config, command_state)
    control_loop = SpeedControlLoop(robot=robot, config=config, command_state=command_state, dry_run=args.dry_run)

    current_episode = empty_episode()
    current_raw_episode: List[Dict[str, Any]] = []
    recording = False
    episode_start_time = None
    should_quit = False
    prev_observation = None
    prev_snapshot = None
    stale_snapshot_warning_count = 0

    print(f"zarr 数据集路径: {config['dataset']['zarr_path']}")
    if raw_writer is not None:
        print(f"raw session 根目录: {raw_writer.session_dir}")
    print(f"task: {task_cfg['task_id']} | {task_cfg['name']}")
    if task_cfg.get("description"):
        print(f"task_desc: {task_cfg['description']}")
    print_common_help(sample_hz=sample_hz, control_hz=control_hz, obs_mode=obs_mode, action_mode=action_mode)
    backend.print_help()

    next_tick = time.monotonic()
    try:
        camera_manager.start()
        robot.connect()
        control_loop.start()
        control_loop.pause()

        if move_home_on_start:
            print("正在回到 home 位...")
            robot.move_home()
            time.sleep(home_settle_s)
        control_loop.resume()

        with KeyPoller() as key_poller:
            while not should_quit:
                error_message = control_loop.get_last_error()
                if error_message is not None:
                    raise RuntimeError(f"速度控制线程异常: {error_message}")

                while True:
                    key = key_poller.poll()
                    if key is None:
                        break
                    key = key.lower()

                    if key == "q":
                        print("收到退出指令，结束采集。")
                        should_quit = True
                        break

                    if key == "h":
                        if recording:
                            print("当前正在录制，禁止在轨迹中回 home。请先按 v / n / x 结束当前轨迹。")
                            continue
                        backend.stop_motion()
                        control_loop.pause()
                        print("正在回到 home 位...")
                        robot.move_home()
                        time.sleep(home_settle_s)
                        control_loop.resume()
                        prev_observation = None
                        prev_snapshot = None
                        continue

                    if key == "b":
                        if recording:
                            print("当前轨迹还未结束。请先按 v 保存、n 标记失败或 x 丢弃。")
                            continue
                        backend.stop_motion()
                        control_loop.pause()
                        if auto_home_before_episode:
                            print("开始录制前先回到 home 位...")
                            robot.move_home()
                            time.sleep(home_settle_s)
                        current_episode = empty_episode()
                        current_raw_episode = []
                        recording = True
                        episode_start_time = time.time()
                        prev_observation = None
                        prev_snapshot = None
                        control_loop.resume()
                        print("已开始记录新轨迹。")
                        continue

                    if key == "x":
                        current_episode = empty_episode()
                        current_raw_episode = []
                        recording = False
                        episode_start_time = None
                        prev_observation = None
                        prev_snapshot = None
                        backend.stop_motion()
                        print("当前轨迹已丢弃。")
                        continue

                    if key in {"v", "n"}:
                        if not recording or len(current_episode["state"]) == 0:
                            print("当前没有可保存的轨迹。")
                            continue
                        backend.stop_motion()
                        time.sleep(min(sample_period, 0.05))
                        if prev_observation is not None and prev_snapshot is not None:
                            final_host_timestamp = time.time()
                            final_snapshot = robot.get_robot_snapshot()
                            final_observation = capture_robot_observation(
                                robot,
                                camera_manager,
                                robot_snapshot=final_snapshot,
                                host_timestamp=final_host_timestamp,
                            )
                            final_motion_integral = control_loop.consume_motion_integral()
                            final_action, final_action_source_flag, final_snapshot_stale = infer_action_with_fallback(
                                prev_snapshot=prev_snapshot,
                                curr_snapshot=final_snapshot,
                                config=config,
                                motion_integral=final_motion_integral,
                            )
                            final_motion_changed = not is_motion_effectively_zero(final_action, action_mode, eps=1.0e-7)
                            final_gripper_changed = has_gripper_target_changed(prev_snapshot, final_snapshot)
                            if final_motion_changed or final_gripper_changed:
                                dataset_step = build_dataset_step(prev_observation, final_action)
                                dataset_step["action_source_flag"] = final_action_source_flag
                                append_step(current_episode, dataset_step)
                                raw_step = build_raw_step(prev_observation, final_action)
                                raw_step["action_source_flag"] = final_action_source_flag
                                current_raw_episode.append(raw_step)
                                print(f"保存前补写尾帧: {format_action_summary(final_action, action_mode)}")
                                prev_observation = final_observation
                                prev_snapshot = final_snapshot
                            elif final_snapshot_stale:
                                print("警告: 保存前检测到 RTDE 快照时间戳未更新，尾帧未写入。")
                        success = key == "v"
                        episode_metadata = {
                            "success": success,
                            "saved_at_unix": time.time(),
                            "started_at_unix": episode_start_time,
                            "dry_run": bool(args.dry_run),
                            "task_id": task_cfg["task_id"],
                            "task_name": task_cfg["name"],
                            "obs_mode": obs_mode,
                            "action_mode": action_mode,
                            "backend": "speed_only_keyboard",
                            "sample_hz": sample_hz,
                            "control_hz": control_hz,
                        }
                        wrote_to_zarr = False
                        if success or save_failures_to_zarr:
                            episode_metadata = writer.add_episode(current_episode, metadata=episode_metadata)
                            wrote_to_zarr = True
                        if raw_writer is not None:
                            raw_metadata = dict(episode_metadata)
                            raw_metadata["stored_in_zarr"] = wrote_to_zarr
                            raw_writer.add_episode(current_raw_episode, metadata=raw_metadata)
                        print(
                            f"轨迹已保存: success={success}, length={len(current_episode['state'])}, "
                            f"zarr={'yes' if wrote_to_zarr else 'no'}"
                        )
                        current_episode = empty_episode()
                        current_raw_episode = []
                        recording = False
                        episode_start_time = None
                        prev_observation = None
                        prev_snapshot = None
                        backend.stop_motion()
                        continue

                    if backend.handle_key(key, robot):
                        continue

                if should_quit:
                    break

                now_monotonic = time.monotonic()
                if now_monotonic < next_tick:
                    time.sleep(next_tick - now_monotonic)
                next_tick += sample_period

                host_timestamp = time.time()
                robot_snapshot = robot.get_robot_snapshot()
                motion_integral = control_loop.consume_motion_integral()
                observation = capture_robot_observation(
                    robot,
                    camera_manager,
                    robot_snapshot=robot_snapshot,
                    host_timestamp=host_timestamp,
                )

                if show_camera_preview:
                    render_preview(
                        observation=observation,
                        recording=recording,
                        sample_hz=sample_hz,
                        backend_lines=backend.get_status_lines(),
                        window_name=preview_window_name,
                    )

                if prev_observation is None:
                    prev_observation = observation
                    prev_snapshot = robot_snapshot
                    if recording:
                        print("speed-only 模式已对齐首帧，从下一采样周期开始写入轨迹。")
                    continue

                inferred_action, action_source_flag, controller_timestamp_stale = infer_action_with_fallback(
                    prev_snapshot=prev_snapshot,
                    curr_snapshot=robot_snapshot,
                    config=config,
                    motion_integral=motion_integral,
                )
                if int(action_source_flag[0]) == 1:
                    stale_snapshot_warning_count += 1
                    if stale_snapshot_warning_count <= 5 or stale_snapshot_warning_count % 100 == 0:
                        print(
                            "警告: RTDE 接收快照未更新，但速度控制线程仍在持续输出笛卡尔速度；"
                            "本条样本已使用 speed-only 控制积分作为动作回退。"
                        )
                elif controller_timestamp_stale:
                    stale_snapshot_warning_count += 1
                    if stale_snapshot_warning_count <= 5 or stale_snapshot_warning_count % 100 == 0:
                        print("警告: RTDE 快照时间戳未更新，本条样本动作可能失真。")
                else:
                    stale_snapshot_warning_count = 0
                if recording:
                    dataset_step = build_dataset_step(prev_observation, inferred_action)
                    dataset_step["action_source_flag"] = action_source_flag
                    append_step(current_episode, dataset_step)
                    raw_step = build_raw_step(prev_observation, inferred_action)
                    raw_step["action_source_flag"] = action_source_flag
                    current_raw_episode.append(raw_step)
                    summary = format_action_summary(inferred_action, action_mode)
                    debug_summary = format_snapshot_debug(prev_snapshot, robot_snapshot, action_source_flag)
                    print(f"已记录第 {len(current_episode['state'])} 步: {summary} | {debug_summary}")

                prev_observation = observation
                prev_snapshot = robot_snapshot
    finally:
        backend.stop_motion()
        control_loop.stop()
        camera_manager.stop()
        robot.disconnect()
        if show_camera_preview and cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
