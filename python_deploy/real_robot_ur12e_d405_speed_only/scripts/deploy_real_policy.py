import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import dill
import numpy as np
import torch

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
MP1_ROOT = PROJECT_ROOT / "MP1"
for path in [THIS_DIR, PROJECT_ROOT, MP1_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_robot_utils import (  # noqa: E402
    KeyPoller,
    MultiRealSenseManager,
    align_rotvec_to_reference,
    capture_robot_observation,
    clip_rotation,
    clip_translation,
    format_action_summary,
    get_action_mode,
    load_json,
    make_robot_controller,
    resolve_collection_paths,
    rotation_matrix_to_rotvec,
    rotvec_to_rotation_matrix,
)

from train_real import TrainMP1Workspace  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a trained MP1 real-robot policy and record real-world success rate."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a .ckpt checkpoint saved by train_real.py.")
    parser.add_argument("--config", required=True, help="Real robot collection/deployment JSON config.")
    parser.add_argument("--output-dir", default="deploy_results", help="Directory for trial logs.")
    parser.add_argument("--device", default="cuda:0", help="Torch device used for policy inference.")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--max-episode-s", type=float, default=180.0)
    parser.add_argument("--control-hz", type=float, default=5.0, help="Policy/action execution frequency.")
    parser.add_argument("--move-home-before-trial", action="store_true", default=True)
    parser.add_argument("--no-move-home-before-trial", dest="move_home_before_trial", action="store_false")
    parser.add_argument("--home-settle-s", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--no-use-ema", dest="use_ema", action="store_false")
    parser.add_argument("--show-preview", action="store_true", default=True)
    parser.add_argument("--no-show-preview", dest="show_preview", action="store_false")
    parser.add_argument("--global-camera", default="global_d405")
    parser.add_argument("--wrist-camera", default="wrist_d405")
    parser.add_argument("--convert-bgr-to-rgb", action="store_true", default=True)
    parser.add_argument("--keep-bgr", dest="convert_bgr_to_rgb", action="store_false")
    parser.add_argument("--max-translation-per-step-m", type=float, default=None)
    parser.add_argument("--max-rotation-per-step-rad", type=float, default=None)
    parser.add_argument(
        "--translation-ema-alpha",
        type=float,
        default=1.0,
        help="EMA smoothing factor for delta_xyz. 1.0 disables smoothing.",
    )
    parser.add_argument(
        "--rotation-ema-alpha",
        type=float,
        default=1.0,
        help="EMA smoothing factor for delta_rotvec. 1.0 disables smoothing.",
    )
    parser.add_argument(
        "--gripper-ema-alpha",
        type=float,
        default=1.0,
        help="EMA smoothing factor for gripper target. 1.0 disables smoothing.",
    )
    parser.add_argument(
        "--translation-deadband-m",
        type=float,
        default=0.0,
        help="Zero delta_xyz when its norm is below this threshold after smoothing.",
    )
    parser.add_argument(
        "--rotation-deadband-rad",
        type=float,
        default=0.0,
        help="Zero delta_rotvec when its norm is below this threshold after smoothing.",
    )
    parser.add_argument(
        "--gripper-deadband",
        type=float,
        default=0.0,
        help="Hold previous gripper target when change is below this threshold after smoothing.",
    )
    parser.add_argument("--save-step-trace", action="store_true", default=True)
    parser.add_argument("--no-save-step-trace", dest="save_step_trace", action="store_false")
    parser.add_argument("--save-policy-dumps", action="store_true", default=True)
    parser.add_argument(
        "--policy-dump-every-steps",
        type=int,
        default=10,
        help="When save-policy-dumps is enabled, save one compressed input/output dump every N control steps.",
    )
    parser.add_argument(
        "--action-rotation-correction-rpy-deg",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("ROLL", "PITCH", "YAW"),
        help="Optional correction applied to predicted delta_xyz and delta_rotvec before execution.",
    )
    parser.add_argument(
        "--success-definition",
        default="task completed without collision or object drop",
        help="Text written into the result log. Success is still marked by operator key press.",
    )
    return parser.parse_args()


def load_workspace_policy(checkpoint_path: Path, device: torch.device, use_ema: bool):
    if checkpoint_path.suffix.lower() not in {".ckpt", ".pth", ".pt"}:
        raise ValueError(f"Unsupported checkpoint suffix: {checkpoint_path.suffix}")

    payload = torch.load(checkpoint_path.open("rb"), pickle_module=dill, map_location="cpu")
    if not isinstance(payload, dict) or "cfg" not in payload or "state_dicts" not in payload:
        raise ValueError(
            "This file is not a full train_real.py checkpoint. "
            "Deployment needs cfg + policy weights + normalizer. Use latest.ckpt or a top-k .ckpt."
        )

    workspace = TrainMP1Workspace(payload["cfg"])
    workspace.load_payload(payload)
    policy = workspace.ema_model if use_ema and workspace.ema_model is not None else workspace.model
    policy.eval()
    policy.to(device)
    return workspace.cfg, policy


def rpy_deg_to_matrix(rpy_deg: List[float]) -> np.ndarray:
    roll, pitch, yaw = [math.radians(float(v)) for v in rpy_deg]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def resize_chw_rgb(image_bgr: np.ndarray, shape_chw: List[int], convert_bgr_to_rgb: bool) -> np.ndarray:
    if cv2 is None:
        raise ModuleNotFoundError("OpenCV is required for live image resizing.")
    channels, height, width = [int(v) for v in shape_chw]
    if channels != 3:
        raise ValueError(f"Only 3-channel RGB images are supported, got shape {shape_chw}")
    image = np.asarray(image_bgr, dtype=np.uint8)
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if convert_bgr_to_rgb:
        resized = resized[..., ::-1]
    return np.transpose(resized, (2, 0, 1)).astype(np.uint8)


def camera_color(observation: Dict[str, Any], camera_name: str) -> np.ndarray:
    raw_cameras = observation.get("raw_cameras", {})
    if camera_name not in raw_cameras:
        raise KeyError(f"Camera {camera_name!r} missing. Available: {sorted(raw_cameras.keys())}")
    color = raw_cameras[camera_name].get("color")
    if color is None:
        raise RuntimeError(f"Camera {camera_name!r} returned no color frame.")
    return np.asarray(color, dtype=np.uint8)


def build_policy_observation(
    observation: Dict[str, Any],
    obs_shape_meta: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    result: Dict[str, np.ndarray] = {}
    for key, meta in obs_shape_meta.items():
        shape = list(meta.get("shape", []))
        if key == "point_cloud":
            value = np.asarray(observation["point_cloud"], dtype=np.float32)
        elif key == "agent_pos":
            value = np.asarray(observation["agent_pos"], dtype=np.float32)
            expected_dim = int(shape[0]) if len(shape) == 1 else None
            if expected_dim is not None and value.ndim == 1 and value.shape[0] + 1 == expected_dim:
                gripper_value = observation.get("gripper_target_fraction", observation.get("gripper_fraction"))
                if gripper_value is None:
                    raise ValueError(
                        f"Observation {key} expected {expected_dim} dims, got {value.shape[0]} and no gripper state is available."
                    )
                gripper_scalar = float(np.asarray(gripper_value, dtype=np.float32).reshape(-1)[0])
                value = np.concatenate([value, np.asarray([gripper_scalar], dtype=np.float32)], axis=0)
        elif key == "global_image":
            value = resize_chw_rgb(
                camera_color(observation, args.global_camera),
                shape_chw=shape,
                convert_bgr_to_rgb=args.convert_bgr_to_rgb,
            )
        elif key == "wrist_image":
            value = resize_chw_rgb(
                camera_color(observation, args.wrist_camera),
                shape_chw=shape,
                convert_bgr_to_rgb=args.convert_bgr_to_rgb,
            )
        else:
            raise KeyError(f"Live observation key {key!r} is not implemented.")

        if tuple(value.shape) != tuple(shape):
            raise ValueError(f"Observation {key} shape mismatch: expected {shape}, got {value.shape}")
        result[key] = value
    return result


def stack_obs_buffer(obs_buffer: Deque[Dict[str, np.ndarray]], device: torch.device) -> Dict[str, torch.Tensor]:
    keys = obs_buffer[0].keys()
    batch = {}
    for key in keys:
        array = np.stack([obs[key] for obs in obs_buffer], axis=0)
        tensor = torch.from_numpy(array).unsqueeze(0).to(device)
        if tensor.dtype == torch.uint8:
            batch[key] = tensor
        else:
            batch[key] = tensor.float()
    return batch


def correct_action(action: np.ndarray, correction_matrix: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if action.shape[0] < 6:
        return action
    action[:3] = (correction_matrix @ action[:3].astype(np.float64)).astype(np.float32)
    action[3:6] = (correction_matrix @ action[3:6].astype(np.float64)).astype(np.float32)
    return action


def safety_clip_action(
    action: np.ndarray,
    max_translation_m: Optional[float],
    max_rotation_rad: Optional[float],
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if max_translation_m is not None and action.shape[0] >= 3:
        action[:3] = clip_translation(action[:3], max_norm=float(max_translation_m))
    if max_rotation_rad is not None and action.shape[0] >= 6:
        action[3:6] = clip_rotation(action[3:6], max_norm=float(max_rotation_rad))
    return action


def _ema_blend(current: np.ndarray, previous: Optional[np.ndarray], alpha: float) -> np.ndarray:
    current = np.asarray(current, dtype=np.float32)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if previous is None or alpha >= 1.0:
        return current.copy()
    previous = np.asarray(previous, dtype=np.float32)
    if previous.shape != current.shape:
        return current.copy()
    return (alpha * current + (1.0 - alpha) * previous).astype(np.float32)


def smooth_action(
    action: np.ndarray,
    previous_action: Optional[np.ndarray],
    args: argparse.Namespace,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    previous = None if previous_action is None else np.asarray(previous_action, dtype=np.float32).reshape(-1)
    if previous is not None and previous.shape != action.shape:
        previous = None

    if action.shape[0] >= 3:
        prev_xyz = None if previous is None else previous[:3]
        action[:3] = _ema_blend(action[:3], prev_xyz, args.translation_ema_alpha)
        if float(np.linalg.norm(action[:3])) < float(args.translation_deadband_m):
            action[:3] = 0.0

    if action.shape[0] >= 6:
        prev_rot = None if previous is None else previous[3:6]
        action[3:6] = _ema_blend(action[3:6], prev_rot, args.rotation_ema_alpha)
        if float(np.linalg.norm(action[3:6])) < float(args.rotation_deadband_rad):
            action[3:6] = 0.0

    if action.shape[0] in (4, 7):
        gripper_idx = action.shape[0] - 1
        prev_gripper = None if previous is None else previous[gripper_idx : gripper_idx + 1]
        smoothed_gripper = _ema_blend(action[gripper_idx : gripper_idx + 1], prev_gripper, args.gripper_ema_alpha)
        if (
            prev_gripper is not None
            and float(np.abs(smoothed_gripper[0] - prev_gripper[0])) < float(args.gripper_deadband)
        ):
            smoothed_gripper[0] = prev_gripper[0]
        action[gripper_idx] = smoothed_gripper[0]

    return action


def safe_stop_robot(robot) -> None:
    control = getattr(robot, "control", None)
    if control is None:
        return
    for name in ["speedStop", "stopL", "stopJ"]:
        fn = getattr(control, name, None)
        if fn is None:
            continue
        try:
            fn()
            return
        except TypeError:
            try:
                fn(0.5)
                return
            except Exception:
                pass
        except Exception:
            pass


def render_preview(observation: Dict[str, Any], action: np.ndarray, trial_idx: int, elapsed_s: float) -> None:
    if cv2 is None:
        return
    panels = []
    for camera_name in sorted(observation.get("raw_cameras", {}).keys()):
        color = observation["raw_cameras"][camera_name].get("color")
        if color is None:
            continue
        panel = np.asarray(color, dtype=np.uint8)
        h, w = panel.shape[:2]
        scale = 320.0 / max(h, 1)
        panel = cv2.resize(panel, (max(1, int(w * scale)), 320), interpolation=cv2.INTER_AREA)
        cv2.putText(panel, camera_name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    if not panels:
        return
    canvas = np.hstack(panels)
    footer = np.zeros((64, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        footer,
        f"trial={trial_idx} elapsed={elapsed_s:.1f}s | s=success f=failure space=abort q=quit",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        "action " + np.array2string(action, precision=4, suppress_small=True),
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow("MP1 Real Policy Deployment", np.vstack([canvas, footer]))
    cv2.waitKey(1)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def summarize_numeric_array(value: Any) -> Dict[str, Any]:
    array = np.asarray(value)
    if array.size == 0:
        return {
            "shape": list(array.shape),
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    array_f = array.astype(np.float32, copy=False)
    return {
        "shape": list(array.shape),
        "min": float(np.min(array_f)),
        "max": float(np.max(array_f)),
        "mean": float(np.mean(array_f)),
        "std": float(np.std(array_f)),
    }


def sign_flip_axes(current: np.ndarray, previous: Optional[np.ndarray], threshold: float = 1.0e-4) -> List[int]:
    if previous is None:
        return []
    current = np.asarray(current, dtype=np.float32).reshape(-1)
    previous = np.asarray(previous, dtype=np.float32).reshape(-1)
    if current.shape != previous.shape:
        return []
    mask = (
        (np.abs(current) > float(threshold))
        & (np.abs(previous) > float(threshold))
        & (np.sign(current) != np.sign(previous))
    )
    return [int(idx) for idx in np.flatnonzero(mask)]


def camera_trace_summary(observation: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for camera_name, camera_value in observation.get("raw_cameras", {}).items():
        color = camera_value.get("color")
        depth = camera_value.get("depth")
        point_cloud = camera_value.get("point_cloud")
        summary[camera_name] = {
            "color_shape": list(np.asarray(color).shape) if color is not None else None,
            "depth_shape": list(np.asarray(depth).shape) if depth is not None else None,
            "point_cloud_shape": list(np.asarray(point_cloud).shape) if point_cloud is not None else None,
            "host_capture_unix": float(camera_value.get("host_capture_unix", np.nan)),
            "depth_frame_timestamp_ms": float(camera_value.get("depth_frame_timestamp_ms", np.nan)),
            "color_frame_timestamp_ms": float(camera_value.get("color_frame_timestamp_ms", np.nan)),
            "depth_frame_number": int(camera_value.get("depth_frame_number", -1)),
            "color_frame_number": int(camera_value.get("color_frame_number", -1)),
        }
    return summary


def save_policy_dump(
    dump_dir: Path,
    trial_idx: int,
    step_idx: int,
    model_obs: Dict[str, torch.Tensor],
    policy_result: Dict[str, torch.Tensor],
    observation: Dict[str, Any],
    action_raw: np.ndarray,
    action_executed: np.ndarray,
    execute_info: Dict[str, Any],
    elapsed_s: float,
) -> Path:
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / f"step_{step_idx:05d}.npz"
    payload: Dict[str, Any] = {
        "trial_idx": np.asarray([trial_idx], dtype=np.int32),
        "step_idx": np.asarray([step_idx], dtype=np.int32),
        "elapsed_s": np.asarray([elapsed_s], dtype=np.float32),
        "action_raw": np.asarray(action_raw, dtype=np.float32),
        "action_executed": np.asarray(action_executed, dtype=np.float32),
        "point_cloud_live": np.asarray(observation["point_cloud"], dtype=np.float32),
        "agent_pos_live": np.asarray(observation["agent_pos"], dtype=np.float32),
        "tcp_pose_live": np.asarray(observation["tcp_pose"], dtype=np.float32),
        "joint_positions_live": np.asarray(observation["joint_positions"], dtype=np.float32),
        "tcp_speed_live": np.asarray(observation["tcp_speed"], dtype=np.float32),
        "gripper_fraction_live": np.asarray(observation["gripper_fraction"], dtype=np.float32),
        "gripper_target_fraction_live": np.asarray(observation["gripper_target_fraction"], dtype=np.float32),
        "controller_timestamp_live": np.asarray(observation["controller_timestamp"], dtype=np.float64),
        "observation_timestamp_live": np.asarray(observation["timestamp"], dtype=np.float64),
        "capture_completed_timestamp_live": np.asarray(observation["capture_completed_timestamp"], dtype=np.float64),
    }
    for key, value in model_obs.items():
        payload[f"model_obs_{key}"] = value.detach().cpu().numpy()[0]
    for key, value in policy_result.items():
        if torch.is_tensor(value):
            payload[f"policy_{key}"] = value.detach().cpu().numpy()
    for key, value in execute_info.items():
        payload[f"execute_{key}"] = np.asarray(value)
    for camera_name, camera_value in observation.get("raw_cameras", {}).items():
        camera_key = camera_name.replace("/", "_")
        color = camera_value.get("color")
        depth = camera_value.get("depth")
        if color is not None:
            payload[f"camera_{camera_key}_color"] = np.asarray(color, dtype=np.uint8)
        if depth is not None:
            payload[f"camera_{camera_key}_depth"] = np.asarray(depth, dtype=np.float32)
    np.savez_compressed(dump_path, **payload)
    return dump_path


def run_trial(
    trial_idx: int,
    cfg,
    policy,
    robot,
    camera_manager: MultiRealSenseManager,
    args: argparse.Namespace,
    log_path: Path,
    trace_root: Path,
    device: torch.device,
) -> bool:
    obs_shape_meta = dict(cfg.shape_meta.obs)
    n_obs_steps = int(cfg.n_obs_steps)
    action_dim = int(cfg.shape_meta.action.shape[0])
    robot_action_dim = 7 if get_action_mode(robot.config) == "delta_tcp_pose_gripper" else 6
    if action_dim != robot_action_dim:
        raise ValueError(
            f"Policy action_dim={action_dim}, robot action_mode={get_action_mode(robot.config)} expects {robot_action_dim}. "
            "Check training task yaml and deployment JSON representation.action_mode."
        )

    if args.move_home_before_trial:
        print(f"[trial {trial_idx}] moving home")
        robot.move_home()
        time.sleep(float(args.home_settle_s))

    first_snapshot = robot.get_robot_snapshot()
    first_observation = capture_robot_observation(robot, camera_manager, robot_snapshot=first_snapshot)
    first_policy_obs = build_policy_observation(first_observation, obs_shape_meta, args)
    obs_buffer: Deque[Dict[str, np.ndarray]] = deque([first_policy_obs] * n_obs_steps, maxlen=n_obs_steps)
    current_observation = first_observation

    correction = rpy_deg_to_matrix(args.action_rotation_correction_rpy_deg)
    start_time = time.monotonic()
    step_idx = 0
    outcome: Optional[str] = None
    reason = "running"
    last_action = np.zeros((action_dim,), dtype=np.float32)
    previous_action_raw: Optional[np.ndarray] = None
    previous_action_smoothed: Optional[np.ndarray] = None
    trial_trace_dir = trace_root / f"trial_{trial_idx:03d}"
    step_trace_path = trial_trace_dir / "step_trace.jsonl"
    dump_dir = trial_trace_dir / "policy_dumps"
    if args.save_step_trace or args.save_policy_dumps:
        trial_trace_dir.mkdir(parents=True, exist_ok=True)

    print(f"[trial {trial_idx}] started. Keys: s=success, f=failure, space=abort/failure, q=quit")
    with KeyPoller() as key_poller:
        while True:
            elapsed_s = time.monotonic() - start_time
            if elapsed_s >= float(args.max_episode_s):
                outcome = "failure"
                reason = "timeout"
                safe_stop_robot(robot)
                break

            while True:
                key = key_poller.poll()
                if key is None:
                    break
                key = key.lower()
                if key == "s":
                    outcome = "success"
                    reason = "operator_success"
                    safe_stop_robot(robot)
                    break
                if key == "f":
                    outcome = "failure"
                    reason = "operator_failure"
                    safe_stop_robot(robot)
                    break
                if key == " ":
                    outcome = "failure"
                    reason = "operator_abort"
                    safe_stop_robot(robot)
                    break
                if key == "q":
                    outcome = "quit"
                    reason = "operator_quit"
                    safe_stop_robot(robot)
                    break
            if outcome in {"success", "failure", "quit"}:
                break

            try:
                infer_start = time.monotonic()
                model_obs = stack_obs_buffer(obs_buffer, device=device)
                with torch.no_grad():
                    result = policy.predict_action(model_obs)
                infer_elapsed_s = time.monotonic() - infer_start
                action_seq = result["action"].detach().cpu().numpy()[0]
                action_raw = np.asarray(action_seq[0], dtype=np.float32)
                action_corrected = correct_action(action_raw, correction)
                action_smoothed = smooth_action(
                    action_corrected,
                    previous_action=previous_action_smoothed,
                    args=args,
                )
                action_executed = safety_clip_action(
                    action_smoothed,
                    max_translation_m=args.max_translation_per_step_m,
                    max_rotation_rad=args.max_rotation_per_step_rad,
                )
                execute_start = time.monotonic()
                execute_info = robot.execute_action(action_executed)
                execute_elapsed_s = time.monotonic() - execute_start
                last_action = action_executed
                previous_action_raw = action_raw.copy()
                previous_action_smoothed = action_smoothed.copy()

                snapshot = robot.get_robot_snapshot()
                observation = capture_robot_observation(robot, camera_manager, robot_snapshot=snapshot)
                current_observation = observation
                obs_buffer.append(build_policy_observation(observation, obs_shape_meta, args))

                if args.show_preview:
                    render_preview(observation, last_action, trial_idx=trial_idx, elapsed_s=elapsed_s)

                dump_path = None
                completed_step = step_idx + 1
                if args.save_policy_dumps and completed_step % max(int(args.policy_dump_every_steps), 1) == 0:
                    dump_path = save_policy_dump(
                        dump_dir=dump_dir,
                        trial_idx=trial_idx,
                        step_idx=completed_step,
                        model_obs=model_obs,
                        policy_result=result,
                        observation=observation,
                        action_raw=action_raw,
                        action_executed=action_executed,
                        execute_info=execute_info,
                        elapsed_s=elapsed_s,
                    )

                if args.save_step_trace:
                    gripper_action_idx = action_dim - 1 if action_dim in (4, 7) else None
                    step_record = {
                        "trial": trial_idx,
                        "step": completed_step,
                        "elapsed_s": elapsed_s,
                        "policy_inference_s": infer_elapsed_s,
                        "robot_execute_s": execute_elapsed_s,
                        "action_raw": action_raw.tolist(),
                        "action_corrected": action_corrected.tolist(),
                        "action_smoothed": action_smoothed.tolist(),
                        "action_executed": action_executed.tolist(),
                        "execute_info": {
                            key: np.asarray(value).tolist() for key, value in execute_info.items()
                        },
                        "action_translation_norm": float(np.linalg.norm(action_executed[:3])) if action_dim >= 3 else 0.0,
                        "action_rotation_norm": float(np.linalg.norm(action_executed[3:6])) if action_dim >= 6 else 0.0,
                        "action_gripper_value": float(action_executed[gripper_action_idx]) if gripper_action_idx is not None else None,
                        "action_change_l2": (
                            float(np.linalg.norm(action_executed - previous_action_smoothed))
                            if previous_action_smoothed is not None
                            else None
                        ),
                        "action_raw_change_l2": (
                            float(np.linalg.norm(action_raw - previous_action_raw))
                            if previous_action_raw is not None
                            else None
                        ),
                        "translation_sign_flip_axes": sign_flip_axes(
                            action_executed[:3],
                            None if previous_action_smoothed is None else previous_action_smoothed[:3],
                        ),
                        "rotation_sign_flip_axes": sign_flip_axes(
                            action_executed[3:6],
                            None if previous_action_smoothed is None or action_dim < 6 else previous_action_smoothed[3:6],
                        ),
                        "agent_pos": np.asarray(observation["agent_pos"]).tolist(),
                        "tcp_pose": np.asarray(observation["tcp_pose"]).tolist(),
                        "tcp_speed": np.asarray(observation["tcp_speed"]).tolist(),
                        "gripper_fraction": np.asarray(observation["gripper_fraction"]).tolist(),
                        "gripper_target_fraction": np.asarray(observation["gripper_target_fraction"]).tolist(),
                        "point_cloud_stats": summarize_numeric_array(observation["point_cloud"]),
                        "camera_summary": camera_trace_summary(observation),
                        "policy_dump_path": str(dump_path) if dump_path is not None else None,
                    }
                    append_jsonl(step_trace_path, step_record)

                step_idx += 1
                print(
                    f"[trial {trial_idx}] step={step_idx} t={elapsed_s:.1f}s "
                    f"{format_action_summary(last_action, get_action_mode(robot.config))}"
                )
            except Exception as exc:
                outcome = "failure"
                reason = f"exception:{type(exc).__name__}:{exc}"
                safe_stop_robot(robot)
                break

            target_dt = 1.0 / max(float(args.control_hz), 1.0)
            sleep_s = start_time + step_idx * target_dt - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    elapsed_s = time.monotonic() - start_time
    record = {
        "trial": trial_idx,
        "outcome": outcome or "failure",
        "success": bool(outcome == "success"),
        "reason": reason,
        "elapsed_s": elapsed_s,
        "steps": step_idx,
        "last_action": last_action.tolist(),
        "success_definition": args.success_definition,
        "max_episode_s": float(args.max_episode_s),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "step_trace_path": str(step_trace_path) if args.save_step_trace else None,
        "policy_dump_dir": str(dump_dir) if args.save_policy_dumps else None,
        "time_unix": time.time(),
    }
    append_jsonl(log_path, record)
    print(f"[trial {trial_idx}] outcome={outcome} reason={reason} elapsed={elapsed_s:.1f}s")
    return outcome != "quit"


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    deploy_config = resolve_collection_paths(load_json(args.config), args.config)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    cfg, policy = load_workspace_policy(checkpoint, device=device, use_ema=args.use_ema)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"deploy_trials_{run_id}.jsonl"
    summary_path = output_dir / f"deploy_summary_{run_id}.json"
    trace_root = output_dir / f"policy_trace_{run_id}"

    camera_manager = MultiRealSenseManager(deploy_config)
    robot = make_robot_controller(deploy_config, dry_run=args.dry_run)

    records: List[Dict[str, Any]] = []
    try:
        camera_manager.start()
        robot.connect()
        for trial_idx in range(int(args.num_trials)):
            input(f"\n准备第 {trial_idx} 次测试，人工复位场景后按 Enter 开始...")
            should_continue = run_trial(
                trial_idx=trial_idx,
                cfg=cfg,
                policy=policy,
                robot=robot,
                camera_manager=camera_manager,
                args=args,
                log_path=log_path,
                trace_root=trace_root,
                device=device,
            )
            with log_path.open("r", encoding="utf-8") as file:
                records = [json.loads(line) for line in file if line.strip()]
            if not should_continue:
                break
    finally:
        safe_stop_robot(robot)
        camera_manager.stop()
        robot.disconnect()
        if cv2 is not None:
            cv2.destroyAllWindows()

    total = len(records)
    successes = sum(1 for item in records if item.get("success"))
    summary = {
        "total_trials": total,
        "successes": successes,
        "success_rate": float(successes / total) if total > 0 else 0.0,
        "log_path": str(log_path),
        "trace_root": str(trace_root),
        "checkpoint": str(checkpoint),
        "config": str(Path(args.config).resolve()),
        "max_episode_s": float(args.max_episode_s),
        "success_definition": args.success_definition,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] success_rate={summary['success_rate']:.3f} ({successes}/{total})")
    print(f"[summary] log={log_path}")
    print(f"[summary] summary={summary_path}")


if __name__ == "__main__":
    main()
