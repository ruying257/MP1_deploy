import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
try:
    import zarr
except Exception:
    zarr = None

try:
    from scipy.signal import savgol_filter
except Exception:
    savgol_filter = None


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent.parent
MP1_ROOT = ROOT_DIR / "MP1"
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from real_robot_utils import (
    RealRobotDatasetWriter,
    fraction_to_gripper_target,
    rotation_matrix_to_rot6d,
    rotation_matrix_to_rotvec,
    rotvec_to_rotation_matrix,
)


def load_episode_manifest(zarr_path: Path) -> List[Dict[str, Any]]:
    manifest_path = Path(str(zarr_path) + ".manifest.json")
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload.get("episodes", []))


def open_input_zarr(zarr_path: Path):
    if zarr is None:
        raise ModuleNotFoundError("open_input_zarr 需要 zarr。请先在当前环境安装 zarr。")
    root = zarr.open(str(zarr_path), mode="r")
    episode_ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
    return root, episode_ends


def load_episode_from_group(root, episode_ends: np.ndarray, episode_idx: int) -> Dict[str, np.ndarray]:
    start_idx = 0
    if episode_idx > 0:
        start_idx = int(episode_ends[episode_idx - 1])
    end_idx = int(episode_ends[episode_idx])
    data_group = root["data"]
    result: Dict[str, np.ndarray] = {}
    for key in data_group.keys():
        result[key] = np.asarray(data_group[key][start_idx:end_idx])
    return result


def odd_window(window: int, length: int) -> int:
    window = max(1, int(window))
    window = min(window, int(length))
    if window % 2 == 0:
        window -= 1
    return max(1, window)


def moving_average_reflect(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(series) <= 2:
        return series.copy()
    half = window // 2
    padded = np.pad(series, ((half, half), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    out = np.zeros_like(series, dtype=np.float64)
    for dim in range(series.shape[1]):
        out[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
    return out


def smooth_series(series: np.ndarray, window: int) -> np.ndarray:
    window = odd_window(window, len(series))
    if window <= 1 or len(series) <= 2:
        return series.astype(np.float64, copy=True)
    if savgol_filter is not None and window >= 5:
        polyorder = min(2, window - 1)
        return savgol_filter(series, window_length=window, polyorder=polyorder, axis=0, mode="interp")
    return moving_average_reflect(series.astype(np.float64), window)


def unwrap_rotvec_sequence(rotvecs: np.ndarray) -> np.ndarray:
    if len(rotvecs) <= 1:
        return rotvecs.astype(np.float64, copy=True)
    out = rotvecs.astype(np.float64, copy=True)
    for i in range(1, len(out)):
        prev = out[i - 1]
        curr = out[i]
        norm = np.linalg.norm(curr)
        if norm < 1e-8:
            continue
        axis = curr / norm
        candidates = [curr, curr + 2.0 * math.pi * axis, curr - 2.0 * math.pi * axis]
        out[i] = min(candidates, key=lambda x: np.linalg.norm(x - prev))
    return out


def smooth_tcp_pose_sequence(tcp_pose: np.ndarray, translation_window: int, rotation_window: int) -> np.ndarray:
    smoothed = tcp_pose.astype(np.float64, copy=True)
    smoothed[:, :3] = smooth_series(smoothed[:, :3], translation_window)
    smooth_rot = smooth_series(unwrap_rotvec_sequence(smoothed[:, 3:6]), rotation_window)
    smoothed[:, 3:6] = smooth_rot
    return smoothed.astype(np.float32)


def choose_gripper_state(
    episode: Dict[str, np.ndarray], mode: str
) -> Tuple[np.ndarray, str]:
    target = episode.get("gripper_target_fraction")
    fraction = episode.get("gripper_fraction")
    target = None if target is None else np.asarray(target, dtype=np.float32).reshape(-1)
    fraction = None if fraction is None else np.asarray(fraction, dtype=np.float32).reshape(-1)

    if mode == "target":
        if target is None:
            raise KeyError("输入 zarr 不包含 gripper_target_fraction，无法使用 target 作为 gripper state。")
        return target, "target"
    if mode == "fraction":
        if fraction is None:
            raise KeyError("输入 zarr 不包含 gripper_fraction，无法使用 fraction 作为 gripper state。")
        return fraction, "fraction"

    if target is None and fraction is None:
        raise KeyError("输入 zarr 中既没有 gripper_target_fraction，也没有 gripper_fraction。")
    if target is None:
        return fraction, "fraction"
    if fraction is None:
        return target, "target"

    target_range = float(np.max(target) - np.min(target))
    fraction_range = float(np.max(fraction) - np.min(fraction))
    if target_range > max(1e-4, 5.0 * fraction_range):
        return target, "target"
    return fraction, "fraction"


def build_agent_state(tcp_pose: np.ndarray, gripper_state: np.ndarray, include_gripper_state: bool) -> np.ndarray:
    rotation = np.stack([rotvec_to_rotation_matrix(rot) for rot in tcp_pose[:, 3:6]], axis=0)
    rot6d = np.stack([rotation_matrix_to_rot6d(rot) for rot in rotation], axis=0).astype(np.float32)
    state = np.concatenate([tcp_pose[:, :3].astype(np.float32), rot6d], axis=1)
    if include_gripper_state:
        state = np.concatenate([state, gripper_state.reshape(-1, 1).astype(np.float32)], axis=1)
    return state.astype(np.float32)


def compute_pose_gripper_action(
    curr_pose: np.ndarray,
    next_pose: np.ndarray,
    next_gripper_target: float,
    rotation_delta_frame: str,
) -> np.ndarray:
    delta_xyz = (next_pose[:3] - curr_pose[:3]).astype(np.float32)
    curr_rot = rotvec_to_rotation_matrix(curr_pose[3:6])
    next_rot = rotvec_to_rotation_matrix(next_pose[3:6])
    if rotation_delta_frame == "tool":
        delta_rot = curr_rot.T @ next_rot
    else:
        delta_rot = next_rot @ curr_rot.T
    delta_rotvec = rotation_matrix_to_rotvec(delta_rot).astype(np.float32)
    return np.asarray(
        [
            delta_xyz[0],
            delta_xyz[1],
            delta_xyz[2],
            delta_rotvec[0],
            delta_rotvec[1],
            delta_rotvec[2],
            fraction_to_gripper_target(float(next_gripper_target)),
        ],
        dtype=np.float32,
    )


def compute_pose_action(
    curr_pose: np.ndarray,
    next_pose: np.ndarray,
    rotation_delta_frame: str,
) -> np.ndarray:
    delta_xyz = (next_pose[:3] - curr_pose[:3]).astype(np.float32)
    curr_rot = rotvec_to_rotation_matrix(curr_pose[3:6])
    next_rot = rotvec_to_rotation_matrix(next_pose[3:6])
    if rotation_delta_frame == "tool":
        delta_rot = curr_rot.T @ next_rot
    else:
        delta_rot = next_rot @ curr_rot.T
    delta_rotvec = rotation_matrix_to_rotvec(delta_rot).astype(np.float32)
    return np.asarray(
        [
            delta_xyz[0],
            delta_xyz[1],
            delta_xyz[2],
            delta_rotvec[0],
            delta_rotvec[1],
            delta_rotvec[2],
        ],
        dtype=np.float32,
    )


def compute_transition_actions(
    tcp_pose: np.ndarray,
    gripper_target: np.ndarray,
    rotation_delta_frame: str,
) -> np.ndarray:
    count = len(tcp_pose) - 1
    actions = np.zeros((max(count, 0), 7), dtype=np.float32)
    for i in range(count):
        actions[i] = compute_pose_gripper_action(
            curr_pose=tcp_pose[i],
            next_pose=tcp_pose[i + 1],
            next_gripper_target=float(gripper_target[i + 1]),
            rotation_delta_frame=rotation_delta_frame,
        )
    return actions


def compute_output_action(
    curr_pose: np.ndarray,
    next_pose: np.ndarray,
    next_gripper_target: float,
    rotation_delta_frame: str,
    output_action_mode: str,
) -> np.ndarray:
    if output_action_mode == "delta_tcp_pose":
        return compute_pose_action(
            curr_pose=curr_pose,
            next_pose=next_pose,
            rotation_delta_frame=rotation_delta_frame,
        )
    if output_action_mode == "delta_tcp_pose_gripper":
        return compute_pose_gripper_action(
            curr_pose=curr_pose,
            next_pose=next_pose,
            next_gripper_target=next_gripper_target,
            rotation_delta_frame=rotation_delta_frame,
        )
    raise ValueError(f"不支持的输出动作模式: {output_action_mode}")


def select_observation_indices(
    tcp_pose: np.ndarray,
    gripper_target: np.ndarray,
    translation_idle_threshold_m: float,
    rotation_idle_threshold_rad: float,
    rotation_delta_frame: str,
) -> List[int]:
    if len(tcp_pose) <= 1:
        return [0]
    keep = [0]
    transitions = compute_transition_actions(tcp_pose, gripper_target, rotation_delta_frame)
    transition_norm = np.linalg.norm(transitions[:, :3], axis=1)
    rotation_norm = np.linalg.norm(transitions[:, 3:6], axis=1)
    gripper_delta = np.abs(np.diff(gripper_target.astype(np.float32), axis=0))
    idle = (
        (transition_norm < float(translation_idle_threshold_m))
        & (rotation_norm < float(rotation_idle_threshold_rad))
        & (gripper_delta < 1e-6)
    )

    i = 0
    while i < len(idle):
        if idle[i]:
            j = i
            while j + 1 < len(idle) and idle[j + 1]:
                j += 1
            keep.append(j + 1)
            i = j + 1
        else:
            keep.append(i + 1)
            i += 1
    if keep[-1] != len(tcp_pose) - 1:
        keep.append(len(tcp_pose) - 1)
    return keep


def select_active_window(
    tcp_pose: np.ndarray,
    gripper_target: np.ndarray,
    translation_idle_threshold_m: float,
    rotation_idle_threshold_rad: float,
    rotation_delta_frame: str,
    context_frames: int = 1,
) -> Tuple[int, int]:
    if len(tcp_pose) <= 1:
        return 0, max(len(tcp_pose) - 1, 0)
    transitions = compute_transition_actions(tcp_pose, gripper_target, rotation_delta_frame)
    transition_norm = np.linalg.norm(transitions[:, :3], axis=1)
    rotation_norm = np.linalg.norm(transitions[:, 3:6], axis=1)
    gripper_delta = np.abs(np.diff(gripper_target.astype(np.float32), axis=0))
    active = (
        (transition_norm >= float(translation_idle_threshold_m))
        | (rotation_norm >= float(rotation_idle_threshold_rad))
        | (gripper_delta >= 1e-6)
    )
    active_indices = np.flatnonzero(active)
    if active_indices.size == 0:
        return 0, len(tcp_pose) - 1
    start = max(int(active_indices[0]) - int(context_frames), 0)
    end = min(int(active_indices[-1]) + 1 + int(context_frames), len(tcp_pose) - 1)
    if end <= start:
        end = min(start + 1, len(tcp_pose) - 1)
    return start, end


def apply_action_deadband(
    actions: np.ndarray,
    translation_deadband_m: float,
    rotation_deadband_rad: float,
) -> np.ndarray:
    output = np.asarray(actions, dtype=np.float32).copy()
    if output.size == 0:
        return output
    translation_norm = np.linalg.norm(output[:, :3], axis=1)
    rotation_norm = np.linalg.norm(output[:, 3:6], axis=1)
    output[translation_norm < float(translation_deadband_m), :3] = 0.0
    output[rotation_norm < float(rotation_deadband_rad), 3:6] = 0.0
    return output


def build_clean_episode(
    episode: Dict[str, np.ndarray],
    include_gripper_state: bool,
    gripper_state_mode: str,
    translation_window: int,
    rotation_window: int,
    translation_idle_threshold_m: float,
    rotation_idle_threshold_rad: float,
    rotation_delta_frame: str,
    timeline_mode: str,
    trim_boundary_idle: bool,
    translation_deadband_m: float,
    rotation_deadband_rad: float,
    output_action_mode: str = "delta_tcp_pose_gripper",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    tcp_pose = np.asarray(episode["tcp_pose"], dtype=np.float32)
    gripper_target = np.asarray(
        episode.get("gripper_target_fraction", episode.get("gripper_fraction")),
        dtype=np.float32,
    ).reshape(-1)
    gripper_state, gripper_source = choose_gripper_state(episode, gripper_state_mode)
    smoothed_pose = smooth_tcp_pose_sequence(
        tcp_pose=tcp_pose,
        translation_window=translation_window,
        rotation_window=rotation_window,
    )
    if timeline_mode == "preserve":
        if trim_boundary_idle:
            start_idx, end_idx = select_active_window(
                tcp_pose=smoothed_pose,
                gripper_target=gripper_target,
                translation_idle_threshold_m=translation_idle_threshold_m,
                rotation_idle_threshold_rad=rotation_idle_threshold_rad,
                rotation_delta_frame=rotation_delta_frame,
                context_frames=1,
            )
        else:
            start_idx, end_idx = 0, len(smoothed_pose) - 1
        if end_idx <= start_idx:
            raise ValueError("保时间轴清洗后的有效区间不足 2 帧，无法构建轨迹。")
        obs_idx = np.arange(start_idx, end_idx, dtype=np.int64)
        next_idx = obs_idx + 1
    elif timeline_mode == "compress":
        keep_obs = select_observation_indices(
            tcp_pose=smoothed_pose,
            gripper_target=gripper_target,
            translation_idle_threshold_m=translation_idle_threshold_m,
            rotation_idle_threshold_rad=rotation_idle_threshold_rad,
            rotation_delta_frame=rotation_delta_frame,
        )
        if len(keep_obs) < 2:
            raise ValueError("压缩后观测不足 2 帧，无法构建轨迹。")
        obs_idx = np.asarray(keep_obs[:-1], dtype=np.int64)
        next_idx = np.asarray(keep_obs[1:], dtype=np.int64)
        start_idx = int(obs_idx[0]) if obs_idx.size > 0 else 0
        end_idx = int(next_idx[-1]) if next_idx.size > 0 else 0
    else:
        raise ValueError(f"不支持的 timeline_mode: {timeline_mode}")

    actions = np.stack(
        [
            compute_output_action(
                curr_pose=smoothed_pose[i],
                next_pose=smoothed_pose[j],
                next_gripper_target=float(gripper_target[j]),
                rotation_delta_frame=rotation_delta_frame,
                output_action_mode=output_action_mode,
            )
            for i, j in zip(obs_idx, next_idx)
        ],
        axis=0,
    ).astype(np.float32)
    actions = apply_action_deadband(
        actions=actions,
        translation_deadband_m=translation_deadband_m,
        rotation_deadband_rad=rotation_deadband_rad,
    )
    state = build_agent_state(
        tcp_pose=smoothed_pose[obs_idx],
        gripper_state=gripper_state[obs_idx],
        include_gripper_state=include_gripper_state,
    )

    clean_episode = {
        "point_cloud": np.asarray(episode["point_cloud"][obs_idx], dtype=np.float32),
        "state": state,
        "action": actions,
        "tcp_pose": np.asarray(smoothed_pose[obs_idx], dtype=np.float32),
        "gripper_fraction": np.asarray(episode.get("gripper_fraction", gripper_state.reshape(-1, 1))[obs_idx], dtype=np.float32),
        "gripper_target_fraction": np.asarray(gripper_target[obs_idx], dtype=np.float32).reshape(-1, 1),
        "joint_positions": np.asarray(episode.get("joint_positions", np.zeros((len(tcp_pose), 6), dtype=np.float32))[obs_idx], dtype=np.float32),
        "tcp_speed": np.asarray(episode.get("tcp_speed", np.zeros((len(tcp_pose), 6), dtype=np.float32))[obs_idx], dtype=np.float32),
        "controller_timestamp": np.asarray(episode.get("controller_timestamp", np.full((len(tcp_pose), 1), np.nan))[obs_idx], dtype=np.float64),
        "timestamp": np.asarray(episode.get("timestamp", np.full((len(tcp_pose), 1), np.nan))[obs_idx], dtype=np.float64),
        "capture_completed_timestamp": np.asarray(
            episode.get("capture_completed_timestamp", np.full((len(tcp_pose), 1), np.nan))[obs_idx],
            dtype=np.float64,
        ),
        "source_step_index": obs_idx.reshape(-1, 1).astype(np.int64),
    }

    source_length = int(max(len(tcp_pose) - 1, 0))
    summary = {
        "source_length": source_length,
        "clean_length": int(len(obs_idx)),
        "removed_steps": int(source_length - len(obs_idx)),
        "keep_ratio": float(len(obs_idx) / max(1, source_length)),
        "gripper_state_source": gripper_source,
        "include_gripper_state": bool(include_gripper_state),
        "output_action_mode": output_action_mode,
        "action_dim": int(actions.shape[1]) if actions.ndim == 2 else int(actions.shape[-1]),
        "timeline_mode": timeline_mode,
        "trim_boundary_idle": bool(trim_boundary_idle),
        "source_window_start": int(start_idx),
        "source_window_end": int(end_idx),
    }
    return clean_episode, summary


def to_writer_episode(episode: Dict[str, np.ndarray]) -> Dict[str, List[np.ndarray]]:
    return {key: [value[i] for i in range(value.shape[0])] for key, value in episode.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将真实机器人 speed-only 数据清洗成训练用 zarr。")
    parser.add_argument("--input-zarr", required=True, help="原始 zarr 路径。")
    parser.add_argument("--output-zarr", required=True, help="清洗后 zarr 路径。")
    parser.add_argument("--overwrite", action="store_true", help="若输出 zarr 已存在则覆盖。")
    parser.add_argument("--translation-window", type=int, default=7, help="TCP 平移平滑窗口。")
    parser.add_argument("--rotation-window", type=int, default=5, help="TCP 旋转平滑窗口。")
    parser.add_argument(
        "--translation-idle-threshold-m",
        type=float,
        default=5e-4,
        help="判定为 idle 的平移阈值，默认 0.5 mm。",
    )
    parser.add_argument(
        "--rotation-idle-threshold-rad",
        type=float,
        default=5e-4,
        help="判定为 idle 的旋转阈值。",
    )
    parser.add_argument(
        "--rotation-delta-frame",
        choices=["base", "tool"],
        default="base",
        help="动作旋转增量坐标系。",
    )
    parser.add_argument(
        "--timeline-mode",
        choices=["preserve", "compress"],
        default="preserve",
        help="清洗时是否保留原始固定节拍。推荐 preserve；compress 仅用于对照。",
    )
    parser.add_argument(
        "--trim-boundary-idle",
        dest="trim_boundary_idle",
        action="store_true",
        default=True,
        help="仅裁掉轨迹首尾长时间静止段，保留中间时间轴。",
    )
    parser.add_argument(
        "--no-trim-boundary-idle",
        dest="trim_boundary_idle",
        action="store_false",
        help="完全保留首尾时间轴。",
    )
    parser.add_argument(
        "--translation-deadband-m",
        type=float,
        default=2e-4,
        help="对极小平移增量做去噪置零，默认 0.2 mm。",
    )
    parser.add_argument(
        "--rotation-deadband-rad",
        type=float,
        default=1e-3,
        help="对极小旋转增量做去噪置零，默认 1e-3 rad。",
    )
    parser.add_argument(
        "--gripper-state-mode",
        choices=["auto", "target", "fraction"],
        default="auto",
        help="构造低维状态时使用哪种 gripper 状态。默认 auto。",
    )
    parser.add_argument(
        "--include-gripper-state",
        dest="include_gripper_state",
        action="store_true",
        default=True,
        help="将 gripper 状态并入 low-dim state。",
    )
    parser.add_argument(
        "--no-include-gripper-state",
        dest="include_gripper_state",
        action="store_false",
        help="不将 gripper 状态并入 low-dim state。",
    )
    parser.add_argument(
        "--output-action-mode",
        choices=["delta_tcp_pose", "delta_tcp_pose_gripper"],
        default="delta_tcp_pose_gripper",
        help="输出训练动作维度。无夹爪任务用 delta_tcp_pose，旧抓取任务用 delta_tcp_pose_gripper。",
    )
    parser.add_argument("--report-json", default=None, help="清洗报告输出路径。默认写到 output_zarr 同目录。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_zarr = Path(args.input_zarr).resolve()
    output_zarr = Path(args.output_zarr).resolve()
    output_zarr.parent.mkdir(parents=True, exist_ok=True)

    root, episode_ends = open_input_zarr(input_zarr)
    input_manifest = load_episode_manifest(input_zarr)
    writer = RealRobotDatasetWriter(str(output_zarr), overwrite=args.overwrite)

    report: Dict[str, Any] = {
        "input_zarr": str(input_zarr),
        "output_zarr": str(output_zarr),
        "translation_window": int(args.translation_window),
        "rotation_window": int(args.rotation_window),
        "translation_idle_threshold_m": float(args.translation_idle_threshold_m),
        "rotation_idle_threshold_rad": float(args.rotation_idle_threshold_rad),
        "rotation_delta_frame": args.rotation_delta_frame,
        "timeline_mode": args.timeline_mode,
        "trim_boundary_idle": bool(args.trim_boundary_idle),
        "translation_deadband_m": float(args.translation_deadband_m),
        "rotation_deadband_rad": float(args.rotation_deadband_rad),
        "gripper_state_mode": args.gripper_state_mode,
        "include_gripper_state": bool(args.include_gripper_state),
        "output_action_mode": args.output_action_mode,
        "episodes": [],
    }

    total_source_steps = 0
    total_clean_steps = 0
    for episode_idx in range(len(episode_ends)):
        source_episode = load_episode_from_group(root, episode_ends, episode_idx)
        cleaned_episode, summary = build_clean_episode(
            episode=source_episode,
            include_gripper_state=args.include_gripper_state,
            gripper_state_mode=args.gripper_state_mode,
            translation_window=args.translation_window,
            rotation_window=args.rotation_window,
            translation_idle_threshold_m=args.translation_idle_threshold_m,
            rotation_idle_threshold_rad=args.rotation_idle_threshold_rad,
            rotation_delta_frame=args.rotation_delta_frame,
            timeline_mode=args.timeline_mode,
            trim_boundary_idle=args.trim_boundary_idle,
            translation_deadband_m=args.translation_deadband_m,
            rotation_deadband_rad=args.rotation_deadband_rad,
            output_action_mode=args.output_action_mode,
        )

        metadata = dict(input_manifest[episode_idx]) if episode_idx < len(input_manifest) else {"episode_index": episode_idx}
        metadata.update(
            {
                "cleaned_from_episode_index": episode_idx,
                "clean_translation_window": int(args.translation_window),
                "clean_rotation_window": int(args.rotation_window),
                "clean_translation_idle_threshold_m": float(args.translation_idle_threshold_m),
                "clean_rotation_idle_threshold_rad": float(args.rotation_idle_threshold_rad),
                "clean_timeline_mode": args.timeline_mode,
                "clean_trim_boundary_idle": bool(args.trim_boundary_idle),
                "clean_translation_deadband_m": float(args.translation_deadband_m),
                "clean_rotation_deadband_rad": float(args.rotation_deadband_rad),
                "clean_gripper_state_mode": summary["gripper_state_source"],
                "clean_include_gripper_state": bool(args.include_gripper_state),
                "clean_output_action_mode": args.output_action_mode,
                "clean_action_dim": int(summary["action_dim"]),
            }
        )
        writer.add_episode(to_writer_episode(cleaned_episode), metadata=metadata)

        total_source_steps += int(summary["source_length"])
        total_clean_steps += int(summary["clean_length"])
        report["episodes"].append({"episode_index": episode_idx, **summary})
        print(
            f"[clean] episode={episode_idx} "
            f"source={summary['source_length']} clean={summary['clean_length']} "
            f"removed={summary['removed_steps']} gripper={summary['gripper_state_source']}"
        )

    report["total_source_steps"] = int(total_source_steps)
    report["total_clean_steps"] = int(total_clean_steps)
    report["total_removed_steps"] = int(total_source_steps - total_clean_steps)
    report["global_keep_ratio"] = float(total_clean_steps / max(1, total_source_steps))

    report_path = Path(args.report_json) if args.report_json else output_zarr.with_suffix(output_zarr.suffix + ".clean_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[done] cleaned_zarr={output_zarr}")
    print(f"[done] report={report_path}")


if __name__ == "__main__":
    main()
