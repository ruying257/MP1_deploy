import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from prepare_clean_real_dataset import (
    apply_action_deadband,
    compute_pose_gripper_action,
    select_active_window,
    select_observation_indices,
    smooth_tcp_pose_sequence,
)


def load_csv_array(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


def build_actions_from_pose(
    tcp_pose: np.ndarray,
    gripper_target: np.ndarray,
    rotation_delta_frame: str,
) -> np.ndarray:
    count = max(len(tcp_pose) - 1, 0)
    actions = np.zeros((count, 7), dtype=np.float32)
    for i in range(count):
        actions[i] = compute_pose_gripper_action(
            curr_pose=tcp_pose[i],
            next_pose=tcp_pose[i + 1],
            next_gripper_target=float(gripper_target[i + 1]),
            rotation_delta_frame=rotation_delta_frame,
        )
    return actions


def action_change_metrics(actions: np.ndarray) -> Dict[str, float]:
    if len(actions) <= 1:
        return {
            "translation_change_median": 0.0,
            "translation_change_p95": 0.0,
            "rotation_change_median": 0.0,
            "rotation_change_p95": 0.0,
        }
    diff = np.diff(actions[:, :6], axis=0)
    trans = np.linalg.norm(diff[:, :3], axis=1)
    rot = np.linalg.norm(diff[:, 3:6], axis=1)
    return {
        "translation_change_median": float(np.median(trans)),
        "translation_change_p95": float(np.percentile(trans, 95)),
        "rotation_change_median": float(np.median(rot)),
        "rotation_change_p95": float(np.percentile(rot, 95)),
    }


def pose_idle_ratio(
    tcp_pose: np.ndarray,
    translation_idle_threshold_m: float,
    rotation_idle_threshold_rad: float,
) -> float:
    if len(tcp_pose) <= 1:
        return 0.0
    delta_xyz = np.linalg.norm(np.diff(tcp_pose[:, :3], axis=0), axis=1)
    delta_rot = np.linalg.norm(np.diff(tcp_pose[:, 3:6], axis=0), axis=1)
    idle = (delta_xyz < translation_idle_threshold_m) & (delta_rot < rotation_idle_threshold_rad)
    return float(np.mean(idle))


def summarize_metric_list(rows: List[Dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows]
    return float(statistics.median(values)) if values else 0.0


def analyze_episode(
    episode_dir: Path,
    translation_window: int,
    rotation_window: int,
    translation_idle_threshold_m: float,
    rotation_idle_threshold_rad: float,
    rotation_delta_frame: str,
    translation_deadband_m: float,
    rotation_deadband_rad: float,
) -> Dict[str, Any]:
    robot_dir = episode_dir / "robot"
    tcp_pose = load_csv_array(robot_dir / "tcp_pose.csv").astype(np.float32)
    gripper_target = load_csv_array(robot_dir / "gripper_target_fraction.csv").reshape(-1).astype(np.float32)

    smoothed_pose = smooth_tcp_pose_sequence(
        tcp_pose=tcp_pose,
        translation_window=translation_window,
        rotation_window=rotation_window,
    )

    raw_actions = build_actions_from_pose(
        tcp_pose=tcp_pose,
        gripper_target=gripper_target,
        rotation_delta_frame=rotation_delta_frame,
    )

    preserve_start, preserve_end = select_active_window(
        tcp_pose=smoothed_pose,
        gripper_target=gripper_target,
        translation_idle_threshold_m=translation_idle_threshold_m,
        rotation_idle_threshold_rad=rotation_idle_threshold_rad,
        rotation_delta_frame=rotation_delta_frame,
        context_frames=1,
    )
    preserve_pose = smoothed_pose[preserve_start : preserve_end + 1]
    preserve_gripper = gripper_target[preserve_start : preserve_end + 1]
    preserve_actions = build_actions_from_pose(
        tcp_pose=preserve_pose,
        gripper_target=preserve_gripper,
        rotation_delta_frame=rotation_delta_frame,
    )
    preserve_actions = apply_action_deadband(
        actions=preserve_actions,
        translation_deadband_m=translation_deadband_m,
        rotation_deadband_rad=rotation_deadband_rad,
    )

    keep_obs = select_observation_indices(
        tcp_pose=smoothed_pose,
        gripper_target=gripper_target,
        translation_idle_threshold_m=translation_idle_threshold_m,
        rotation_idle_threshold_rad=rotation_idle_threshold_rad,
        rotation_delta_frame=rotation_delta_frame,
    )
    compress_obs = np.asarray(keep_obs[:-1], dtype=np.int64)
    compress_next = np.asarray(keep_obs[1:], dtype=np.int64)
    compress_actions = np.stack(
        [
            compute_pose_gripper_action(
                curr_pose=smoothed_pose[i],
                next_pose=smoothed_pose[j],
                next_gripper_target=float(gripper_target[j]),
                rotation_delta_frame=rotation_delta_frame,
            )
            for i, j in zip(compress_obs, compress_next)
        ],
        axis=0,
    ).astype(np.float32)
    compress_actions = apply_action_deadband(
        actions=compress_actions,
        translation_deadband_m=translation_deadband_m,
        rotation_deadband_rad=rotation_deadband_rad,
    )

    return {
        "episode": episode_dir.name,
        "raw_steps": int(len(raw_actions)),
        "preserve_steps": int(len(preserve_actions)),
        "compress_steps": int(len(compress_actions)),
        "raw_idle_ratio": pose_idle_ratio(tcp_pose, translation_idle_threshold_m, rotation_idle_threshold_rad),
        "preserve_idle_ratio": pose_idle_ratio(preserve_pose, translation_idle_threshold_m, rotation_idle_threshold_rad),
        "raw": action_change_metrics(raw_actions),
        "preserve": action_change_metrics(preserve_actions),
        "compress": action_change_metrics(compress_actions),
    }


def aggregate_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {"episodes": rows}
    for mode in ["raw", "preserve", "compress"]:
        mode_rows = [row[mode] for row in rows]
        summary[f"{mode}_median_translation_change_median"] = summarize_metric_list(mode_rows, "translation_change_median")
        summary[f"{mode}_median_translation_change_p95"] = summarize_metric_list(mode_rows, "translation_change_p95")
        summary[f"{mode}_median_rotation_change_median"] = summarize_metric_list(mode_rows, "rotation_change_median")
        summary[f"{mode}_median_rotation_change_p95"] = summarize_metric_list(mode_rows, "rotation_change_p95")
    summary["raw_idle_ratio_median"] = float(statistics.median([row["raw_idle_ratio"] for row in rows])) if rows else 0.0
    summary["preserve_idle_ratio_median"] = (
        float(statistics.median([row["preserve_idle_ratio"] for row in rows])) if rows else 0.0
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析真实机器人 raw 轨迹的平滑程度，并比较不同清洗策略。")
    parser.add_argument("--raw-task-dir", required=True, help="例如 data_all/raw/pole_pickoff")
    parser.add_argument("--translation-window", type=int, default=7)
    parser.add_argument("--rotation-window", type=int, default=5)
    parser.add_argument("--translation-idle-threshold-m", type=float, default=5e-4)
    parser.add_argument("--rotation-idle-threshold-rad", type=float, default=5e-4)
    parser.add_argument("--rotation-delta-frame", choices=["base", "tool"], default="base")
    parser.add_argument("--translation-deadband-m", type=float, default=2e-4)
    parser.add_argument("--rotation-deadband-rad", type=float, default=1e-3)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_task_dir = Path(args.raw_task_dir).resolve()
    episode_dirs = sorted(path for path in raw_task_dir.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        raise FileNotFoundError(f"未在 {raw_task_dir} 下找到 episode_* 目录。")

    rows = []
    for episode_dir in episode_dirs:
        rows.append(
            analyze_episode(
                episode_dir=episode_dir,
                translation_window=args.translation_window,
                rotation_window=args.rotation_window,
                translation_idle_threshold_m=args.translation_idle_threshold_m,
                rotation_idle_threshold_rad=args.rotation_idle_threshold_rad,
                rotation_delta_frame=args.rotation_delta_frame,
                translation_deadband_m=args.translation_deadband_m,
                rotation_deadband_rad=args.rotation_deadband_rad,
            )
        )

    report = aggregate_report(rows)
    report["raw_task_dir"] = str(raw_task_dir)
    report["recommendation"] = (
        "如果目标是训练时保持时间戳和多模态严格对齐，优先使用 preserve 模式；"
        "compress 只适合做停顿压缩对照，不建议作为默认训练数据。"
    )

    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else raw_task_dir.parent / f"{raw_task_dir.name}_trajectory_quality.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] report={output_json}")
    print(json.dumps({k: v for k, v in report.items() if k != 'episodes'}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
