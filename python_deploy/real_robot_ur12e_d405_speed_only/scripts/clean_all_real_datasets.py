import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from prepare_clean_real_dataset import (
    RealRobotDatasetWriter,
    build_clean_episode,
    load_episode_manifest,
    load_episode_from_group,
    open_input_zarr,
    to_writer_episode,
)


def discover_input_zarrs(root: Path) -> List[Path]:
    inputs: List[Path] = []
    for task_dir in sorted((root / "zarr").glob("*")):
        if not task_dir.is_dir():
            continue
        for zarr_dir in sorted(task_dir.glob("*.zarr")):
            if zarr_dir.is_dir():
                inputs.append(zarr_dir)
    return inputs


def clean_one_dataset(
    input_zarr: Path,
    output_zarr: Path,
    overwrite: bool,
    translation_window: int,
    rotation_window: int,
    translation_idle_threshold_m: float,
    rotation_idle_threshold_rad: float,
    rotation_delta_frame: str,
    timeline_mode: str,
    trim_boundary_idle: bool,
    translation_deadband_m: float,
    rotation_deadband_rad: float,
    include_gripper_state: bool,
    gripper_state_mode: str,
    output_action_mode: str,
) -> Dict[str, object]:
    root, episode_ends = open_input_zarr(input_zarr)
    input_manifest = load_episode_manifest(input_zarr)
    writer = RealRobotDatasetWriter(str(output_zarr), overwrite=overwrite)

    summary: Dict[str, object] = {
        "input_zarr": str(input_zarr),
        "output_zarr": str(output_zarr),
        "episodes": [],
    }
    total_source_steps = 0
    total_clean_steps = 0

    for episode_idx in range(len(episode_ends)):
        source_episode = load_episode_from_group(root, episode_ends, episode_idx)
        cleaned_episode, episode_summary = build_clean_episode(
            episode=source_episode,
            include_gripper_state=include_gripper_state,
            gripper_state_mode=gripper_state_mode,
            translation_window=translation_window,
            rotation_window=rotation_window,
            translation_idle_threshold_m=translation_idle_threshold_m,
            rotation_idle_threshold_rad=rotation_idle_threshold_rad,
            rotation_delta_frame=rotation_delta_frame,
            timeline_mode=timeline_mode,
            trim_boundary_idle=trim_boundary_idle,
            translation_deadband_m=translation_deadband_m,
            rotation_deadband_rad=rotation_deadband_rad,
            output_action_mode=output_action_mode,
        )

        metadata = dict(input_manifest[episode_idx]) if episode_idx < len(input_manifest) else {"episode_index": episode_idx}
        metadata.update(
            {
                "cleaned_from_episode_index": episode_idx,
                "clean_translation_window": int(translation_window),
                "clean_rotation_window": int(rotation_window),
                "clean_translation_idle_threshold_m": float(translation_idle_threshold_m),
                "clean_rotation_idle_threshold_rad": float(rotation_idle_threshold_rad),
                "clean_timeline_mode": timeline_mode,
                "clean_trim_boundary_idle": bool(trim_boundary_idle),
                "clean_translation_deadband_m": float(translation_deadband_m),
                "clean_rotation_deadband_rad": float(rotation_deadband_rad),
                "clean_gripper_state_mode": episode_summary["gripper_state_source"],
                "clean_include_gripper_state": bool(include_gripper_state),
                "clean_output_action_mode": output_action_mode,
                "clean_action_dim": int(episode_summary["action_dim"]),
            }
        )
        writer.add_episode(to_writer_episode(cleaned_episode), metadata=metadata)

        total_source_steps += int(episode_summary["source_length"])
        total_clean_steps += int(episode_summary["clean_length"])
        summary["episodes"].append({"episode_index": episode_idx, **episode_summary})
        print(
            f"[clean] {input_zarr.name} episode={episode_idx} "
            f"source={episode_summary['source_length']} clean={episode_summary['clean_length']} "
            f"removed={episode_summary['removed_steps']} gripper={episode_summary['gripper_state_source']}"
        )

    summary["total_source_steps"] = int(total_source_steps)
    summary["total_clean_steps"] = int(total_clean_steps)
    summary["total_removed_steps"] = int(total_source_steps - total_clean_steps)
    summary["global_keep_ratio"] = float(total_clean_steps / max(1, total_source_steps))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动清洗 data_all 下的所有真实 zarr 数据集。")
    parser.add_argument(
        "--root",
        default=str((THIS_DIR.parent / "data_all").resolve()),
        help="总数据目录，要求内部至少包含 zarr/ 与 clean_zarr/。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 clean zarr。")
    parser.add_argument("--translation-window", type=int, default=7)
    parser.add_argument("--rotation-window", type=int, default=5)
    parser.add_argument("--translation-idle-threshold-m", type=float, default=5e-4)
    parser.add_argument("--rotation-idle-threshold-rad", type=float, default=5e-4)
    parser.add_argument("--rotation-delta-frame", choices=["base", "tool"], default="base")
    parser.add_argument("--timeline-mode", choices=["preserve", "compress"], default="preserve")
    parser.add_argument("--trim-boundary-idle", dest="trim_boundary_idle", action="store_true", default=True)
    parser.add_argument("--no-trim-boundary-idle", dest="trim_boundary_idle", action="store_false")
    parser.add_argument("--translation-deadband-m", type=float, default=2e-4)
    parser.add_argument("--rotation-deadband-rad", type=float, default=1e-3)
    parser.add_argument("--gripper-state-mode", choices=["auto", "target", "fraction"], default="auto")
    parser.add_argument(
        "--include-gripper-state",
        dest="include_gripper_state",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-include-gripper-state",
        dest="include_gripper_state",
        action="store_false",
    )
    parser.add_argument(
        "--output-action-mode",
        choices=["delta_tcp_pose", "delta_tcp_pose_gripper"],
        default="delta_tcp_pose_gripper",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    zarr_root = root / "zarr"
    clean_root = root / "clean_zarr"
    clean_root.mkdir(parents=True, exist_ok=True)

    input_zarrs = discover_input_zarrs(root)
    if not input_zarrs:
        raise FileNotFoundError(f"在 {zarr_root} 下没有发现任何 *.zarr 任务目录。")

    all_reports = []
    for input_zarr in input_zarrs:
        task_name = input_zarr.stem
        output_zarr = clean_root / f"{task_name}_clean.zarr"
        print(f"[start] {input_zarr} -> {output_zarr}")
        report = clean_one_dataset(
            input_zarr=input_zarr,
            output_zarr=output_zarr,
            overwrite=args.overwrite,
            translation_window=args.translation_window,
            rotation_window=args.rotation_window,
            translation_idle_threshold_m=args.translation_idle_threshold_m,
            rotation_idle_threshold_rad=args.rotation_idle_threshold_rad,
            rotation_delta_frame=args.rotation_delta_frame,
            timeline_mode=args.timeline_mode,
            trim_boundary_idle=args.trim_boundary_idle,
            translation_deadband_m=args.translation_deadband_m,
            rotation_deadband_rad=args.rotation_deadband_rad,
            include_gripper_state=args.include_gripper_state,
            gripper_state_mode=args.gripper_state_mode,
            output_action_mode=args.output_action_mode,
        )
        report_path = clean_root / f"{task_name}_clean_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[done] report={report_path}")
        all_reports.append(report)

    summary_path = clean_root / "clean_all_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"datasets": all_reports}, f, ensure_ascii=False, indent=2)
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
