import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from prepare_multimodal_real_dataset import clean_one_dataset


def discover_input_zarrs(root: Path) -> List[Path]:
    inputs: List[Path] = []
    for task_dir in sorted((root / "zarr").glob("*")):
        if not task_dir.is_dir():
            continue
        for zarr_dir in sorted(task_dir.glob("*.zarr")):
            if zarr_dir.is_dir():
                inputs.append(zarr_dir)
    return inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multimodal clean zarrs for every task under data_all/zarr.")
    parser.add_argument("--root", default=str((THIS_DIR.parent / "data_all").resolve()))
    parser.add_argument("--overwrite", action="store_true")
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
    parser.add_argument("--include-gripper-state", dest="include_gripper_state", action="store_true", default=True)
    parser.add_argument("--no-include-gripper-state", dest="include_gripper_state", action="store_false")
    parser.add_argument("--global-image-key", default="camera_global_d405_img")
    parser.add_argument("--wrist-image-key", default="camera_wrist_d405_img")
    parser.add_argument("--wrist-point-cloud-key", default="camera_wrist_d405_point_cloud")
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--image-width", type=int, default=96)
    parser.add_argument("--global-image-height", type=int, default=None)
    parser.add_argument("--global-image-width", type=int, default=None)
    parser.add_argument("--wrist-image-height", type=int, default=None)
    parser.add_argument("--wrist-image-width", type=int, default=None)
    parser.add_argument("--convert-bgr-to-rgb", dest="convert_bgr_to_rgb", action="store_true", default=True)
    parser.add_argument("--keep-bgr", dest="convert_bgr_to_rgb", action="store_false")
    parser.add_argument(
        "--output-action-mode",
        choices=["delta_tcp_pose", "delta_tcp_pose_gripper"],
        default="delta_tcp_pose_gripper",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    clean_root = root / "clean_zarr"
    clean_root.mkdir(parents=True, exist_ok=True)
    global_image_height = int(args.global_image_height or args.image_height)
    global_image_width = int(args.global_image_width or args.image_width)
    wrist_image_height = int(args.wrist_image_height or args.image_height)
    wrist_image_width = int(args.wrist_image_width or args.image_width)

    input_zarrs = discover_input_zarrs(root)
    if not input_zarrs:
        raise FileNotFoundError(f"No *.zarr directories found under {root / 'zarr'}")

    all_reports: List[Dict[str, object]] = []
    for input_zarr in input_zarrs:
        task_name = input_zarr.stem
        output_zarr = clean_root / f"{task_name}_multimodal_clean.zarr"
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
            global_image_key=args.global_image_key,
            wrist_image_key=args.wrist_image_key,
            wrist_point_cloud_key=args.wrist_point_cloud_key,
            global_image_height=global_image_height,
            global_image_width=global_image_width,
            wrist_image_height=wrist_image_height,
            wrist_image_width=wrist_image_width,
            convert_bgr_to_rgb=args.convert_bgr_to_rgb,
            output_action_mode=args.output_action_mode,
        )
        report_path = clean_root / f"{task_name}_multimodal_clean_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[done] report={report_path}")
        all_reports.append(report)

    summary_path = clean_root / "clean_all_multimodal_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"datasets": all_reports}, f, ensure_ascii=False, indent=2)
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
