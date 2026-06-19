import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None

from prepare_clean_real_dataset import (
    RealRobotDatasetWriter,
    build_clean_episode,
    load_episode_from_group,
    load_episode_manifest,
    open_input_zarr,
    to_writer_episode,
)


def _first_available_key(episode: Dict[str, np.ndarray], candidates: List[str]) -> str:
    for key in candidates:
        if key in episode:
            return key
    raise KeyError(f"None of the candidate keys exist: {candidates}")


def _resize_hwc_uint8(image: np.ndarray, image_height: int, image_width: int) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {image.shape}")
    if image.shape[0] == image_height and image.shape[1] == image_width:
        return image
    if Image is not None:
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((image_width, image_height), Image.BILINEAR)
        return np.asarray(pil_image, dtype=np.uint8)

    y_idx = np.linspace(0, image.shape[0] - 1, image_height).round().astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, image_width).round().astype(np.int64)
    return image[np.ix_(y_idx, x_idx)]


def _prepare_image_sequence(
    image_array: np.ndarray,
    source_step_index: np.ndarray,
    image_height: int,
    image_width: int,
    convert_bgr_to_rgb: bool,
) -> np.ndarray:
    frames = np.asarray(image_array[source_step_index], dtype=np.uint8)
    output = []
    for frame in frames:
        resized = _resize_hwc_uint8(frame, image_height=image_height, image_width=image_width)
        if convert_bgr_to_rgb:
            resized = resized[..., ::-1]
        output.append(np.transpose(resized, (2, 0, 1)).astype(np.uint8))
    return np.stack(output, axis=0)


def build_multimodal_clean_episode(
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
    global_image_key: str,
    wrist_image_key: str,
    wrist_point_cloud_key: str,
    global_image_height: int,
    global_image_width: int,
    wrist_image_height: int,
    wrist_image_width: int,
    convert_bgr_to_rgb: bool,
    output_action_mode: str = "delta_tcp_pose_gripper",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    clean_episode, summary = build_clean_episode(
        episode=episode,
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

    source_step_index = clean_episode["source_step_index"].reshape(-1)
    resolved_global_image_key = _first_available_key(
        episode,
        [global_image_key, "camera_global_d405_img"],
    )
    resolved_wrist_image_key = _first_available_key(
        episode,
        [wrist_image_key, "camera_wrist_d405_img", "img"],
    )
    resolved_point_cloud_key = _first_available_key(
        episode,
        [wrist_point_cloud_key, "camera_wrist_d405_point_cloud", "point_cloud"],
    )

    clean_episode["point_cloud"] = np.asarray(
        episode[resolved_point_cloud_key][source_step_index],
        dtype=np.float32,
    )
    clean_episode["global_image"] = _prepare_image_sequence(
        episode[resolved_global_image_key],
        source_step_index=source_step_index,
        image_height=global_image_height,
        image_width=global_image_width,
        convert_bgr_to_rgb=convert_bgr_to_rgb,
    )
    clean_episode["wrist_image"] = _prepare_image_sequence(
        episode[resolved_wrist_image_key],
        source_step_index=source_step_index,
        image_height=wrist_image_height,
        image_width=wrist_image_width,
        convert_bgr_to_rgb=convert_bgr_to_rgb,
    )

    summary.update(
        {
            "global_image_key": resolved_global_image_key,
            "wrist_image_key": resolved_wrist_image_key,
            "point_cloud_key": resolved_point_cloud_key,
            "global_image_height": int(global_image_height),
            "global_image_width": int(global_image_width),
            "wrist_image_height": int(wrist_image_height),
            "wrist_image_width": int(wrist_image_width),
            "convert_bgr_to_rgb": bool(convert_bgr_to_rgb),
        }
    )
    return clean_episode, summary


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
    global_image_key: str,
    wrist_image_key: str,
    wrist_point_cloud_key: str,
    global_image_height: int,
    global_image_width: int,
    wrist_image_height: int,
    wrist_image_width: int,
    convert_bgr_to_rgb: bool,
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
        cleaned_episode, episode_summary = build_multimodal_clean_episode(
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
            global_image_key=global_image_key,
            wrist_image_key=wrist_image_key,
            wrist_point_cloud_key=wrist_point_cloud_key,
            global_image_height=global_image_height,
            global_image_width=global_image_width,
            wrist_image_height=wrist_image_height,
            wrist_image_width=wrist_image_width,
            convert_bgr_to_rgb=convert_bgr_to_rgb,
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
                "global_image_key": episode_summary["global_image_key"],
                "wrist_image_key": episode_summary["wrist_image_key"],
                "point_cloud_key": episode_summary["point_cloud_key"],
                "global_image_height": int(global_image_height),
                "global_image_width": int(global_image_width),
                "wrist_image_height": int(wrist_image_height),
                "wrist_image_width": int(wrist_image_width),
                "clean_output_action_mode": output_action_mode,
                "clean_action_dim": int(episode_summary["action_dim"]),
            }
        )
        writer.add_episode(to_writer_episode(cleaned_episode), metadata=metadata)

        total_source_steps += int(episode_summary["source_length"])
        total_clean_steps += int(episode_summary["clean_length"])
        summary["episodes"].append({"episode_index": episode_idx, **episode_summary})
        print(
            f"[multimodal-clean] {input_zarr.name} episode={episode_idx} "
            f"source={episode_summary['source_length']} clean={episode_summary['clean_length']} "
            f"removed={episode_summary['removed_steps']} "
            f"global={episode_summary['global_image_key']} wrist={episode_summary['wrist_image_key']}"
        )

    summary["total_source_steps"] = int(total_source_steps)
    summary["total_clean_steps"] = int(total_clean_steps)
    summary["total_removed_steps"] = int(total_source_steps - total_clean_steps)
    summary["global_keep_ratio"] = float(total_clean_steps / max(1, total_source_steps))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multimodal clean real-robot zarr for global RGB + wrist RGB/point-cloud training.")
    parser.add_argument("--input-zarr", required=True, help="Input raw/full zarr path.")
    parser.add_argument("--output-zarr", required=True, help="Output multimodal clean zarr path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output zarr if it already exists.")
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
    parser.add_argument("--image-height", type=int, default=96, help="Legacy default height applied to both views unless per-view sizes are specified.")
    parser.add_argument("--image-width", type=int, default=96, help="Legacy default width applied to both views unless per-view sizes are specified.")
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
    input_zarr = Path(args.input_zarr).resolve()
    output_zarr = Path(args.output_zarr).resolve()
    global_image_height = int(args.global_image_height or args.image_height)
    global_image_width = int(args.global_image_width or args.image_width)
    wrist_image_height = int(args.wrist_image_height or args.image_height)
    wrist_image_width = int(args.wrist_image_width or args.image_width)
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
    report_path = output_zarr.parent / f"{output_zarr.stem}_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[done] report={report_path}")


if __name__ == "__main__":
    main()
