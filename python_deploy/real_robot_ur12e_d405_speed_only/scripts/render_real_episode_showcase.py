import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from prepare_clean_real_dataset import (
    select_active_window,
    select_observation_indices,
    smooth_tcp_pose_sequence,
)


BG = (18, 20, 26)
HEADER = (24, 27, 35)
PANEL = (34, 38, 49)
PANEL_ALT = (28, 31, 40)
BORDER = (63, 71, 87)
WHITE = (238, 240, 244)
MUTED = (174, 181, 193)
ACCENT = (255, 191, 83)
ACCENT_2 = (80, 182, 255)
GREEN = (124, 217, 124)
ORANGE = (78, 158, 255)
RED = (90, 110, 255)
TRAJ_SOFT = (182, 224, 255)
TRAJ_RECENT = (88, 183, 255)
SHADOW = (82, 90, 104)
ROD = (98, 210, 116)
FIXTURE = (86, 176, 255)
FIXTURE_SLOT = (150, 220, 255)
CABLE = (232, 235, 240)
CABLE_SHADOW = (128, 138, 154)
GHOST_A = (208, 214, 255)
GHOST_B = (187, 224, 255)
GHOST_C = (176, 208, 246)
ROD_GRASP_POINT_IN_TCP = np.asarray([0.0, 0.0, 0.12], dtype=np.float32)
ROD_TOP_ABOVE_GRASP_M = 0.085
ROD_BOTTOM_BELOW_GRASP_M = 0.165
DISTURBANCE_FREQUENCY_HZ = 0.3333333333


def is_fixture_hanging_task(task_label: str) -> bool:
    return "fixture hanging" in str(task_label).lower()


def load_csv_array(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)


def read_ascii_ply_points(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    start = 0
    for idx, line in enumerate(lines):
        if line.strip() == "end_header":
            start = idx + 1
            break
    if start <= 0 or start >= len(lines):
        return np.zeros((0, 3), dtype=np.float32)
    points = []
    for line in lines[start:]:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            continue
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def make_card(width: int, height: int, fill_color: Tuple[int, int, int] = PANEL) -> np.ndarray:
    card = np.full((height, width, 3), fill_color, dtype=np.uint8)
    cv2.rectangle(card, (0, 0), (width - 1, height - 1), BORDER, 1, cv2.LINE_AA)
    return card


def draw_panel_tag(panel: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
    (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    tag_w = max(120, text_w + 30)
    cv2.rectangle(panel, (14, 12), (14 + tag_w, 40), color, -1, cv2.LINE_AA)
    cv2.putText(panel, text, (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, BG, 2, cv2.LINE_AA)


def fit_image(image: np.ndarray, width: int, height: int, margin: int = 14) -> np.ndarray:
    panel = make_card(width, height, PANEL_ALT)
    if image is None or image.size == 0:
        cv2.putText(panel, "missing frame", (24, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
        return panel
    max_w = max(1, width - 2 * margin)
    max_h = max(1, height - 2 * margin)
    h, w = image.shape[:2]
    scale = min(max_w / max(w, 1), max_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    panel[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    cv2.rectangle(panel, (x0, y0), (x0 + new_w - 1, y0 + new_h - 1), BORDER, 1, cv2.LINE_AA)
    return panel


def render_rgb_panel(image: np.ndarray, width: int, height: int, title: str) -> np.ndarray:
    panel = fit_image(image, width, height)
    draw_panel_tag(panel, title, ACCENT_2)
    return panel


def project_points(points: np.ndarray, projection: str) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if projection == "xz":
        return points[:, [0, 2]]
    if projection == "xy":
        return points[:, [0, 1]]
    if projection == "iso":
        x = points[:, 0] - 0.55 * points[:, 1]
        y = points[:, 2] + 0.35 * points[:, 1]
        return np.stack([x, y], axis=1).astype(np.float32)
    raise ValueError(f"unsupported projection: {projection}")


def build_point_cloud_bounds(
    ply_paths: Iterable[Path],
    projection: str,
    sample_limit: int = 4000,
) -> Tuple[np.ndarray, np.ndarray]:
    projected_chunks = []
    for path in ply_paths:
        if not path.exists():
            continue
        pts = read_ascii_ply_points(path)
        if pts.size == 0:
            continue
        if len(pts) > sample_limit:
            step = max(1, len(pts) // sample_limit)
            pts = pts[::step]
        projected_chunks.append(project_points(pts, projection))
    if not projected_chunks:
        return np.asarray([-1.0, -1.0], dtype=np.float32), np.asarray([1.0, 1.0], dtype=np.float32)
    merged = np.concatenate(projected_chunks, axis=0)
    low = np.percentile(merged, 2.0, axis=0)
    high = np.percentile(merged, 98.0, axis=0)
    span = np.maximum(high - low, 1e-3)
    pad = 0.1 * span
    return (low - pad).astype(np.float32), (high + pad).astype(np.float32)


def normalize_points_to_box(
    pts2: np.ndarray,
    box: Tuple[int, int, int, int],
    bounds: Tuple[np.ndarray, np.ndarray],
    symmetric: bool = False,
) -> np.ndarray:
    x0, y0, width, height = box
    if pts2.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if symmetric:
        radius = float(np.max(np.abs(pts2)))
        radius = max(radius, 1e-3)
        min_xy = np.asarray([-radius, -radius], dtype=np.float32)
        max_xy = np.asarray([radius, radius], dtype=np.float32)
    else:
        min_xy, max_xy = bounds
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = min((width - 2) / span[0], (height - 2) / span[1])
    draw_w = span[0] * scale
    draw_h = span[1] * scale
    left = x0 + int(round((width - draw_w) / 2.0))
    top = y0 + int(round((height - draw_h) / 2.0))
    xy = (pts2 - min_xy) * scale
    px = (xy[:, 0] + left).astype(np.int32)
    py = (height - 1 - xy[:, 1] + top + y0 - top).astype(np.int32)
    py = (y0 + height - 1 - (xy[:, 1] + (top - y0))).astype(np.int32)
    return np.stack([px, py], axis=1)


def draw_text(panel: np.ndarray, text: str, x: int, y: int, scale: float, color: Tuple[int, int, int], thickness: int) -> None:
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_status_chip(
    panel: np.ndarray,
    text: str,
    x: int,
    y: int,
    fill_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = BG,
    scale: float = 0.48,
) -> int:
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    chip_w = text_w + 20
    chip_h = text_h + 12
    cv2.rectangle(panel, (x, y), (x + chip_w, y + chip_h), fill_color, -1, cv2.LINE_AA)
    cv2.putText(panel, text, (x + 10, y + chip_h - 6), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 1, cv2.LINE_AA)
    return chip_w


def render_point_cloud_panel(
    ply_path: Path,
    width: int,
    height: int,
    projection: str,
    bounds: Tuple[np.ndarray, np.ndarray],
    color_axis: int,
    title: str,
) -> np.ndarray:
    panel = make_card(width, height)
    draw_panel_tag(panel, title, ACCENT)
    if not ply_path.exists():
        draw_text(panel, "point cloud missing", 22, height // 2, 0.7, WHITE, 2)
        return panel
    pts = read_ascii_ply_points(ply_path)
    if pts.size == 0:
        draw_text(panel, "point cloud empty", 22, height // 2, 0.7, WHITE, 2)
        return panel

    projected = project_points(pts, projection)
    box = (18, 54, width - 36, height - 88)
    pixels = normalize_points_to_box(projected, box, bounds)
    color_value = pts[:, color_axis]
    color_norm = (color_value - np.min(color_value)) / max(float(np.max(color_value) - np.min(color_value)), 1e-6)
    colors = cv2.applyColorMap((color_norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)

    cv2.rectangle(panel, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), BORDER, 1, cv2.LINE_AA)
    for (x, y), color in zip(pixels, colors):
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(panel, (int(x), int(y)), 2, tuple(int(v) for v in color), -1, cv2.LINE_AA)

    draw_text(panel, f"{len(pts)} pts | {projection}", 22, height - 20, 0.52, MUTED, 1)
    return panel


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values.copy()
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    if len(values) < window:
        return np.repeat(np.mean(values, axis=0, keepdims=True), len(values), axis=0)
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.empty_like(values)
    for dim in range(values.shape[1]):
        padded = np.pad(values[:, dim], (pad, pad), mode="edge")
        smoothed[:, dim] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def draw_polyline(panel: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], thickness: int) -> None:
    if len(points) < 2:
        return
    for idx in range(1, len(points)):
        p0 = tuple(int(v) for v in points[idx - 1])
        p1 = tuple(int(v) for v in points[idx])
        cv2.line(panel, p0, p1, color, thickness, cv2.LINE_AA)


def rotation_matrix_x(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def rotation_matrix_z(angle: float) -> np.ndarray:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float32).reshape(3)
    matrix, _ = cv2.Rodrigues(rotvec.astype(np.float64))
    return np.asarray(matrix, dtype=np.float32)


def transform_points_from_tcp_frame(
    tcp_xyz: np.ndarray,
    tcp_rotvec: np.ndarray,
    points_in_tcp: np.ndarray,
) -> np.ndarray:
    tool_r = rotvec_to_matrix(tcp_rotvec)
    local_points = np.asarray(points_in_tcp, dtype=np.float32).reshape(-1, 3)
    tcp_xyz = np.asarray(tcp_xyz, dtype=np.float32).reshape(1, 3)
    return (tcp_xyz + local_points @ tool_r.T).astype(np.float32)


def apply_platform_transform_points(
    points: np.ndarray,
    platform_center: np.ndarray,
    center_offset: np.ndarray,
    yaw_deg: float,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    center = np.asarray(platform_center, dtype=np.float32).reshape(1, 3)
    offset = np.asarray(center_offset, dtype=np.float32).reshape(1, 3)
    if abs(float(yaw_deg)) < 1e-6 and np.max(np.abs(offset)) < 1e-9:
        return (pts + offset).astype(np.float32)
    rot = rotation_matrix_z(np.deg2rad(yaw_deg))
    return (center + offset + (pts - center) @ rot.T).astype(np.float32)


def estimate_robot_base(tcp_xyz: np.ndarray) -> np.ndarray:
    mean_xyz = np.mean(tcp_xyz, axis=0)
    min_xyz = np.min(tcp_xyz, axis=0)
    return np.asarray(
        [mean_xyz[0] - 0.18, mean_xyz[1] - 0.10, min_xyz[2] - 0.15],
        dtype=np.float32,
    )


def build_workspace_bounds(tcp_xyz: np.ndarray, base_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    low = np.minimum(np.min(tcp_xyz, axis=0), base_pos - np.asarray([0.04, 0.04, 0.01], dtype=np.float32))
    high = np.maximum(np.max(tcp_xyz, axis=0), base_pos + np.asarray([0.14, 0.14, 0.32], dtype=np.float32))
    center = 0.5 * (low + high)
    span = float(np.max(high - low))
    span = max(span, 0.35)
    half = 0.53 * span
    return (center - half).astype(np.float32), (center + half).astype(np.float32)


def cube_corners(bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    low, high = bounds
    corners = []
    for x in [low[0], high[0]]:
        for y in [low[1], high[1]]:
            for z in [low[2], high[2]]:
                corners.append([x, y, z])
    return np.asarray(corners, dtype=np.float32)


def project_points_3d(
    points: np.ndarray,
    box: Tuple[int, int, int, int],
    bounds: Tuple[np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return np.zeros((0, 2), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    low, high = bounds
    center = 0.5 * (low + high)
    span = float(np.max(high - low))
    span = max(span, 1e-6)
    normalized = (points - center) / span
    view_r = rotation_matrix_x(np.deg2rad(26.0)) @ rotation_matrix_z(np.deg2rad(-38.0))
    camera = normalized @ view_r.T
    uv = np.stack([camera[:, 0], -camera[:, 2]], axis=1)
    depth = camera[:, 1]

    corners = cube_corners(bounds)
    corners_cam = ((corners - center) / span) @ view_r.T
    corners_uv = np.stack([corners_cam[:, 0], -corners_cam[:, 2]], axis=1)
    uv_min = np.min(corners_uv, axis=0)
    uv_max = np.max(corners_uv, axis=0)
    uv_span = np.maximum(uv_max - uv_min, 1e-6)

    x0, y0, width, height = box
    scale = min((width - 8) / uv_span[0], (height - 8) / uv_span[1])
    draw_w = uv_span[0] * scale
    draw_h = uv_span[1] * scale
    left = x0 + int(round((width - draw_w) / 2.0))
    top = y0 + int(round((height - draw_h) / 2.0))
    mapped_x = left + (uv[:, 0] - uv_min[0]) * scale
    mapped_y = top + (uv[:, 1] - uv_min[1]) * scale
    pixels = np.stack([mapped_x.astype(np.int32), mapped_y.astype(np.int32)], axis=1)
    return pixels, depth.astype(np.float32)


def draw_cube(panel: np.ndarray, box: Tuple[int, int, int, int], bounds: Tuple[np.ndarray, np.ndarray]) -> None:
    corners = cube_corners(bounds)
    pixels, depth = project_points_3d(corners, box, bounds)
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]
    edge_depth = sorted(edges, key=lambda e: float(depth[e[0]] + depth[e[1]]))
    for idx0, idx1 in edge_depth:
        p0 = tuple(int(v) for v in pixels[idx0])
        p1 = tuple(int(v) for v in pixels[idx1])
        cv2.line(panel, p0, p1, BORDER, 1, cv2.LINE_AA)


def draw_axes(panel: np.ndarray, box: Tuple[int, int, int, int], bounds: Tuple[np.ndarray, np.ndarray]) -> None:
    low, high = bounds
    origin = np.asarray([low[0], low[1], low[2]], dtype=np.float32)
    dx = np.asarray([high[0], low[1], low[2]], dtype=np.float32)
    dy = np.asarray([low[0], high[1], low[2]], dtype=np.float32)
    dz = np.asarray([low[0], low[1], high[2]], dtype=np.float32)
    pts = np.stack([origin, dx, dy, dz], axis=0)
    pixels, _ = project_points_3d(pts, box, bounds)
    o = tuple(int(v) for v in pixels[0])
    px = tuple(int(v) for v in pixels[1])
    py = tuple(int(v) for v in pixels[2])
    pz = tuple(int(v) for v in pixels[3])
    cv2.line(panel, o, px, (92, 182, 255), 2, cv2.LINE_AA)
    cv2.line(panel, o, py, (130, 205, 120), 2, cv2.LINE_AA)
    cv2.line(panel, o, pz, (255, 169, 86), 2, cv2.LINE_AA)
    draw_text(panel, "x", px[0] + 4, px[1], 0.5, (92, 182, 255), 1)
    draw_text(panel, "y", py[0] + 4, py[1], 0.5, (130, 205, 120), 1)
    draw_text(panel, "z", pz[0] + 4, pz[1], 0.5, (255, 169, 86), 1)


def build_robot_skeleton_points(
    tcp_xyz: np.ndarray,
    tcp_rotvec: np.ndarray,
    gripper_target: float,
    base_pos: np.ndarray,
    joint_row: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    shoulder = base_pos + np.asarray([0.0, 0.0, 0.18], dtype=np.float32)
    direction = tcp_xyz - shoulder
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-6:
        direction = np.asarray([0.25, 0.0, 0.18], dtype=np.float32)
        direction_norm = float(np.linalg.norm(direction))
    direction_unit = direction / direction_norm

    if joint_row is not None and len(joint_row) > 0:
        q0 = float(joint_row[0])
        side = np.asarray([np.cos(q0 + np.pi / 2.0), np.sin(q0 + np.pi / 2.0), 0.0], dtype=np.float32)
    else:
        side = np.cross(direction_unit, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    side_norm = float(np.linalg.norm(side))
    if side_norm < 1e-6:
        side = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        side_norm = 1.0
    side /= side_norm

    bend = 0.12 * side + 0.06 * np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    elbow = shoulder + 0.42 * direction + bend
    wrist = shoulder + 0.78 * direction + 0.40 * bend

    base_top = base_pos + np.asarray([0.0, 0.0, 0.08], dtype=np.float32)
    arm_points = np.stack([base_pos, base_top, shoulder, elbow, wrist, tcp_xyz], axis=0).astype(np.float32)

    tool_r = rotvec_to_matrix(tcp_rotvec)
    finger_axis = tool_r[:, 0]
    tool_axis = tool_r[:, 2]
    jaw = 0.02 + 0.02 * float(np.clip(gripper_target, 0.0, 1.0))
    palm = tcp_xyz - 0.018 * tool_axis
    finger_base_l = palm + jaw * finger_axis
    finger_base_r = palm - jaw * finger_axis
    finger_tip_l = finger_base_l + 0.040 * tool_axis
    finger_tip_r = finger_base_r + 0.040 * tool_axis
    gripper_points = np.stack([finger_base_l, finger_tip_l, finger_base_r, finger_tip_r], axis=0).astype(np.float32)
    return arm_points, gripper_points


def draw_robot_model_3d(
    panel: np.ndarray,
    box: Tuple[int, int, int, int],
    bounds: Tuple[np.ndarray, np.ndarray],
    tcp_xyz: np.ndarray,
    tcp_rotvec: np.ndarray,
    gripper_target: float,
    base_pos: np.ndarray,
    joint_row: np.ndarray = None,
    translation_offset: np.ndarray = None,
    platform_center: np.ndarray = None,
    platform_offset: np.ndarray = None,
    platform_yaw_deg: float = 0.0,
    link_color: Tuple[int, int, int] = (77, 88, 103),
    joint_fill: Tuple[int, int, int] = (64, 73, 86),
    edge_color: Tuple[int, int, int] = WHITE,
    gripper_color: Tuple[int, int, int] = ACCENT,
    link_scale: float = 1.0,
) -> None:
    arm_points, gripper_points = build_robot_skeleton_points(
        tcp_xyz=tcp_xyz,
        tcp_rotvec=tcp_rotvec,
        gripper_target=gripper_target,
        base_pos=base_pos,
        joint_row=joint_row,
    )
    if translation_offset is not None:
        offset = np.asarray(translation_offset, dtype=np.float32).reshape(1, 3)
        arm_points = arm_points + offset
        gripper_points = gripper_points + offset
    if platform_center is not None and platform_offset is not None:
        arm_points = apply_platform_transform_points(arm_points, platform_center, platform_offset, platform_yaw_deg)
        gripper_points = apply_platform_transform_points(gripper_points, platform_center, platform_offset, platform_yaw_deg)
    arm_pixels, arm_depth = project_points_3d(arm_points, box, bounds)
    gripper_pixels, _ = project_points_3d(gripper_points, box, bounds)

    for idx in range(1, len(arm_pixels)):
        p0 = tuple(int(v) for v in arm_pixels[idx - 1])
        p1 = tuple(int(v) for v in arm_pixels[idx])
        thickness = 10 if idx < 2 else 7
        thickness = max(2, int(round(thickness * link_scale)))
        cv2.line(panel, p0, p1, link_color, thickness, cv2.LINE_AA)
        cv2.line(panel, p0, p1, edge_color, 1, cv2.LINE_AA)

    order = np.argsort(arm_depth)
    for idx in order:
        point = tuple(int(v) for v in arm_pixels[idx])
        radius = 8 if idx >= 2 else 10
        radius = max(2, int(round(radius * link_scale)))
        cv2.circle(panel, point, radius, joint_fill, -1, cv2.LINE_AA)
        cv2.circle(panel, point, radius, edge_color, 1, cv2.LINE_AA)

    cv2.line(panel, tuple(int(v) for v in gripper_pixels[0]), tuple(int(v) for v in gripper_pixels[1]), gripper_color, max(2, int(round(3 * link_scale))), cv2.LINE_AA)
    cv2.line(panel, tuple(int(v) for v in gripper_pixels[2]), tuple(int(v) for v in gripper_pixels[3]), gripper_color, max(2, int(round(3 * link_scale))), cv2.LINE_AA)


def draw_robot_silhouette_3d(
    panel: np.ndarray,
    box: Tuple[int, int, int, int],
    bounds: Tuple[np.ndarray, np.ndarray],
    tcp_xyz: np.ndarray,
    tcp_rotvec: np.ndarray,
    gripper_target: float,
    base_pos: np.ndarray,
    joint_row: np.ndarray = None,
    platform_center: np.ndarray = None,
    platform_offset: np.ndarray = None,
    platform_yaw_deg: float = 0.0,
    color: Tuple[int, int, int] = (150, 162, 182),
    link_scale: float = 1.0,
) -> None:
    arm_points, gripper_points = build_robot_skeleton_points(
        tcp_xyz=tcp_xyz,
        tcp_rotvec=tcp_rotvec,
        gripper_target=gripper_target,
        base_pos=base_pos,
        joint_row=joint_row,
    )
    if platform_center is not None and platform_offset is not None:
        arm_points = apply_platform_transform_points(arm_points, platform_center, platform_offset, platform_yaw_deg)
        gripper_points = apply_platform_transform_points(gripper_points, platform_center, platform_offset, platform_yaw_deg)

    arm_pixels, _ = project_points_3d(arm_points, box, bounds)
    gripper_pixels, _ = project_points_3d(gripper_points, box, bounds)
    for idx in range(1, len(arm_pixels)):
        p0 = tuple(int(v) for v in arm_pixels[idx - 1])
        p1 = tuple(int(v) for v in arm_pixels[idx])
        thickness = 18 if idx < 2 else 13
        thickness = max(3, int(round(thickness * link_scale)))
        cv2.line(panel, p0, p1, color, thickness, cv2.LINE_AA)
    cv2.line(panel, tuple(int(v) for v in gripper_pixels[0]), tuple(int(v) for v in gripper_pixels[1]), color, max(3, int(round(6 * link_scale))), cv2.LINE_AA)
    cv2.line(panel, tuple(int(v) for v in gripper_pixels[2]), tuple(int(v) for v in gripper_pixels[3]), color, max(3, int(round(6 * link_scale))), cv2.LINE_AA)


def build_reference_rod_points_sequence(
    tcp_pose: np.ndarray,
    gripper_target: np.ndarray,
) -> Tuple[np.ndarray, int]:
    count = int(len(tcp_pose))
    if count <= 0:
        return np.zeros((0, 2, 3), dtype=np.float32), -1

    gripper_target = np.asarray(gripper_target, dtype=np.float32).reshape(-1)
    rod_in_tcp = np.asarray(
        [
            ROD_GRASP_POINT_IN_TCP + np.asarray([0.0, 0.0, -ROD_BOTTOM_BELOW_GRASP_M], dtype=np.float32),
            ROD_GRASP_POINT_IN_TCP + np.asarray([0.0, 0.0, ROD_TOP_ABOVE_GRASP_M], dtype=np.float32),
        ],
        dtype=np.float32,
    )

    attach_idx = -1
    if gripper_target.size > 1:
        close_candidates = np.flatnonzero(np.diff(gripper_target) < -0.2)
        if close_candidates.size > 0:
            attach_idx = int(close_candidates[0] + 1)
    if attach_idx < 0 and gripper_target.size > 0:
        grip_span = float(np.max(gripper_target) - np.min(gripper_target))
        if grip_span > 1e-4:
            closed_threshold = float(np.min(gripper_target) + 0.35 * grip_span)
            closed_mask = gripper_target <= closed_threshold
            if np.any(closed_mask):
                attach_idx = int(np.flatnonzero(closed_mask)[0])

    static_pose_idx = attach_idx if attach_idx >= 0 else count - 1
    static_rod_points = transform_points_from_tcp_frame(
        tcp_xyz=tcp_pose[static_pose_idx, :3],
        tcp_rotvec=tcp_pose[static_pose_idx, 3:6],
        points_in_tcp=rod_in_tcp,
    )

    rod_points = np.repeat(static_rod_points.reshape(1, 2, 3), count, axis=0).astype(np.float32)
    if attach_idx >= 0:
        for idx in range(attach_idx, count):
            rod_points[idx] = transform_points_from_tcp_frame(
                tcp_xyz=tcp_pose[idx, :3],
                tcp_rotvec=tcp_pose[idx, 3:6],
                points_in_tcp=rod_in_tcp,
            )
    return rod_points, attach_idx


def draw_reference_rod_3d(
    panel: np.ndarray,
    box: Tuple[int, int, int, int],
    bounds: Tuple[np.ndarray, np.ndarray],
    rod_pts: np.ndarray,
    color: Tuple[int, int, int] = ROD,
    thickness: int = 4,
    tip_radius: int = 6,
) -> None:
    rod_pts = np.asarray(rod_pts, dtype=np.float32).reshape(-1, 3)
    if len(rod_pts) != 2:
        return
    pixels, _ = project_points_3d(rod_pts, box, bounds)
    cv2.line(panel, tuple(int(v) for v in pixels[0]), tuple(int(v) for v in pixels[1]), color, thickness, cv2.LINE_AA)


def fixture_segments_in_tcp() -> np.ndarray:
    # CAD-free sketch of the real fixture: a rectangular body with a C-shaped cable slot.
    segments = [
        # Tool shank.
        ([0.0, 0.0, 0.000], [0.0, 0.0, 0.024]),
        # Rectangular body, front face.
        ([-0.030, -0.014, 0.024], [0.030, -0.014, 0.024]),
        ([0.030, -0.014, 0.024], [0.030, -0.014, 0.086]),
        ([0.030, -0.014, 0.086], [-0.030, -0.014, 0.086]),
        ([-0.030, -0.014, 0.086], [-0.030, -0.014, 0.024]),
        # Rectangular body, rear face.
        ([-0.030, 0.014, 0.024], [0.030, 0.014, 0.024]),
        ([0.030, 0.014, 0.024], [0.030, 0.014, 0.086]),
        ([0.030, 0.014, 0.086], [-0.030, 0.014, 0.086]),
        ([-0.030, 0.014, 0.086], [-0.030, 0.014, 0.024]),
        # Body depth edges.
        ([-0.030, -0.014, 0.024], [-0.030, 0.014, 0.024]),
        ([0.030, -0.014, 0.024], [0.030, 0.014, 0.024]),
        ([-0.030, -0.014, 0.086], [-0.030, 0.014, 0.086]),
        ([0.030, -0.014, 0.086], [0.030, 0.014, 0.086]),
        # Neck from the body into the C-slot.
        ([0.000, 0.0, 0.086], [0.038, 0.0, 0.098]),
        # C-shaped slot: closed spine on +X, open mouth toward -X.
        ([0.038, 0.0, 0.094], [0.038, 0.0, 0.180]),
        ([0.038, 0.0, 0.180], [-0.064, 0.0, 0.180]),
        ([0.038, 0.0, 0.094], [-0.064, 0.0, 0.094]),
    ]
    return np.asarray(segments, dtype=np.float32)


def fixture_c_slot_center_in_tcp() -> np.ndarray:
    return np.asarray([[-0.038, 0.0, 0.137]], dtype=np.float32)


def transform_fixture_segments_from_tcp(tcp_xyz: np.ndarray, tcp_rotvec: np.ndarray) -> np.ndarray:
    local_segments = fixture_segments_in_tcp()
    flat_points = local_segments.reshape(-1, 3)
    tcp_xyz = np.asarray(tcp_xyz, dtype=np.float32).reshape(1, 3)
    # For the fixture-hanging visualization, keep the fixture upright in world Z.
    # The real tool can be remounted, so the TCP rotation is not a reliable CAD frame.
    world_points = tcp_xyz + flat_points
    return world_points.reshape(local_segments.shape).astype(np.float32)


def transform_fixture_c_slot_center_from_tcp(tcp_xyz: np.ndarray, tcp_rotvec: np.ndarray) -> np.ndarray:
    tcp_xyz = np.asarray(tcp_xyz, dtype=np.float32).reshape(1, 3)
    return (tcp_xyz + fixture_c_slot_center_in_tcp())[0].astype(np.float32)


def draw_fixture_3d(
    panel: np.ndarray,
    box: Tuple[int, int, int, int],
    bounds: Tuple[np.ndarray, np.ndarray],
    fixture_segments: np.ndarray,
    color: Tuple[int, int, int] = FIXTURE,
    thickness: int = 4,
) -> None:
    segments = np.asarray(fixture_segments, dtype=np.float32).reshape(-1, 2, 3)
    c_slot_start = max(0, len(segments) - 3)
    for seg_idx, seg in enumerate(segments):
        pixels, _ = project_points_3d(seg, box, bounds)
        p0 = tuple(int(v) for v in pixels[0])
        p1 = tuple(int(v) for v in pixels[1])
        line_color = FIXTURE_SLOT if seg_idx >= c_slot_start else color
        line_thickness = thickness + 1 if seg_idx >= c_slot_start else thickness
        cv2.line(panel, p0, p1, SHADOW, line_thickness + 3, cv2.LINE_AA)
        cv2.line(panel, p0, p1, line_color, line_thickness, cv2.LINE_AA)
    if len(segments) > 0:
        slot_center = segments.reshape(-1, 3)[-3:].mean(axis=0)
        slot_px, _ = project_points_3d(slot_center.reshape(1, 3), box, bounds)
        cv2.circle(panel, tuple(int(v) for v in slot_px[0]), max(5, thickness + 2), WHITE, -1, cv2.LINE_AA)
        cv2.circle(panel, tuple(int(v) for v in slot_px[0]), max(4, thickness), color, -1, cv2.LINE_AA)


def build_fixed_bus_cable(fixture_c_slot_centers_world: np.ndarray) -> np.ndarray:
    slot_centers = np.asarray(fixture_c_slot_centers_world, dtype=np.float32).reshape(-1, 3)
    hook_center = slot_centers[-1] if len(slot_centers) > 0 else np.zeros(3, dtype=np.float32)
    half_length = 0.20
    # The physical bus cable is fixed in the world and runs parallel to robot Y.
    return np.asarray(
        [
            hook_center + np.asarray([0.0, -half_length, 0.0], dtype=np.float32),
            hook_center + np.asarray([0.0, half_length, 0.0], dtype=np.float32),
        ],
        dtype=np.float32,
    )


def draw_fixed_bus_cable_3d(
    panel: np.ndarray,
    box: Tuple[int, int, int, int],
    bounds: Tuple[np.ndarray, np.ndarray],
    cable_points: np.ndarray,
    color: Tuple[int, int, int] = CABLE,
) -> None:
    points = np.asarray(cable_points, dtype=np.float32).reshape(2, 3)
    pixels, _ = project_points_3d(points, box, bounds)
    cv2.line(panel, tuple(int(v) for v in pixels[0]), tuple(int(v) for v in pixels[1]), CABLE_SHADOW, 9, cv2.LINE_AA)
    cv2.line(panel, tuple(int(v) for v in pixels[0]), tuple(int(v) for v in pixels[1]), color, 5, cv2.LINE_AA)
    for px in pixels:
        cv2.circle(panel, tuple(int(v) for v in px), 5, color, -1, cv2.LINE_AA)


def build_platform_motion(
    relative_times_s: np.ndarray,
    disturbed: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    time_s = np.asarray(relative_times_s, dtype=np.float32).reshape(-1)
    frame_count = int(len(time_s))
    if frame_count <= 0 or not disturbed:
        return (
            np.zeros((max(frame_count, 1), 3), dtype=np.float32)[:frame_count],
            np.zeros((max(frame_count, 1),), dtype=np.float32)[:frame_count],
        )

    phase = 2.0 * np.pi * DISTURBANCE_FREQUENCY_HZ * (time_s - float(time_s[0]))
    # Keep the platform center fixed and let the short-edge mount move on a smooth oscillatory arc.
    yaw_deg = 4.8 * np.sin(phase)
    vertical = 0.00587 * np.sin(phase + np.pi / 2.0)
    lateral = 0.00253 * np.sin(phase)
    offsets = np.stack(
        [
            np.zeros_like(vertical),
            lateral.astype(np.float32),
            vertical.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return offsets, yaw_deg.astype(np.float32)


def build_platform_outline(
    platform_center: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    center_offset: np.ndarray,
    yaw_deg: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    low, high = bounds
    span = float(np.max(high - low))
    length = 0.42 * span
    width = 0.22 * span
    center = np.asarray(platform_center, dtype=np.float32).copy() + np.asarray(center_offset, dtype=np.float32)

    local = np.asarray(
        [
            [-0.5 * length, -0.5 * width, 0.0],
            [0.5 * length, -0.5 * width, 0.0],
            [0.5 * length, 0.5 * width, 0.0],
            [-0.5 * length, 0.5 * width, 0.0],
        ],
        dtype=np.float32,
    )
    rot = rotation_matrix_z(np.deg2rad(yaw_deg))
    corners = center + local @ rot.T
    mount_local = np.asarray([[-0.5 * length, 0.0, 0.0]], dtype=np.float32)
    mount = center + mount_local @ rot.T
    return corners.astype(np.float32), mount.reshape(3).astype(np.float32), center.astype(np.float32)


def draw_platform_outline(
    panel: np.ndarray,
    box: Tuple[int, int, int, int],
    bounds: Tuple[np.ndarray, np.ndarray],
    platform_center: np.ndarray,
    center_offset: np.ndarray,
    yaw_deg: float,
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> None:
    corners, mount, center = build_platform_outline(platform_center, bounds, center_offset, yaw_deg)
    pixels, _ = project_points_3d(corners, box, bounds)
    mount_px, _ = project_points_3d(mount.reshape(1, 3), box, bounds)
    center_px, _ = project_points_3d(center.reshape(1, 3), box, bounds)
    if len(pixels) != 4:
        return
    loop = [0, 1, 2, 3, 0]
    for idx in range(1, len(loop)):
        p0 = tuple(int(v) for v in pixels[loop[idx - 1]])
        p1 = tuple(int(v) for v in pixels[loop[idx]])
        cv2.line(panel, p0, p1, color, thickness, cv2.LINE_AA)
    cv2.line(panel, tuple(int(v) for v in pixels[0]), tuple(int(v) for v in pixels[2]), color, 1, cv2.LINE_AA)
    cv2.line(panel, tuple(int(v) for v in pixels[1]), tuple(int(v) for v in pixels[3]), color, 1, cv2.LINE_AA)
    cv2.circle(panel, tuple(int(v) for v in mount_px[0]), max(3, thickness + 1), color, -1, cv2.LINE_AA)
    cv2.circle(panel, tuple(int(v) for v in center_px[0]), max(3, thickness), WHITE, 1, cv2.LINE_AA)


def draw_disturbance_ghosts(
    panel: np.ndarray,
    box: Tuple[int, int, int, int],
    bounds: Tuple[np.ndarray, np.ndarray],
    platform_center: np.ndarray,
    platform_offsets: np.ndarray,
    platform_yaw: np.ndarray,
    tcp_pose: np.ndarray,
    joint_positions: np.ndarray,
    gripper_target: np.ndarray,
    rod_points_world: np.ndarray,
    rod_attach_idx: int,
    base_mount_ref: np.ndarray,
    current_idx: int,
) -> None:
    count = len(platform_offsets)
    sway_mag = float(np.max(np.linalg.norm(platform_offsets[:, [1, 2]], axis=1))) if count > 0 else 0.0
    if count == 0 or (np.max(np.abs(platform_yaw)) < 1e-6 and sway_mag < 1e-6):
        return
    overlay = panel.copy()
    local_offsets = np.asarray([-6, -3, 3, 6], dtype=np.int32)
    sample_ids = np.unique(np.clip(current_idx + local_offsets, 0, count - 1))
    outline_color = (120, 132, 150)
    shadow_color = (110, 122, 142)
    rod_shadow = (108, 168, 118)
    for ghost_idx in sample_ids:
        draw_platform_outline(
            panel=overlay,
            box=box,
            bounds=bounds,
            platform_center=platform_center,
            center_offset=platform_offsets[ghost_idx],
            yaw_deg=float(platform_yaw[ghost_idx]),
            color=outline_color,
            thickness=1,
        )
        ghost_joint = joint_positions[ghost_idx] if joint_positions is not None and len(joint_positions) > ghost_idx else None
        ghost_gripper = float(gripper_target[ghost_idx]) if len(gripper_target) > ghost_idx else 0.0
        draw_robot_silhouette_3d(
            panel=overlay,
            box=box,
            bounds=bounds,
            tcp_xyz=tcp_pose[ghost_idx, :3],
            tcp_rotvec=tcp_pose[ghost_idx, 3:6],
            gripper_target=ghost_gripper,
            base_pos=base_mount_ref,
            joint_row=ghost_joint,
            platform_center=platform_center,
            platform_offset=platform_offsets[ghost_idx],
            platform_yaw_deg=float(platform_yaw[ghost_idx]),
            color=shadow_color,
            link_scale=0.92,
        )
        if rod_attach_idx >= 0 and current_idx >= rod_attach_idx and ghost_idx >= rod_attach_idx:
            draw_reference_rod_3d(
                panel=overlay,
                box=box,
                bounds=bounds,
                rod_pts=rod_points_world[ghost_idx],
                color=rod_shadow,
                thickness=6,
                tip_radius=5,
            )
    cv2.addWeighted(overlay, 0.28, panel, 0.72, 0.0, panel)


def render_disturbance_status_inset(
    panel: np.ndarray,
    box: Tuple[int, int, int, int],
    disturbed: bool,
    rod_attached: bool,
    current_idx: int,
    fixture_task: bool = False,
) -> None:
    cv2.rectangle(panel, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), BORDER, 1, cv2.LINE_AA)
    draw_text(panel, "state cue", box[0] + 10, box[1] + 22, 0.55, MUTED, 1)
    if not disturbed:
        chip_x = box[0] + 10
        chip_y = box[1] + 34
        chip_x += draw_status_chip(panel, "static", chip_x, chip_y, ACCENT_2, WHITE) + 8
        phase_chip = "fixture-on-tool" if fixture_task else ("post-grasp" if rod_attached else "pre-grasp")
        draw_status_chip(panel, phase_chip, chip_x, chip_y, GREEN if rod_attached or fixture_task else ACCENT_2, WHITE)
        headline = "fixed cable, fixture follows TCP" if fixture_task else ("rod rigidly attached to TCP" if rod_attached else "rod fixed in task frame")
        detail_lines = [
            "robot, base, and arm remain still",
            "no disturbance injected",
            f"frame {current_idx + 1}",
        ]
    else:
        chip_x = box[0] + 10
        chip_y = box[1] + 34
        chip_x += draw_status_chip(panel, "sway cue", chip_x, chip_y, ACCENT) + 8
        phase_chip = "fixture-on-tool" if fixture_task else ("post-grasp" if rod_attached else "pre-grasp")
        draw_status_chip(
            panel,
            phase_chip,
            chip_x,
            chip_y,
            GREEN if rod_attached or fixture_task else ACCENT_2,
            WHITE,
        )
        if fixture_task:
            headline = "fixture/robot sway; cable stays fixed"
            detail_lines = [
                "disturbance applies to robot-side motion only",
                "target bus cable is fixed in world frame",
                f"frame {current_idx + 1}",
            ]
        else:
            headline = "rod rigidly follows robot motion" if rod_attached else "rod fixed in task frame"
            detail_lines = [
                "platform, base, arm, and TCP oscillate together",
                "same world-frame disturbance",
                f"frame {current_idx + 1}",
            ]
    draw_text(panel, headline, box[0] + 12, box[1] + 84, 0.5, WHITE, 1)
    y = box[1] + 112
    for line in detail_lines:
        draw_text(panel, line, box[0] + 12, y, 0.48, MUTED, 1)
        y += 22


def render_trajectory_panel(
    tcp_pose: np.ndarray,
    joint_positions: np.ndarray,
    gripper_target: np.ndarray,
    relative_times_s: np.ndarray,
    current_idx: int,
    width: int,
    height: int,
    task_label: str,
    disturbed: bool,
    mode: str,
) -> np.ndarray:
    panel = make_card(width, height)
    draw_panel_tag(panel, "3D robot workspace", ACCENT_2)
    fixture_task = is_fixture_hanging_task(task_label)

    body_top = 56
    info_height = 46
    body_height = max(80, height - body_top - info_height - 12)
    inset_w = max(170, int(round(width * 0.17)))
    main_box = (18, body_top, width - inset_w - 30, body_height)
    inset_box = (main_box[0] + main_box[2] + 10, body_top, inset_w, body_height)
    cv2.rectangle(panel, (main_box[0], main_box[1]), (main_box[0] + main_box[2], main_box[1] + main_box[3]), BORDER, 1, cv2.LINE_AA)
    cv2.rectangle(panel, (inset_box[0], inset_box[1]), (inset_box[0] + inset_box[2], inset_box[1] + inset_box[3]), BORDER, 1, cv2.LINE_AA)
    if len(tcp_pose) > 0:
        xyz = tcp_pose[:, :3]
        current_gripper_target = float(gripper_target[min(current_idx, len(gripper_target) - 1)]) if len(gripper_target) > 0 else 0.0
        base_pos = estimate_robot_base(xyz)
        prelim_bounds = build_workspace_bounds(xyz, base_pos)
        span = float(np.max(prelim_bounds[1] - prelim_bounds[0]))
        platform_center = base_pos.copy()
        platform_center[0] += 0.21 * span
        platform_center[2] -= 0.05 * span
        platform_offsets, platform_yaw = build_platform_motion(relative_times_s=relative_times_s, disturbed=disturbed)
        world_xyz = np.zeros_like(xyz)
        for idx in range(len(xyz)):
            world_xyz[idx] = apply_platform_transform_points(
                xyz[idx : idx + 1],
                platform_center=platform_center,
                center_offset=platform_offsets[idx],
                yaw_deg=float(platform_yaw[idx]),
            )[0]

        if fixture_task:
            attach_idx = -1
            rod_attached = False
            fixture_segments_world = []
            fixture_c_slot_centers_world = []
            for idx in range(len(tcp_pose)):
                fixture_segments = transform_fixture_segments_from_tcp(
                    tcp_xyz=tcp_pose[idx, :3],
                    tcp_rotvec=tcp_pose[idx, 3:6],
                )
                c_slot_center = transform_fixture_c_slot_center_from_tcp(
                    tcp_xyz=tcp_pose[idx, :3],
                    tcp_rotvec=tcp_pose[idx, 3:6],
                ).reshape(1, 3)
                fixture_segments = apply_platform_transform_points(
                    fixture_segments.reshape(-1, 3),
                    platform_center=platform_center,
                    center_offset=platform_offsets[idx],
                    yaw_deg=float(platform_yaw[idx]),
                ).reshape(fixture_segments.shape)
                c_slot_center = apply_platform_transform_points(
                    c_slot_center,
                    platform_center=platform_center,
                    center_offset=platform_offsets[idx],
                    yaw_deg=float(platform_yaw[idx]),
                )[0]
                fixture_segments_world.append(fixture_segments)
                fixture_c_slot_centers_world.append(c_slot_center)
            fixture_segments_world = np.stack(fixture_segments_world, axis=0).astype(np.float32)
            fixture_c_slot_centers_world = np.stack(fixture_c_slot_centers_world, axis=0).astype(np.float32)
            cable_points = build_fixed_bus_cable(fixture_c_slot_centers_world)
            rod_points_world = np.zeros((len(tcp_pose), 2, 3), dtype=np.float32)
            scene_points = np.concatenate(
                [world_xyz, fixture_segments_world.reshape(-1, 3), cable_points.reshape(-1, 3)],
                axis=0,
            )
        else:
            fixture_segments_world = None
            cable_points = None
            rod_points_base, attach_idx = build_reference_rod_points_sequence(tcp_pose=tcp_pose, gripper_target=gripper_target)
            rod_points_world = np.zeros_like(rod_points_base)
            for idx in range(len(rod_points_base)):
                transform_idx = attach_idx if attach_idx >= 0 and idx < attach_idx else idx
                rod_points_world[idx] = apply_platform_transform_points(
                    rod_points_base[idx],
                    platform_center=platform_center,
                    center_offset=platform_offsets[transform_idx],
                    yaw_deg=float(platform_yaw[transform_idx]),
                )
            scene_points = np.concatenate([world_xyz, rod_points_world.reshape(-1, 3)], axis=0)
        bounds = build_workspace_bounds(scene_points, base_pos)
        _, base_mount_ref, _ = build_platform_outline(
            platform_center=platform_center,
            bounds=bounds,
            center_offset=np.zeros(3, dtype=np.float32),
            yaw_deg=0.0,
        )
        platform_offsets_mm = platform_offsets[:, [1, 2]] * 1000.0
        render_box = (main_box[0] + 6, main_box[1] + 26, main_box[2] - 12, main_box[3] - 34)
        draw_cube(panel, render_box, bounds)
        draw_axes(panel, render_box, bounds)
        if not fixture_task:
            rod_attached = bool(attach_idx >= 0 and current_idx >= attach_idx)
        phase_text = (
            "fixed bus cable parallel to robot Y | fixture follows TCP"
            if fixture_task
            else ("short-edge mount | post-grasp: rod rigidly follows robot motion" if rod_attached else "short-edge mount | pre-grasp: rod fixed in task frame")
        )
        draw_text(panel, phase_text, main_box[0] + 10, main_box[1] + 22, 0.54, MUTED, 1)
        draw_disturbance_ghosts(
            panel=panel,
            box=render_box,
            bounds=bounds,
            platform_center=platform_center,
            platform_offsets=platform_offsets,
            platform_yaw=platform_yaw,
            tcp_pose=tcp_pose,
            joint_positions=joint_positions,
            gripper_target=gripper_target,
            rod_points_world=rod_points_world,
            rod_attach_idx=attach_idx,
            base_mount_ref=base_mount_ref,
            current_idx=current_idx,
        )
        if fixture_task and cable_points is not None:
            draw_fixed_bus_cable_3d(panel, render_box, bounds, cable_points)
        draw_platform_outline(
            panel=panel,
            box=render_box,
            bounds=bounds,
            platform_center=platform_center,
            center_offset=platform_offsets[current_idx],
            yaw_deg=float(platform_yaw[current_idx]),
            color=(205, 214, 226),
            thickness=2,
        )
        if fixture_task and fixture_segments_world is not None:
            draw_fixture_3d(panel, render_box, bounds, fixture_segments_world[current_idx])
        else:
            draw_reference_rod_3d(panel, render_box, bounds, rod_points_world[current_idx])

        current_joint = joint_positions[current_idx] if joint_positions is not None and len(joint_positions) > current_idx else None
        draw_robot_model_3d(
            panel=panel,
            box=render_box,
            bounds=bounds,
            tcp_xyz=xyz[current_idx],
            tcp_rotvec=tcp_pose[current_idx, 3:6],
            gripper_target=current_gripper_target,
            base_pos=base_mount_ref,
            joint_row=current_joint,
            platform_center=platform_center,
            platform_offset=platform_offsets[current_idx],
            platform_yaw_deg=float(platform_yaw[current_idx]),
            link_color=(82, 92, 108),
            joint_fill=(68, 76, 88),
            edge_color=WHITE,
            gripper_color=ACCENT,
            link_scale=1.05,
        )
        tcp_px, _ = project_points_3d(world_xyz[current_idx : current_idx + 1], render_box, bounds)
        if len(tcp_px) > 0:
            cv2.circle(panel, tuple(int(v) for v in tcp_px[0]), 7, WHITE, -1, cv2.LINE_AA)
            cv2.circle(panel, tuple(int(v) for v in tcp_px[0]), 4, ACCENT_2, -1, cv2.LINE_AA)

        sway_rms = float(np.sqrt(np.mean(np.sum(np.square(platform_offsets_mm), axis=1)))) if len(platform_offsets_mm) > 0 else 0.0
        render_disturbance_status_inset(
            panel=panel,
            box=inset_box,
            disturbed=disturbed,
            rod_attached=rod_attached,
            current_idx=current_idx,
            fixture_task=fixture_task,
        )
        if fixture_task:
            draw_text(panel, "fixed cable", main_box[0] + main_box[2] - 172, main_box[1] + 24, 0.46, CABLE, 1)
            draw_text(panel, "fixture", main_box[0] + main_box[2] - 70, main_box[1] + 24, 0.46, FIXTURE, 1)
            rod_state_text = "target cable fixed in world; fixture moves with robot"
        else:
            draw_text(panel, "platform", main_box[0] + main_box[2] - 132, main_box[1] + 24, 0.46, WHITE, 1)
            draw_text(panel, "rod", main_box[0] + main_box[2] - 34, main_box[1] + 24, 0.46, ROD, 1)
            if rod_attached:
                rod_state_text = "post-grasp: rod rigidly follows world-frame robot motion" if disturbed else "post-grasp: rod rigidly attached to TCP"
            else:
                rod_state_text = "pre-grasp: rod fixed until grasp"
    else:
        sway_rms = 0.0
        current_gripper_target = 0.0
        rod_state_text = "rod state unavailable"

    info_y = body_top + body_height + 24
    current_time_s = float(relative_times_s[min(current_idx, len(relative_times_s) - 1)]) if len(relative_times_s) > 0 else 0.0
    line_1 = f"task: {task_label} | mode: {mode} | frame: {current_idx + 1}/{len(tcp_pose)} | time: {current_time_s:.2f} s"
    state_text = "no disturbance" if not disturbed else "periodic world-frame sway cue"
    if fixture_task:
        line_2 = f"sway RMS: {sway_rms:.1f} mm | {state_text} | {rod_state_text}"
    else:
        line_2 = f"gripper target: {current_gripper_target:.3f} | sway RMS: {sway_rms:.1f} mm | {state_text} | {rod_state_text}"
    draw_text(panel, line_1, 20, info_y, 0.54, WHITE, 1)
    draw_text(panel, line_2, 20, info_y + 18, 0.52, MUTED, 1)
    return panel


def canonical_task_label(task_id: str, task_name: str, fallback_name: str) -> str:
    mapping = {
        "pole_pickoff": "Pole pickup",
        "pole_pickoff_shake": "Pole pickup (disturbed)",
        "pole_hang_on_line": "Pole hanging",
        "pole_hang_on_line_shake": "Pole hanging (disturbed)",
        "guagongzhuang": "Fixture hanging",
        "guagongzhuang_huangdong": "Fixture hanging (disturbed)",
        "挂工装_晃动": "Fixture hanging (disturbed)",
    }
    fallback_lower = str(fallback_name or "").lower()
    task_name_text = str(task_name or "")
    if "guagongzhuang_huangdong" in fallback_lower or "晃动" in task_name_text:
        return "Fixture hanging (disturbed)"
    if "guagongzhuang" in fallback_lower or "挂工装" in task_name_text:
        return "Fixture hanging"
    if task_id in mapping:
        return mapping[task_id]
    candidates = [task_name, fallback_name, task_id]
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text:
            continue
        ascii_text = text.encode("ascii", errors="ignore").decode("ascii").strip()
        if ascii_text:
            return ascii_text.replace("_", " ").title()
        if all(ord(ch) < 128 for ch in text):
            return text.replace("_", " ").title()
    return "Task rollout"


def choose_frame_indices(
    tcp_pose: np.ndarray,
    gripper_target: np.ndarray,
    mode: str,
    translation_window: int,
    rotation_window: int,
    translation_idle_threshold_m: float,
    rotation_idle_threshold_rad: float,
    rotation_delta_frame: str,
) -> np.ndarray:
    if mode == "raw":
        return np.arange(len(tcp_pose), dtype=np.int64)

    smoothed_pose = smooth_tcp_pose_sequence(
        tcp_pose=tcp_pose,
        translation_window=translation_window,
        rotation_window=rotation_window,
    )

    if mode == "preserve":
        start_idx, end_idx = select_active_window(
            tcp_pose=smoothed_pose,
            gripper_target=gripper_target,
            translation_idle_threshold_m=translation_idle_threshold_m,
            rotation_idle_threshold_rad=rotation_idle_threshold_rad,
            rotation_delta_frame=rotation_delta_frame,
            context_frames=1,
        )
        return np.arange(start_idx, end_idx + 1, dtype=np.int64)

    if mode == "compress":
        keep = select_observation_indices(
            tcp_pose=smoothed_pose,
            gripper_target=gripper_target,
            translation_idle_threshold_m=translation_idle_threshold_m,
            rotation_idle_threshold_rad=rotation_idle_threshold_rad,
            rotation_delta_frame=rotation_delta_frame,
        )
        return np.asarray(keep, dtype=np.int64)

    raise ValueError(f"unsupported mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a raw real-robot episode into a supplementary showcase video."
    )
    parser.add_argument("--episode-dir", required=True, help="Example: data_all/raw/pole_pickoff/episode_0000")
    parser.add_argument("--output-mp4", required=True, help="Output MP4 path")
    parser.add_argument("--mode", choices=["raw", "preserve", "compress"], default="preserve")
    parser.add_argument("--global-camera", default="global_d405")
    parser.add_argument("--wrist-camera", default="wrist_d405")
    parser.add_argument("--translation-window", type=int, default=7)
    parser.add_argument("--rotation-window", type=int, default=5)
    parser.add_argument("--translation-idle-threshold-m", type=float, default=5e-4)
    parser.add_argument("--rotation-idle-threshold-rad", type=float, default=5e-4)
    parser.add_argument("--rotation-delta-frame", choices=["base", "tool"], default="base")
    parser.add_argument("--fps", type=float, default=0.0, help="If zero, estimate FPS from timestamps")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--layout", choices=["full", "supp_clean"], default="supp_clean")
    return parser.parse_args()


def main() -> None:
    if cv2 is None:
        raise ModuleNotFoundError("render_real_episode_showcase.py requires cv2.")

    args = parse_args()
    episode_dir = Path(args.episode_dir).resolve()
    output_mp4 = Path(args.output_mp4).resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    robot_dir = episode_dir / "robot"
    metadata_path = episode_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    task_id = str(metadata.get("task_id", episode_dir.parent.name))
    task_name = str(metadata.get("task_name", ""))
    task_label = canonical_task_label(task_id=task_id, task_name=task_name, fallback_name=episode_dir.parent.name)
    task_key = f"{task_id} {task_name} {episode_dir.parent.name}".lower()
    disturbed = (
        ("shake" in task_key)
        or ("disturbed" in task_label.lower())
        or ("huangdong" in task_key)
        or ("晃动" in task_key)
        or ("扰动" in task_key)
    )

    tcp_pose = load_csv_array(robot_dir / "tcp_pose.csv").astype(np.float32)
    joint_positions_path = robot_dir / "joint_positions.csv"
    joint_positions = load_csv_array(joint_positions_path).astype(np.float32) if joint_positions_path.exists() else None
    timestamps = load_csv_array(robot_dir / "timestamp.csv").reshape(-1).astype(np.float64)
    gripper_target = load_csv_array(robot_dir / "gripper_target_fraction.csv").reshape(-1).astype(np.float32)

    frame_indices = choose_frame_indices(
        tcp_pose=tcp_pose,
        gripper_target=gripper_target,
        mode=args.mode,
        translation_window=args.translation_window,
        rotation_window=args.rotation_window,
        translation_idle_threshold_m=args.translation_idle_threshold_m,
        rotation_idle_threshold_rad=args.rotation_idle_threshold_rad,
        rotation_delta_frame=args.rotation_delta_frame,
    )
    if frame_indices.size == 0:
        raise RuntimeError("No frames were selected for rendering.")

    dt = np.diff(timestamps[frame_indices])
    dt = dt[np.abs(dt) > 1e-6]
    fps = float(args.fps) if args.fps > 0 else float(1.0 / np.median(dt)) if dt.size > 0 else 15.0
    fps = float(np.clip(fps, 5.0, 30.0))
    relative_timestamps = timestamps - timestamps[frame_indices[0]]

    global_rgb_dir = episode_dir / "cameras" / args.global_camera / "rgb"
    wrist_rgb_dir = episode_dir / "cameras" / args.wrist_camera / "rgb"
    global_ply_dir = episode_dir / "cameras" / args.global_camera / "point_cloud"
    wrist_ply_dir = episode_dir / "cameras" / args.wrist_camera / "point_cloud"

    global_ply_paths = [global_ply_dir / f"{idx:06d}.ply" for idx in frame_indices]
    wrist_ply_paths = [wrist_ply_dir / f"{idx:06d}.ply" for idx in frame_indices]
    global_bounds = build_point_cloud_bounds(global_ply_paths, projection="xz")
    wrist_bounds = build_point_cloud_bounds(wrist_ply_paths, projection="iso")

    writer = cv2.VideoWriter(
        str(output_mp4),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(args.width), int(args.height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_mp4}")

    header_h = 72
    content_h = args.height - header_h
    top_h = int(round(content_h * 0.34))
    bottom_h = content_h - top_h
    left_w = args.width // 2
    right_w = args.width - left_w
    bottom_col_w = int(round(args.width * 0.29))

    for local_idx, frame_idx in enumerate(frame_indices):
        global_rgb = cv2.imread(str(global_rgb_dir / f"{frame_idx:06d}.png"), cv2.IMREAD_COLOR)
        wrist_rgb = cv2.imread(str(wrist_rgb_dir / f"{frame_idx:06d}.png"), cv2.IMREAD_COLOR)

        global_panel = render_rgb_panel(global_rgb, left_w, top_h, "global RGB")
        wrist_panel = render_rgb_panel(wrist_rgb, right_w, top_h, "wrist RGB")

        global_pc_panel = render_point_cloud_panel(
            ply_path=global_ply_dir / f"{frame_idx:06d}.ply",
            width=bottom_col_w,
            height=bottom_h,
            projection="xz",
            bounds=global_bounds,
            color_axis=1,
            title="global cloud",
        )
        wrist_pc_panel = render_point_cloud_panel(
            ply_path=wrist_ply_dir / f"{frame_idx:06d}.ply",
            width=bottom_col_w,
            height=bottom_h,
            projection="iso",
            bounds=wrist_bounds,
            color_axis=1,
            title="wrist-local cloud",
        )
        traj_width = args.width - (2 * bottom_col_w if args.layout == "full" else bottom_col_w)
        traj_panel = render_trajectory_panel(
            tcp_pose=tcp_pose[frame_indices],
            joint_positions=joint_positions[frame_indices] if joint_positions is not None else None,
            gripper_target=gripper_target[frame_indices],
            relative_times_s=relative_timestamps[frame_indices],
            current_idx=local_idx,
            width=traj_width,
            height=bottom_h,
            task_label=task_label,
            disturbed=disturbed,
            mode=args.mode,
        )

        canvas = np.full((args.height, args.width, 3), BG, dtype=np.uint8)
        canvas[:header_h, :] = np.asarray(HEADER, dtype=np.uint8)
        draw_text(canvas, task_label, 18, 30, 0.95, WHITE, 2)
        subtitle = f"{episode_dir.name} | {args.mode} timeline | {fps:.1f} FPS | frame {local_idx + 1}/{len(frame_indices)}"
        draw_text(canvas, subtitle, 18, 58, 0.56, MUTED, 1)
        if args.layout == "supp_clean":
            draw_text(canvas, "supplementary multimedia", max(18, args.width - 226), 58, 0.46, ACCENT, 1)
        else:
            draw_text(canvas, "full layout", args.width - 120, 58, 0.52, ACCENT, 1)

        canvas[header_h : header_h + top_h, :left_w] = global_panel
        canvas[header_h : header_h + top_h, left_w:] = wrist_panel
        if args.layout == "full":
            canvas[header_h + top_h :, :bottom_col_w] = global_pc_panel
            canvas[header_h + top_h :, bottom_col_w : 2 * bottom_col_w] = wrist_pc_panel
            canvas[header_h + top_h :, 2 * bottom_col_w :] = traj_panel
        else:
            canvas[header_h + top_h :, :bottom_col_w] = wrist_pc_panel
            canvas[header_h + top_h :, bottom_col_w:] = traj_panel

        writer.write(canvas)

    writer.release()
    print(f"[done] video={output_mp4}")


if __name__ == "__main__":
    main()
