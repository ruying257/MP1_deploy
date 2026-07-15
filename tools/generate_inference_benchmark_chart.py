#!/usr/bin/env python3
"""生成 README 使用的 Jetson 推理性能 SVG 图表。"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Iterable, Sequence


SVG_WIDTH = 1100
SVG_HEIGHT = 560

LATENCY_METRICS = (
    ("TorchScript CPU", 331.012, "#64748B"),
    ("TorchScript CUDA", 56.908, "#16A34A"),
    ("TensorRT FP16", 14.686, "#EA580C"),
)

SPEEDUP_METRICS = (
    ("CPU → CUDA", 5.82, "#16A34A"),
    ("CUDA → TensorRT", 3.87, "#EA580C"),
)


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 16,
    weight: int = 400,
    anchor: str = "start",
    color: str = "#111827",
) -> str:
    """生成单行 SVG 文本元素。

    Args:
        x: 文本横坐标。
        y: 文本基线纵坐标。
        value: 文本内容。
        size: 字号。
        weight: 字重。
        anchor: 文本锚点。
        color: 文本颜色。

    Returns:
        转义后的 SVG 文本字符串。
    """
    escaped = html.escape(value)
    return (
        f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}">'
        f"{escaped}</text>"
    )


def _lines(values: Iterable[str]) -> str:
    """按固定换行规则拼接 SVG 元素。"""
    return "\n".join(values)


def _render_latency_panel() -> Sequence[str]:
    """渲染 p50 延迟柱状图。"""
    elements = [
        '<rect x="40" y="108" width="650" height="390" rx="8" fill="#FFFFFF" stroke="#D1D5DB"/>',
        _text(64, 145, "p50 延迟（越低越好）", size=20, weight=700),
    ]

    chart_left = 86
    chart_right = 660
    chart_top = 178
    chart_bottom = 420
    max_latency = 350.0

    for tick in (0, 100, 200, 300):
        y = chart_bottom - tick / max_latency * (chart_bottom - chart_top)
        elements.append(
            f'<line x1="{chart_left}" y1="{y:.2f}" x2="{chart_right}" y2="{y:.2f}" '
            'stroke="#E5E7EB" stroke-width="1"/>'
        )
        elements.append(_text(chart_left - 12, y + 5, str(tick), size=13, anchor="end", color="#6B7280"))

    bar_width = 108
    bar_centers = (180, 374, 568)
    for (label, value, color), center_x in zip(LATENCY_METRICS, bar_centers):
        height = value / max_latency * (chart_bottom - chart_top)
        y = chart_bottom - height
        elements.append(
            f'<rect x="{center_x - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width}" '
            f'height="{height:.2f}" rx="4" fill="{color}"/>'
        )
        label_y = max(y - 12, chart_top - 2)
        elements.append(_text(center_x, label_y, f"{value:.3f} ms", size=15, weight=700, anchor="middle"))
        elements.append(_text(center_x, 451, label, size=14, weight=600, anchor="middle"))
        if label == "TensorRT FP16":
            elements.append(_text(center_x, 472, "离线 full-loop", size=13, anchor="middle", color="#9A3412"))

    return elements


def _render_speedup_panel() -> Sequence[str]:
    """渲染阶段加速比横向柱状图。"""
    elements = [
        '<rect x="716" y="108" width="344" height="390" rx="8" fill="#FFFFFF" stroke="#D1D5DB"/>',
        _text(740, 145, "阶段加速比（越高越好）", size=20, weight=700),
    ]

    bar_left = 740
    bar_max_width = 270
    max_speedup = 6.0
    rows = (220, 345)

    for (label, value, color), y in zip(SPEEDUP_METRICS, rows):
        elements.append(_text(bar_left, y - 24, label, size=15, weight=600))
        elements.append(
            f'<rect x="{bar_left}" y="{y}" width="{bar_max_width}" height="38" '
            'rx="4" fill="#F3F4F6"/>'
        )
        width = value / max_speedup * bar_max_width
        elements.append(
            f'<rect x="{bar_left}" y="{y}" width="{width:.2f}" height="38" '
            f'rx="4" fill="{color}"/>'
        )
        elements.append(_text(bar_left + width - 10, y + 26, f"{value:.2f}x", size=17, weight=700, anchor="end", color="#FFFFFF"))

    elements.append(_text(740, 421, "仅展示两段实测 p50 比值", size=14, color="#4B5563"))
    elements.append(_text(740, 445, "不合并为端到端总加速", size=14, color="#4B5563"))
    return elements


def build_svg() -> str:
    """构建完整 SVG 文本。"""
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" '
        f'viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img" aria-labelledby="title desc">',
        '<title id="title">MP1 Jetson 端侧推理性能</title>',
        '<desc id="desc">TorchScript CPU、TorchScript CUDA 与 TensorRT FP16 的 p50 延迟，以及两段实测加速比。</desc>',
        '<rect width="1100" height="560" fill="#F8FAFC"/>',
        _text(40, 48, "MP1 Jetson 端侧推理性能", size=28, weight=700),
        _text(40, 78, "Jetson Orin，5 次 warmup，200 次正式样本", size=15, color="#4B5563"),
        *_render_latency_panel(),
        *_render_speedup_panel(),
        _text(
            550,
            532,
            "TensorRT FP16 为冻结 case 离线验证，当前真机主链路仍使用 TorchScript / C++。",
            size=14,
            anchor="middle",
            color="#4B5563",
        ),
        "</svg>",
    ]
    return _lines(elements) + "\n"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    default_output = Path(__file__).resolve().parents[1] / "asserts" / "inference_benchmark.svg"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="SVG 输出路径，默认写入 asserts/inference_benchmark.svg。",
    )
    return parser.parse_args()


def main() -> None:
    """生成 SVG 图表并写入目标文件。"""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(build_svg())
    print(f"已生成图表：{args.output}")


if __name__ == "__main__":
    main()
