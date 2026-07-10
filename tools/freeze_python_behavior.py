"""
冻结 MP1 Python 部署样本工具

本程序用于在 Python 到 C++ 部署对齐验证前，收集和记录策略执行的输入输出数据。

核心功能:
1. 扫描策略执行生成的 .npz 样本文件
2. 提取每个样本的统计摘要（形状、类型、最小值、最大值、均值、标准差）
3. 生成行为冻结清单 (manifest.json)，记录：
   - 模型信息：检查点路径、配置文件路径
   - 任务信息：任务 ID、动作模式
   - 数据契约：预期输入输出形状
   - 样本清单：实际样本的统计摘要

使用场景:
- Python/C++ 策略行为对齐验证
- 确保部署前后策略行为一致
- 作为 C++ 推理的验证基准

输入:
- 策略 dump 文件 (policy_trace_*/trial_*/policy_dumps/*.npz)
- 任务配置文件 (pole_pickoff.json)

输出:
- python_behavior_manifest.json (默认路径：deploy_artifacts/)

用法:
    python tools/freeze_python_behavior.py
    python tools/freeze_python_behavior.py --max-samples 50 --output my_manifest.json

注意事项:
- 样本数据应来自性能达标的 Python 策略（用于验证对齐）
- 仅用于对齐验证，样本质量不影响验证准确性
- 使用 npz 中的 model_obs_* 和 action_* 字段
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


# 默认配置文件路径：机器人任务配置 JSON
DEFAULT_CONFIG = "python_deploy/real_robot_ur12e_d405_speed_only/configs/pole_pickoff.json"
# 默认 dump 文件匹配模式：策略执行数据的路径通配符
DEFAULT_DUMP_GLOB = "python_deploy/deploy_results/quganzi/policy_trace_*/trial_*/policy_dumps/*.npz"


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    
    返回:
        argparse.Namespace: 包含所有命令行参数的命名空间对象
    """
    # 解析命令行参数，配置冻结 Python 部署样本的选项
    parser = argparse.ArgumentParser(description="Freeze MP1 Python deployment samples before C++ alignment.")
    parser.add_argument("--checkpoint", default="python_deploy/checkpoints/latest.ckpt",
                        help="模型检查点文件路径")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="任务配置文件路径")
    parser.add_argument("--dump-glob", default=DEFAULT_DUMP_GLOB,
                        help="策略 dump 文件的 glob 匹配模式")
    parser.add_argument("--output", default="deploy_artifacts/python_behavior_manifest.json",
                        help="输出的 manifest JSON 文件路径")
    parser.add_argument("--max-samples", type=int, default=20,
                        help="最大样本数量")
    return parser.parse_args()


def summarize_array(value: np.ndarray) -> Dict[str, Any]:
    """
    将 numpy 数组转换为统计摘要字典。
    
    参数:
        value: 输入的 numpy 数组
        
    返回:
        Dict[str, Any]: 包含数组统计信息的字典，包括：
            - shape: 数组形状列表
            - dtype: 数据类型
            - min/max/mean/std: 数值统计信息（仅当数组为数值类型时包含）
    """
    # 将 numpy 数组转换为统计摘要字典
    arr = np.asarray(value)
    result: Dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    # 如果数组非空且是数值类型，计算统计信息
    if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
        arr_f = arr.astype(np.float32, copy=False)
        result.update(
            {
                "min": float(np.min(arr_f)),
                "max": float(np.max(arr_f)),
                "mean": float(np.mean(arr_f)),
                "std": float(np.std(arr_f)),
            }
        )
    return result


def load_json(path: Path) -> Dict[str, Any]:
    """
    读取并解析 JSON 配置文件。
    
    参数:
        path: JSON 文件路径
        
    返回:
        Dict[str, Any]: 解析后的 JSON 数据字典
        
    异常:
        JSONDecodeError: 如果文件格式不是有效的 JSON
    """
    # 读取并解析 JSON 配置文件
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sample_summary(path: Path) -> Dict[str, Any]:
    """
    从单个 npz 样本文件中提取摘要信息。
    
    参数:
        path: npz 文件路径
        
    返回:
        Dict[str, Any]: 包含样本摘要信息的字典，包括：
            - path: 文件路径字符串
            - step_idx: 步骤索引
            - trial_idx: 试验索引
            - inputs: 输入数据摘要（图像、点云、位置）
            - outputs: 输出数据摘要（动作）
            
    异常:
        KeyError: 如果缺少必需的数据字段
    """
    # 从单个 npz 样本文件中提取摘要信息
    data = np.load(path)
    # 必需的关键数据字段
    required_keys = [
        "model_obs_global_image",
        "model_obs_wrist_image",
        "model_obs_point_cloud",
        "model_obs_agent_pos",
        "action_raw",
        "action_executed",
    ]
    # 检查是否存在所有必需字段
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")

    return {
        "path": str(path),
        "step_idx": int(np.asarray(data.get("step_idx", [-1])).reshape(-1)[0]),
        "trial_idx": int(np.asarray(data.get("trial_idx", [-1])).reshape(-1)[0]),
        # 输入数据：全局图像、手腕图像、点云、代理位置
        "inputs": {
            "global_image": summarize_array(data["model_obs_global_image"]),
            "wrist_image": summarize_array(data["model_obs_wrist_image"]),
            "point_cloud": summarize_array(data["model_obs_point_cloud"]),
            "agent_pos": summarize_array(data["model_obs_agent_pos"]),
        },
        # 输出数据：原始动作、执行动作
        "outputs": {
            "action_raw": summarize_array(data["action_raw"]),
            "action_executed": summarize_array(data["action_executed"]),
        },
    }


def main() -> None:
    """
    主函数：生成 Python 行为冻结清单。
    
    执行流程:
        1. 解析命令行参数
        2. 扫描策略 dump 文件
        3. 加载任务配置
        4. 生成样本摘要
        5. 构建并写入 manifest JSON 文件
        
    输出:
        在指定路径生成 python_behavior_manifest.json 文件
        
    异常:
        FileNotFoundError: 如果没有找到匹配的 dump 文件
    """
    # 主函数：生成 Python 行为冻结清单
    args = parse_args()
    config_path = Path(args.config)
    output_path = Path(args.output)
    # 创建输出目录（如果不存在）
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 查找匹配的 dump 文件
    dump_paths = sorted(Path(".").glob(args.dump_glob))
    if not dump_paths:
        raise FileNotFoundError(f"No policy dumps matched: {args.dump_glob}")
    # 选择前 N 个样本
    selected = dump_paths[: max(1, int(args.max_samples))]
    # 加载配置获取任务信息
    config = load_json(config_path)

    # 构建 manifest JSON 结构
    manifest: Dict[str, Any] = {
        "format_version": 1,
        "checkpoint": str(Path(args.checkpoint)),
        "config": str(config_path),
        "task": config.get("task", {}).get("task_id", "pole_pickoff"),
        "action_mode": config.get("representation", {}).get("action_mode"),
        "expected_shapes": {
            "global_image": [1, 2, 3, 128, 128],
            "wrist_image": [1, 2, 3, 96, 96],
            "point_cloud": [1, 2, 512, 3],
            "agent_pos": [1, 2, 10],
            "action": [1, 4, 7],
        },
        "sample_count": len(selected),
        "samples": [sample_summary(path) for path in selected],
    }

    # 写入 JSON 清单文件
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)

    print(f"wrote {output_path}")
    print(f"frozen samples: {len(selected)}")


if __name__ == "__main__":
    main()
