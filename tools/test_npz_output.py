"""测试脚本：查看 .npz 文件的实际输出内容"""
import numpy as np
from pathlib import Path


def inspect_npz_file(file_path: str):
    """
    检查单个 npz 文件的所有数据内容。
    
    参数:
        file_path: npz 文件路径
    """
    path = Path(file_path)
    print(f"\n{'='*80}")
    print(f"文件路径：{path}")
    print(f"{'='*80}\n")
    
    # 加载数据
    data = np.load(path)
    
    # 列出所有键
    print("[INFO] 文件中的键列表:")
    for key in data.keys():
        arr = data[key]
        print(f"  - {key}")
        print(f"    形状：{arr.shape}")
        print(f"    类型：{arr.dtype}")
        
        # 如果是数值数组，显示统计信息
        if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
            arr_f = arr.astype(np.float32, copy=False)
            print(f"    最小值：{np.min(arr_f):.6f}")
            print(f"    最大值：{np.max(arr_f):.6f}")
            print(f"    平均值：{np.mean(arr_f):.6f}")
            print(f"    标准差：{np.std(arr_f):.6f}")
            print(f"    元素数：{arr.size}")
        print()
    
    # 显示每个数组的前几个值
    print("\n[INFO] 数据内容预览 (前 3 个元素):")
    for key in data.keys():
        arr = data[key]
        print(f"\n  {key}:")
        if arr.size > 0:
            # 展平后取前 3 个
            flat = arr.flatten()[:3]
            for i, val in enumerate(flat):
                if np.issubdtype(arr.dtype, np.number):
                    print(f"    [{i}] = {val:.6f}")
                else:
                    print(f"    [{i}] = {val}")


def main():
    """主函数：测试所有找到的 npz 文件"""
    # 查找第一个 npz 文件
    dump_glob = "python_deploy/deploy_results/quganzi/policy_trace_*/trial_*/policy_dumps/*.npz"
    npz_files = sorted(Path(".").glob(dump_glob))
    
    if not npz_files:
        print(f"未找到匹配的文件：{dump_glob}")
        return
    
    print(f"找到 {len(npz_files)} 个 .npz 文件")
    print(f"使用第一个文件进行测试：{npz_files[0]}\n")
    
    # 检查第一个文件
    inspect_npz_file(str(npz_files[0]))
    
    # 询问是否检查更多文件
    if len(npz_files) > 1:
        print(f"\n是否要检查其他文件？(y/n) ", end="")
        choice = input().strip().lower()
        if choice == 'y':
            for f in npz_files[1:5]:  # 最多再检查 4 个
                print("\n" + ">"*40)
                inspect_npz_file(str(f))
                print("<"*40)


if __name__ == "__main__":
    main()
