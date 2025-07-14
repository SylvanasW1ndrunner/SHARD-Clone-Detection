# evaluate_baselines.py (路径硬编码版)

import os
import numpy as np
import pandas as pd
import json
from sklearn.metrics import classification_report, recall_score
from tqdm import tqdm

# ==============================================================================
# 1. 配置区域: 请在这里修改您的文件路径
# ==============================================================================

# 定义存放所有基线工具数据的根目录
BASELINE_DATA_DIR = "../../2_tools_label/dataset_II_contract_level/"

# 自动拼接各个文件的完整路径
TEST_CSV_PATH = "../test.csv"
CONTRACT_MAP_PATH = os.path.join(BASELINE_DATA_DIR, "index_map/contract_label_380.json")
OUTPUT_REPORT_PATH = "baseline_performance_report.txt"

# 定义所有基线工具及其结果矩阵的路径
# 脚本会自动使用 BASELINE_DATA_DIR 作为前缀
TOOLS = {
    "Deckard": "Deckard/compressed_matrix.npy",
    "EClone": "EClone/compressed_matrix.npy",
    "Nicad": "Nicad/compressed_matrix.npy",
    "SmartEmbed": "SmartEmbed/compressed_matrix.npy",
    "SourcererCC": "SourcererCC/compressed_matrix.npy",
}

# ==============================================================================
# 2. 主评估函数
# ==============================================================================

def explore_npy_file(filepath, max_rows=5):
    """
    探索 .npy 文件的内容，打印基本信息和前几行数据

    参数:
    filepath: .npy 文件路径
    max_rows: 显示的最大行数，默认为5
    """

    # 检查文件是否存在
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return

    try:
        print(f"📁 正在加载文件: {filepath}")
        print("=" * 50)

        # 加载数据
        data = np.load(filepath, allow_pickle=True)

        # 基本信息
        print("📊 基本信息:")
        print(f"   数据类型: {type(data)}")
        print(f"   NumPy 数据类型: {data.dtype}")
        print(f"   数据形状: {data.shape}")
        print(f"   维度数: {data.ndim}")
        print(f"   总元素数: {data.size}")
        print(f"   内存占用: {data.nbytes / 1024:.2f} KB")
        print()

        # 根据维度显示不同信息
        if data.ndim == 0:
            # 标量
            print("📋 数据内容 (标量):")
            print(f"   值: {data}")

        elif data.ndim == 1:
            # 一维数组
            print("📋 数据内容 (一维数组):")
            print(f"   长度: {len(data)}")
            print(f"   前 {min(max_rows, len(data))} 个元素:")
            for i, item in enumerate(data[:max_rows]):
                print(f"   [{i}]: {item}")
            if len(data) > max_rows:
                print(f"   ... (还有 {len(data) - max_rows} 个元素)")

        elif data.ndim == 2:
            # 二维数组 (类似表格)
            print("📋 数据内容 (二维数组/表格):")
            print(f"   行数: {data.shape[0]}")
            print(f"   列数: {data.shape[1]}")

            # 显示列索引作为"表头"
            print("\n   列索引:", end="")
            for j in range(min(10, data.shape[1])):  # 最多显示10列
                print(f"{j:>10}", end="")
            if data.shape[1] > 10:
                print("       ...")
            else:
                print()

            # 显示前几行数据
            rows_to_show = min(max_rows, data.shape[0])
            print(f"\n   前 {rows_to_show} 行数据:")
            for i in range(rows_to_show):
                print(f"   [{i}]:", end="")
                cols_to_show = min(10, data.shape[1])  # 最多显示10列
                for j in range(cols_to_show):
                    # 格式化数字显示
                    if np.issubdtype(data.dtype, np.floating):
                        print(f"{data[i, j]:>10.3f}", end="")
                    else:
                        print(f"{data[i, j]:>10}", end="")
                if data.shape[1] > 10:
                    print("       ...")
                else:
                    print()

            if data.shape[0] > max_rows:
                print(f"   ... (还有 {data.shape[0] - max_rows} 行)")

        else:
            # 多维数组
            print(f"📋 数据内容 ({data.ndim}维数组):")
            print("   这是一个高维数组，显示整体统计信息:")

            # 尝试显示统计信息
            if np.issubdtype(data.dtype, np.number):
                print(f"   最小值: {np.min(data)}")
                print(f"   最大值: {np.max(data)}")
                print(f"   平均值: {np.mean(data):.3f}")
                print(f"   标准差: {np.std(data):.3f}")

            # 显示第一个"切片"
            print(f"\n   第一个切片 [0] 的形状: {data[0].shape}")
            print(f"   第一个切片的前几个元素:")
            flat_slice = data[0].flatten()
            for i, item in enumerate(flat_slice[:10]):
                print(f"   [{i}]: {item}")
            if len(flat_slice) > 10:
                print(f"   ... (该切片还有 {len(flat_slice) - 10} 个元素)")

        # 特殊数据类型处理
        if data.dtype == 'object':
            print("\n⚠️  注意: 这是一个对象数组，可能包含复杂的Python对象")
            print("   第一个元素的类型:", type(data.flat[0]) if data.size > 0 else "空数组")

        print("\n" + "=" * 50)
        print("✅ 文件探索完成!")

        return data

    except Exception as e:
        print(f"❌ 加载文件时出错: {e}")
        return None

def evaluate_all_baselines():
    """
    主函数，用于评估所有基线工具的性能。
    """
    # --- 加载必要的文件 ---
    print("正在加载测试数据和映射文件...")
    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        with open(CONTRACT_MAP_PATH, 'r', encoding='utf-8') as f:
            name_to_idx = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ 错误: 无法找到必要文件，请检查配置区域的路径 - {e}")
        return

    # 打开报告文件准备写入
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write("="*30 + " 基线工具性能评估报告 " + "="*30 + "\n\n")
        print(f"报告将保存至: {OUTPUT_REPORT_PATH}")

        # --- 遍历每一个工具进行评估 ---
        for tool_name, matrix_suffix in TOOLS.items():
            matrix_path = os.path.join(BASELINE_DATA_DIR, matrix_suffix)

            print(f"\n--- 正在评估工具: {tool_name} ---")
            report_file.write(f"--- 工具: {tool_name} ---\n")

            if not os.path.exists(matrix_path):
                print(f"  - 警告: 找不到结果矩阵 {matrix_path}，跳过该工具。")
                report_file.write("  - 结果文件不存在，跳过评估。\n\n")
                continue
            with open(matrix_path,'rb') as f:
                result_matrix = np.unpackbits(np.load(f))[:144400].reshape((380, 380))
                print(f"  - 读取结果矩阵 {matrix_path} 成功。")
                print(f"  - 矩阵大小：", result_matrix.ndim)
            y_true = []
            y_pred = []

            # 使用 .to_dict('records') 可以更高效地迭代DataFrame
            for row in tqdm(test_df.to_dict('records'), desc=f"Processing {tool_name}"):
                id1, id2 = row['contract_id'], row['clone_contract_id']
                groundtruth = int(row['groundtruth'])

                idx1 = name_to_idx.get(id1)
                idx2 = name_to_idx.get(id2)

                if idx1 is None or idx2 is None:
                    continue

                predicted_type = result_matrix[idx1, idx2]
                pred_binary = 1 if predicted_type > 0 else 0

                y_true.append(groundtruth)
                y_pred.append(pred_binary)

            if not y_true:
                print("  - 错误: 未能处理任何有效的数据对。请检查CSV和JSON文件的合约名称是否匹配。")
                report_file.write("  - 未能处理任何有效的数据对。\n\n")
                continue

            # --- 计算并保存整体指标 ---
            report_str = classification_report(y_true, y_pred, target_names=["Not Clone (Type 0)", "Clone (Type 1-4)"], digits=4)
            print("\n  --- 整体性能指标 ---")
            print(report_str)
            report_file.write("\n整体性能指标:\n")
            report_file.write(report_str + "\n")

            # --- 计算并保存各类型独立的召回率 ---
            test_df['predicted'] = y_pred
            clone_df = test_df[test_df['groundtruth'] == 1]

            print("\n  --- 各类型克隆的独立召回率 (Recall) ---")
            report_file.write("\n各类型克隆的独立召回率 (Recall):\n")

            for clone_type in [1.0, 2.0, 3.0, 4.0]: # 直接迭代所有可能的类型
                type_df = clone_df[clone_df['type'] == clone_type]
                if len(type_df) == 0:
                    recall_str = f"  - Type-{int(clone_type)} Recall: N/A (测试集中无此类型样本)\n"
                else:
                    recall = recall_score(type_df['groundtruth'], type_df['predicted'], zero_division=0)
                    correctly_found = type_df['predicted'].sum()
                    total = len(type_df)
                    recall_str = f"  - Type-{int(clone_type)} Recall: {recall:.4f} ({correctly_found} / {total})\n"

                print(recall_str.strip())
                report_file.write(recall_str)

            report_file.write("\n" + "="*70 + "\n\n")
            print(f"--- {tool_name} 评估完成 ---")

    print("\n所有基线工具评估完毕！")


# ==============================================================================
# 3. 脚本主入口
# ==============================================================================

if __name__ == '__main__':
    # 直接调用主函数，不再需要argparse
    evaluate_all_baselines()