# evaluate_combined_baselines.py

import os
import numpy as np
import pandas as pd
import json
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from itertools import combinations
from tqdm import tqdm

# ==============================================================================
# 1. 配置区域: 请在这里核对您的文件路径
# ==============================================================================

# 定义存放所有基线工具数据的根目录
BASELINE_DATA_DIR = "../../2_tools_label/dataset_II_contract_level/"
SIM_DATA_DIR = "../../3_tools_similarity/dataset_II_contract_level/"

# 测试集CSV路径
TEST_CSV_PATH = "../test.csv" # 假设与脚本在同一目录，或提供完整路径

# 合约名到矩阵索引的JSON映射文件路径
CONTRACT_MAP_PATH = os.path.join(BASELINE_DATA_DIR, "index_map/contract_label_380.json")

# 输出报告文件名
OUTPUT_REPORT_PATH = "combined_tools_performance_report.txt"

def evaluate_combined_tools():
    """
    主函数，评估所有基线工具组合在test.csv上的性能。
    """
    # --- 1. 加载所有数据和映射文件 ---
    print("正在加载测试数据、映射文件和基线工具矩阵...")
    try:
        test_df = pd.read_csv(TEST_CSV_PATH)
        with open(CONTRACT_MAP_PATH, 'r', encoding='utf-8') as f:
            name_to_idx = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ 错误: 无法找到必要文件 - {e}")
        return

    # 定义工具、原始相似度矩阵路径和阈值
    tool_info = {
        'deckard': {'path': 'Deckard/similarity_matrix.npy', 'threshold': 79},
        'nicad': {'path': 'Nicad/similarity_matrix.npy', 'threshold': 70},
        'smartembed': {'path': 'SmartEmbed/similarity_matrix.npy', 'threshold': 95},
        'sourcercc': {'path': 'SourcererCC/similarity_matrix.npy', 'threshold': 70},
        'eclone': {'path': 'eclone/similarity_matrix.npy', 'threshold': 84}
    }

    # --- 2. 加载并二值化所有工具的预测矩阵 ---
    binary_matrices = {}
    for name, info in tool_info.items():
        matrix_path = os.path.join(SIM_DATA_DIR, info['path'])
        if os.path.exists(matrix_path):
            sim_matrix = np.load(matrix_path)
            binary_matrices[name] = (sim_matrix >= info['threshold'])
            print(f"  - 工具 {name} 的矩阵已加载并二值化。")
        else:
            print(f"  - 警告: 找不到工具 {name} 的矩阵文件，将跳过。")

    if not binary_matrices:
        print("❌ 错误: 未能加载任何有效的工具矩阵。")
        return

    # --- 3. 生成所有工具的组合 ---
    tool_labels = list(binary_matrices.keys())
    combined_matrices = {}
    for i in range(1, len(tool_labels) + 1):
        for combo in combinations(tool_labels, i):
            combo_name = '_'.join(sorted(combo))
            # 对组合内的所有二值矩阵进行逻辑“或”操作，代表“任何一个工具认为是克隆，则组合也认为是克隆”
            # 这是最常见的组合策略，等价于投票数>0
            combined_matrix = np.zeros_like(next(iter(binary_matrices.values())), dtype=bool)
            for tool_name in combo:
                combined_matrix = np.logical_or(combined_matrix, binary_matrices[tool_name])
            combined_matrices[combo_name] = combined_matrix

    print(f"\n已生成 {len(combined_matrices)} 种工具组合。开始评估...")

    # --- 4. 对每一种组合，在test.csv上进行评估 ---
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as report_file:
        report_file.write("="*30 + " 基线工具组合性能评估报告 " + "="*30 + "\n")
        report_file.write(f"测试集: {TEST_CSV_PATH}\n\n")

        for combo_name, combo_matrix in tqdm(combined_matrices.items(), desc="Evaluating Combinations"):
            report_file.write(f"--- 组合: {combo_name} ---\n")

            y_true = []
            y_pred = []

            for _, row in test_df.iterrows():
                id1, id2, groundtruth = row['contract_id'], row['clone_contract_id'], int(row['groundtruth'])
                idx1, idx2 = name_to_idx.get(id1), name_to_idx.get(id2)
                if idx1 is None or idx2 is None: continue

                # 从组合矩阵中获取二元预测
                pred_binary = int(combo_matrix[idx1, idx2])
                y_true.append(groundtruth)
                y_pred.append(pred_binary)

            if not y_true:
                report_file.write("  - 未能处理任何有效的数据对。\n\n")
                continue

            # --- 计算并保存指标 ---
            # a. 整体指标
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            report_file.write(f"整体性能:\n")
            report_file.write(f"  - Precision: {p:.4f}\n")
            report_file.write(f"  - Recall:    {r:.4f}\n")
            report_file.write(f"  - F1-Score:  {f1:.4f}\n\n")

            # b. 各类型独立召回率
            temp_df = test_df.copy()
            temp_df['predicted'] = y_pred
            clone_df = temp_df[temp_df['groundtruth'] == 1]

            report_file.write("各类型克隆的独立召回率 (Recall):\n")
            for clone_type in [1.0, 2.0, 3.0, 4.0]:
                type_df = clone_df[clone_df['type'] == clone_type]
                if len(type_df) == 0:
                    recall_str = f"  - Type-{int(clone_type)} Recall: N/A (测试集中无此类型样本)\n"
                else:
                    recall_type = recall_score(type_df['groundtruth'], type_df['predicted'], zero_division=0)
                    correctly_found = type_df['predicted'].sum()
                    total = len(type_df)
                    recall_str = f"  - Type-{int(clone_type)} Recall: {recall_type:.4f} ({correctly_found} / {total})\n"
                report_file.write(recall_str)

            report_file.write("\n" + "="*70 + "\n\n")

    print(f"\n评估完成！所有结果已保存到 {OUTPUT_REPORT_PATH}")

if __name__ == '__main__':
    evaluate_combined_tools()