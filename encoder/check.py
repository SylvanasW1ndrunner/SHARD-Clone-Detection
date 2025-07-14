# debug_pair.py

import pickle
import numpy as np
import networkx as nx
import pydot
import os

# --- 依赖 ---
# 确保这个脚本可以找到并导入您的Vocabulary类
# 如果您的类在不同文件中，请相应修改
from prepare_data import Vocabulary


def forensic_analysis(id1, id2, embeddings_cache_path, dot_dir_path):
    print(f"\n{'='*25}\n法证分析开始: {id1} vs {id2}\n{'='*25}")

    print("\n--- 阶段一: 对比已缓存的“块向量集合” ---")
    if not os.path.exists(embeddings_cache_path):
        print(f"❌ 错误: 编码缓存文件 '{embeddings_cache_path}' 不存在。")
        return

    with open(embeddings_cache_path, 'rb') as f:
        embeddings = pickle.load(f)

    vec_set1 = embeddings.get(id1)
    vec_set2 = embeddings.get(id2)

    if vec_set1 is None or vec_set2 is None:
        print(f"❌ 错误: 在缓存中找不到合约ID。ID1: {'找到' if vec_set1 is not None else '未找到'}, ID2: {'找到' if vec_set2 is not None else '未找到'}")
        return

    print(f"  向量集1形状: {vec_set1.shape}, 向量集2形状: {vec_set2.shape}")

    are_vectors_identical = False
    if vec_set1.shape == vec_set2.shape:
        sorted_vec1 = vec_set1[np.lexsort(vec_set1.T)]
        sorted_vec2 = vec_set2[np.lexsort(vec_set2.T)]
        diff = np.linalg.norm(sorted_vec1 - sorted_vec2)
        print(f"  排序后向量集的差异 (L2范数): {diff}")
        if diff < 1e-6:
            print("  ✅ 结论: 两个块向量集合在数值上完全相同！")
            are_vectors_identical = True
        else:
            print("  ❌ 结论: 两个块向量集合在数值上存在差异！这是导致OT距离不为0的直接原因。")
    else:
        print("  ❌ 结论: 两个块向量集合的形状不同！这是导致OT距离不为0的直接原因。")

    if are_vectors_identical:
        print("\n🔥 诊断: 如果向量集相同但OT距离仍不为0，问题可能出在OT计算库的数值稳定性上，但这极其罕见。请检查您的OT距离计算函数。")
        return

    # --- 阶段二：如果向量不同，则深入对比源DOT文件 ---
    print("\n--- 阶段二: 深度对比源 .dot 文件内容 ---")
    dot_path1 = os.path.join(dot_dir_path, f"{id1}_cfg.dot") # 假设文件名格式
    dot_path2 = os.path.join(dot_dir_path, f"{id2}_cfg.dot")

    if not (os.path.exists(dot_path1) and os.path.exists(dot_path2)):
        print(f"❌ 错误: 找不到DOT文件。路径1: '{dot_path1}', 路径2: '{dot_path2}'")
        return

    try:
        graph1 = nx.drawing.nx_pydot.from_pydot(pydot.graph_from_dot_file(dot_path1)[0])
        graph2 = nx.drawing.nx_pydot.from_pydot(pydot.graph_from_dot_file(dot_path2)[0])
    except Exception as e:
        print(f"加载DOT文件时出错: {e}")
        return
    edges1 = sorted([e for e in graph1.edges()])
    edges2 = sorted([e for e in graph2.edges()])
    print(f"  DOT文件1的边数量: {len(edges1)}")
    print(f"  DOT文件2的边数量: {len(edges2)}")
    nodes1 = sorted([n.strip('"') for n in graph1.nodes()])
    nodes2 = sorted([n.strip('"') for n in graph2.nodes()])
    print(f"  DOT文件1的节点数量: {len(nodes1)}")
    print(f"  DOT文件2的节点数量: {len(nodes2)}")
    if nodes1 != nodes2:
        print("❌ 致命差异: 两个文件的节点名称列表不同！")
        return

    print("  ✅ 节点名称和数量一致。开始逐一比对节点标签...")

    mismatch_found = False
    for node_name in nodes1:
        label1 = graph1.nodes[node_name].get('label', '').strip('"')
        label2 = graph2.nodes[node_name].get('label', '').strip('"')

        if label1 != label2:
            mismatch_found = True
            print(f"\n  ❌ 发现标签不匹配！节点: {node_name}")
            print("-" * 40)
            print(f"    文件1 (repr): {repr(label1)}")
            print(f"    文件2 (repr): {repr(label2)}")
            print("-" * 40)

    if not mismatch_found:
        print("  ✅ 所有节点的标签在深度比对后完全一致。")
        print("\n🔥 最终诊断: 如果节点标签完全一致，但编码出的向量集不同（如阶段一所示），")
        print("     这强烈暗示您的编码过程存在随机性。请务必检查您的encoder加载后是否调用了 `.eval()` 模式！")
    else:
        print("\n🔥 最终诊断: 已找到根本原因！两个合约的节点标签内容存在微小但致命的差异，导致编码结果不同。")


# --- 主程序入口 ---
if __name__ == '__main__':
    # --- 配置 ---
    # 请将这里的路径和ID修改为您要调试的目标
    EMBEDDINGS_CACHE_PATH = 'embeddings.pkl'
    DOT_FILES_DIR = '../GNNdata/proccessed_cfg' # 存放.dot文件的文件夹

    # 您发现问题的那一对合约ID
    CONTRACT_ID_1 = "0xd58a2e914f31c708442ff58871deb3a57c3322fc"
    CONTRACT_ID_2 = "0x506ce57a0050ffce5fe9437f606cb1d9db17a7b5"

    # --- 运行分析 ---
    forensic_analysis(CONTRACT_ID_1, CONTRACT_ID_2, EMBEDDINGS_CACHE_PATH, DOT_FILES_DIR)