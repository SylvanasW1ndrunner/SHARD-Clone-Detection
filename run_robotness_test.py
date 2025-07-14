# run_robustness_test.py

import os
import sys
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
import networkx as nx
import pydot
import random
import argparse

from matplotlib import pyplot as plt
from ot.backend import str_type_error
from sklearn.metrics import classification_report, roc_curve
from torch_geometric.data import Batch
from tqdm import tqdm
from scipy.spatial.distance import cosine, euclidean

from GNNMethod.dataset import collate_fn_hierarchical
from GNNMethod.encoder.evaluate import get_encoder, encode_sequence
# 确保可以从您的项目中导入所有必要的模块
from GNNMethod.encoder.prepare_data import Vocabulary # 假设在prepare_data.py中
from GNNMethod.model import HierarchicalGNN, FlatGNN # 假设在model.py中
from GNNMethod.preprocess_graphs import create_graph_data # 假设在data_gnn.py中
import seaborn as sns
# --- 模拟的编码器部分，请替换为您真实加载的代码 ---

output_path = "GNNdata/robustdata"
if not os.path.exists(output_path):
    print("make dir: ", output_path)
    os.makedirs(output_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Encoder:
    def __init__(self,encoder_path,tokenizer_path):
        self.vocab = Vocabulary.load(tokenizer_path)
        self.encoder = get_encoder(len(self.vocab), encoder_path, device)
    def encode(self, token_str):

        encoder = self.encoder
        vocab = self.vocab
        return encode_sequence(encoder, token_str, vocab, device) # 假设您的编码器输出256维向量
encoder = Encoder(tokenizer_path='encoder/tokenizer.pkl', encoder_path='encoder/models/autoencoder_epoch_10.pt')
import os
import json
import pydot
import networkx as nx
import random
import argparse
import copy
from tqdm import tqdm

def add_real_orphan_islands_noise(
        graph_nx: nx.DiGraph,
        corpus: list[str],
        ratio: float = 0.2
) -> nx.DiGraph:
    noisy_graph = graph_nx.copy()

    num_original_nodes = len(graph_nx.nodes())
    num_nodes_to_add = int(num_original_nodes * ratio)

    if num_nodes_to_add == 0 or not corpus:
        return noisy_graph

    print(f"    - 计划添加 {num_nodes_to_add} 个内容真实的 [孤儿块噪声] 节点...")

    new_orphan_nodes = []
    for i in range(num_nodes_to_add):
        noise_node_name = f"real_orphan_noise_{i}"

        # --- 核心修改点：从真实语料库中随机抽取一个基本块作为标签 ---
        noise_label_content = random.choice(corpus)
        noise_label = f'"{noise_label_content}"' # 保持DOT格式的引号

        noisy_graph.add_node(
            noise_node_name,
            label=noise_label,
            shape="box",
            style="filled",
            fillcolor="palegreen" # 换个颜色以示区分
        )
        new_orphan_nodes.append(noise_node_name)

    if num_nodes_to_add > 1:
        num_edges_to_add = num_nodes_to_add // 2
        for _ in range(num_edges_to_add):
            u, v = random.sample(new_orphan_nodes, 2)
            if u != v:
                noisy_graph.add_edge(u, v)

    # 同样，不将孤儿块连接到主图，保证语义不变
    return noisy_graph

def add_noise_to_graph_diverse(graph_nx: nx.DiGraph,args) -> nx.DiGraph:
    noisy_graph = graph_nx.copy()
    corpus_path = args.corpus
    with open(corpus_path,"r") as f:
        corpus = f.readlines()
    # 3. 应用孤儿块 (结构级噪声)
    noisy_graph = add_real_orphan_islands_noise(noisy_graph, corpus=corpus,ratio=0.5)

    # print(f"    - 密集扰动完成。")
    return noisy_graph

def run_robustness_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed)
    np.random.seed(args.seed)
    data_list = {}
    print("正在加载模型...")
    # TODO: 确保这里的维度与您的真实模型匹配

    df_test_full = pd.read_csv(args.test_csv)
    df_type4 = df_test_full[df_test_full['type'] == 4]
    df_type0 = df_test_full[df_test_full['type'] == 0]
    num_samples_per_type = args.num_samples // 2
    sampled_type4 = df_type4.sample(n=min(num_samples_per_type, len(df_type4)), random_state=args.seed)
    sampled_type0 = df_type0.sample(n=min(num_samples_per_type, len(df_type0)), random_state=args.seed)
    test_df = pd.concat([sampled_type4, sampled_type0]).reset_index(drop=True)
    print(f"采样完成，将对 {len(test_df)} 个数据对进行扰动和评估...")
    test_df.to_csv("robust_test.csv", index=False)
    # 3. 运行扰动实验并收集数据
    results_list = []
    # 获取所有需要处理的独立合约ID
    all_contract_ids = pd.concat([test_df['contract_id'], test_df['clone_contract_id']]).unique()

    # a. 对所有独立合约进行一次性编码（原始和加噪）并缓存
    print("开始对选取的独立合约进行编码和扰动...")
    embedding_cache = {}
    for contract_id in all_contract_ids:
        dot_path = os.path.join(args.dot_dir, f"{contract_id}_cfg.dot")
        json_path = os.path.join(args.json_dir, f"{contract_id}_function_mapping.json")
        if not (os.path.exists(dot_path) and os.path.exists(json_path)): continue

        nx_graph_orig = pydot.graph_from_dot_file(dot_path)[0]
        nx_graph_orig = nx.drawing.nx_pydot.from_pydot(nx_graph_orig)
        nx_graph_noisy = add_noise_to_graph_diverse(nx_graph_orig,args)
        dot_filepath = os.path.join(output_path,"processed_cfg")
        pt_filepath = os.path.join(output_path,"ptdata")
        if os.path.exists(dot_filepath) is False:
            os.makedirs(dot_filepath, exist_ok=True)
        if os.path.exists(pt_filepath) is False:
            os.makedirs(pt_filepath, exist_ok=True)
        outputfile = os.path.join(dot_filepath, f"{contract_id}_noise.dot")
        print(f"  - 生成噪声图 {outputfile}")
        nx_graph_noisy = nx.drawing.nx_pydot.to_pydot(nx_graph_noisy)
        nx_graph_noisy.write_dot(outputfile)

        data_noisy,_ = create_graph_data(outputfile, json_path, encoder)

        torch.save(data_noisy, os.path.join(pt_filepath, f"{contract_id}.pt"))
        if data_noisy is None:
            print(f"  - 警告: 无法创建数据对象，跳过 {contract_id}")
            continue

        # except Exception as e:
        #     print(f"处理合约 {contract_id} 时出错: {e.with_traceback(e.__traceback__)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='运行GNN模型抗结构噪声的鲁棒性测试')
    parser.add_argument('--hier_model_path', type=str, required=False, default='model/best_model.pth')
    parser.add_argument('--flat_model_path', type=str, required=False, default='basemodel/best_model.pth')
    parser.add_argument('--tokenizer_path', type=str, default='encoder/tokenizer.pkl')
    parser.add_argument('--dot_dir', type=str, required=False, default='GNNdata/proccessed_cfg')
    parser.add_argument('--json_dir', type=str, required=False, default='GNNdata/function')
    parser.add_argument('--num_samples', type=int, default=500, help='用于测试的随机抽样合约数量')
    parser.add_argument('--test_csv', type=str, default='test.csv', help='测试的CSV文件路径')
    parser.add_argument('--result', type=str, default='',help='测试结果的文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--corpus',type=str,default= 'corpus_deduplicated.txt')

    args = parser.parse_args()
    run_robustness_experiment(args)