# 1_preprocess_graphs.py (已根据您的JSON结构修正)

import os
import json
import torch
import pydot
import networkx as nx
from torch_geometric.data import Data
from tqdm import tqdm
import argparse

from GNNMethod.encoder.evaluate import get_encoder, encode_sequence
from GNNMethod.encoder.prepare_data import Vocabulary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Encoder:
    def __init__(self,encoder_path,tokenizer_path):
        self.vocab = Vocabulary.load(tokenizer_path)
        self.encoder = get_encoder(len(self.vocab), encoder_path, device)
    def encode(self, token_str):

        encoder = self.encoder
        vocab = self.vocab
        return encode_sequence(encoder, token_str, vocab, device) # 假设您的编码器输出256维向量
encoder = Encoder(tokenizer_path='encoder/noAbstract_tokenizer.pkl', encoder_path='encoder/noAbstractModel/autoencoder_epoch_10.pt')
# ----------------------------------------------------------------------------


def create_graph_data(dot_path, json_path, encoder):
    """
    (已修改) 为单个合约创建PyG的Data对象，并返回函数-ID映射。
    """
    try:
        graphs = pydot.graph_from_dot_file(dot_path)
        graph_nx = nx.drawing.nx_pydot.from_pydot(graphs[0])
    except Exception as e:
        print(f"  - 无法解析 {dot_path}: {e}")
        return None, None

    with open(json_path, 'r', encoding='utf-8') as f:
        func_map_data = json.load(f)

    sorted_nodes = sorted(list(graph_nx.nodes()))
    node_name_to_idx = {name: i for i, name in enumerate(sorted_nodes)}

    edge_list = []
    for u, v in graph_nx.edges():
        u, v = u.strip('"'), v.strip('"')
        u_idx, v_idx = node_name_to_idx.get(u), node_name_to_idx.get(v)
        if u_idx is not None and v_idx is not None:
            edge_list.append([u_idx, v_idx])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    node_features = []
    function_assignments = []

    # --- 保证函数ID是连续的 ---
    block_to_func_id = {}
    function_membership = func_map_data.get('function_membership', {})
    sorted_func_names = sorted(function_membership.keys())
    func_name_to_id = {name: i for i, name in enumerate(sorted_func_names)}

    for func_name, func_data in function_membership.items():
        func_id = func_name_to_id[func_name]
        for block_name in func_data.get('basic_block_ids', []):
            block_to_func_id[block_name] = func_id

    for node_name in sorted_nodes:
        node_name_clean = node_name.strip('"')
        label = graph_nx.nodes[node_name].get('label', '').strip('"')
        node_vec = encoder.encode(label) # 这是一个模拟调用，返回Tensor
        node_features.append(node_vec)
        func_id = block_to_func_id.get(node_name_clean, -1)
        function_assignments.append(func_id)

    if not node_features:
        return None, None

    x = torch.stack([torch.as_tensor(vec, dtype=torch.float) for vec in node_features])
    function_mapping = torch.tensor(function_assignments, dtype=torch.long)
    num_functions = len(sorted_func_names)

    graph_data = Data(
        x=x,
        edge_index=edge_index,
        function_mapping=function_mapping,
        num_functions=num_functions
    )

    return graph_data, func_name_to_id


def main(args):
    # 创建所有需要的输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    # os.makedirs(args.output_func_map_dir, exist_ok=True)

    dot_files = [f for f in os.listdir(args.dot_dir) if f.endswith('.dot')]

    for filename in tqdm(dot_files, desc="Preprocessing Graphs"):
        # --- 核心修改点：使用更安全的文件名作为ID ---
        contract_id = os.path.splitext(filename)[0]

        dot_path = os.path.join(args.dot_dir, filename)
        json_filename = f"{contract_id.replace('_cfg', '')}_function_mapping.json"
        json_path = os.path.join(args.json_dir, json_filename)

        if not os.path.exists(json_path):
            print(f"  - 警告: 找不到对应的JSON文件 {json_path}，跳过 {filename}")
            continue

        # --- 核心修改点：接收两个返回值 ---
        graph_data, func_name_map = create_graph_data(dot_path, json_path, encoder)

        if graph_data and func_name_map is not None:
            # 1. 保存为PyTorch的.pt文件，供模型使用
            pt_save_path = os.path.join(args.output_dir, f"{contract_id}.pt")
            torch.save(graph_data, pt_save_path)

            # 2. 保存函数名到ID的映射为JSON文件，供分析使用
            # map_save_path = os.path.join(args.output_func_map_dir, f"{contract_id}_func_idx.json")
            # with open(map_save_path, 'w', encoding='utf-8') as f:
            #     json.dump(func_name_map, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='预处理DOT和JSON文件为PyG的Data对象')
    parser.add_argument('--dot_dir', type=str, required=False, help='包含DOT文件的文件夹路径',default='GNNdata/noAbstact_proccessed_cfg')
    parser.add_argument('--json_dir', type=str, required=False, help='包含函数映射JSON文件的文件夹路径',default='GNNdata/function')
    parser.add_argument('--output_dir', type=str, required=False, help='保存处理后图数据(.pt)的输出文件夹路径',default='GNNdata/noAbstract_ptdata')
   ## parser.add_argument('--output_func_map_dir', type=str, required=False ,help='保存函数名到ID映射(.json)的输出文件夹')
    args = parser.parse_args()
    main(args)