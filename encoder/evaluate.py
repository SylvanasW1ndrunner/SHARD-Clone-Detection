# run_encoder_ablation_final.py

import os
import pickle
import torch
import pydot
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from GNNMethod.encoder.model import Decoder, Seq2SeqAutoencoder, Encoder
from GNNMethod.encoder.prepare_data import Vocabulary


# 假设您的模块和类已正确定义或可导入
# from GNNMethod.encoder.prepare_data import Vocabulary
# from GNNMethod.encoder.model import Encoder, Decoder, Seq2SeqAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_encoder(vocab_size, model_path, device):
    enc = Encoder(vocab_size, 128, 256, 2, 0.5)

    dec = Decoder(vocab_size, 128, 256 * 2, 2, 0.5)

    model = Seq2SeqAutoencoder(enc, dec, device).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    print(f"从 '{model_path}' 加载模型权重成功。")

    return model.encoder.eval()

def encode_sequence(encoder, sequence_str, vocab, device):

    encoder.eval()

    tokens = sequence_str.strip().split()

    numericalized = [vocab.stoi["<SOS>"]] + vocab.numericalize(tokens) + [vocab.stoi["<EOS>"]]
    src_tensor = torch.LongTensor(numericalized).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(numericalized)])

    with torch.no_grad():

        hidden, cell = encoder(src_tensor, src_len)

    return hidden[-1].squeeze(0).cpu().numpy()


def get_contract_centroid_vector(contract_id: str, cfg_dir: str, encoder,vocab):
    """加载一个合约的CFG，将其所有基本块编码为向量，并返回质心（平均向量）。"""
    dot_path = os.path.join(cfg_dir, f"{contract_id}_cfg.dot") # 假设文件名格式
    if not os.path.exists(dot_path): return None

    try:
        graph_nx = nx.drawing.nx_pydot.from_pydot(pydot.graph_from_dot_file(dot_path)[0])
    except Exception: return None

    block_vectors = [encode_sequence(encoder,node_data.get('label', '').strip('"'),vocab,device)
                     for _, node_data in graph_nx.nodes(data=True)
                     if node_data.get('label', '')]

    return np.mean(np.array(block_vectors), axis=0) if block_vectors else None


def run_single_experiment(config: dict, all_pairs_df: pd.DataFrame):
    """对单个实验配置（语义版或原始版）执行完整的评估流程"""
    setup_name = config['name']
    print("\n" + "="*30)
    print(f"开始评估: {setup_name}")
    print("="*30)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载此分支对应的编码器
    vocab = Vocabulary.load(config['tokenizer_path'])
    encoder = get_encoder(len(vocab), config['encoder_path'], device)

    # 2. 一次性计算所有合约的质心向量并缓存
    print("正在为所有合约生成质心向量...")
    all_contract_ids = pd.concat([all_pairs_df['contract_id'], all_pairs_df['clone_contract_id']]).unique()
    centroid_vectors = {}
    for contract_id in tqdm(all_contract_ids, desc=f"Encoding ({setup_name})"):
        centroid = get_contract_centroid_vector(contract_id, config['cfg_dir'], encoder,vocab)
        if centroid is not None:
            centroid_vectors[contract_id] = centroid

    # 3. 为每个数据对构建分类器特征
    print("正在为数据对构建分类特征...")
    X, y = [], []
    for _, row in tqdm(all_pairs_df.iterrows(), total=len(all_pairs_df), desc="Building Features"):
        id1, id2 = row['contract_id'], row['clone_contract_id']
        if id1 in centroid_vectors and id2 in centroid_vectors:
            v1, v2 = centroid_vectors[id1], centroid_vectors[id2]
            feature_vec = np.concatenate([np.abs(v1 - v2), v1 * v2])
            X.append(feature_vec)
            y.append(row['groundtruth'])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print("错误: 未能为任何数据对构建特征。")
        return None

    # 4. 划分训练集、验证集、测试集 (70/10/20)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(2/3.0), random_state=42, stratify=y_temp)
    print(f"数据划分完成: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 5. 训练并评估线性分类器
    classifier = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    print("正在训练线性分类器...")
    classifier.fit(X_train, y_train)

    print("在测试集上进行评估...")
    y_pred = classifier.predict(X_test)

    report_str = classification_report(y_test, y_pred, target_names=["Not Clone", "Clone"], digits=4)
    print(f"\n--- {setup_name} 模型性能报告 ---")
    print(report_str)

    return report_str


if __name__ == '__main__':
    CONFIG = {
        "pairs_csv_path": "../encoder_test.csv",
        "output_dir": "encoder_ablation_reports",
        "setups": [
            {
                "name": "Semantic Abstraction (Ours)",
                "encoder_path": "models/autoencoder_epoch_10.pt",
                "tokenizer_path": "tokenizer.pkl",
                "cfg_dir": "../GNNdata/proccessed_cfg"
            },
            {
                "name": "Raw Token (Baseline)",
                "encoder_path": "noAbstractModel/autoencoder_epoch_10.pt",
                "tokenizer_path": "noAbstract_tokenizer.pkl",
                "cfg_dir": "../GNNdata/noAbstact_proccessed_cfg"
            }
        ]
    }
    # ----------------

    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    all_pairs_df = pd.read_csv(CONFIG['pairs_csv_path'])

    final_report = ""
    for setup_config in CONFIG['setups']:
        report = run_single_experiment(setup_config, all_pairs_df)
        if report:
            final_report += "="*30 + f"\n    评估报告: {setup_config['name']}\n" + "="*30 + "\n"
            final_report += report + "\n\n"

    report_path = os.path.join(CONFIG['output_dir'], "encoder_ablation_summary.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(final_report)

    print(f"所有消融实验评估完成！汇总报告已保存到: {report_path}")