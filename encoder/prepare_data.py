# prepare_data.py (最终修正版)

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import pickle
import os

class Vocabulary:
    """词汇表构建与管理的类"""
    def __init__(self, min_freq=2):
        # 初始化特殊token和映射字典
        # <PAD>: 填充符，<SOS>: 句子起始符，<EOS>: 句子结束符，<UNK>: 未知词
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.min_freq = min_freq

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        """根据句子列表构建词汇表"""
        print("词汇表不存在，正在从头构建...")
        # 使用生成器表达式和Counter高效统计词频
        word_counts = Counter(word for sentence in tqdm(sentence_list, desc="Counting tokens") for word in sentence)

        idx = 4 # 从4开始，因为0-3是特殊token
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        print(f"词汇表构建完成，总词数: {len(self.itos)}")

    def numericalize(self, text_tokens: list) -> list:
        """将token序列（字符串列表）转换为数字序列（ID列表）"""
        # 正确地使用列表推导，'token'变量的作用域被限制在推导式内部
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in text_tokens]

    def save(self, path="tokenizer.pkl"):
        """将词汇表对象保存到文件"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"词汇表已保存到: {path}")

    @classmethod
    def load(cls, path="tokenizer.pkl"):
        """从文件加载词汇表对象"""
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"从 '{path}' 加载词汇表成功。")
        return vocab


class OpcodeDataset(Dataset):
    """自定义的PyTorch数据集类"""
    def __init__(self, corpus_path, vocab: Vocabulary):
        self.vocab = vocab
        with open(corpus_path, 'r', encoding='utf-8') as f:
            # 读取所有行，并切分成token列表
            self.lines = [line.strip().split() for line in f.readlines() if line.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        tokens = self.lines[index]
        numericalized_tokens = self.vocab.numericalize(tokens)
        # 为每个序列添加起始和结束符
        return torch.tensor([self.vocab.stoi["<SOS>"]] + numericalized_tokens + [self.vocab.stoi["<EOS>"]])


class PadCollate:
    """自定义的collate_fn，用于处理变长序列的padding"""
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        source_seqs = [item for item in batch]
        # 使用pad_sequence进行填充
        padded_sources = pad_sequence(source_seqs, batch_first=False, padding_value=self.pad_idx)

        # 计算每个序列的原始长度，用于后续的pack_padded_sequence
        source_lens = torch.tensor([len(seq) for seq in source_seqs], dtype=torch.int64)

        return padded_sources, source_lens


def get_dataloader(corpus_path: str, tokenizer_path: str, batch_size: int = 32, min_freq: int = 2) -> (DataLoader, Vocabulary):
    """
    获取数据加载器和词汇表。如果词汇表文件存在则加载，否则创建。
    """
    if os.path.exists(tokenizer_path):
        vocab = Vocabulary.load(tokenizer_path)
    else:
        print(f"词汇表文件 '{tokenizer_path}' 未找到。")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip().split() for line in f.readlines() if line.strip()]
        vocab = Vocabulary(min_freq)
        vocab.build_vocabulary(sentences)
        vocab.save(tokenizer_path)

    dataset = OpcodeDataset(corpus_path, vocab)
    pad_idx = vocab.stoi["<PAD>"]

    # 创建DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=PadCollate(pad_idx=pad_idx)
    )

    return dataloader, vocab

if __name__ == '__main__':
    # 使用示例，确保脚本可以独立运行以进行测试
    # CORPUS_FILE = r'C:\Users\zkjg\Downloads\13756064\clone_detection_replication_package\GNNMethod\corpus_deduplicated.txt' # <-- 替换为您的语料库文件名
    # TOKENIZER_FILE = 'tokenizer.pkl'

    CORPUS_FILE = r'C:\Users\zkjg\Downloads\13756064\clone_detection_replication_package\GNNMethod\no_abstract_corpus_deduplicated.txt' # <-- 替换为您的语料库文件名
    TOKENIZER_FILE = 'noAbstract_tokenizer.pkl'

    if not os.path.exists(CORPUS_FILE):
        print(f"错误: 语料库文件 '{CORPUS_FILE}' 不存在，请先准备好数据。")
    else:
        # 运行get_dataloader来构建或加载词汇表和数据
        dataloader, vocab = get_dataloader(CORPUS_FILE, TOKENIZER_FILE, batch_size=4)
        print(f"\n成功获取DataLoader。词汇表大小: {len(vocab)}")
        # 打印一个batch来检查输出是否正确
        src_batch, src_len_batch = next(iter(dataloader))
        print(f"\n一个batch的源数据形状 (Seq_Len, Batch_Size): {src_batch.shape}")
        print(f"一个batch的源数据长度 (Batch_Size): {src_len_batch}")
        print("\n第一个样本 (数字形式):")
        print(src_batch[:, 0])
        print("\n第一个样本 (Token形式):")
        print([vocab.itos[idx.item()] for idx in src_batch[:, 0]])