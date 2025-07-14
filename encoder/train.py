# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from prepare_data import get_dataloader, Vocabulary

from prepare_data import get_dataloader
from model import Encoder, Decoder, Seq2SeqAutoencoder

# --- 超参数 ---
##CORPUS_FILE = '../corpus_deduplicated.txt' # <-- 请使用您最终去重后的语料库文件
CORPUS_FILE = '../no_abstract_corpus_deduplicated.txt'
# TOKENIZER_FILE = 'tokenizer.pkl'
TOKENIZER_FILE = 'noAbstract_tokenizer.pkl'
BATCH_SIZE = 64
EMB_DIM = 128
HID_DIM = 256 # 编码器单向的hidden dim
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 10 # 先用10轮看看效果
CLIP = 1 # 梯度裁剪
model_path = 'noAbstractModel'


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, src_len) in enumerate(tqdm(iterator, desc="Training")):
        src = src.to(device)
        # target 和 src 是一样的，因为是自编码器
        trg = src

        optimizer.zero_grad()

        # target作为teacher forcing的依据
        output = model(src, src_len, trg)

        # output: [seq_len, batch_size, vocab_size]
        # trg: [seq_len, batch_size]
        output_dim = output.shape[-1]

        # 去掉<SOS> token，不计算它的损失
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载数据
    train_dataloader, vocab = get_dataloader(CORPUS_FILE, TOKENIZER_FILE, BATCH_SIZE)
    INPUT_DIM = len(vocab)
    OUTPUT_DIM = len(vocab)
    PAD_IDX = vocab.stoi['<PAD>']

    # 2. 初始化模型
    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    # 解码器的hid_dim是编码器的两倍
    dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM * 2, N_LAYERS, DEC_DROPOUT)
    model = Seq2SeqAutoencoder(enc, dec, device).to(device)

    # 3. 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 4. 开始训练
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.0f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | PPL: {torch.exp(torch.tensor(train_loss)):7.3f}')

        # 每轮结束后保存模型
        torch.save(model.state_dict(), model_path + f'/autoencoder_epoch_{epoch+1}.pt')
        print(f"模型已保存: autoencoder_epoch_{epoch+1}.pt")