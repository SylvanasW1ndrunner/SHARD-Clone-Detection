# model.py

import torch
import torch.nn as nn
import random



class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        # src: [seq_len, batch_size]
        # src_len: [batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded: [seq_len, batch_size, emb_dim]

        # 打包变长序列
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=False)

        # packed_outputs是所有时间步的隐状态，我们在这里不需要用它
        # hidden 和 cell 是最后一个时间步的隐状态
        packed_outputs, (hidden, cell) = self.rnn(packed_embedded)

        # hidden/cell 的形状是: [n_layers * 2, batch_size, hid_dim]
        # 因为是双向的，所以第一个维度是层数*2

        # --- 核心修正开始 ---
        # 我们需要将前向和后向的hidden/cell state拼接起来，
        # 以匹配单向解码器的输入要求。
        # 旧的 .view().transpose()... 写法过于复杂且容易出错。
        # 我们用更清晰的方式重写。

        # 首先，将 hidden state 重塑为 [n_layers, num_directions, batch_size, hid_dim]
        hidden = hidden.view(self.rnn.num_layers, 2, -1, self.rnn.hidden_size)
        cell = cell.view(self.rnn.num_layers, 2, -1, self.rnn.hidden_size)

        # 将前向 (dim 1, index 0) 和后向 (dim 1, index 1) 的隐状态在最后一个维度上拼接
        # hidden_forward shape: [n_layers, batch_size, hid_dim]
        # hidden_backward shape: [n_layers, batch_size, hid_dim]
        # 最终 hidden_cat shape: [n_layers, batch_size, hid_dim * 2]
        hidden_cat = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        cell_cat = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)

        # --- 核心修正结束 ---

        # 返回的hidden_cat和cell_cat现在拥有了正确的形状，可以被Decoder正确接收
        return hidden_cat, cell_cat

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        # 解码器的hid_dim必须是encoder的两倍，因为encoder是双向的
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input: [batch_size] (当前时刻的输入token)
        # hidden, cell: [n_layers, batch_size, hid_dim]
        input = input.unsqueeze(0) # -> [1, batch_size]

        embedded = self.dropout(self.embedding(input))
        # embedded: [1, batch_size, emb_dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: [1, batch_size, hid_dim]

        prediction = self.fc_out(output.squeeze(0))
        # prediction: [batch_size, output_dim]

        return prediction, hidden, cell

class Seq2SeqAutoencoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

# model.py -> class Seq2SeqAutoencoder(nn.Module)

# ... __init__ 方法保持不变 ...

# 这是需要被完整替换的函数
    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src, trg: [seq_len, batch_size]
        # src_len: [batch_size]

        batch_size = src.shape[1]
        trg_len = trg.shape[0] # 注意：目标长度应与源相同（自编码器）
        trg_vocab_size = self.decoder.output_dim

        # 创建一个张量来存储解码器的所有输出
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # 1. 编码器处理输入序列，得到上下文向量 (最后一个隐状态)
        hidden, cell = self.encoder(src, src_len)

        # 2. 解码器的第一个输入是 <SOS> (start of sequence) token
        # src[0, :] 就是所有batch样本的第一个token，即<SOS>
        input = src[0, :] # input shape: [batch_size]

        # 3. 按时间步进行解码循环
        # 从第二个token开始，因为第一个是<SOS>
        for t in range(1, trg_len):

            # 将当前输入token和前一步的隐状态送入解码器
            # 这里的 input 是 1D 的 [batch_size]，是正确的
            output, hidden, cell = self.decoder(input, hidden, cell)

            # 在outputs张量中保存当前时间步的预测结果
            outputs[t] = output

            # 决定是否使用 teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # 获取模型当前最可能的预测词
            top1 = output.argmax(1) # top1 shape: [batch_size]

            # 如果是teacher forcing，下一个输入是真实的词；否则是模型自己预测的词
            input = trg[t] if teacher_force else top1

        return outputs