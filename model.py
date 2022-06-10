import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np


device = torch.device("cuda")

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.f = nn.Linear(input_size + hidden_size, hidden_size)
        self.i = nn.Linear(input_size + hidden_size, hidden_size)
        self.o = nn.Linear(input_size + hidden_size, hidden_size)
        self.g = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, ht, ct, xt):
        # ht: 1 * hidden_size
        # ct: 1 * hidden_size
        # xt: 1 * input_size
        input_combined = torch.cat((xt, ht), 1)
        ft = torch.sigmoid(self.f(input_combined))
        it = torch.sigmoid(self.i(input_combined))
        ot = torch.sigmoid(self.o(input_combined))
        gt = torch.tanh(self.g(input_combined))
        ct = ft * ct + it * gt
        ht = ot * torch.tanh(ct)
        return ht, ct


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(BiLSTM, self).__init__()
        # TODO
        self.fLSTM = LSTM(input_size, hidden_size)
        self.bLSTM = LSTM(input_size, hidden_size)
        self.register_buffer("_batch", torch.zeros(batch_size, hidden_size))
        self.register_buffer("_valid", torch.zeros(1408, hidden_size))
        self.register_buffer("_test", torch.zeros(1410, hidden_size))
        self.batch_size = batch_size
        self.hidden_size = hidden_size
    
    def init_h_and_c(self, mode):
        if mode == "Train":
            h = torch.zeros_like(self._batch)
            c = torch.zeros_like(self._batch)
        elif mode == "Valid":
            h = torch.zeros_like(self._valid)
            c = torch.zeros_like(self._valid)
        elif mode == "Test":
            h = torch.zeros_like(self._test)
            c = torch.zeros_like(self._test)
        
        return h, c
    
    def forward(self, x, mode="Train"):
        """
        输入
            x: 1 * length * input_size
        输出
            hiddens: 1 * length * (hidden_size*2)
        """
        # TODO

        B, length = x.shape[0], x.shape[1]
        hf, cf = self.init_h_and_c(mode)
        hb, cb = self.init_h_and_c(mode)
        hidden_f, hidden_b = [], []

        for i in range(length):
            hf, cf = self.fLSTM(hf, cf, x[:, i, :])
            hb, cb = self.bLSTM(hb, cb, x[:, length-i-1, :])
            hidden_f.append(hf)
            hidden_b.append(hb)

        hidden_b.reverse()
        hidden_f = torch.stack(hidden_f)    # len*B*d
        hidden_b = torch.stack(hidden_b)

        hidden_f = hidden_f.reshape(-1, hidden_f.shape[2])
        hidden_b = hidden_b.reshape(-1, hidden_b.shape[2])

        hiddens = torch.hstack([hidden_f, hidden_b]) # (len*B)*(2*d)
        hiddens = hiddens.reshape(B, length, -1)
        
        return hiddens


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # TODO
        self.feat2att = nn.Linear(hidden_size, hidden_size)
        self.to_alpha = nn.Linear(hidden_size, 1, bias=False)

    
    def forward(self, hiddens, mask=None):
        """
        输入
            hiddens: 1 * length * hidden_size
            mask: 1 * length
        输出
            attn_outputs: 1 * hidden_size
        """
        # TODO
        attn_f = self.feat2att(hiddens) # 1*length*hidden_size
        dot = torch.tanh(attn_f) # 1*length*hidden_size
        alpha = self.to_alpha(dot) # 1*length*1
        if mask is not None:
            alpha = alpha.masked_fill(mask.float().unsqueeze(2).eq(0), -1e9)
        attw = F.softmax(alpha.transpose(1, 2), dim=2) # 1*1*length
        attn_outputs = attw @ hiddens # 1*1*hidden_size
        attn_outputs = attn_outputs.squeeze(1) # 1*hidden_size

        return attn_outputs


class EncoderRNN(nn.Module):
    def __init__(self, num_vocab, embedding_dim, hidden_size, num_classes, batch_size):
        """
        参数
            num_vocab: 词表大小
            embedding_dim: 词向量维数
            hidden_size: 隐状态维数
            num_classes: 类别数量
        """
        super(EncoderRNN, self).__init__()
        # TODO
        self.Encoder = BiLSTM(embedding_dim, hidden_size, batch_size)
        self.selfatt = Attention(hidden_size*2)
        self.linear_layers = nn.Sequential(
            nn.Linear(hidden_size*2, num_classes), 
            nn.LogSoftmax()
        )
    
    def forward(self, x, mode, mask=None):
        """
        输入
            x: 1 * length, LongTensor -> 1 * length * input_size
        输出
            outputs: 1 * num_classes
        """
        # TODO
        wordfeats = self.Encoder(x, mode) # 1 * length * (hidden_size*2)
        sentfeat = self.selfatt(wordfeats, mask)  # 1 * (hidden_size*2)
        outputs = self.linear_layers(sentfeat)  # 1 * num_classes
        

        return outputs