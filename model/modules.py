import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_model, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_model, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_model, bias=False)

        self.fc = nn.Linear(n_head * d_model, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_model ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, self.n_head,  self.d_model)
        k = self.w_ks(k).view(sz_b, len_k, self.n_head,  self.d_model)
        v = self.w_vs(v).view(sz_b, len_v, self.n_head,  self.d_model)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, attn = self.attention(q, k, v)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_embedding=64, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_model)
        self.w_2 = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x
