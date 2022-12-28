

import torch
import torch.nn as nn
import numpy as np
import math
from model.modules import MultiHeadAttention, PositionwiseFeedForward


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(n_position, d_hid)
        position = torch.arange(0, n_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_hid, 2).float() * (-math.log(10000.0)/d_hid))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].clone().detach()


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model,  dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class Encoder(nn.Module):

    def __init__(self,  n_layers, n_head, d_model, dropout=0.1, n_position=200):
        super().__init__()
        self.sensor_embedding = nn.Linear(1, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, n_head, dropout=dropout)
            for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.std = 0.02

    def forward(self, src_seq):

        src_seq = src_seq.unsqueeze(-1)
        emb_vector = self.sensor_embedding(src_seq)
        enc_output = self.dropout(self.position_enc(emb_vector))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(enc_output)

        return enc_output, _


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=64, d_embedding=64,
                 n_layers=3, n_head=4, dropout=0.1, n_position=200, window_size=16):

        super().__init__()

        self.transformer_encoder = Encoder(
            n_position=n_position,
            d_model=d_model,
            n_layers=n_layers, n_head=n_head,
            dropout=dropout)

        self.linear_block = nn.Sequential(
            nn.Linear(d_model,  d_embedding*2),
            nn.LeakyReLU(),
            nn.Linear(d_embedding*2,  d_embedding*4),
            nn.LeakyReLU(),)

        self.phase_block = nn.Sequential(
            nn.Linear(d_embedding*4 * window_size, d_embedding*4),
            nn.LeakyReLU(),
            nn.Linear(d_embedding*4, d_embedding*2),
            nn.LeakyReLU(),
            nn.Linear(d_embedding*2, 2),
            nn.Sigmoid())

        self.mlp_block = nn.Sequential(
            nn.Linear(d_embedding*4 * window_size, d_embedding*4),
            nn.LeakyReLU(),
            nn.Linear(d_embedding*4, d_embedding*2),
            nn.LeakyReLU(),
            nn.Linear(d_embedding*2, d_embedding))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_seq):
        enc_output, *_ = self.transformer_encoder(src_seq)
        enc_output = self.linear_block(enc_output)
        phase_variable = self.phase_block(
            enc_output.reshape(enc_output.size(0), -1))
        embedding_vector = self.mlp_block(
            enc_output.reshape(enc_output.size(0), -1))

        return embedding_vector.reshape(embedding_vector.size(0), -1), phase_variable
