import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math

'''
    Context-Aware Prototype-Guided State-Space Model 
    for Efficient Wearable Human Activity Recognition in IoT Devices
  
    Gyuyeon Lim and Myung-Kyu Yi
'''

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model, self.num_heads, self.d_k = d_model, num_heads, d_model // num_heads
        self.w_q, self.w_k, self.w_v = nn.Linear(d_model, d_model, bias=False), nn.Linear(d_model, d_model, bias=False), nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(self, query, key, value, mask=None):
        bs, sl = query.size(0), query.size(1)
        Q = self.w_q(query).view(bs, sl, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, V).transpose(1, 2).contiguous().view(bs, sl, self.d_model)
        return self.w_o(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        pos = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class BasicTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class BasicTransformer(nn.Module):
    def __init__(self, input_channels, num_classes, d_model, num_heads, num_layers, d_ff, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([BasicTransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x.transpose(1, 2))
        x = self.dropout(self.pos_enc(x))
        for layer in self.layers:
            x = layer(x)
        x = torch.mean(self.norm(x), dim=1)
        return self.classifier(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Basic Transformer for HAR.")
    parser.add_argument('--input_channels', type=int, required=True, help='Number of input sensor channels.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes.')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer blocks.')
    parser.add_argument('--d_ff', type=int, default=512, help='Dimension of the feed-forward layer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    args = parser.parse_args()

    print("\n" + "="*50 + "\nBuilding BasicTransformer with Configuration:\n" + "="*50)
    for k, v in vars(args).items(): print(f"{k:<15}: {v}")
    print("="*50 + "\n")

    model = BasicTransformer(**vars(args))
