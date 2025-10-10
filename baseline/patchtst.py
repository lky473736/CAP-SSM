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

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x):
        bs, c, sl = x.shape
        num_patches = sl // self.patch_size
        x = x[:, :, :num_patches * self.patch_size]
        x = x.view(bs, c, num_patches, self.patch_size).permute(0, 2, 1, 3)
        x = x.reshape(bs * num_patches, c * self.patch_size)
        x = self.proj(x).view(bs, num_patches, -1)
        return x

class PatchTST(nn.Module):
    def __init__(self, input_channels, num_classes, patch_size, d_model, num_heads, num_layers, dropout):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([BasicTransformerBlock(d_model, num_heads, d_model * 4, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bs, c, sl = x.shape
        channel_outputs = []
        for i in range(c):
            channel_data = x[:, i:i+1, :]
            patches = self.patch_embedding(channel_data)
            patches = self.dropout(self.pos_encoding(patches))
            for layer in self.layers:
                patches = layer(patches)
            channel_repr = torch.mean(self.norm(patches), dim=1)
            channel_outputs.append(channel_repr)
        x = torch.mean(torch.stack(channel_outputs, dim=1), dim=1)
        return self.classifier(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a PatchTST for HAR.")
    parser.add_argument('--input_channels', type=int, required=True, help='Number of input sensor channels.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes.')
    parser.add_argument('--patch_size', type=int, default=16, help='Size of each patch.')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer blocks.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    args = parser.parse_args()

    print("\n" + "="*50 + "\nBuilding PatchTST with Configuration:\n" + "="*50)
    for k, v in vars(args).items(): print(f"{k:<15}: {v}")
    print("="*50 + "\n")

    model = PatchTST(**vars(args))
