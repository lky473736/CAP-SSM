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

class FourierAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = (d_model // num_heads) ** -0.5

    def forward(self, x):
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, V)
        return self.out_proj(output)

class FEDformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.fourier_attn = FourierAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.fourier_attn(x)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

class FEDformer(nn.Module):
    def __init__(self, input_channels, num_classes, d_model, num_heads, num_layers, d_ff, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([FEDformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
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
    parser = argparse.ArgumentParser(description="Build a FEDformer for HAR.")
    parser.add_argument('--input_channels', type=int, required=True, help='Number of input sensor channels.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes.')
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer blocks.')
    parser.add_argument('--d_ff', type=int, default=512, help='Dimension of the feed-forward layer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    args = parser.parse_args()

    print("\n" + "="*50 + "\nBuilding FEDformer with Configuration:\n" + "="*50)
    for k, v in vars(args).items(): print(f"{k:<15}: {v}")
    print("="*50 + "\n")

    model = FEDformer(**vars(args))
