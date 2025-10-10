import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

'''
    Context-Aware Prototype-Guided State-Space Model 
    for Efficient Wearable Human Activity Recognition in IoT Devices
  
    Gyuyeon Lim and Myung-Kyu Yi
'''

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

class ChannelAttention1D(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 4):
        super().__init__()
        hidden_channels = max(8, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.BatchNorm1d(hidden_channels), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, hidden_channels), nn.BatchNorm1d(hidden_channels), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        avg_pool = self.avg_pool(x).view(B, C)
        max_pool = self.max_pool(x).view(B, C)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        attention = self.sigmoid(avg_out + max_out).view(B, C, 1)
        return attention

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv_layers = nn.Sequential(
            nn.Conv1d(2, 8, kernel_size, padding=padding, bias=False), nn.BatchNorm1d(8), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(8, 8, kernel_size, padding=padding, bias=False), nn.BatchNorm1d(8), nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(8, 1, kernel_size, padding=padding, bias=False), nn.BatchNorm1d(1)
        )
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.conv_layers:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv_layers(concat)
        attention = self.sigmoid(attention)
        return attention

class CBAM1D(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 4, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention1D(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention1D(kernel_size)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out + identity * self.residual_weight

class PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 8, overlap: bool = True):
        super().__init__()
        stride = patch_size // 2 if overlap else patch_size
        self.projection = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=0)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x

class MambaSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int, eps: float = 1e-5):
        super().__init__()
        self.d_model, self.d_state, self.eps = d_model, d_state, eps
        self.A_log = nn.Parameter(torch.log(torch.linspace(1.0, float(d_state), d_state)))
        self.chunk_size = 16

    def forward(self, u_input, delta, b_mat, c_mat):
        B, T, D = u_input.shape
        N = self.d_state
        A = -torch.exp(self.A_log).view(1, 1, 1, N)
        chunk_size = min(self.chunk_size, T)
        num_chunks = (T + chunk_size - 1) // chunk_size
        output = torch.zeros(B, T, D, device=u_input.device, dtype=u_input.dtype)
        h = torch.zeros(B, D, N, device=u_input.device, dtype=u_input.dtype)

        for chunk_idx in range(num_chunks):
            start_idx, end_idx = chunk_idx * chunk_size, min(start_idx + chunk_size, T)
            u_chunk = u_input[:, start_idx:end_idx, :]
            delta_chunk = delta[:, start_idx:end_idx, :]
            b_chunk = b_mat[:, start_idx:end_idx, :, :]
            c_chunk = c_mat[:, start_idx:end_idx, :, :]
            deltaA = torch.clamp(delta_chunk.unsqueeze(-1) * A, min=-10, max=10)
            exp_deltaA = torch.exp(deltaA)
            frac_exact = (exp_deltaA - 1.0) / (A + 1e-12)
            frac = torch.where(deltaA.abs() < 1e-4, delta_chunk.unsqueeze(-1), frac_exact)
            Bu = frac * (b_chunk * u_chunk.unsqueeze(-1))
            Ah = exp_deltaA
            Ah_cum = torch.cumprod(Ah, dim=1)
            h_expand = h.unsqueeze(1) * Ah_cum
            flip_weights = torch.cat([torch.ones_like(Ah[:, :1]), Ah[:, :-1].flip(1)], dim=1).flip(1)
            Bu_term = torch.cumsum(Bu * flip_weights, dim=1).flip(1)
            h_new = h_expand + Bu_term
            y_chunk = (h_new * c_chunk).sum(dim=-1)
            output[:, start_idx:end_idx, :] = y_chunk
            h = h_new[:, -1]
            del h_expand, Bu_term, Ah_cum, exp_deltaA
        return output

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, expansion: int, groups: int, dropout: float, droppath: float, bidirectional: bool):
        super().__init__()
        assert (expansion * d_model) % groups == 0
        self.d_model, self.d_state, self.expansion, self.d_inner, self.groups = d_model, d_state, expansion, expansion * d_model, groups
        self.group_size = self.d_inner // groups
        self.bidirectional = bidirectional
        self.norm1 = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.param_proj = nn.Linear(self.d_inner, self.groups * (1 + 2 * d_state), bias=False)
        self.ssm = MambaSSM(d_model=self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )

    def _expand_group_params(self, xz, params):
        B, T, _ = xz.shape
        G, N = self.groups, self.d_state
        delta_g, b_g, c_g = torch.split(params, [1, N, N], dim=-1)
        repeat = self.group_size
        delta = delta_g.repeat_interleave(repeat, dim=2).squeeze(-1)
        b_mat = b_g.unsqueeze(3).expand(B, T, G, repeat, N).reshape(B, T, -1, N)
        c_mat = c_g.unsqueeze(3).expand(B, T, G, repeat, N).reshape(B, T, -1, N)
        return delta, b_mat, c_mat

    def _run_ssm_once(self, xz, gate):
        B, T, Din = xz.shape
        raw_params = self.param_proj(xz)
        raw_params = raw_params.view(B, T, self.groups, (1 + 2 * self.d_state))
        delta, b_mat, c_mat = self._expand_group_params(xz, raw_params)
        delta = F.softplus(delta)
        y_ssm = self.ssm(xz, delta, b_mat, c_mat)
        y = y_ssm * F.silu(gate)
        return y

    def forward(self, x):
        residual = x
        z = self.norm1(x)
        xz, gate = self.in_proj(z).chunk(2, dim=-1)
        xz = F.silu(xz)
        if not self.bidirectional:
            y = self._run_ssm_once(xz, gate)
        else:
            y_f = self._run_ssm_once(xz, gate)
            y_b = self._run_ssm_once(torch.flip(xz, dims=[1]), torch.flip(gate, dims=[1]))
            y_b = torch.flip(y_b, dims=[1])
            y = 0.5 * (y_f + y_b)
        y = self.out_proj(y)
        x = residual + self.drop_path(self.drop(y))
        x = x + self.drop_path(self.drop(self.ffn(self.norm2(x))))
        return x

class LiteMambaEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, d_state: int, expansion: int, groups: int, dropout: float, droppath: float, bidirectional: bool, use_local: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model, d_state=d_state, expansion=expansion, groups=groups,
                dropout=dropout, droppath=droppath * (i / (num_layers - 1)) if num_layers > 1 else droppath,
                bidirectional=bidirectional
            ) for i in range(num_layers)
        ])
        self.use_local = use_local
        if use_local:
            self.local_conv = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model),
                nn.BatchNorm1d(d_model), nn.SiLU()
            )
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        if self.use_local:
            x_conv = x.transpose(1, 2)
            x_conv = self.local_conv(x_conv)
            x = x_conv.transpose(1, 2) + x
        x = self.norm(x)
        return x

class AttentionPooling(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        valid_heads = [h for h in [1, 2, 3, 4, 6, 8] if dim % h == 0 and h <= num_heads]
        self.num_heads = max(valid_heads) if valid_heads else 1
        self.head_dim = dim // self.num_heads
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        query = self.query_proj(cls_tokens).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous().view(B, 1, D)
        output = self.output_proj(attn_output.squeeze(1))
        return output

class PrototypeClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float, num_context_heads: int, temperature: float, aux_loss_weight: float):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, input_dim))
        nn.init.xavier_normal_(self.prototypes.data)
        self.feature_proj = nn.Linear(input_dim, input_dim)
        self.context_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 4), nn.BatchNorm1d(input_dim // 4), nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate), nn.Linear(input_dim // 4, input_dim)
            ) for _ in range(num_context_heads)
        ])
        self.context_fusion = nn.Sequential(
            nn.Linear(input_dim * num_context_heads, input_dim), nn.SiLU(),
            nn.Linear(input_dim, input_dim)
        )
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def l2_normalize(self, x, dim=-1, eps=1e-8):
        return F.normalize(x, p=2, dim=dim, eps=eps)

    def forward(self, x, labels=None, epoch=0):
        projected_features = self.feature_proj(x)
        normalized_features = self.l2_normalize(projected_features)
        context_outputs = [head(x) for head in self.context_heads]
        concatenated_context = torch.cat(context_outputs, dim=-1)
        context_vector = self.context_fusion(concatenated_context)
        normalized_prototypes = self.l2_normalize(self.prototypes)
        adjusted_prototypes = self.l2_normalize(normalized_prototypes + context_vector.unsqueeze(1))
        similarities = torch.matmul(normalized_features.unsqueeze(1), adjusted_prototypes.transpose(1, 2)).squeeze(1)
        logits = self.temperature * similarities
        return logits, {} # Dummy aux_info for simplicity

class CBAMPatchMambaAttAVPPrototype(nn.Module):
    def __init__(self, in_channels, num_classes, patch_size, embed_dim, num_layers, d_state, expansion, groups, dropout, droppath, bidirectional, num_heads, classifier_dropout, num_context_heads, temperature, aux_loss_weight):
        super().__init__()
        self.cbam = CBAM1D(in_channels)
        self.patch_embed = PatchEmbedding1D(in_channels, embed_dim, patch_size, overlap=True)
        self.encoder = LiteMambaEncoder(
            d_model=embed_dim, num_layers=num_layers, d_state=d_state, expansion=expansion, groups=groups,
            dropout=dropout, droppath=droppath, bidirectional=bidirectional
        )
        self.pooling = AttentionPooling(embed_dim, num_heads=num_heads)
        self.classifier = PrototypeClassifier(
            input_dim=embed_dim, num_classes=num_classes, dropout_rate=classifier_dropout,
            num_context_heads=num_context_heads, temperature=temperature, aux_loss_weight=aux_loss_weight
        )

    def forward(self, x, labels=None, epoch=0):
        x = self.cbam(x)
        x = self.patch_embed(x)
        x = self.encoder(x)
        x = self.pooling(x)
        logits, aux_info = self.classifier(x, labels=labels, epoch=epoch)
        return logits, aux_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a CBAM-Mamba HAR model with specified hyperparameters.")
    
    parser.add_argument('-ic', '--in_channels', type=int, required=True, help='Number of input sensor channels.')
    parser.add_argument('-nc', '--num_classes', type=int, required=True, help='Number of output activity classes.')
    parser.add_argument('--patch_size', type=int, default=8, help='Patch size for the embedding layer.')
    parser.add_argument('--embed_dim', type=int, default=140, help='Embedding dimension of the model.')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of Mamba layers in the encoder.')
    parser.add_argument('--d_state', type=int, default=56, help='State dimension (N) of the Mamba SSM.')
    parser.add_argument('--expansion', type=int, default=2, help='Expansion factor in the Mamba block.')
    parser.add_argument('--groups', type=int, default=4, help='Number of groups in the Mamba block.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for Mamba and FFN.')
    parser.add_argument('--droppath', type=float, default=0.1, help='Stochastic depth rate (droppath).')
    parser.add_argument('--bidirectional', action='store_true', help='Use a bidirectional Mamba encoder.')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for Attention Pooling.')
    parser.add_argument('--classifier_dropout', type=float, default=0.1, help='Dropout rate for the prototype classifier.')
    parser.add_argument('--num_context_heads', type=int, default=4, help='Number of context heads in the classifier.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Initial temperature for the classifier.')
    parser.add_argument('--aux_loss_weight', type=float, default=0.01, help='Weight for auxiliary losses in the classifier.')

    args = parser.parse_args()

    print("\n" + "="*50)
    print("Building Model with a Custom Configuration")
    print("="*50)
    for key, value in vars(args).items():
        print(f"{key:<20}: {value}")
    print("="*50 + "\n")

    model = CBAMPatchMambaAttAVPPrototype(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        d_state=args.d_state,
        expansion=args.expansion,
        groups=args.groups,
        dropout=args.dropout,
        droppath=args.droppath,
        bidirectional=args.bidirectional,
        num_heads=args.num_heads,
        classifier_dropout=args.classifier_dropout,
        num_context_heads=args.num_context_heads,
        temperature=args.temperature,
        aux_loss_weight=args.aux_loss_weight
    )
