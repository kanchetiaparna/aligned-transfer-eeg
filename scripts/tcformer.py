"""TCFormer: Temporal Convolutional Transformer for EEG motor imagery decoding.

Standalone port (no Lightning) of the official implementation:
  H. Altaheri, F. Karray, A.H. Karimi (2025). Temporal convolutional transformer for EEG based
  motor imagery decoding. Scientific Reports. https://github.com/Altaheri/TCFormer
  Original code: MIT License, Copyright (c) 2025 Hamdi Altaheri.

Architecture: multi-kernel CNN front end (+ channel group attention) -> 5 layer pre-LN transformer
with grouped-query attention + RoPE + stochastic depth -> concat(conv, transformer) features ->
causal dilated TCN head with grouped max-norm classifiers averaged over groups.
Defaults are the paper's BCI IV-2a settings (configs/tcformer.yaml).
"""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import torch
from torch import nn, Tensor
import torch.nn.functional as F


def glorot_weight_zero_bias(model):
    for m in model.modules():
        if hasattr(m, "weight") and m.weight is not None and "norm" not in m.__class__.__name__.lower():
            nn.init.xavier_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)


class CausalConv1d(nn.Conv1d):
    def __init__(self, cin, cout, kernel_size, dilation=1, groups=1, bias=True):
        super().__init__(cin, cout, kernel_size, dilation=dilation, groups=groups, bias=bias)
        self._pad = (kernel_size - 1) * dilation

    def forward(self, x):
        return super().forward(F.pad(x, (self._pad, 0)))


class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *a, max_norm=None, **k):
        self.max_norm = max_norm
        super().__init__(*a, **k)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class ChannelGroupAttention(nn.Module):
    """Squeeze-excitation style gating with one weight per kernel group."""
    def __init__(self, in_channels, num_groups, reduction=4):
        super().__init__()
        self.G, self.gs = num_groups, in_channels // num_groups
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, groups=num_groups, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_groups, 1, groups=num_groups, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(self.pool(x)))))
        return (x.view(B, self.G, self.gs, H, W) * w.view(B, self.G, 1, 1, 1)).view(B, C, H, W)


class MultiKernelConvBlock(nn.Module):
    def __init__(self, n_channels, kernels=(20, 32, 64), F1=32, D=2, pool1=8, pool2=7,
                 dropout=0.4, d_group=16, use_group_attn=True):
        super().__init__()
        self.temporal_convs = nn.ModuleList([nn.Sequential(
            nn.ConstantPad2d((k // 2 - 1, k // 2, 0, 0) if k % 2 == 0 else (k // 2, k // 2, 0, 0), 0),
            nn.Conv2d(1, F1, (1, k), bias=False), nn.BatchNorm2d(F1)) for k in kernels])
        G = len(kernels)
        self.d_model = d_group * G
        F2 = F1 * G * D
        self.channel_DW_conv = nn.Sequential(
            nn.Conv2d(F1 * G, F2, (n_channels, 1), groups=F1 * G, bias=False), nn.BatchNorm2d(F2), nn.ELU())
        self.pool1, self.drop1 = nn.AvgPool2d((1, pool1)), nn.Dropout(dropout)
        self.reduce = (nn.Sequential(nn.Conv2d(F2, self.d_model, 1, groups=G, bias=False),
                                     nn.BatchNorm2d(self.d_model)) if F2 != self.d_model else nn.Identity())
        self.temporal_conv_2 = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, (1, 16), padding="same", groups=G, bias=False),
            nn.BatchNorm2d(self.d_model), nn.ELU())
        self.group_attn = ChannelGroupAttention(self.d_model, G) if (use_group_attn and G > 1) else None
        self.pool2, self.drop2 = nn.AvgPool2d((1, pool2)), nn.Dropout(dropout)
        glorot_weight_zero_bias(self)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.cat([c(x) for c in self.temporal_convs], 1)
        x = self.drop1(self.pool1(self.channel_DW_conv(x)))
        x = self.temporal_conv_2(self.reduce(x))
        if self.group_attn is not None:
            x = x + self.group_attn(x)
        return self.drop2(self.pool2(x)).squeeze(2)


class TCNBlock(nn.Module):
    def __init__(self, k, nf, dilation, groups, dropout):
        super().__init__()
        self.c1, self.b1 = CausalConv1d(nf, nf, k, dilation, groups), nn.BatchNorm1d(nf)
        self.c2, self.b2 = CausalConv1d(nf, nf, k, dilation, groups), nn.BatchNorm1d(nf)
        self.d1, self.d2 = nn.Dropout(dropout), nn.Dropout(dropout)
        nn.init.zeros_(self.c1.bias); nn.init.zeros_(self.c2.bias)

    def forward(self, inp):
        x = self.d1(F.elu(self.b1(self.c1(inp))))
        x = self.d2(F.elu(self.b2(self.c2(x))))
        return F.elu(inp + x)


class TCNHead(nn.Module):
    def __init__(self, d_features, n_groups, depth, k, dropout, n_classes):
        super().__init__()
        self.G, self.n_classes = n_groups, n_classes
        self.tcn = nn.Sequential(*[TCNBlock(k, d_features, 2 ** i, n_groups, dropout) for i in range(depth)])
        self.linear = Conv1dWithConstraint(d_features, n_classes * n_groups, 1, groups=n_groups, max_norm=0.25)

    def forward(self, x):
        x = self.linear(self.tcn(x)[:, :, -1:]).squeeze(-1)
        return x.view(x.size(0), self.G, self.n_classes).mean(1)


def _rope_cache(head_dim, seq_len, device):
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    f = torch.outer(torch.arange(seq_len, device=device).float(), theta)
    emb = torch.cat((f, f), -1)
    return emb.cos(), emb.sin()


def _rope(q, k, cos, sin):
    def rot(x):
        return torch.stack((-x[..., 1::2], x[..., ::2]), -1).flatten(-2)
    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


class GQAttention(nn.Module):
    def __init__(self, d_model, q_heads, kv_heads, dropout):
        super().__init__()
        self.hq, self.hk, self.hd = q_heads, kv_heads, d_model // q_heads
        self.scale = self.hd ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * kv_heads * self.hd, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.hq, self.hd).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, self.hk, 2, self.hd)
        r = self.hq // self.hk
        k = kv[..., 0, :].transpose(1, 2).repeat_interleave(r, 1)
        v = kv[..., 1, :].transpose(1, 2).repeat_interleave(r, 1)
        q, k = _rope(q, k, cos[:T], sin[:T])
        a = self.drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(-1))
        return self.o_proj((a @ v).transpose(1, 2).reshape(B, T, C))


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__(); self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training: return x
        keep = 1 - self.p
        m = (keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device)).floor_()
        return x.div(keep) * m


class TransformerBlock(nn.Module):
    def __init__(self, d, q_heads, kv_heads, mlp_ratio=2, dropout=0.4, drop_path=0.25):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = GQAttention(d, q_heads, kv_heads, dropout)
        self.dp = DropPath(drop_path)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d), nn.Dropout(dropout))

    def forward(self, x, cos, sin):
        x = x + self.dp(self.attn(self.n1(x), cos, sin))
        return x + self.dp(self.mlp(self.n2(x)))


class TCFormer(nn.Module):
    def __init__(self, n_channels=22, n_classes=4, F1=32, kernels=(20, 32, 64), pool1=8, pool2=7, D=2,
                 dropout_conv=0.4, d_group=16, tcn_depth=2, k_tcn=4, dropout_tcn=0.3, use_group_attn=True,
                 q_heads=4, kv_heads=2, trans_depth=5, trans_dropout=0.4, drop_path_max=0.25):
        super().__init__()
        G = len(kernels); self.d_model = d_group * G
        self.conv_block = MultiKernelConvBlock(n_channels, kernels, F1, D, pool1, pool2, dropout_conv,
                                               d_group, use_group_attn)
        self.mix = nn.Sequential(nn.Conv1d(self.d_model, self.d_model, 1, bias=False),
                                 nn.BatchNorm1d(self.d_model), nn.SiLU())
        rates = torch.linspace(0, 1, trans_depth) ** 2 * drop_path_max
        self.transformer = nn.ModuleList([TransformerBlock(self.d_model, q_heads, kv_heads, dropout=trans_dropout,
                                                           drop_path=rates[i].item()) for i in range(trans_depth)])
        self.reduce = nn.Sequential(nn.Conv1d(self.d_model, d_group, 1, bias=False), nn.BatchNorm1d(d_group), nn.SiLU())
        self.tcn_head = TCNHead(d_group * (G + 1), G + 1, tcn_depth, k_tcn, dropout_tcn, n_classes)
        self.head_dim = self.d_model // q_heads

    def forward(self, x):
        conv = self.conv_block(x)                      # B, d_model, T
        tok = self.mix(conv).transpose(1, 2)           # B, T, d_model
        cos, sin = _rope_cache(self.head_dim, tok.shape[1], tok.device)
        for blk in self.transformer:
            tok = blk(tok, cos, sin)
        tr = self.reduce(tok.transpose(1, 2))
        return self.tcn_head(torch.cat((conv, tr), 1))
