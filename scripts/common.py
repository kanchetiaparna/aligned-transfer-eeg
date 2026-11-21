"""Shared data, preprocessing, augmentation and model code for BCI IV 2a experiments."""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import math, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
DATA = f"{ROOT}/data"
N_CH, N_T, N_CLS, SFREQ = 22, 1000, 4, 250


# ----------------------------------------------------------------------------- data
def load_subject(s):
    d = np.load(f"{DATA}/A{s:02d}.npz")
    return d["Xtr"], d["ytr"], d["Xte"], d["yte"]


def ea_matrix(X):
    """Euclidean Alignment reference R^{-1/2} from mean trial covariance (label free)."""
    C = np.mean([x @ x.T / x.shape[1] for x in X.astype(np.float64)], axis=0)
    w, V = np.linalg.eigh(C)
    return (V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-10))) @ V.T).astype(np.float32)


def apply_ea(X, R):
    return np.einsum("ij,njt->nit", R, X).astype(np.float32)


def preprocess(Xtr, Xte, ea):
    """ea=True: align each session to identity covariance using its own (unlabelled) trials.
    ea=False: per-channel z-score with statistics from the training trials only."""
    if ea:
        return apply_ea(Xtr, ea_matrix(Xtr)), apply_ea(Xte, ea_matrix(Xte))
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def stratified_split(y, frac, seed):
    rng = np.random.RandomState(seed)
    va = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        va += list(rng.choice(idx, int(round(len(idx) * frac)), replace=False))
    va = np.array(sorted(va))
    tr = np.setdiff1d(np.arange(len(y)), va)
    return tr, va


# ----------------------------------------------------------------------------- augmentation
def segment_recombine(X, y, n_seg=8, n_new=None, rng=np.random):
    """S&R: build artificial trials by stitching same-class time segments from random trials."""
    n_new = n_new or len(y)
    seg = X.shape[-1] // n_seg
    outX, outy = [], []
    per = n_new // len(np.unique(y))
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        pick = rng.choice(idx, size=(per, n_seg))
        new = np.empty((per,) + X.shape[1:], np.float32)
        for k in range(n_seg):
            new[:, :, k * seg:(k + 1) * seg] = X[pick[:, k], :, k * seg:(k + 1) * seg]
        outX.append(new); outy.append(np.full(per, c))
    return np.concatenate(outX), np.concatenate(outy)


def gpu_augment(x, p_shift=0.5, max_shift=50, p_scale=0.5, noise=0.05):
    """Cheap on-GPU jitter: circular time shift, per-trial amplitude scale, gaussian noise."""
    B = x.shape[0]
    if p_shift > 0:
        sh = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
        sh = sh * (torch.rand(B, device=x.device) < p_shift)
        ar = torch.arange(x.shape[-1], device=x.device)
        idx = (ar[None] - sh[:, None]) % x.shape[-1]
        x = torch.gather(x, 2, idx[:, None, :].expand(-1, x.shape[1], -1))
    if p_scale > 0:
        sc = 1 + 0.1 * torch.randn(B, 1, 1, device=x.device) * (torch.rand(B, 1, 1, device=x.device) < p_scale)
        x = x * sc
    if noise > 0:
        x = x + noise * torch.randn_like(x) * x.std(dim=-1, keepdim=True)
    return x


# ----------------------------------------------------------------------------- custom model
class MaxNormLinear(nn.Linear):
    def __init__(self, *a, max_norm=0.25, **k):
        super().__init__(*a, **k); self.max_norm = max_norm

    def forward(self, x):
        with torch.no_grad():
            self.weight.data = torch.renorm(self.weight.data, 2, 0, self.max_norm)
        return super().forward(x)


class MSCFormer(nn.Module):
    """Multi-Scale Conv + Transformer.
    1) parallel temporal convs (kernels 64/32/16 ~ 256/128/64 ms) act as a learnable filter bank
    2) depthwise spatial conv per band (EEGNet style, few params, strong prior)
    3) pooling -> ~25 tokens; light pre-norm transformer models temporal dynamics
    4) residual mix of conv tokens and attention output, mean pooled, max-norm classifier.
    Deliberately small (~40k params) to limit overfitting on 288 trials/subject."""

    def __init__(self, n_chans=N_CH, n_outputs=N_CLS, n_times=N_T, F1=16, kernels=(64, 32, 16),
                 D=2, dim=64, depth=2, heads=4, drop=0.3, pool1=8, pool2=5):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False), nn.BatchNorm2d(F1))
            for k in kernels])
        Fb = F1 * len(kernels)
        self.spatial = nn.Sequential(
            nn.Conv2d(Fb, Fb * D, (n_chans, 1), groups=Fb, bias=False), nn.BatchNorm2d(Fb * D), nn.ELU(),
            nn.AvgPool2d((1, pool1)), nn.Dropout(drop))
        self.sep = nn.Sequential(
            nn.Conv2d(Fb * D, Fb * D, (1, 16), padding=(0, 8), groups=Fb * D, bias=False),
            nn.Conv2d(Fb * D, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.ELU(),
            nn.AvgPool2d((1, pool2)), nn.Dropout(drop))
        with torch.no_grad():
            n_tok = self._tokens(torch.zeros(1, n_chans, n_times)).shape[1]
        self.pos = nn.Parameter(torch.zeros(1, n_tok, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Dropout(0.5), MaxNormLinear(dim * 2, n_outputs))

    def _tokens(self, x):
        x = x.unsqueeze(1)
        n = x.shape[-1]
        x = torch.cat([b(x)[..., :n] for b in self.branches], 1)
        x = self.sep(self.spatial(x))
        return x.squeeze(2).transpose(1, 2)  # B, tokens, dim

    def forward(self, x):
        t = self._tokens(x)
        z = self.norm(self.tf(t + self.pos))
        return self.head(torch.cat([z.mean(1), t.mean(1)], 1))


# ----------------------------------------------------------------------------- factory
def build_model(name):
    if name == "MSCFormer":
        return MSCFormer()
    if name == "MSCFormerL":
        return MSCFormer(F1=24, dim=96, depth=3)
    if name == "TCFormer":
        from tcformer import TCFormer
        return TCFormer()                # repo config (configs/tcformer.yaml): 5 transformer layers, 127k params
    if name == "TCFormer_d2":
        from tcformer import TCFormer
        return TCFormer(trans_depth=2)   # matches the 77.8k params reported in the paper
    import braindecode.models as M
    return getattr(M, name)(n_chans=N_CH, n_outputs=N_CLS, n_times=N_T, sfreq=SFREQ)
