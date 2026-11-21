"""New model experiments built on the paper-faithful TCFormer pipeline (train_tcformer_paper.py).

Components (each motivated by the failure analysis in analyze_baselines.py):
  --ea 1      Euclidean Alignment per recording session (label free), replaces z-scoring.
              Targets the ~8-point cross-session drop. Test session R uses its own unlabeled trials.
  --cov 1     Filter-bank Riemannian branch fused with TCFormer ("TCFormer-RB"):
              6 learnable temporal filters initialised as band-pass 4-8, 8-12, 12-16, 16-24, 24-32, 32-40 Hz,
              per-band learnable spatial projection 22 -> 12, band covariance, log-Euclidean map (matrix log),
              upper-triangle tangent vector -> dropout -> linear; logits added to TCFormer logits.
              Targets tongue/feet confusions, which are spatial-spectral (band power topography) differences.
  --pretrain  Two-stage transfer: stage "pre" trains on the other 8 subjects of 2a (both sessions, each
              aligned separately) and saves a checkpoint; stage "ft" fine-tunes it on the target's session T.
Protocol (unchanged, honest): train on session T only, evaluate the last-epoch weights once on session E.
"""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import argparse, json, math, os, random, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import firwin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score
import warnings; warnings.filterwarnings("ignore")
from tcformer import TCFormer

p = argparse.ArgumentParser()
p.add_argument("--subject", type=int, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--ea", type=int, default=1)
p.add_argument("--cov", type=int, default=0)
p.add_argument("--stage", choices=["single", "pre", "ft"], default="single")
p.add_argument("--epochs", type=int, default=1000)
p.add_argument("--lr", type=float, default=9e-4)
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--aug", type=int, default=1)
p.add_argument("--model", choices=["tcformer", "atcnet", "eegnet"], default="tcformer", help="backbone; atcnet/eegnet test whether the AT recipe generalises")
p.add_argument("--pseed", type=int, default=0, help="pretraining seed; 0 = original checkpoint name, k>0 adds _p{k}")
p.add_argument("--data", default=f"{ROOT}/data/tcf")
p.add_argument("--tag", required=True)
p.add_argument("--out", default=f"{ROOT}/results/newmodel")
a = p.parse_args()
CK = f"{ROOT}/checkpoints/newmodel"; os.makedirs(CK, exist_ok=True)
DS = os.path.basename(a.data.rstrip("/"))
ckpt = f"{CK}/pre_ea{a.ea}_cov{a.cov}_S{a.subject}.pt" if DS == "tcf" else f"{CK}/pre_{DS}_ea{a.ea}_cov{a.cov}_S{a.subject}.pt"
if a.pseed: ckpt = ckpt.replace(".pt", f"_p{a.pseed}.pt")
if a.model != "tcformer": ckpt = ckpt.replace("/pre_", f"/pre_{a.model}_")
if a.stage == "pre": a.seed = a.pseed  # pretraining run is seeded by its pretraining seed
import glob as _g
ALL_SUBJ = sorted(int(os.path.basename(f)[1:3]) for f in _g.glob(f"{a.data}/A*.npz"))
out_dir = f"{a.out}/{a.tag}"; os.makedirs(out_dir, exist_ok=True)
res_file = f"{out_dir}/S{a.subject}_seed{a.seed}.json"
if (a.stage == "pre" and os.path.exists(ckpt)) or (a.stage != "pre" and os.path.exists(res_file)):
    print("skip, exists"); raise SystemExit

dev = "cuda"; torch.backends.cudnn.benchmark = True; torch.set_float32_matmul_precision("medium")
for _ in range(240):  # shared GPUs: wait for ~6 GB free before starting
    if torch.cuda.mem_get_info()[0] > 6 * 2**30: break
    time.sleep(30 + 30 * random.SystemRandom().random())


# ---------------------------------------------------------------- data
def ea_apply(X):
    C = np.mean([x @ x.T / x.shape[1] for x in X.astype(np.float64)], 0)
    w, V = np.linalg.eigh(C)
    return np.einsum("ij,njt->nit", V @ np.diag(w ** -0.5) @ V.T, X).astype(np.float32)


def zscore_pair(Xtr, Xte):
    s, c, t = Xtr.shape
    sc = StandardScaler().fit(Xtr.transpose(1, 0, 2).reshape(c, -1).T)
    f = lambda X: sc.transform(X.transpose(1, 0, 2).reshape(c, -1).T).T.reshape(c, X.shape[0], t).transpose(1, 0, 2)
    return f(Xtr).astype(np.float32), f(Xte).astype(np.float32)


def load(s):
    d = np.load(f"{a.data}/A{s:02d}.npz")
    return d["Xtr"].astype(np.float32), d["ytr"], d["Xte"].astype(np.float32), d["yte"]


if a.stage == "pre":  # other subjects, both sessions, each session aligned on its own
    subs = [s for s in ALL_SUBJ if s != a.subject]; Xs, ys = [], []
    for s in subs:
        Xtr, ytr, Xte, yte = load(s)
        if Xtr.shape[-1] % 8 and Xtr.shape[-1] % 7:
            cut = Xtr.shape[-1] - Xtr.shape[-1] % 8; Xtr, Xte = Xtr[..., :cut], Xte[..., :cut]
        for X, y in ((Xtr, ytr), (Xte, yte)):
            Xs.append(ea_apply(X) if a.ea else zscore_pair(X, X)[0]); ys.append(y)
    Xtr, ytr = np.concatenate(Xs), np.concatenate(ys); Xte, yte = None, None
else:
    Xtr, ytr, Xte, yte = load(a.subject)
    if Xtr.shape[-1] % 8 and Xtr.shape[-1] % 7:          # interaug needs equal chunks (HaLT 375 -> 368)
        cut = Xtr.shape[-1] - Xtr.shape[-1] % 8; Xtr, Xte = Xtr[..., :cut], Xte[..., :cut]
    Xtr_raw, Xte_raw = Xtr.copy(), Xte.copy()
    Xtr, Xte = (ea_apply(Xtr), ea_apply(Xte)) if a.ea else zscore_pair(Xtr, Xte)
C_, T_ = Xtr.shape[1], Xtr.shape[2]
n_cls = 4
Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev, dtype=torch.long)
if Xte is not None:
    XE, yE = torch.tensor(Xte, device=dev), torch.tensor(yte, device=dev, dtype=torch.long)
N_CHUNKS = 9 if T_ == 1125 else (8 if T_ % 8 == 0 else 7)


def interaug(x, y):  # identical to the TCFormer paper's interaug (vectorised)
    newx, newy = torch.empty_like(x), torch.empty_like(y); cur = 0; L = T_ // N_CHUNKS
    for cls in torch.unique(y):
        xc = x[y == cls]; n = len(xc); idx = torch.randint(0, n, (n, N_CHUNKS), device=x.device)
        newx[cur:cur + n] = torch.cat([xc[idx[:, k], :, k * L:(k + 1) * L] for k in range(N_CHUNKS)], -1)
        newy[cur:cur + n] = cls; cur += n
    X2, Y2 = torch.cat((x, newx)), torch.cat((y, newy)); perm = torch.randperm(len(X2), device=x.device)
    return X2[perm], Y2[perm]


# ---------------------------------------------------------------- model
class RiemannBranch(nn.Module):
    """Filter bank -> per-band spatial projection -> covariance -> log-Euclidean tangent vector -> logits."""
    BANDS = [(4, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 40)]

    def __init__(self, C, n_cls, d=12, klen=65, fs=250):
        super().__init__()
        K = len(self.BANDS)
        taps = np.stack([firwin(klen, [lo, hi], pass_zero=False, fs=fs) for lo, hi in self.BANDS])
        self.fb = nn.Conv2d(1, K, (1, klen), padding=(0, klen // 2), bias=False)
        with torch.no_grad(): self.fb.weight.copy_(torch.tensor(taps, dtype=torch.float32)[:, None, None, :])
        self.W = nn.Parameter(torch.linalg.qr(torch.randn(K, C, d))[0])        # orthonormal init
        iu = torch.triu_indices(d, d); self.register_buffer("iu", iu)
        scale = torch.where(iu[0] == iu[1], 1.0, math.sqrt(2.0)); self.register_buffer("scale", scale)
        self.bn = nn.BatchNorm1d(K * iu.shape[1])
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(K * iu.shape[1], n_cls))

    def forward(self, x):                                  # x: B, C, T
        z = self.fb(x.unsqueeze(1))                        # B, K, C, T  (band-passed copies)
        z = torch.einsum("bkct,kcd->bkdt", z, self.W)      # B, K, d, T  (spatial filters per band)
        z = z - z.mean(-1, keepdim=True)
        cov = z @ z.transpose(-1, -2) / (z.shape[-1] - 1)
        cov = cov.double() + 1e-5 * torch.eye(cov.shape[-1], device=x.device, dtype=torch.float64)
        e, V = torch.linalg.eigh(cov)
        L = V @ torch.diag_embed(torch.log(e.clamp_min(1e-8))) @ V.transpose(-1, -2)   # matrix log
        f = (L[..., self.iu[0], self.iu[1]] * self.scale).float().flatten(1)          # tangent vector
        return self.head(self.bn(f))


class TCFormerRB(nn.Module):
    def __init__(self, C, n_cls, cov):
        super().__init__()
        self.tcf = TCFormer(n_channels=C, n_classes=n_cls, trans_depth=2)
        self.rb = RiemannBranch(C, n_cls) if cov else None

    def forward(self, x):
        o = self.tcf(x)
        return o + self.rb(x) if self.rb is not None else o


def build_net():
    if a.model == "tcformer":
        return TCFormerRB(C_, n_cls, a.cov).to(dev)
    import braindecode.models as BM
    if a.model == "eegnet":
        return BM.EEGNetv4(n_chans=C_, n_outputs=n_cls, n_times=T_, sfreq=250).to(dev)
    m = BM.ATCNet(n_chans=C_, n_outputs=n_cls, n_times=T_, sfreq=250).to(dev)
    for mod in m.modules():                    # Keras-matched settings, as in train_atcnet_paper.py (kerasify)
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)): mod.momentum, mod.eps = 0.01, 1e-3
    for n, mod in m.named_modules():
        if n.startswith("conv_block.conv") or ".mha.fc_" in n:
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None: nn.init.zeros_(mod.bias)
    for i in range(len(m.final_layer)):
        lin = nn.Linear(m.final_layer[i].in_features, m.final_layer[i].out_features).to(dev)
        nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias); m.final_layer[i] = lin
    return m


def keras_constraints(m):  # ATCNet's Keras weight limits, applied after every optimizer step
    with torch.no_grad():
        m.conv_block.conv2.weight.clamp_(-1.0, 1.0)
        for lin in m.final_layer:
            n_ = lin.weight.norm(2, dim=1, keepdim=True); lin.weight.mul_(n_.clamp(max=0.25) / (n_ + 1e-7))


def evaluate(model, X, y, bs=48):
    model.eval(); C = torch.zeros((), device=dev); P = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            o = model(X[i:i + bs]); C += (o.argmax(1) == y[i:i + bs]).sum(); P.append(F.softmax(o, -1))
    return C.item() / len(X), torch.cat(P).cpu().numpy()


def train_once():
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    model = build_net()
    if a.stage == "ft":
        for _ in range(180):  # wait for the pretraining job of this target subject
            if os.path.exists(ckpt): break
            time.sleep(60)
        time.sleep(3); model.load_state_dict(torch.load(ckpt, map_location=dev))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.5, 0.999), weight_decay=1e-3)
    W, E = a.warmup, a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: e / max(1, W) if e < W else 0.5 * (1 + np.cos(np.pi * (e - W) / max(1, E - W))))
    hist, t0, BS = [], time.time(), 48
    for ep in range(E):
        model.train(); perm = torch.randperm(len(yt), device=dev); tc = torch.zeros((), device=dev); tn = 0
        for i in range(0, len(yt), BS):
            xb, yb = Xt[perm[i:i + BS]], yt[perm[i:i + BS]]
            if a.aug: xb, yb = interaug(xb, yb)
            o = model(xb); loss = F.cross_entropy(o, yb)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # guards the eigen-decomposition gradients
            opt.step(); tc += (o.argmax(1) == yb).sum(); tn += len(yb)
            if a.model == "atcnet": keras_constraints(model)
        sched.step()
        rec = {"ep": ep, "train_acc": tc.item() / tn}
        if Xte is not None and (ep % 50 == 0 or ep == E - 1):
            rec["test_acc_monitor"] = evaluate(model, XE, yE)[0]   # curves only, never used for selection
        hist.append(rec)
    return model, hist, t0


for attempt in range(30):  # on CUDA OOM: free, wait 1-3 min, restart from scratch (same seed)
    try:
        model, hist, t0 = train_once(); break
    except torch.OutOfMemoryError:
        if attempt == 29: raise
        torch.cuda.empty_cache(); time.sleep(60 + 120 * random.SystemRandom().random())

if a.stage == "pre":
    torch.save(model.state_dict(), ckpt + ".tmp"); os.replace(ckpt + ".tmp", ckpt)
    print(json.dumps({"pretrain_done": a.subject, "cov": a.cov, "train_acc": hist[-1]["train_acc"], "min": (time.time() - t0) / 60}))
    raise SystemExit
acc, P = evaluate(model, XE, yE)
online = {}
if a.ea:
    covs = np.stack([x @ x.T / x.shape[1] for x in Xte_raw.astype(np.float64)])
    isq = lambda C: (lambda w, V: V @ np.diag(w ** -0.5) @ V.T)(*np.linalg.eigh(C))
    R_T = isq(np.mean([x @ x.T / x.shape[1] for x in Xtr_raw.astype(np.float64)], 0))
    csum = np.cumsum(covs, 0)
    Xon = np.stack([(R_T if i < 5 else isq(csum[i] / (i + 1))) @ Xte_raw[i] for i in range(len(Xte_raw))]).astype(np.float32)
    acc_on, P_on = evaluate(model, torch.tensor(Xon, device=dev), yE)
    online = {"test_acc_online_ea": acc_on, "kappa_online_ea": cohen_kappa_score(yte, P_on.argmax(1))}
res = {"model": a.model, "subject": a.subject, "seed": a.seed, "tag": a.tag, "ea": a.ea, "cov": a.cov, "stage": a.stage, "epochs": a.epochs,
       "test_acc": acc, "kappa": cohen_kappa_score(yte, P.argmax(1)), "train_acc_last": hist[-1]["train_acc"],
       "params": sum(p.numel() for p in model.parameters()), "minutes": (time.time() - t0) / 60, **online}
np.save(f"{out_dir}/S{a.subject}_seed{a.seed}_P.npy", P)
if a.ea: np.save(f"{out_dir}/S{a.subject}_seed{a.seed}_Pon.npy", P_on)  # real-time (online EA) probabilities
os.makedirs(f"{CK}/ft", exist_ok=True); torch.save(model.state_dict(), f"{CK}/ft/{a.tag}_S{a.subject}_seed{a.seed}.pt")
json.dump({"res": res, "hist": hist}, open(res_file, "w"))
print(json.dumps(res))
