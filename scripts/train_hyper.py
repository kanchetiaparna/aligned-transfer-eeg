"""Idea A: TCFormer with a covariance-conditioned hypernetwork ("HyperTCF").

A small network reads an unlabelled description of the whole recording session (log of the mean band-wise
covariance of the RAW, unaligned trials) and personalises TCFormer to that session:
  1. a low-rank channel re-mixing of the (EA-aligned) input:  x <- x + A(s) B(s)^T x   (rank 4)
  2. FiLM (scale and shift) on the 192 outputs of TCFormer's depthwise spatial convolution.
The last hypernetwork layer starts at zero, so at initialisation the model is exactly TCFormer-AT.
EA already removes the session mean covariance from the input; the descriptor is computed BEFORE EA, so it
carries the information that EA throws away (per band, per session).

Stages (same protocol as train_new.py: train on session T, last epoch, test once on session E):
  pre : continue from the existing TCFormer-AT pretrained checkpoint (other people, both sessions, each aligned
        on its own) for --epochs epochs. Batches come from one session at a time so each batch has one
        descriptor; descriptors are bootstrapped (random half of the session's trials) as augmentation.
  ft  : fine-tune on the target's session T with the hypernetwork frozen (it only ever sees one session here).
  --ctrl 1 : identical model, batching and training, but the descriptor is always zero = no session information.
Evaluation also reports, from the same weights: descriptor of session T (no day adaptation), descriptor of another
person (wrong information) and the real-time version (running EA + running descriptor).
"""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import argparse, glob, json, os, random, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import cohen_kappa_score
import warnings; warnings.filterwarnings("ignore")
from tcformer import TCFormer, _rope_cache

p = argparse.ArgumentParser()
p.add_argument("--subject", type=int, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--pseed", type=int, default=0)
p.add_argument("--stage", choices=["pre", "ft"], required=True)
p.add_argument("--ctrl", type=int, default=0)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--data", default=f"{ROOT}/data/tcf")
p.add_argument("--tag", required=True)
p.add_argument("--out", default=f"{ROOT}/results/newmodel")
a = p.parse_args()
CK = f"{ROOT}/checkpoints/newmodel"
DS = os.path.basename(a.data.rstrip("/"))
base_ckpt = f"{CK}/pre_ea1_cov0_S{a.subject}.pt" if DS == "tcf" else f"{CK}/pre_{DS}_ea1_cov0_S{a.subject}.pt"
if a.pseed: base_ckpt = base_ckpt.replace(".pt", f"_p{a.pseed}.pt")
ckpt = base_ckpt.replace("/pre_", f"/prehyper{'ctrl' if a.ctrl else ''}_")
if a.stage == "pre": a.seed = a.pseed
ALL_SUBJ = sorted(int(os.path.basename(f)[1:3]) for f in glob.glob(f"{a.data}/A*.npz"))
out_dir = f"{a.out}/{a.tag}"; os.makedirs(out_dir, exist_ok=True)
res_file = f"{out_dir}/S{a.subject}_seed{a.seed}.json"
if (a.stage == "pre" and os.path.exists(ckpt)) or (a.stage == "ft" and os.path.exists(res_file)):
    print("skip, exists"); raise SystemExit
dev = "cuda"; torch.backends.cudnn.benchmark = True; torch.set_float32_matmul_precision("medium")
for _ in range(240):
    if torch.cuda.mem_get_info()[0] > 6 * 2**30: break
    time.sleep(30 + 30 * random.SystemRandom().random())

# ---------------------------------------------------------------- data and session descriptors
BANDS = [(4, 8), (8, 13), (13, 20), (20, 30), (30, 40)]
SOS = [butter(4, b, btype="band", fs=250, output="sos") for b in BANDS]
iu = np.triu_indices(0)


def isq(C):
    w, V = np.linalg.eigh(C); return V @ np.diag(w ** -0.5) @ V.T


def logm_vec(C):
    w, V = np.linalg.eigh(C); L = V @ np.diag(np.log(np.maximum(w, 1e-12))) @ V.T
    i, j = np.triu_indices(C.shape[0]); return L[i, j] * np.where(i == j, 1.0, np.sqrt(2))


def band_covs(X):  # X raw: n, C, T  ->  n, K, C, C
    X = X.astype(np.float64)
    return np.stack([np.einsum("nct,ndt->ncd", Z, Z) / Z.shape[-1] for Z in (sosfiltfilt(s, X, axis=-1) for s in SOS)], 1)


def desc(covs_subset):  # mean band covariances of some trials -> one descriptor vector
    M = covs_subset.mean(0); return np.concatenate([logm_vec(M[k]) for k in range(M.shape[0])]).astype(np.float32)


def boot(covs, n=24, rng=None):
    rng = rng or np.random.default_rng(0); m = len(covs) // 2
    return np.stack([desc(covs[rng.choice(len(covs), m, replace=False)]) for _ in range(n)])


def ea(X):
    R = isq(np.mean([x @ x.T / x.shape[1] for x in X.astype(np.float64)], 0))
    return np.einsum("ij,njt->nit", R, X).astype(np.float32)


def load(s):
    d = np.load(f"{a.data}/A{s:02d}.npz"); Xtr, Xte = d["Xtr"].astype(np.float32), d["Xte"].astype(np.float32)
    if Xtr.shape[-1] % 8 and Xtr.shape[-1] % 7:
        cut = Xtr.shape[-1] - Xtr.shape[-1] % 8; Xtr, Xte = Xtr[..., :cut], Xte[..., :cut]
    return Xtr, d["ytr"], Xte, d["yte"]


rng = np.random.default_rng(1000 + a.seed)
if a.stage == "pre":
    sessions = []  # (X aligned, y, bootstrap descriptors)
    for s in ALL_SUBJ:
        if s == a.subject: continue
        Xtr, ytr, Xte, yte = load(s)
        for X, y in ((Xtr, ytr), (Xte, yte)):
            sessions.append((ea(X), y, boot(band_covs(X), rng=rng)))
    allD = np.concatenate([S[2] for S in sessions]); mu, sd = allD.mean(0), allD.std(0) + 1e-6
else:
    Xtr, ytr, Xte, yte = load(a.subject)
    covT, covE = band_covs(Xtr), band_covs(Xte)
    sessions = [(ea(Xtr), ytr, boot(covT, rng=rng))]
C_, T_ = sessions[0][0].shape[1], sessions[0][0].shape[2]
N_CHUNKS = 8 if T_ % 8 == 0 else 7
D_IN = len(BANDS) * C_ * (C_ + 1) // 2


def interaug(x, y):
    newx, newy = torch.empty_like(x), torch.empty_like(y); cur = 0; L = T_ // N_CHUNKS
    for cls in torch.unique(y):
        xc = x[y == cls]; n = len(xc); idx = torch.randint(0, n, (n, N_CHUNKS), device=x.device)
        newx[cur:cur + n] = torch.cat([xc[idx[:, k], :, k * L:(k + 1) * L] for k in range(N_CHUNKS)], -1)
        newy[cur:cur + n] = cls; cur += n
    X2, Y2 = torch.cat((x, newx)), torch.cat((y, newy)); perm = torch.randperm(len(X2), device=x.device)
    return X2[perm], Y2[perm]


# ---------------------------------------------------------------- model
class HyperTCF(nn.Module):
    def __init__(self, C, n_cls=4, rank=4, hidden=32):
        super().__init__()
        self.tcf = TCFormer(n_channels=C, n_classes=n_cls, trans_depth=2)
        self.C, self.r = C, rank
        self.F2 = self.tcf.conv_block.channel_DW_conv[0].out_channels
        self.register_buffer("mu", torch.zeros(D_IN)); self.register_buffer("sd", torch.ones(D_IN))
        self.hyper = nn.Sequential(nn.Dropout(0.3), nn.Linear(D_IN, hidden), nn.GELU(),
                                   nn.Linear(hidden, 2 * C * rank + 2 * self.F2))
        nn.init.zeros_(self.hyper[-1].weight); nn.init.zeros_(self.hyper[-1].bias)

    def forward(self, x, d):  # x: B,C,T (aligned); d: B,D_IN raw descriptor (or zeros = no information)
        h = self.hyper((d - self.mu) / self.sd)
        A = h[:, :self.C * self.r].view(-1, self.C, self.r)
        Bm = h[:, self.C * self.r:2 * self.C * self.r].view(-1, self.C, self.r)
        g, b = h[:, 2 * self.C * self.r:].split(self.F2, 1)
        x = x + 0.1 * A @ (Bm.transpose(1, 2) @ x)
        cb, m = self.tcf.conv_block, self.tcf
        z = x.unsqueeze(1); z = torch.cat([c(z) for c in cb.temporal_convs], 1)
        z = cb.channel_DW_conv(z) * (1 + g[:, :, None, None]) + b[:, :, None, None]
        z = cb.temporal_conv_2(cb.reduce(cb.drop1(cb.pool1(z))))
        if cb.group_attn is not None: z = z + cb.group_attn(z)
        conv = cb.drop2(cb.pool2(z)).squeeze(2)
        tok = m.mix(conv).transpose(1, 2); cos, sin = _rope_cache(m.head_dim, tok.shape[1], tok.device)
        for blk in m.transformer: tok = blk(tok, cos, sin)
        return m.tcn_head(torch.cat((conv, m.reduce(tok.transpose(1, 2))), 1))


def train_once():
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    model = HyperTCF(C_).to(dev)
    if a.stage == "pre":
        sd_ = torch.load(base_ckpt, map_location=dev)
        model.load_state_dict({k: v for k, v in sd_.items() if k.startswith("tcf.")}, strict=False)
        model.mu.copy_(torch.tensor(mu)); model.sd.copy_(torch.tensor(sd))
    else:
        for _ in range(180):
            if os.path.exists(ckpt): break
            time.sleep(60)
        time.sleep(3); model.load_state_dict(torch.load(ckpt, map_location=dev))
        for q in model.hyper.parameters(): q.requires_grad_(False)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.5, 0.999), weight_decay=1e-3)
    W, E = a.warmup, a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e + 1) / max(1, W) if e < W else 0.5 * (1 + np.cos(np.pi * (e - W) / max(1, E - W))))
    S = [(torch.tensor(X, device=dev), torch.tensor(y, device=dev, dtype=torch.long), torch.tensor(Db, device=dev)) for X, y, Db in sessions]
    hist, t0, BS = [], time.time(), 48
    for ep in range(E):
        model.train(); batches = []
        for si, (X, y, _) in enumerate(S):
            perm = torch.randperm(len(y), device=dev)
            batches += [(si, perm[i:i + BS]) for i in range(0, len(y), BS)]
        random.shuffle(batches); tc = tn = 0
        for si, idx in batches:
            X, y, Db = S[si]; xb, yb = interaug(X[idx], y[idx])
            d = Db[random.randrange(len(Db))].expand(len(xb), -1)
            if a.ctrl: d = model.mu.expand(len(xb), -1)          # normalised descriptor = 0: no session info
            o = model(xb, d); loss = F.cross_entropy(o, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params, 5.0); opt.step()
            tc += (o.argmax(1) == yb).sum().item(); tn += len(yb)
        sched.step(); hist.append({"ep": ep, "train_acc": tc / tn})
    return model, hist, t0


for attempt in range(30):
    try:
        model, hist, t0 = train_once(); break
    except torch.OutOfMemoryError:
        if attempt == 29: raise
        torch.cuda.empty_cache(); time.sleep(60 + 120 * random.SystemRandom().random())

if a.stage == "pre":
    torch.save(model.state_dict(), ckpt + ".tmp"); os.replace(ckpt + ".tmp", ckpt)
    print(json.dumps({"pretrain_done": a.subject, "ctrl": a.ctrl, "train_acc": hist[-1]["train_acc"], "min": (time.time() - t0) / 60}))
    raise SystemExit


@torch.no_grad()
def predict(X, D):  # X: n,C,T aligned; D: n,D_IN descriptors (one per trial)
    model.eval(); X, D = torch.tensor(X, device=dev), torch.tensor(D, device=dev)
    return torch.cat([F.softmax(model(X[i:i + 48], D[i:i + 48]), -1) for i in range(0, len(X), 48)]).cpu().numpy()


n = len(yte); rep = lambda v: np.repeat(v[None], n, 0)
dE, dT = desc(covE), desc(covT)
other = ALL_SUBJ[(ALL_SUBJ.index(a.subject) + 1) % len(ALL_SUBJ)]
dO = desc(band_covs(load(other)[2]))
XE = ea(Xte); zero = model.mu.cpu().numpy()
variants = {"P": dE, "P_descT": dT, "P_descOther": dO, "P_descZero": zero}
if a.ctrl: variants = {"P": zero}
acc = {}
for k, dv in variants.items():
    P = predict(XE, rep(dv)); np.save(f"{out_dir}/S{a.subject}_seed{a.seed}_{k}.npy", P); acc[k] = float((P.argmax(1) == yte).mean())
# real time: running EA (first 5 trials use session T's reference) and running descriptor (first 5 use session T's)
covs = np.stack([x @ x.T / x.shape[1] for x in Xte.astype(np.float64)]); csum = np.cumsum(covs, 0)
R_T = isq(np.mean([x @ x.T / x.shape[1] for x in Xtr.astype(np.float64)], 0))
Xon = np.stack([(R_T if i < 5 else isq(csum[i] / (i + 1))) @ Xte[i] for i in range(n)]).astype(np.float32)
bsum = np.cumsum(covE, 0)
Don = zero[None].repeat(n, 0) if a.ctrl else np.stack([dT if i < 5 else np.concatenate([logm_vec(bsum[i, k] / (i + 1)) for k in range(len(BANDS))]).astype(np.float32) for i in range(n)])
Pon = predict(Xon, Don); np.save(f"{out_dir}/S{a.subject}_seed{a.seed}_Pon.npy", Pon)
res = {"subject": a.subject, "seed": a.seed, "tag": a.tag, "ctrl": a.ctrl, "test_acc": acc["P"], "acc_variants": acc,
       "test_acc_online": float((Pon.argmax(1) == yte).mean()), "kappa": cohen_kappa_score(yte, np.load(f"{out_dir}/S{a.subject}_seed{a.seed}_P.npy").argmax(1)),
       "train_acc_last": hist[-1]["train_acc"], "params": sum(q.numel() for q in model.parameters()), "minutes": (time.time() - t0) / 60}
json.dump({"res": res, "hist": hist}, open(res_file, "w")); print(json.dumps(res))
