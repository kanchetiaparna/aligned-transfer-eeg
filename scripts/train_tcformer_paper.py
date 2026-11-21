"""TCFormer reproduced as in the official code (github.com/Altaheri/TCFormer, train_pipeline.py + configs/tcformer.yaml).

Model : scripts/tcformer.py (port of models/tcformer.py), config values F1=32, kernels (20,32,64), d_group 16, D 2,
        pools 8/7, conv dropout 0.4, group attention on, GQA 4 query / 2 kv heads, transformer dropout 0.4,
        drop-path 0.25, TCN depth 2 kernel 4 dropout 0.3. Transformer depth 2 = the 77.8k-param model whose
        BCI IV-2a results are reported in the README / paper (depth 5 is only used for HGD LOSO).
Data  : data/tcf/A0x.npz (prep_2a_tcformer.py): 0..4 s after the cue, 1000 samples, no filtering.
Scale : z_scale "Method 2": per channel, mean/std over all training trials and time points (fit on session T).
Train : whole session T, 1000 epochs, batch 48 (shuffled), Adam lr 9e-4 betas (0.5, 0.999) weight decay 1e-3,
        LambdaLR linear warm-up 20 epochs then cosine decay, stepped once per epoch (Lightning default),
        plain cross-entropy. interaug (optional): per batch, per class, new trials stitched from random
        same-class trials chunk by chunk (8 chunks for 1000 samples), appended to the batch, then shuffled.
Test  : last-epoch weights on session E (no early stopping, no model selection). Seeds 0..4.
Speed-only deviation: cudnn.benchmark=True instead of deterministic kernels.
"""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import argparse, json, os, random, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")
from tcformer import TCFormer

p = argparse.ArgumentParser()
p.add_argument("--subject", type=int, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--aug", type=int, default=1, help="interaug on/off (paper reports both)")
p.add_argument("--depth", type=int, default=2, help="transformer depth (2 = 77.8k params, paper's 2a model)")
p.add_argument("--epochs", type=int, default=1000)
p.add_argument("--data", default=f"{ROOT}/data/tcf")
p.add_argument("--tag", default="")
p.add_argument("--out", default=f"{ROOT}/results/tcformer_paper")
a = p.parse_args()
name = a.tag or f"2a_aug{a.aug}_d{a.depth}"
out_dir = f"{a.out}/{name}"; os.makedirs(out_dir, exist_ok=True)
res_file = f"{out_dir}/S{a.subject}_seed{a.seed}.json"
if os.path.exists(res_file):
    print("skip, exists", res_file); raise SystemExit

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
dev = "cuda"; torch.backends.cudnn.benchmark = True
for _ in range(240):  # shared GPUs: wait (up to 2 h) until ~6 GB is free before starting, instead of crashing
    if torch.cuda.mem_get_info()[0] > 6 * 2**30: break
    time.sleep(30)
torch.set_float32_matmul_precision("medium")  # as in utils/seed.py

d = np.load(f"{a.data}/A{a.subject:02d}.npz")
Xtr, ytr, Xte, yte = d["Xtr"].astype(np.float64), d["ytr"], d["Xte"].astype(np.float64), d["yte"]
T = Xtr.shape[-1]
if T % 8 and T % 7:  # interaug splits into 8 (or 7) equal chunks; crop to a multiple of 8 if needed (HaLT: 375 -> 368)
    T = T - T % 8; Xtr, Xte = Xtr[..., :T], Xte[..., :T]
s, c, t = Xtr.shape  # z_scale Method 2
sc = StandardScaler().fit(Xtr.transpose(1, 0, 2).reshape(c, -1).T)
zs = lambda X: sc.transform(X.transpose(1, 0, 2).reshape(c, -1).T).T.reshape(c, X.shape[0], t).transpose(1, 0, 2)
Xtr, Xte = zs(Xtr).astype(np.float32), zs(Xte).astype(np.float32)
n_cls = int(max(ytr.max(), yte.max()) + 1)
Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev, dtype=torch.long)
XE, yE = torch.tensor(Xte, device=dev), torch.tensor(yte, device=dev, dtype=torch.long)
N_CHUNKS = 9 if t == 1125 else (8 if t % 8 == 0 else 7)


def interaug(x, y):
    """utils/interaug.py, vectorised: same sampling (per class, chunk k of new trial i from a random same-class trial)."""
    newx, newy = torch.empty_like(x), torch.empty_like(y); cur = 0; L = t // N_CHUNKS
    for cls in torch.unique(y):
        xc = x[y == cls]; n = len(xc)
        idx = torch.randint(0, n, (n, N_CHUNKS), device=x.device)
        parts = [xc[idx[:, k], :, k * L:(k + 1) * L] for k in range(N_CHUNKS)]
        newx[cur:cur + n] = torch.cat(parts, -1); newy[cur:cur + n] = cls; cur += n
    X2, Y2 = torch.cat((x, newx)), torch.cat((y, newy)); perm = torch.randperm(len(X2), device=x.device)
    return X2[perm], Y2[perm]


def evaluate(X, y, bs=48):
    model.eval(); C = torch.zeros((), device=dev); P = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            o = model(X[i:i + bs]); C += (o.argmax(1) == y[i:i + bs]).sum(); P.append(F.softmax(o, -1))
    return C.item() / len(X), torch.cat(P).cpu().numpy()


def train_once():
    global model
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    model = TCFormer(n_channels=c, n_classes=n_cls, trans_depth=a.depth).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=9e-4, betas=(0.5, 0.999), weight_decay=1e-3)
    WARM = 20
    lr_fn = lambda e: e / max(1, WARM) if e < WARM else 0.5 * (1 + np.cos(np.pi * (e - WARM) / max(1, a.epochs - WARM)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)


    BS = 48; hist = []; t0 = time.time()
    for ep in range(a.epochs):
        model.train(); perm = torch.randperm(len(yt), device=dev); tc = torch.zeros((), device=dev); tn = 0
        for i in range(0, len(yt), BS):
            xb, yb = Xt[perm[i:i + BS]], yt[perm[i:i + BS]]
            if a.aug: xb, yb = interaug(xb, yb)
            o = model(xb); loss = F.cross_entropy(o, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tc += (o.argmax(1) == yb).sum(); tn += len(yb)
        sched.step()
        rec = {"ep": ep, "train_acc": tc.item() / tn, "lr": opt.param_groups[0]["lr"]}
        if ep % 50 == 0 or ep == a.epochs - 1:
            rec["test_acc_monitor"] = evaluate(XE, yE)[0]   # logged for curves only, never used for selection
        hist.append(rec)
    return hist, t0


for attempt in range(30):  # shared GPUs: on CUDA OOM free memory, wait a random 1-3 min, restart from scratch (same seed)
    try:
        hist, t0 = train_once(); break
    except torch.OutOfMemoryError:
        if attempt == 29: raise
        model = None; torch.cuda.empty_cache()
        print(json.dumps({"oom_restart": attempt + 1}), flush=True); time.sleep(60 + 120 * random.SystemRandom().random())  # unseeded, so restarts do not collide

acc, P = evaluate(XE, yE)
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(yte, P.argmax(1))
res = {"subject": a.subject, "seed": a.seed, "aug": a.aug, "depth": a.depth, "epochs": a.epochs, "n_times": t,
       "test_acc": acc, "kappa": kappa, "train_acc_last": hist[-1]["train_acc"],
       "params": sum(p.numel() for p in model.parameters()), "minutes": (time.time() - t0) / 60}
np.save(f"{out_dir}/S{a.subject}_seed{a.seed}_P.npy", P)
json.dump({"res": res, "hist": hist}, open(res_file, "w"))
print(json.dumps(res))
