"""ATCNet reproduced as in the paper's official code (github.com/Altaheri/EEG-ATCNet), PyTorch version.

Model: braindecode ATCNet (same defaults as the paper's ATCNet_: F1=16, D=2, kernel 64, pools 8/7,
5 sliding windows, MHA 2 heads x 8, TCN depth 2 kernel 4, dropout 0.3, average fusion, max-norm 0.25).
Data:  data/paper/A0x.npz (prep_2a_paper.py): -0.5..4 s around cue, 1125 samples, no filtering.
Scaling: per channel sklearn StandardScaler fitted on (trials x time) of the training session,
         i.e. every time point of every channel standardized (exactly preprocess.standardize_data).
Training (both protocols): Adam lr 1e-3 (Keras eps 1e-7), batch 64, shuffle every epoch,
         categorical cross-entropy, ReduceLROnPlateau(val_loss, factor 0.9, patience 20, min_lr 1e-4).

--protocol paper  = main_TrainTest.py (how the published 85.4% was obtained):
         train on all of session T, "validation" = session E (the test set), 1000 epochs,
         keep the weights with the best validation (= test) accuracy. Optimistic: selection uses test data.
--protocol honest = main_TrainValTest.py: 80/20 split of session T (train_test_split, random_state=42),
         500 epochs, keep weights with lowest validation loss, then evaluate once on session E.
Both report also the accuracy of the last-epoch weights.
"""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import argparse, json, os, copy, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, torch, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings; warnings.filterwarnings("ignore")

p = argparse.ArgumentParser()
p.add_argument("--subject", type=int, required=True)
p.add_argument("--run", type=int, default=0, help="repetition index (paper: n_train=10)")
p.add_argument("--protocol", choices=["paper", "honest"], required=True)
p.add_argument("--data", default=f"{ROOT}/data/paper")
p.add_argument("--keras", type=int, default=0, help="1 = match the Keras original (BN momentum/eps, Glorot init, Keras max-norm axes)")
p.add_argument("--tag", default="")
p.add_argument("--out", default=f"{ROOT}/results/atcnet_paper")
a = p.parse_args()
name = a.tag or a.protocol
out_dir = f"{a.out}/{name}"; os.makedirs(out_dir, exist_ok=True)
res_file = f"{out_dir}/S{a.subject}_run{a.run}.json"
if os.path.exists(res_file):
    print("skip, exists", res_file); raise SystemExit

seed = 1000 * a.subject + a.run
torch.manual_seed(seed); np.random.seed(seed)
dev = "cuda"
torch.backends.cudnn.benchmark = True  # pick fastest conv algorithms (same math)
EPOCHS = 1000 if a.protocol == "paper" else 500
BS, LR = 64, 1e-3

d = np.load(f"{a.data}/A{a.subject:02d}.npz")
Xtr, ytr, Xte, yte = d["Xtr"].copy(), d["ytr"], d["Xte"].copy(), d["yte"]
n_ch, n_t = Xtr.shape[1], Xtr.shape[2]
n_cls = int(max(ytr.max(), yte.max()) + 1)
for j in range(n_ch):  # preprocess.standardize_data
    sc = StandardScaler().fit(Xtr[:, j, :])
    Xtr[:, j, :] = sc.transform(Xtr[:, j, :]); Xte[:, j, :] = sc.transform(Xte[:, j, :])

if a.protocol == "paper":
    Xt, yt, Xv, yv = Xtr, ytr, Xte, yte
else:  # main_TrainValTest.py: split after standardization, fixed random_state=42
    Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=42)

T = lambda z, dt=torch.float32: torch.as_tensor(z, dtype=dt, device=dev)
Xt, yt, Xv, yv, XE, yE = T(Xt), T(yt, torch.long), T(Xv), T(yv, torch.long), T(Xte), T(yte, torch.long)

import braindecode.models as M
for attempt in range(40):  # shared GPUs: wait for memory instead of failing
    try:
        model = M.ATCNet(n_chans=n_ch, n_outputs=n_cls, n_times=n_t, sfreq=250).to(dev)
        break
    except Exception as e:
        if "out of memory" not in str(e): raise
        torch.cuda.empty_cache(); time.sleep(90)
import torch.nn as nn
def kerasify(m):
    """Match Keras defaults of the official ATCNet_ code without changing the architecture."""
    for mod in m.modules():                              # Keras BatchNormalization: momentum 0.99, epsilon 1e-3
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d)):
            mod.momentum, mod.eps = 0.01, 1e-3
    for n, mod in m.named_modules():                     # Keras default kernel init = glorot_uniform
        if n.startswith("conv_block.conv") or ".mha.fc_" in n:
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None: nn.init.zeros_(mod.bias)
    for i in range(len(m.final_layer)):                  # plain Dense; max_norm(0.25) applied after each step
        lin = nn.Linear(m.final_layer[i].in_features, m.final_layer[i].out_features).to(dev)
        nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias); m.final_layer[i] = lin
    return m
def keras_constraints(m):
    """Keras applies kernel constraints after every optimizer update."""
    with torch.no_grad():
        # DepthwiseConv2D((1, C)) depthwise_constraint=max_norm(1.), axis=0 over a size-1 axis -> |w| <= 1
        m.conv_block.conv2.weight.clamp_(-1.0, 1.0)
        for lin in m.final_layer:                        # Dense max_norm(0.25), axis=0 of (in, out) = per output unit
            n_ = lin.weight.norm(2, dim=1, keepdim=True)
            lin.weight.mul_(n_.clamp(max=0.25) / (n_ + 1e-7))
if a.keras:
    model = kerasify(model)
opt = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-7)


def evaluate(X, y, bs=64, probs=False):
    # totals stay on the GPU; a single sync at the end (no per-batch .item() stalls)
    model.eval(); L = torch.zeros((), device=dev); C = torch.zeros((), device=dev); P = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            o = model(X[i:i + bs]); L += F.cross_entropy(o, y[i:i + bs], reduction="sum")
            C += (o.argmax(1) == y[i:i + bs]).sum()
            if probs: P.append(F.softmax(o, -1))
    return L.item() / len(X), C.item() / len(X), (torch.cat(P).cpu().numpy() if probs else None)


# Keras ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=1e-4), min_delta=1e-4 absolute
rl_best, rl_wait = np.inf, 0
monitor_best = -np.inf if a.protocol == "paper" else np.inf   # ModelCheckpoint: val_accuracy max | val_loss min
best_state, best_ep, hist = None, -1, []
t0 = time.time()
for ep in range(EPOCHS):
    model.train(); perm = torch.randperm(len(yt), device=dev)
    tl = torch.zeros((), device=dev); tc = torch.zeros((), device=dev)
    for i in range(0, len(yt), BS):
        b = perm[i:i + BS]; o = model(Xt[b]); loss = F.cross_entropy(o, yt[b])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if a.keras: keras_constraints(model)
        tl += loss.detach() * len(b); tc += (o.argmax(1) == yt[b]).sum()
    tl, tc = tl.item(), tc.item()
    vl, va, _ = evaluate(Xv, yv)
    m = va if a.protocol == "paper" else vl
    if (a.protocol == "paper" and m > monitor_best) or (a.protocol == "honest" and m < monitor_best):
        monitor_best, best_state, best_ep = m, copy.deepcopy(model.state_dict()), ep
    if vl < rl_best - 1e-4:
        rl_best, rl_wait = vl, 0
    else:
        rl_wait += 1
        if rl_wait >= 20:
            for g in opt.param_groups:
                g["lr"] = max(g["lr"] * 0.9, 1e-4)
            rl_wait = 0
    hist.append({"ep": ep, "train_loss": tl / len(yt), "train_acc": tc / len(yt), "val_loss": vl, "val_acc": va,
                 "lr": opt.param_groups[0]["lr"]})

_, test_last, P_last = evaluate(XE, yE, probs=True)
model.load_state_dict(best_state)
_, test_best, P_best = evaluate(XE, yE, probs=True)
res = {"subject": a.subject, "run": a.run, "protocol": a.protocol, "keras": a.keras, "epochs": EPOCHS, "best_ep": best_ep,
       "train_acc_last": hist[-1]["train_acc"], "test_acc_selected": test_best, "test_acc_last": test_last,
       "kappa_selected": (test_best - 0.25) / 0.75, "minutes": (time.time() - t0) / 60,
       "params": sum(p.numel() for p in model.parameters())}
np.save(f"{out_dir}/S{a.subject}_run{a.run}_P.npy", P_best)
json.dump({"res": res, "hist": hist}, open(res_file, "w"))
print(json.dumps(res))
