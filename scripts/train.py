"""Train one (model, subject, seed, recipe) job on BCI IV 2a.

Protocol (no test-set peeking):
  * train on session T only, 20% of T held out as validation (stratified)
  * test on session E exactly once per checkpoint; reported numbers:
      test_last  : weights after the fixed training budget
      test_bestval: weights at the epoch with best validation accuracy (selection uses val only)
  * --full: train on 100% of T for the fixed budget (final protocol, no validation)

Modes:
  within   : subject-specific training from scratch
  pretrain : train on the 8 other subjects (both sessions) -> saves checkpoint
  finetune : load the leave-target-out pretrained checkpoint, fine-tune on target T
"""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import argparse, json, os, time, copy
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation on shared GPUs
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import (load_subject, preprocess, stratified_split, segment_recombine, gpu_augment,
                    build_model, apply_ea, ea_matrix)

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--subject", type=int, required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--mode", default="within", choices=["within", "pretrain", "finetune"])
p.add_argument("--ea", type=int, default=1)
p.add_argument("--sr", type=int, default=1, help="segment&recombine augmentation")
p.add_argument("--jitter", type=int, default=1)
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=1e-2)
p.add_argument("--ls", type=float, default=0.1, help="label smoothing")
p.add_argument("--bs", type=int, default=64)
p.add_argument("--opt", default="adamw", choices=["adamw", "adam"])
p.add_argument("--beta1", type=float, default=0.9)
p.add_argument("--full", type=int, default=0)
p.add_argument("--tag", default="")
p.add_argument("--ext", default="", help="comma list of external datasets for pretraining, e.g. PhysionetMI,Weibo2014")
p.add_argument("--pre_seed", type=int, default=-1, help="seed of the pretrained checkpoint to load (default: --seed)")
p.add_argument("--out", default=f"{ROOT}/results")
a = p.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda"
torch.backends.cudnn.benchmark = True
EXT = "_ext" if a.ext else ""
name = a.tag or f"{a.model}_{a.mode}{EXT}_ea{a.ea}_sr{a.sr}"
ckpt = lambda seed: f"{ROOT}/checkpoints/{a.model}_pre{EXT}_ea{a.ea}_S{a.subject}_seed{seed}.pt"
CLASS_W = None  # optional class weights for the loss (used when pretraining on unbalanced pooled data)
os.makedirs(f"{a.out}/{name}", exist_ok=True)
ck_dir = f"{ROOT}/checkpoints"; os.makedirs(ck_dir, exist_ok=True)
# skip jobs that already finished (lets the runner be restarted safely)
_done = (ckpt(a.seed) if a.mode == "pretrain"
         else f"{a.out}/{name}/S{a.subject}_seed{a.seed}.json")
if os.path.exists(_done):
    print(f"skip, exists: {_done}"); raise SystemExit


def evaluate(model, X, y, bs=48):  # small eval batch keeps peak GPU memory low on the shared GPUs
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(F.softmax(model(X[i:i + bs]), -1))
    P = torch.cat(out)
    return (P.argmax(1) == y).float().mean().item(), P.cpu().numpy()


def fit(model, Xtr, ytr, epochs, lr, Xva=None, yva=None, Xte=None, yte=None, log_every=25):
    Opt = torch.optim.AdamW if a.opt == "adamw" else torch.optim.Adam
    opt = Opt(model.parameters(), lr=lr, weight_decay=a.wd, betas=(a.beta1, 0.999))
    steps_per = max(1, int(np.ceil(len(ytr) * (2 if a.sr else 1) / a.bs)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * steps_per, pct_start=0.1)
    Xtr_np = Xtr.cpu().numpy(); ytr_np = ytr.cpu().numpy()
    best = (-1, None, -1); hist = []
    for ep in range(epochs):
        model.train()
        if a.sr:
            Xa, ya = segment_recombine(Xtr_np, ytr_np, n_new=len(ytr_np))
            X = torch.cat([Xtr, torch.from_numpy(Xa).to(dev)]); Y = torch.cat([ytr, torch.from_numpy(ya).to(dev)])
        else:
            X, Y = Xtr, ytr
        perm = torch.randperm(len(Y), device=dev)
        corr = tot = 0
        for i in range(0, len(Y), a.bs):
            b = perm[i:i + a.bs]
            if len(b) < 2: continue
            xb = gpu_augment(X[b]) if a.jitter else X[b]
            logit = model(xb)
            loss = F.cross_entropy(logit, Y[b], weight=CLASS_W, label_smoothing=a.ls)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            corr += (logit.argmax(1) == Y[b]).sum().item(); tot += len(b)
        rec = {"ep": ep, "train_acc": corr / tot}
        if Xva is not None:
            va, _ = evaluate(model, Xva, yva); rec["val_acc"] = va
            if va >= best[0]: best = (va, copy.deepcopy(model.state_dict()), ep)
        if Xte is not None and (ep % log_every == 0 or ep == epochs - 1):
            rec["test_acc_monitor"] = evaluate(model, Xte, yte)[0]  # logged only, never used for selection
        hist.append(rec)
    return best, hist


t0 = time.time()
if a.mode == "pretrain":
    # pool of all subjects except the target, each session aligned separately
    Xs, ys = [], []
    for s in range(1, 10):
        if s == a.subject: continue
        Xtr, ytr, Xte, yte = load_subject(s)
        for X, y in ((Xtr, ytr), (Xte, yte)):
            X = apply_ea(X, ea_matrix(X)) if a.ea else (X - X.mean((0, 2), keepdims=True)) / X.std((0, 2), keepdims=True)
            Xs.append(X); ys.append(y)
    # external datasets (left hand / right hand / feet only), each subject-session aligned separately
    import glob
    for dsn in [d for d in a.ext.split(",") if d]:
        for f in sorted(glob.glob(f"{ROOT}/data/ext/{dsn}_S*.npz")):
            d = np.load(f, allow_pickle=True)  # own files; session labels are stored as strings
            for ses in np.unique(d["session"]):
                m = d["session"] == ses
                if m.sum() < 10: continue
                Xs.append(apply_ea(d["X"][m], ea_matrix(d["X"][m]))); ys.append(d["y"][m])
    yall = np.concatenate(ys)
    cnt = np.bincount(yall, minlength=4).astype(np.float32)
    CLASS_W = torch.tensor(cnt.sum() / (4 * np.maximum(cnt, 1)), device=dev)  # balance tongue (2a only) vs the rest
    print(json.dumps({"pretrain_pool_trials": int(len(yall)), "per_class": cnt.astype(int).tolist()}), flush=True)
    Xnp, ynp = np.concatenate(Xs), np.concatenate(ys); del Xs, ys
    # The GPUs are shared: if memory is short, free it, wait for other jobs to finish and start over
    # (same settings and seed, so results stay comparable). Up to 40 attempts, 90 s apart.
    for attempt in range(40):
        try:
            torch.manual_seed(a.seed); np.random.seed(a.seed)
            X = torch.from_numpy(Xnp).to(dev); y = torch.from_numpy(ynp).long().to(dev)
            model = build_model(a.model).to(dev)
            _, hist = fit(model, X, y, a.epochs, a.lr)
            break
        except Exception as e:
            if not (isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e)) or attempt == 39:
                raise
            X = y = model = None; torch.cuda.empty_cache()
            print(json.dumps({"oom_wait": attempt + 1}), flush=True); time.sleep(90)
    torch.save(model.state_dict(), ckpt(a.seed))
    print(json.dumps({"pretrain_done": a.subject, "train_acc": hist[-1]["train_acc"], "min": (time.time() - t0) / 60}))
    raise SystemExit

Xtr, ytr, Xte, yte = load_subject(a.subject)
Xtr, Xte = preprocess(Xtr, Xte, bool(a.ea))
if a.full:
    tr_idx, va_idx = np.arange(len(ytr)), None
else:
    tr_idx, va_idx = stratified_split(ytr, 0.2, seed=100 + a.seed)
T = lambda z: torch.from_numpy(z).to(dev)
Xt, yt = T(Xtr[tr_idx]), T(ytr[tr_idx]).long()
Xv, yv = (T(Xtr[va_idx]), T(ytr[va_idx]).long()) if va_idx is not None else (None, None)
XE, yE = T(Xte), T(yte).long()

model = build_model(a.model).to(dev)
if a.mode == "finetune":
    _ck = ckpt(a.pre_seed if a.pre_seed >= 0 else a.seed)
    for _ in range(90):  # pretrain may still be finishing on another worker: wait up to 45 min
        if os.path.exists(_ck): break
        time.sleep(30)
    time.sleep(2)  # let torch.save finish writing
    model.load_state_dict(torch.load(_ck))
best, hist = fit(model, Xt, yt, a.epochs, a.lr, Xv, yv, XE, yE)

test_last, P_last = evaluate(model, XE, yE)
res = {"model": a.model, "subject": a.subject, "seed": a.seed, "mode": a.mode + EXT, "ext": a.ext, "ea": a.ea, "sr": a.sr,
       "epochs": a.epochs, "lr": a.lr, "full": a.full, "train_acc_last": hist[-1]["train_acc"],
       "test_last": test_last, "params": sum(p.numel() for p in model.parameters()),
       "minutes": (time.time() - t0) / 60}
np.save(f"{a.out}/{name}/S{a.subject}_seed{a.seed}_Plast.npy", P_last)
if Xv is not None:
    res["val_last"] = hist[-1]["val_acc"]
    res["val_best"], res["best_ep"] = best[0], best[2]
    model.load_state_dict(best[1])
    res["test_bestval"], P_bv = evaluate(model, XE, yE)
    np.save(f"{a.out}/{name}/S{a.subject}_seed{a.seed}_Pbv.npy", P_bv)
json.dump({"res": res, "hist": hist}, open(f"{a.out}/{name}/S{a.subject}_seed{a.seed}.json", "w"))
print(json.dumps(res))
