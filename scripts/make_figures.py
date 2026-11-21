"""Make the report figures from the saved results (no training).
    .venv/bin/python scripts/make_figures.py      -> figures/*.png
Note: 'test accuracy' curves use the test session only for plotting; no model was selected with it
(every reported number is the last epoch)."""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = f"{ROOT}/results"; OUT = f"{ROOT}/figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY, GREEN = "#2a6fdb", "#e8833a", "#9aa0a6", "#2e9e5b"


def runs(d):
    return [json.load(open(f)) for f in glob.glob(f"{R}/{d}/S*_seed*.json")]


def curves(d):
    """train accuracy every epoch; test accuracy only at the epochs logged in every run (every 50 epochs)."""
    H = [r["hist"] for r in runs(d)]
    n = min(len(h) for h in H)
    tr = np.array([[e["train_acc"] for e in h[:n]] for h in H]).mean(0) * 100
    eps = sorted(set.intersection(*[{e["ep"] for e in h if "test_acc_monitor" in e} for h in H]) | {n - 1})
    te = [np.mean([next(e["test_acc_monitor"] for e in h if e["ep"] == ep) if ep < n - 1 or "test_acc_monitor" in h[n-1]
                   else r["res"]["test_acc"] for h, r in zip(H, runs(d))]) * 100 for ep in eps]
    return tr, np.array(eps), np.array(te)


def per_subject(d, key="test_acc"):
    s = {}
    for r in runs(d):
        s.setdefault(r["res"]["subject"], []).append(r["res"][key] * 100)
    return {k: np.mean(v) for k, v in sorted(s.items())}


# 1. learning curves: plain TCFormer vs TCFormer-AT
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, (name, plain, at) in zip(ax, [("BCI IV 2a", "tcformer_paper/2a_aug1_d2", "newmodel/V4_rt"),
                                     ("HaLT", "tcformer_paper/halt_aug1_d2", "newmodel/H4_rt")]):
    for d, lab, c in [(plain, "TCFormer (paper recipe, 1000 epochs)", GREY), (at, "TCFormer-AT (fine-tune, 300 epochs)", BLUE)]:
        tr, ep, te = curves(d)
        a.plot(tr, c=c, ls="--", lw=1.2, label=f"{lab}: train")
        a.plot(ep, te, c=c, lw=1.8, marker="o", ms=3, label=f"{lab}: test (day 2)")
        a.annotate(f"gap {tr[-1]-te[-1]:.1f}", (len(tr) - 1, (tr[-1] + te[-1]) / 2), color=c, fontsize=8, ha="right")
    a.set(title=name, xlabel="epoch", ylabel="accuracy (%)", ylim=(20, 101))
    a.grid(alpha=.3)
ax[0].legend(fontsize=7, loc="lower right")
fig.suptitle("Train vs test accuracy per epoch (mean over people and seeds)")
fig.tight_layout(); fig.savefig(f"{OUT}/1_learning_curves.png"); plt.close(fig)

# 2. the recipe on three networks (5-model ensembles, from compare scripts)
rows = {"2a": {"EEGNet": (79.55, 85.49), "ATCNet": (84.88, 89.16), "TCFormer": (86.50, 89.97)},
        "HaLT": {"EEGNet": (59.80, 66.26), "ATCNet": (58.07, 63.81), "TCFormer": (62.43, 68.15)}}
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, (ds, r) in zip(ax, rows.items()):
    x = np.arange(len(r)); w = .38
    b1 = a.bar(x - w / 2, [v[0] for v in r.values()], w, color=GREY, label="plain training")
    b2 = a.bar(x + w / 2, [v[1] for v in r.values()], w, color=BLUE, label="Aligned Transfer recipe")
    a.bar_label(b1, fmt="%.1f", fontsize=8); a.bar_label(b2, fmt="%.1f", fontsize=8)
    for i, v in enumerate(r.values()):
        a.text(i + w / 2, v[1] + 3, f"+{v[1]-v[0]:.1f}", ha="center", color=BLUE, fontsize=8, fontweight="bold")
    a.set_xticks(x, list(r)); a.set(title=ds, ylabel="test accuracy (%)", ylim=(50 if ds == "HaLT" else 70, 100 if ds == "2a" else 75))
ax[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Same recipe helps every network (5-model ensembles)")
fig.tight_layout(); fig.savefig(f"{OUT}/2_recipe_all_networks.png"); plt.close(fig)

# 3. per-person accuracy, plain TCFormer vs TCFormer-AT (single model, mean of 5 seeds)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, (ds, plain, at) in zip(ax, [("BCI IV 2a", "tcformer_paper/2a_aug1_d2", "newmodel/V4_rt"),
                                   ("HaLT", "tcformer_paper/halt_aug1_d2", "newmodel/H4_rt")]):
    p, q = per_subject(plain), per_subject(at)
    q = {k: q[k] for k in p if k in q}; p = {k: p[k] for k in q}
    x = np.arange(len(p)); w = .38
    a.bar(x - w / 2, list(p.values()), w, color=GREY, label=f"TCFormer  (mean {np.mean(list(p.values())):.1f})")
    a.bar(x + w / 2, list(q.values()), w, color=BLUE, label=f"TCFormer-AT  (mean {np.mean(list(q.values())):.1f})")
    a.axhline(25, c="k", lw=.8, ls=":"); a.text(len(p) - .5, 26, "chance", fontsize=7, ha="right")
    a.set_xticks(x, [f"S{k}" for k in p]); a.set(title=ds, ylabel="test accuracy (%)", ylim=(0, 100))
    a.legend(fontsize=8, loc="lower left")
fig.suptitle("Accuracy for each person (single model, mean of 5 seeds)")
fig.tight_layout(); fig.savefig(f"{OUT}/3_per_person.png"); plt.close(fig)

# 4. HyperTCF (idea A) result and descriptor check (5-model ensembles, 2a)
names = ["TCFormer-AT", "Control\n(no session info)", "HyperTCF", "HyperTCF fed\nday 1 summary", "HyperTCF fed\nempty summary", "HyperTCF fed\nother person"]
vals = [89.97, 90.20, 89.62, 90.24, 89.39, 85.80]
cols = [BLUE, GREY, ORANGE, ORANGE, ORANGE, ORANGE]
fig, a = plt.subplots(figsize=(9, 4))
b = a.bar(names, vals, color=cols); a.bar_label(b, fmt="%.2f", fontsize=8)
a.set(ylim=(84, 91.5), ylabel="test accuracy (%)", title="Idea A (HyperTCF) on 2a: no gain; it uses the summary to recognise the person, not the day")
a.title.set_fontsize(10)
fig.tight_layout(); fig.savefig(f"{OUT}/4_hypertcf_check.png"); plt.close(fig)

# 5. overfitting gap (last epoch train minus test)
fig, a = plt.subplots(figsize=(7, 4))
labs, tr_, te_ = [], [], []
for ds, plain, at in [("2a", "tcformer_paper/2a_aug1_d2", "newmodel/V4_rt"), ("HaLT", "tcformer_paper/halt_aug1_d2", "newmodel/H4_rt")]:
    for d, n in [(plain, "TCFormer"), (at, "TCFormer-AT")]:
        rr = runs(d)
        labs.append(f"{n}\n{ds}"); tr_.append(np.mean([r["res"]["train_acc_last"] for r in rr]) * 100)
        te_.append(np.mean([r["res"]["test_acc"] for r in rr]) * 100)
x = np.arange(len(labs)); w = .38
b1 = a.bar(x - w / 2, tr_, w, color=GREY, label="train (day 1)"); b2 = a.bar(x + w / 2, te_, w, color=BLUE, label="test (day 2)")
a.bar_label(b1, fmt="%.1f", fontsize=8); a.bar_label(b2, fmt="%.1f", fontsize=8)
for i in range(len(labs)):
    a.text(i, 108, f"gap {tr_[i]-te_[i]:.1f}", ha="center", fontsize=8, color="crimson")
a.set_xticks(x, labs); a.set(ylim=(0, 115), ylabel="accuracy (%)", title="Overfitting gap at the last epoch")
a.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/5_overfitting_gap.png"); plt.close(fig)
print("saved to", OUT, sorted(os.listdir(OUT)))
