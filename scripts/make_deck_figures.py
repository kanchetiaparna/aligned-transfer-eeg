"""Extra figures for the presentation (no training): raw EEG, motor imagery band power,
covariance before/after Euclidean Alignment, trial features before/after EA, failure analysis, ablation.
    .venv/bin/python scripts/make_deck_figures.py   -> figures/deck/*.png"""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.linalg import logm
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

OUT = f"{ROOT}/figures/deck"; os.makedirs(OUT, exist_ok=True)
CH = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
CLS = ["Left hand", "Right hand", "Feet", "Tongue"]
INK, CYAN, ORANGE, GREY = "#14213D", "#0B8F8A", "#E8702A", "#9AA0A6"
plt.rcParams.update({"figure.dpi": 150, "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555", "axes.labelcolor": INK, "xtick.color": "#444", "ytick.color": "#444"})
FS = 250
SOS = butter(4, [8, 30], btype="band", fs=FS, output="sos")


def load(s):
    d = np.load(f"{ROOT}/data/tcf/A{s:02d}.npz")
    return d["Xtr"].astype(np.float64), d["ytr"].astype(int), d["Xte"].astype(np.float64), d["yte"].astype(int)


def covs(X):
    return np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]


def isqrt(C):
    w, V = np.linalg.eigh(C); return V @ np.diag(w ** -0.5) @ V.T


def ea(X):
    return np.einsum("ij,njt->nit", isqrt(covs(X).mean(0)), X)


def logvec(C):
    iu = np.triu_indices(C.shape[-1])
    w = np.sqrt(2) * np.ones(len(iu[0])); w[iu[0] == iu[1]] = 1
    return np.array([np.real(logm(c))[iu] * w for c in C])


# 1. raw EEG, one trial, 6 channels
Xtr, ytr, Xte, yte = load(1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
t = np.arange(Xtr.shape[-1]) / FS
for a, c in zip(ax, [0, 1]):
    i = np.where(ytr == c)[0][0]
    for k, ch in enumerate(["FC3", "C3", "Cz", "C4", "FC4", "CPz"]):
        x = Xtr[i, CH.index(ch)]; x = (x - x.mean()) / (4 * x.std())
        a.plot(t, x - k, lw=.8, color=INK)
        a.text(-0.08, -k, ch, ha="right", va="center", fontsize=11, color=INK)
    a.set(title=f"One trial: {CLS[c].lower()} imagery", xlabel="time after cue (s)", yticks=[])
    a.spines["left"].set_visible(False)
fig.suptitle("What the data looks like: 22 electrodes x 1000 samples per trial (6 shown)", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/raw_eeg.png"); plt.close(fig)

# 2. motor imagery signature: 8-30 Hz power at C3 and C4 by class (all 9 people, training day, EA'd data)
r = []
for s in range(1, 10):
    Xtr, ytr, _, _ = load(s)
    lv = np.log(np.var(sosfiltfilt(SOS, Xtr[:, [CH.index("C3"), CH.index("C4")]], axis=-1), axis=-1))
    lv -= lv.mean(0)                                    # relative to the person's average
    r.append([lv[ytr == c].mean(0) for c in (0, 1)])
r = np.array(r)                                         # people x class x (C3, C4)
fig, a = plt.subplots(figsize=(6.5, 4.2))
x = np.arange(2); w = .36
for j, (c, col) in enumerate([(0, CYAN), (1, ORANGE)]):
    m = r[:, c].mean(0); e = r[:, c].std(0) / np.sqrt(9)
    b = a.bar(x + (j - .5) * w, m, w, yerr=e, color=col, capsize=4, label=f"imagining {CLS[c].lower()}")
a.axhline(0, color="#444", lw=.8)
a.set_xticks(x, ["C3 (left motor cortex)", "C4 (right motor cortex)"])
a.set(ylabel="8 to 30 Hz power vs person's average (log)", title="Imagining one hand lowers rhythm power on the opposite side")
a.title.set_fontsize(11); a.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/mi_signature.png"); plt.close(fig)

# 3. mean covariance of day 1 and day 2, before and after EA (person 1)
Xtr, ytr, Xte, yte = load(1)
Xtr_f, Xte_f = sosfiltfilt(SOS, Xtr, axis=-1), sosfiltfilt(SOS, Xte, axis=-1)
mats = [covs(Xtr_f).mean(0), covs(Xte_f).mean(0), covs(ea(Xtr_f)).mean(0), covs(ea(Xte_f)).mean(0)]
diff_before = np.linalg.norm(mats[0] - mats[1]) / np.linalg.norm(mats[0])
fig, ax = plt.subplots(1, 4, figsize=(14, 3.9))
for a, M, ttl in zip(ax, mats, ["Day 1, raw", "Day 2, raw", "Day 1, after EA", "Day 2, after EA"]):
    v = np.abs(M).max()
    a.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
    a.set(title=ttl, xticks=[], yticks=[]); a.title.set_color(INK)
ax[0].set_ylabel("22 electrodes"); ax[0].set_xlabel("22 electrodes")
fig.suptitle(f"Average covariance of a session (person 1, 8 to 30 Hz). Raw: each day has its own pattern. After EA: both become the identity",
             color=INK, fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/cov_before_after.png"); plt.close(fig)

# 4. every trial of every person and day as a point (log-covariance features, PCA), before vs after EA
feats = {"raw": [], "ea": []}; subj, sess = [], []
for s in range(1, 10):
    Xtr, _, Xte, _ = load(s)
    for k, X in enumerate([Xtr, Xte]):
        Xf = sosfiltfilt(SOS, X, axis=-1)[:, :, ::2]
        sel = np.random.default_rng(s * 10 + k).choice(len(Xf), 96, replace=False)
        for name, Z in [("raw", Xf[sel]), ("ea", ea(Xf)[sel])]:
            C = covs(Z); C = C / np.trace(C, axis1=1, axis2=2)[:, None, None] * 22 if name == "raw" else C
            feats[name].append(logvec(C))
        subj += [s] * 96; sess += [k] * 96
subj, sess = np.array(subj), np.array(sess)
cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
sil = {}
for a, name, ttl in zip(ax, ["raw", "ea"], ["Before alignment", "After Euclidean Alignment"]):
    F = np.vstack(feats[name]); F = (F - F.mean(0))
    Z = PCA(2, random_state=0).fit_transform(F)
    sil[name] = silhouette_score(F, subj)
    for s in range(1, 10):
        for k, mk in [(0, "o"), (1, "^")]:
            m = (subj == s) & (sess == k)
            a.scatter(Z[m, 0], Z[m, 1], s=10, marker=mk, color=cmap(s - 1), alpha=.65, lw=0,
                      label=f"person {s}" if k == 0 else None)
    a.set(title=f"{ttl}  (people separated: silhouette {sil[name]:.2f})", xticks=[], yticks=[], xlabel="PC 1", ylabel="PC 2")
    a.title.set_fontsize(11)
ax[1].legend(fontsize=8, ncol=1, frameon=False, loc="center left", bbox_to_anchor=(1, .5), markerscale=2)
fig.suptitle("Each dot is one trial (circle: day 1, triangle: day 2). Colour = person", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/ea_scatter.png"); plt.close(fig)
print("silhouette by person: raw %.3f  EA %.3f" % (sil["raw"], sil["ea"]))

# same, one person, colour = day
fig, ax = plt.subplots(1, 2, figsize=(10, 4.4))
m1 = subj == 2
for a, name, ttl in zip(ax, ["raw", "ea"], ["Person 2 before alignment", "Person 2 after alignment"]):
    F = np.vstack(feats[name])[m1]; Z = PCA(2, random_state=0).fit_transform(F - F.mean(0))
    for k, col, lab in [(0, CYAN, "day 1 (train)"), (1, ORANGE, "day 2 (test)")]:
        mm = sess[m1] == k
        a.scatter(Z[mm, 0], Z[mm, 1], s=16, color=col, alpha=.7, lw=0, label=lab)
        a.scatter(*Z[mm].mean(0), s=260, color=col, edgecolor=INK, lw=1.5, marker="X")
    a.set(title=ttl, xticks=[], yticks=[], xlabel="PC 1", ylabel="PC 2")
ax[0].legend(frameon=False)
fig.suptitle("Day 2 drifts away from day 1; alignment puts the two centres (X) back together", color=INK, fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/ea_days.png"); plt.close(fig)


# 5. failure analysis: confusion matrices and seed disagreement
def conf(tag, ens):
    M = np.zeros((4, 4)); dis = []
    for s in range(1, 10):
        yte = load(s)[3]
        Ps = [np.load(f) for f in sorted(glob.glob(f"{ROOT}/results/{tag}/S{s}_seed[0-4]_P.npy"))]
        if not Ps: continue
        preds = [p.argmax(1) for p in Ps]
        dis.append(np.mean([len(set(col)) > 1 for col in zip(*preds)]))
        pr = np.mean(Ps, 0).argmax(1) if ens else preds[0]
        for a_, b_ in zip(yte, pr): M[a_, b_] += 1
    return M / M.sum(1, keepdims=True) * 100, np.mean(dis) * 100


Mb, disb = conf("tcformer_paper/2a_aug1_d2", False)
Ma, disa = conf("newmodel/V4_rt", True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for a, M, ttl in zip(ax, [Mb, Ma], ["TCFormer (single model)", "TCFormer-AT (5-model ensemble)"]):
    a.imshow(M, cmap="Blues", vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            a.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", color="white" if M[i, j] > 55 else INK, fontsize=12)
    a.set_xticks(range(4), ["Left", "Right", "Feet", "Tongue"]); a.set_yticks(range(4), ["Left", "Right", "Feet", "Tongue"])
    a.set(xlabel="predicted", ylabel="true", title=ttl)
fig.suptitle("Where errors go (2a, % of each true class, 9 people). Tongue is the hardest class", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/confusion.png"); plt.close(fig)
print(f"seed disagreement: TCFormer {disb:.1f}% of trials, TCFormer-AT {disa:.1f}%")
print("diag before", np.round(np.diag(Mb), 1), "after", np.round(np.diag(Ma), 1))


# 6. ablation: paired gain over TCFormer on the same people and seeds (single models)
def accs(tag):
    out = {}
    for f in glob.glob(f"{ROOT}/results/{tag}/S*_seed*.json"):
        r = json.load(open(f))["res"]; out[(r["subject"], r["seed"])] = r["test_acc"] * 100
    return out


base = accs("tcformer_paper/2a_aug1_d2")
rows = [("Alignment only", "newmodel/V1_ea"), ("Covariance branch only", "newmodel/V2_cov"),
        ("Alignment + covariance branch", "newmodel/V3_ea_cov"), ("Alignment + pretraining", "newmodel/V4_ea_pre"),
        ("Alignment + pretraining + cov. branch", "newmodel/V5_full")]
gains = []
for name, tag in rows:
    v = accs(tag); k = [x for x in v if x in base]
    gains.append(np.mean([v[x] - base[x] for x in k])); print(name, len(k), round(gains[-1], 2))
fig, a = plt.subplots(figsize=(9, 4.2))
cols = [CYAN if g > 1 else (ORANGE if g < 0 else GREY) for g in gains]
b = a.barh([n for n, _ in rows][::-1], gains[::-1], color=cols[::-1])
a.bar_label(b, fmt="%+.2f", padding=4, fontsize=11)
a.axvline(0, color="#444", lw=.8)
a.set(xlabel="accuracy change vs TCFormer (points, same people and seeds)", title="Ablation on 2a: what each part adds")
a.set_xlim(min(gains) - 1.5, max(gains) + 1.5)
fig.tight_layout(); fig.savefig(f"{OUT}/ablation.png"); plt.close(fig)
print("saved", sorted(os.listdir(OUT)))
