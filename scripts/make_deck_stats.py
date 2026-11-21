"""Statistics and figures for the professional deck (no training).
Recomputes per-person accuracies from saved predictions, then: mean +- SD, Cohen's kappa, paired Wilcoxon
(exact), rank-biserial effect size, bootstrap 95% CI over people, Holm correction; plus figures and equations.
    .venv/bin/python scripts/make_deck_stats.py  -> figures/deck2/*.png, figures/deck2/stats.json"""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score

R = f"{ROOT}/results"; D = f"{ROOT}/data"
OUT = f"{ROOT}/figures/deck2"; os.makedirs(OUT, exist_ok=True)
INK, CYAN, ORANGE, GREY = "#14213D", "#0B8F8A", "#E8702A", "#8A94A6"
plt.rcParams.update({"figure.dpi": 150, "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
rng = np.random.default_rng(0)


def y(folder, s):
    return np.load(f"{D}/{folder}/A{s:02d}.npz")["yte"]


def subjects(folder):
    return sorted(int(os.path.basename(f)[1:3]) for f in glob.glob(f"{D}/{folder}/A*.npz"))


def per_person(tag, folder, pf="P", ens=True):
    acc, kap = {}, {}
    for s in subjects(folder):
        fs = sorted(glob.glob(f"{R}/{tag}/S{s}_*[0-9]_{pf}.npy"))[:5]
        if not fs: continue
        yt = y(folder, s)
        if ens:
            pr = np.mean([np.load(f) for f in fs], 0).argmax(1)
            acc[s] = (pr == yt).mean() * 100; kap[s] = cohen_kappa_score(yt, pr)
        else:
            a = [(np.load(f).argmax(1) == yt).mean() * 100 for f in fs]
            k = [cohen_kappa_score(yt, np.load(f).argmax(1)) for f in fs]
            acc[s] = np.mean(a); kap[s] = np.mean(k)
    return acc, kap


def compare(a, b):
    k = sorted(set(a) & set(b)); d = np.array([a[i] - b[i] for i in k])
    w = wilcoxon(d, method="exact" if len(d) < 25 else "auto")
    pos = sum(np.argsort(np.argsort(np.abs(d)))[d > 0] + 1); neg = sum(np.argsort(np.argsort(np.abs(d)))[d < 0] + 1)
    boot = [rng.choice(d, len(d)).mean() for _ in range(10000)]
    return dict(n=len(d), mean=d.mean(), sd=d.std(ddof=1), ci=list(np.percentile(boot, [2.5, 97.5])),
                p=w.pvalue, r_rb=(pos - neg) / (pos + neg), better=int((d > 0).sum()), diffs=d.tolist(), subj=k)


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    return adj


S = {}
sets = {"2a": "tcf", "HaLT": "halt"}
main = {}
for ds, folder in sets.items():
    T = "tcformer_paper/2a_aug1_d2" if ds == "2a" else "tcformer_paper/halt_aug1_d2"
    A = "newmodel/V4_rt" if ds == "2a" else "newmodel/H4_rt"
    rows = {"TCFormer single": per_person(T, folder, ens=False), "TCFormer 5-seed ensemble": per_person(T, folder),
            "TCFormer-AT single": per_person(A, folder, ens=False), "TCFormer-AT ensemble": per_person(A, folder),
            "TCFormer-AT ensemble, real time": per_person(A, folder, "Pon")}
    main[ds] = rows
    S[ds] = {k: dict(acc=np.mean(list(v[0].values())), acc_sd=np.std(list(v[0].values()), ddof=1),
                     kappa=np.mean(list(v[1].values())), n=len(v[0])) for k, v in rows.items()}

# family of 6 recipe comparisons (5-model ensembles): Holm-corrected
fam = []
for ds, folder in sets.items():
    for net, b, a in [("EEGNet", "E_base", "E_AT"), ("ATCNet", "A_base", "A_AT"), ("TCFormer", None, None)]:
        if net == "TCFormer":
            pb = main[ds]["TCFormer 5-seed ensemble"][0]; pa = main[ds]["TCFormer-AT ensemble"][0]
        else:
            pre = "" if ds == "2a" else "H"
            b2 = b[0] + pre + b[1:] if pre else b; a2 = a[0] + pre + a[1:] if pre else a
            pb = per_person(f"newmodel/{b2}", folder)[0]; pa = per_person(f"newmodel/{a2}", folder)[0]
        c = compare(pa, pb); c.update(name=f"{net} · {ds}", base=np.mean(list(pb.values())), ours=np.mean(list(pa.values())))
        fam.append(c)
adj = holm(np.array([c["p"] for c in fam]))
for c, p in zip(fam, adj): c["p_holm"] = p
S["family"] = fam
S["single_AT_vs_single_TCF"] = {ds: compare(main[ds]["TCFormer-AT single"][0], main[ds]["TCFormer single"][0]) for ds in sets}
S["realtime_vs_offline"] = {ds: compare(main[ds]["TCFormer-AT ensemble, real time"][0], main[ds]["TCFormer-AT ensemble"][0]) for ds in sets}

# ablation family (single models, paired by person and seed -> averaged per person)
def accs(tag):
    out = {}
    for f in glob.glob(f"{R}/{tag}/S*_seed*.json"):
        r = json.load(open(f))["res"]; out.setdefault(r["subject"], {})[r["seed"]] = r["test_acc"] * 100
    return out
base = accs("tcformer_paper/2a_aug1_d2"); abl = []
for name, tag in [("Alignment only", "V1_ea"), ("Covariance branch only", "V2_cov"), ("Alignment + covariance branch", "V3_ea_cov"),
                  ("Alignment + pretraining", "V4_ea_pre"), ("Alignment + pretraining + cov. branch", "V5_full")]:
    v = accs(f"newmodel/{tag}")
    a = {s: np.mean([v[s][k] for k in v[s] if k in base.get(s, {})]) for s in v}
    b = {s: np.mean([base[s][k] for k in v[s] if k in base.get(s, {})]) for s in v}
    c = compare(a, b); c["name"] = name; abl.append(c)
for c, p in zip(abl, holm(np.array([c["p"] for c in abl]))): c["p_holm"] = p
S["ablation"] = abl

# run counts
cnt = {}
for d in glob.glob(f"{R}/**/", recursive=True):
    n = len(glob.glob(f"{d}*.json"))
    if n: cnt[os.path.relpath(d, R)] = n
S["runs_total"] = sum(cnt.values()); S["run_dirs"] = len(cnt)
json.dump(S, open(f"{OUT}/stats.json", "w"), indent=1, default=float)

# ---------- figures ----------
# paired slope plot
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
for a, ds in zip(ax, sets):
    pb, pa = main[ds]["TCFormer 5-seed ensemble"][0], main[ds]["TCFormer-AT ensemble"][0]
    for s in pb:
        col = CYAN if pa[s] > pb[s] else ORANGE
        a.plot([0, 1], [pb[s], pa[s]], color=col, lw=1.8, marker="o", ms=6, alpha=.85)
        a.text(1.04, pa[s], f"S{s}", va="center", fontsize=9, color=INK)
    a.plot([0, 1], [np.mean(list(pb.values())), np.mean(list(pa.values()))], color=INK, lw=4, marker="D", ms=10, label="mean")
    c = [f for f in fam if f["name"] == f"TCFormer · {ds}"][0]
    a.set(xticks=[0, 1], xticklabels=["TCFormer\n5-seed ensemble", "TCFormer-AT\n5-model ensemble"], ylabel="test accuracy, day 2 (%)",
          title=f"{ds}: +{c['mean']:.2f} [95% CI {c['ci'][0]:.2f}, {c['ci'][1]:.2f}], {c['better']}/{c['n']} better, p = {c['p']:.4f}")
    a.title.set_fontsize(10.5); a.set_xlim(-.2, 1.25)
ax[0].legend(frameon=False, loc="lower right")
fig.suptitle("Paired per-person comparison (cyan: improved, orange: worse)", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/paired.png"); plt.close(fig)

# forest plot of the 6 recipe comparisons
fig, a = plt.subplots(figsize=(10, 4.6))
for i, c in enumerate(fam[::-1]):
    a.errorbar(c["mean"], i, xerr=[[c["mean"] - c["ci"][0]], [c["ci"][1] - c["mean"]]], fmt="o", color=CYAN, ms=8, capsize=4, lw=2)
    a.text(c["ci"][1] + .3, i, f"+{c['mean']:.2f}  ({c['better']}/{c['n']}, p_Holm = {c['p_holm']:.3f}, r = {c['r_rb']:.2f})", va="center", fontsize=10)
a.set_yticks(range(len(fam)), [c["name"] for c in fam[::-1]]); a.axvline(0, color="#444", lw=1)
a.set(xlabel="gain from the recipe, points (mean over people, bootstrap 95% CI)", xlim=(-2, 17),
      title="Recipe vs plain training, 5-model ensembles: every interval excludes zero")
fig.tight_layout(); fig.savefig(f"{OUT}/forest.png"); plt.close(fig)

# literature comparison (4-class, train day 1 -> test day 2, BCI IV 2a)
lit = [("EEGNet, plain (ours, 5-seed ens.)", 79.55, GREY, "ours"), ("CTNet [26]", 82.5, GREY, "paper"), ("TCFormer [3]", 84.79, GREY, "paper"),
       ("TFANet [25]", 84.9, GREY, "paper"), ("ATCNet [2] (test-set epoch)", 85.4, "#C9CED6", "paper*"),
       ("TCFormer-AT single (ours)", 88.80, CYAN, "ours"), ("TCFormer-AT 5-model (ours)", 89.97, CYAN, "ours")]
fig, a = plt.subplots(figsize=(10, 4.8))
b = a.barh([l[0] for l in lit], [l[1] for l in lit], color=[l[2] for l in lit])
a.bar_label(b, fmt="%.2f", padding=4, fontsize=10); a.set_xlim(70, 93)
a.set(xlabel="accuracy on BCI IV 2a, session E (%)", title="Published 4-class cross-session results vs ours (axis starts at 70%)")
a.title.set_fontsize(11)
fig.tight_layout(); fig.savefig(f"{OUT}/literature.png"); plt.close(fig)

# real time: accuracy by quarter of the test session, offline vs online ensemble
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, ds in zip(ax, sets):
    folder = sets[ds]; A = "newmodel/V4_rt" if ds == "2a" else "newmodel/H4_rt"
    q_off, q_on = [], []
    for s in subjects(folder):
        yt = y(folder, s); n = len(yt)
        Po = np.mean([np.load(f) for f in sorted(glob.glob(f"{R}/{A}/S{s}_*[0-9]_P.npy"))[:5]], 0).argmax(1)
        Pn = np.mean([np.load(f) for f in sorted(glob.glob(f"{R}/{A}/S{s}_*[0-9]_Pon.npy"))[:5]], 0).argmax(1)
        idx = np.array_split(np.arange(n), 4)
        q_off.append([(Po[i] == yt[i]).mean() * 100 for i in idx]); q_on.append([(Pn[i] == yt[i]).mean() * 100 for i in idx])
    x = np.arange(4); w = .38
    a.bar(x - w / 2, np.mean(q_off, 0), w, color=GREY, label="whole session aligned at once")
    a.bar(x + w / 2, np.mean(q_on, 0), w, color=CYAN, label="real time (trials seen so far)")
    a.set_xticks(x, ["1st quarter", "2nd", "3rd", "4th"]); a.set(title=ds, ylabel="accuracy (%)", ylim=(50 if ds == "HaLT" else 80, 75 if ds == "HaLT" else 95))
ax[0].legend(frameon=False, fontsize=9, loc="lower left")
fig.suptitle("Real time vs offline across the test session (5-model ensembles)", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/realtime_quarters.png"); plt.close(fig)

# ---------- equations ----------
EQ = {
 "eq_ea": r"$\bar{R}=\frac{1}{N}\sum_{i=1}^{N}X_iX_i^{\top},\qquad \tilde{X}_i=\bar{R}^{-1/2}X_i,\qquad \frac{1}{N}\sum_{i=1}^{N}\tilde{X}_i\tilde{X}_i^{\top}=I$",
 "eq_online2": r"$\bar{R}_t=\frac{1}{t}\sum_{j=1}^{t}X_jX_j^{\top},\qquad \tilde{X}_t=\bar{R}_t^{-1/2}\,X_t\quad (t>5),\qquad \tilde{X}_t=\bar{R}_{\mathrm{day1}}^{-1/2}\,X_t\quad (t\leq 5)$",
 "eq_loss": r"$\theta_0=\arg\min_{\theta}\sum_{k\neq s}\;\mathcal{L}_{CE}(\theta;\,\tilde{\mathcal{D}}_k),\qquad \theta_s=\arg\min_{\theta\,\leftarrow\,\theta_0}\;\mathcal{L}_{CE}(\theta;\,\tilde{\mathcal{D}}_s^{\,day1})$",
 "eq_ens": r"$\hat{y}=\arg\max_{c}\;\frac{1}{K}\sum_{k=1}^{K}\mathrm{softmax}\left(f_{\theta_s^{(k)}}(\tilde{X})\right)_c,\qquad K=5$",
 "eq_kappa": r"$\kappa=\frac{p_o-p_e}{1-p_e},\qquad p_e=\sum_{c=1}^{4}p_c^{\mathrm{true}}\,p_c^{\mathrm{pred}}\;(=0.25\ \mathrm{for\ balanced\ classes})$",
 "eq_wilcoxon": r"$d_s=a_s^{\mathrm{AT}}-a_s^{\mathrm{base}},\qquad W^{+}=\sum_{d_s>0}\mathrm{rank}(|d_s|),\qquad r_{rb}=\frac{W^{+}-W^{-}}{W^{+}+W^{-}}$",
 "eq_holm": r"$p_{(i)}^{\mathrm{Holm}}=\max_{j\leq i}\;\min\left(1,\,(m-j+1)\,p_{(j)}\right),\qquad m=6$",
}
plt.rcParams["mathtext.fontset"] = "cm"
for name, tex in EQ.items():
    for theme, col in [("dark", "#F4F1EA"), ("light", INK)]:
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0, 0, tex, fontsize=30, color=col)
        fig.savefig(f"{OUT}/{name}_{theme}.png", dpi=150, transparent=True, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)

# print summary
for ds in sets:
    print(f"== {ds}")
    for k, v in S[ds].items(): print(f"  {k:34s} acc {v['acc']:.2f} ± {v['acc_sd']:.2f}   kappa {v['kappa']:.3f}  (n={v['n']})")
print("== recipe family (Holm, m=6)")
for c in fam: print(f"  {c['name']:18s} base {c['base']:.2f} ours {c['ours']:.2f}  +{c['mean']:.2f} ± {c['sd']:.2f}  CI [{c['ci'][0]:.2f},{c['ci'][1]:.2f}]  {c['better']}/{c['n']}  p={c['p']:.4f}  pHolm={c['p_holm']:.4f}  r={c['r_rb']:.2f}")
for ds in sets:
    c = S["single_AT_vs_single_TCF"][ds]; print(f"  single AT vs single TCF {ds}: +{c['mean']:.2f} CI [{c['ci'][0]:.2f},{c['ci'][1]:.2f}] {c['better']}/{c['n']} p={c['p']:.4f} r={c['r_rb']:.2f}")
    c = S["realtime_vs_offline"][ds]; print(f"  real time minus offline {ds}: {c['mean']:+.2f} CI [{c['ci'][0]:.2f},{c['ci'][1]:.2f}] p={c['p']:.3f}")
print("== ablation (Holm, m=5)")
for c in abl: print(f"  {c['name']:40s} {c['mean']:+.2f} CI [{c['ci'][0]:.2f},{c['ci'][1]:.2f}] {c['better']}/{c['n']} p={c['p']:.4f} pHolm={c['p_holm']:.4f}")
print("runs:", S["runs_total"], "in", S["run_dirs"], "folders")
