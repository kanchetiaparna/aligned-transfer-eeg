"""Idea A results: HyperTCF vs identical control (no session info) vs TCFormer-AT, single and 5-copy ensembles."""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, os, sys, numpy as np
from scipy.stats import wilcoxon
R = f"{ROOT}/results/newmodel"
ds, hy, ct, base = (sys.argv[1:5] if len(sys.argv) > 4 else ("tcf", "HY", "HY_ctrl", "V4_rt"))
def load(tag, kind):
    d = {}
    for f in sorted(glob.glob(f"{R}/{tag}/S*_seed?_{kind}.npy")):
        s = int(os.path.basename(f).split("_")[0][1:]); d.setdefault(s, []).append(np.load(f))
    return d
rows = {"TCFormer-AT": (base, "P"), "TCFormer-AT real time": (base, "Pon"), "Control (no session info)": (ct, "P"),
        "Control real time": (ct, "Pon"), "HyperTCF": (hy, "P"), "HyperTCF real time": (hy, "Pon"),
        "HyperTCF, day-1 descriptor": (hy, "P_descT"), "HyperTCF, other person's descriptor": (hy, "P_descOther"),
        "HyperTCF, zero descriptor": (hy, "P_descZero")}
L = {k: load(*v) for k, v in rows.items()}
subs = sorted(s for s in L["HyperTCF"] if all(len(L[k].get(s, [])) == 5 for k in L))
print(f"{len(subs)} people complete")
if not subs: sys.exit()
out = {}
for k in L:
    sg, en = [], []
    for s in subs:
        y = np.load(f"{ROOT}/data/{ds}/A{s:02d}.npz")["yte"]
        sg.append(np.mean([(p.argmax(1) == y).mean() * 100 for p in L[k][s]])); en.append((np.mean(L[k][s], 0).argmax(1) == y).mean() * 100)
    out[k] = (np.array(sg), np.array(en)); print(f"  {k:38s} single {np.mean(sg):6.2f}   5-copy ensemble {np.mean(en):6.2f}")
for a_, b_ in [("HyperTCF", "Control (no session info)"), ("HyperTCF", "TCFormer-AT"), ("HyperTCF real time", "TCFormer-AT real time"),
               ("HyperTCF", "HyperTCF, day-1 descriptor"), ("HyperTCF", "HyperTCF, other person's descriptor")]:
    for i, nm in enumerate(["single", "ensemble"]):
        d = out[a_][i] - out[b_][i]; p = wilcoxon(d).pvalue if np.any(d) else 1.0
        print(f"  {a_} vs {b_} [{nm}]: {d.mean():+.2f}, better {(d>0).sum()}/{len(d)}, worse {(d<0).sum()}, p={p:.3f}")
