"""Real-time (online EA) check of the 5-copy ensemble: V4_rt / H4_rt, offline P vs online Pon."""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, os, numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score
R = f"{ROOT}/results"
def load(p):
    d = {}
    for f in sorted(glob.glob(p)):
        s = int(os.path.basename(f).split("_")[0][1:]); d.setdefault(s, []).append(np.load(f))
    return dict(sorted(d.items()))
for ds, tag, tcf in [("tcf", "V4_rt", "2a_aug1_d2"), ("halt", "H4_rt", "halt_aug1_d2")]:
    Off, On, T = load(f"{R}/newmodel/{tag}/S*_seed?_P.npy"), load(f"{R}/newmodel/{tag}/S*_Pon.npy"), load(f"{R}/tcformer_paper/{tcf}/S*_P.npy")
    r = {k: [] for k in ["single_off", "single_on", "ens_off", "ens_on", "kappa_on", "tcf_ens"]}
    for s in Off:
        y = np.load(f"{ROOT}/data/{ds}/A{s:02d}.npz")["yte"]; acc = lambda P: (P.argmax(1) == y).mean() * 100
        r["single_off"].append(np.mean([acc(p) for p in Off[s]])); r["single_on"].append(np.mean([acc(p) for p in On[s]]))
        r["ens_off"].append(acc(np.mean(Off[s], 0))); r["ens_on"].append(acc(np.mean(On[s], 0)))
        r["kappa_on"].append(cohen_kappa_score(y, np.mean(On[s], 0).argmax(1))); r["tcf_ens"].append(acc(np.mean(T[s], 0)))
    print(f"\n=== {ds}: {len(Off)} subjects, {sum(map(len, Off.values()))} offline / {sum(map(len, On.values()))} online runs")
    for k, v in r.items(): print(f"  {k:10s} {np.mean(v):6.2f}  {[round(x, 1) for x in v]}")
    d = np.array(r["ens_on"]) - np.array(r["tcf_ens"])
    print(f"  real-time ensemble vs TCFormer 5-seed ensemble: {d.mean():+.2f}, better on {(d>0).sum()}/{len(d)}, Wilcoxon p={wilcoxon(d).pvalue:.4f}")
