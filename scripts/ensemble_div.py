"""Final comparison: new model with 5 independently pretrained copies (V4_div / H4_div) vs single runs and TCFormer."""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, os, json, numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import cohen_kappa_score
R = f"{ROOT}/results"
def load(p):
    d = {}
    for f in sorted(glob.glob(p)):
        s = int(os.path.basename(f).split("_")[0][1:]); d.setdefault(s, []).append(np.load(f))
    return dict(sorted(d.items()))
for ds, new, shared, tcf in [("tcf", "V4_div", "V4_final", "2a_aug1_d2"), ("halt", "H4_div", "H4_final", "halt_aug1_d2")]:
    N, S, T = load(f"{R}/newmodel/{new}/S*_P.npy"), load(f"{R}/newmodel/{shared}/S*_P.npy"), load(f"{R}/tcformer_paper/{tcf}/S*_P.npy")
    rows = {k: [] for k in ["single", "ens_shared", "ens_div", "kappa_div", "tcf_single", "tcf_ens", "mix"]}
    for s in N:
        y = np.load(f"{ROOT}/data/{ds}/A{s:02d}.npz")["yte"]
        acc = lambda P: (P.argmax(1) == y).mean() * 100
        rows["single"].append(np.mean([acc(p) for p in N[s]]))
        rows["ens_shared"].append(acc(np.mean(S[s], 0)))
        rows["ens_div"].append(acc(np.mean(N[s], 0)))
        rows["kappa_div"].append(cohen_kappa_score(y, np.mean(N[s], 0).argmax(1)))
        rows["tcf_single"].append(np.mean([acc(p) for p in T[s]]))
        rows["tcf_ens"].append(acc(np.mean(T[s], 0)))
        rows["mix"].append(acc((np.mean(N[s], 0) + np.mean(T[s], 0)) / 2))
    print(f"\n=== {ds}  ({len(N)} subjects, {sum(len(v) for v in N.values())} runs)")
    for k, v in rows.items(): print(f"  {k:11s} {np.mean(v):6.2f}   {[round(x, 1) for x in v]}")
    d = np.array(rows["ens_div"]) - np.array(rows["tcf_ens"])
    print(f"  div-ensemble vs TCFormer 5-seed ensemble: {d.mean():+.2f}, better on {(d>0).sum()}/{len(d)}, Wilcoxon p={wilcoxon(d).pvalue:.4f}")
    d2 = np.array(rows["ens_div"]) - np.array(rows["single"])
    print(f"  div-ensemble vs single new model: {d2.mean():+.2f}, better on {(d2>0).sum()}/{len(d2)}")
