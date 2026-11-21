"""Does the AT recipe (EA + diverse cross-subject pretraining + fine-tune + 5-copy ensemble) help ATCNet and EEGNet too?"""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, os, sys, numpy as np
from scipy.stats import wilcoxon
R = f"{ROOT}/results/newmodel"
def load(p):
    d = {}
    for f in sorted(glob.glob(p)):
        s = int(os.path.basename(f).split("_")[0][1:]); d.setdefault(s, []).append(np.load(f))
    return dict(sorted(d.items()))
SETS = [("2a", "tcf", "A_base", "A_AT", "ATCNet"), ("2a", "tcf", "E_base", "E_AT", "EEGNet"),
        ("HaLT", "halt", "AH_base", "AH_AT", "ATCNet"), ("HaLT", "halt", "EH_base", "EH_AT", "EEGNet")]
for name, ds, bt, at, net in SETS:
    B, A, On = load(f"{R}/{bt}/S*_seed?_P.npy"), load(f"{R}/{at}/S*_seed?_P.npy"), load(f"{R}/{at}/S*_Pon.npy")
    subs = [s for s in A if len(A[s]) == 5 and len(B.get(s, [])) == 5 and len(On.get(s, [])) == 5]
    if not subs: print(f"\n=== {name} {net}: not finished yet"); continue
    r = {k: [] for k in ["base_single", "base_ens", "AT_single", "AT_ens", "AT_ens_realtime"]}
    for s in subs:
        y = np.load(f"{ROOT}/data/{ds}/A{s:02d}.npz")["yte"]; acc = lambda P: (P.argmax(1) == y).mean() * 100
        r["base_single"].append(np.mean([acc(p) for p in B[s]])); r["base_ens"].append(acc(np.mean(B[s], 0)))
        r["AT_single"].append(np.mean([acc(p) for p in A[s]])); r["AT_ens"].append(acc(np.mean(A[s], 0)))
        r["AT_ens_realtime"].append(acc(np.mean(On[s], 0)))
    print(f"\n=== {name} {net}: {len(subs)} subjects")
    for k, v in r.items(): print(f"  {k:16s} {np.mean(v):6.2f}  {[round(x, 1) for x in v]}")
    for a, b in [("AT_single", "base_single"), ("AT_ens", "base_ens"), ("AT_ens_realtime", "base_ens")]:
        d = np.array(r[a]) - np.array(r[b])
        print(f"  {a} vs {b}: {d.mean():+.2f}, better on {(d>0).sum()}/{len(d)}, Wilcoxon p={wilcoxon(d).pvalue:.4f}")
