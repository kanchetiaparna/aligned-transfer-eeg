"""Seed ensemble (average softmax over the 5 seeds) for the new model vs TCFormer, from saved test probabilities."""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, os, numpy as np
R = f"{ROOT}/results"
def ens(pat, ds):
    single, en = {}, {}
    runs = {}
    for f in sorted(glob.glob(pat)):
        s = int(os.path.basename(f).split("_")[0][1:]); runs.setdefault(s, []).append(np.load(f))
    for s, Ps in sorted(runs.items()):
        y = np.load(f"{ROOT}/data/{ds}/A{s:02d}.npz")["yte"]
        single[s] = np.mean([(P.argmax(1) == y).mean() for P in Ps]) * 100
        en[s] = (np.mean(Ps, 0).argmax(1) == y).mean() * 100
    return single, en
for name, pat, ds in [("2a  new model", f"{R}/newmodel/V4_final/S*_P.npy", "tcf"),
                      ("2a  TCFormer ", f"{R}/tcformer_paper/2a_aug1_d2/S*_P.npy", "tcf"),
                      ("HaLT new model", f"{R}/newmodel/H4_final/S*_P.npy", "halt"),
                      ("HaLT TCFormer ", f"{R}/tcformer_paper/halt_aug1_d2/S*_P.npy", "halt")]:
    a, b = ens(pat, ds)
    print(f"{name}: single {np.mean(list(a.values())):.2f}  5-seed ensemble {np.mean(list(b.values())):.2f}  per subject", {k: round(v, 1) for k, v in b.items()})
