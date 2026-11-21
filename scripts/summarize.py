"""Aggregate results: per-config mean accuracy, kappa, overfitting gap, and seed-ensemble accuracy."""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, json, sys, os
import numpy as np, pandas as pd
from common import load_subject

R = f"{ROOT}/results"
rows = []
for f in glob.glob(f"{R}/*/S*_seed*.json"):
    r = json.load(open(f))["res"]; r["cfg"] = os.path.basename(os.path.dirname(f)); rows.append(r)
df = pd.DataFrame(rows)
if df.empty: sys.exit("no results")
kappa = lambda acc: (acc - 0.25) / 0.75

g = df.groupby("cfg")
out = pd.DataFrame({
    "n_runs": g.size(),
    "params": g.params.first(),
    "train_acc": g.train_acc_last.mean() * 100,
    "val_acc": g.val_last.mean() * 100 if "val_last" in df else np.nan,
    "test_last": g.test_last.mean() * 100,
    "test_bestval": g.test_bestval.mean() * 100 if "test_bestval" in df else np.nan,
})
out["gap(train-test)"] = out.train_acc - out.test_last
out["seed_std"] = g.apply(lambda d: d.groupby("seed").test_last.mean().std() * 100)

# ensemble: average softmax over seeds per subject (final-epoch weights, no selection on test)
yte = {s: load_subject(s)[3] for s in range(1, 10)}
ens = {}
for cfg in out.index:
    accs = []
    for s in range(1, 10):
        P = [np.load(p) for p in glob.glob(f"{R}/{cfg}/S{s}_seed*_Plast.npy")]
        if P: accs.append((np.mean(P, 0).argmax(1) == yte[s]).mean())
    ens[cfg] = np.mean(accs) * 100 if len(accs) == 9 else np.nan
out["ens_seeds"] = pd.Series(ens)
out["kappa"] = kappa(out.test_last / 100)
pd.set_option("display.width", 200)
print(out.sort_values("test_last", ascending=False).round(2).to_string())

# per subject table for test_last
ps = df.pivot_table(index="cfg", columns="subject", values="test_last", aggfunc="mean") * 100
print("\nper-subject test accuracy (mean over seeds)")
print(ps.loc[out.sort_values("test_last", ascending=False).index].round(1).to_string())
out.to_csv(f"{R}/summary.csv"); ps.to_csv(f"{R}/per_subject.csv")
