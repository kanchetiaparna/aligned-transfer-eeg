"""Compare new-model variants with the paper-faithful TCFormer / ATCNet baselines on the same seeds."""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, json, os, sys
import numpy as np, pandas as pd

R = f"{ROOT}/results"
SEEDS = {0, 1, 2}
rows = []
for f in glob.glob(f"{R}/newmodel/V*/S*_seed*.json"):
    r = json.load(open(f))["res"]; rows.append({"cfg": r["tag"], "subject": r["subject"], "seed": r["seed"],
                                                 "acc": r["test_acc"] * 100, "kappa": r["kappa"], "train": r["train_acc_last"] * 100})
for f in glob.glob(f"{R}/tcformer_paper/2a_aug1_d2/S*_seed*.json"):
    r = json.load(open(f))["res"]
    if r["seed"] in SEEDS:
        rows.append({"cfg": "B_TCFormer_paper", "subject": r["subject"], "seed": r["seed"], "acc": r["test_acc"] * 100,
                     "kappa": r["kappa"], "train": r["train_acc_last"] * 100})
for f in glob.glob(f"{R}/atcnet_paper/honest_keras/S*_run*.json"):
    r = json.load(open(f))["res"]
    if r["run"] in SEEDS:
        rows.append({"cfg": "B_ATCNet_honest", "subject": r["subject"], "seed": r["run"], "acc": r["test_acc_selected"] * 100,
                     "kappa": r["kappa_selected"], "train": r["train_acc_last"] * 100})
df = pd.DataFrame(rows); pd.set_option("display.width", 220)
base = df[df.cfg == "B_TCFormer_paper"].set_index(["subject", "seed"]).acc
out = []
for cfg, d in df.groupby("cfg"):
    m = d.set_index(["subject", "seed"]).acc
    common = m.index.intersection(base.index)
    diff = (m.loc[common] - base.loc[common])
    subj_means = d.groupby("subject").acc.mean()
    out.append({"cfg": cfg, "runs": len(d), "subjects": d.subject.nunique(), "acc": subj_means.mean(),
                "kappa": d.kappa.mean(), "train": d.train.mean(), "gap": d.train.mean() - subj_means.mean(),
                "vs_TCFormer(paired)": diff.mean() if len(common) else np.nan,
                "subjects_better": int((diff.groupby(level=0).mean() > 0).sum()) if len(common) else 0})
t = pd.DataFrame(out).set_index("cfg").sort_values("acc", ascending=False)
print(t.round(2).to_string())
print("\nper-subject accuracy (mean over seeds)")
print(df.pivot_table(index="cfg", columns="subject", values="acc", aggfunc="mean").loc[t.index].round(1).to_string())
