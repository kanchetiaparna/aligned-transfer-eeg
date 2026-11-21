"""Summary of the paper-faithful ATCNet runs (results/atcnet_paper/<protocol>/S*_run*.json)."""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, json, os
import numpy as np, pandas as pd

R = f"{ROOT}/results/atcnet_paper"
rows = [json.load(open(f))["res"] | {"cfg": os.path.basename(os.path.dirname(f))}
        for f in glob.glob(f"{R}/*/S*_run*.json")]
if not rows: raise SystemExit("no results yet")
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for cfg, d in sorted(df.groupby("cfg")):
    per_sub = d.groupby("subject")
    t = pd.DataFrame({"runs": per_sub.size(),
                      "acc_selected": per_sub.test_acc_selected.mean() * 100,
                      "acc_last_epoch": per_sub.test_acc_last.mean() * 100,
                      "train_acc": per_sub.train_acc_last.mean() * 100})
    run_means = d.groupby("run").test_acc_selected.mean() * 100   # spread across repetitions
    print(f"\n=== {cfg}: {len(d)} runs, {d.subject.nunique()} subjects")
    print(t.round(1).T.to_string())
    sel, last = t.acc_selected.mean(), t.acc_last_epoch.mean()
    print(f"mean acc (selected weights) {sel:.2f}% +- {run_means.std():.2f} | kappa {(sel/100-0.25)/0.75:.3f} | "
          f"last-epoch acc {last:.2f}% | train acc {t.train_acc.mean():.1f}% | gap {t.train_acc.mean()-sel:.1f}")
