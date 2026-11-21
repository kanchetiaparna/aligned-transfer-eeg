"""Summary of the paper-faithful TCFormer runs (results/tcformer_paper/<tag>/S*_seed*.json)."""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, json, os
import pandas as pd

R = f"{ROOT}/results/tcformer_paper"
rows = [json.load(open(f))["res"] | {"cfg": os.path.basename(os.path.dirname(f))} for f in glob.glob(f"{R}/*/S*_seed*.json")]
if not rows: raise SystemExit("no results yet")
df = pd.DataFrame(rows); pd.set_option("display.width", 200)
for cfg, d in sorted(df.groupby("cfg")):
    g = d.groupby("subject")
    t = pd.DataFrame({"runs": g.size(), "test_acc": g.test_acc.mean() * 100, "train_acc": g.train_acc_last.mean() * 100})
    seed_means = d.groupby("seed").test_acc.mean() * 100
    print(f"\n=== {cfg}: {len(d)} runs, {d.subject.nunique()} subjects")
    print(t.round(1).T.to_string())
    print(f"mean test acc {t.test_acc.mean():.2f}% +- {seed_means.std():.2f} (across seeds) | kappa {d.kappa.mean():.3f} | "
          f"train acc {t.train_acc.mean():.1f}% | gap {t.train_acc.mean() - t.test_acc.mean():.1f}")
