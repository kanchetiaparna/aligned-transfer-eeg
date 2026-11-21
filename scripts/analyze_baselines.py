"""Failure analysis of the two paper-faithful baselines on BCI IV 2a (and HaLT).
ATCNet: results/atcnet_paper/{paper,honest}_keras  (P = test probabilities of the selected weights)
TCFormer: results/tcformer_paper/2a_aug{0,1}_d2    (P = test probabilities of the last epoch)
Prints: confusion matrices, per-class recall, per-subject accuracy, model agreement / complementarity,
seed-ensemble gains, learning curves (overfitting timing) and within-session vs cross-session accuracy."""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, json, os
import numpy as np

R = f"{ROOT}/results"
CLS = ["left", "right", "feet", "tongue"]
np.set_printoptions(precision=1, suppress=True)


def y_test(ds, s):
    return np.load(f"{ROOT}/data/{ds}/A{s:02d}.npz")["yte"]


def load_runs(pattern, key):
    """returns {subject: [P_run1, P_run2, ...]}"""
    out = {}
    for f in sorted(glob.glob(pattern)):
        s = int(os.path.basename(f).split("_")[0][1:])
        out.setdefault(s, []).append(np.load(f))
    return out


def report(name, runs, ds="tcf"):
    cm = np.zeros((4, 4)); accs, ens = {}, {}
    for s, Ps in sorted(runs.items()):
        y = y_test(ds, s)
        for P in Ps:
            cm += np.bincount(y * 4 + P.argmax(1), minlength=16).reshape(4, 4)
        accs[s] = np.mean([(P.argmax(1) == y).mean() for P in Ps]) * 100
        ens[s] = (np.mean(Ps, 0).argmax(1) == y).mean() * 100
    cmn = cm / cm.sum(1, keepdims=True) * 100
    print(f"\n### {name}: mean acc {np.mean(list(accs.values())):.2f}%  | seed-ensemble {np.mean(list(ens.values())):.2f}%")
    print("per subject:", {k: round(v, 1) for k, v in accs.items()})
    print("confusion (rows=true, cols=pred, % of true class):  " + "  ".join(CLS))
    for i, c in enumerate(CLS):
        print(f"  {c:7s}", cmn[i])
    return accs, cmn


atc = load_runs(f"{R}/atcnet_paper/honest_keras/S*_run*_P.npy", "P")
atcp = load_runs(f"{R}/atcnet_paper/paper_keras/S*_run*_P.npy", "P")
tcf = load_runs(f"{R}/tcformer_paper/2a_aug1_d2/S*_seed*_P.npy", "P")
tcf0 = load_runs(f"{R}/tcformer_paper/2a_aug0_d2/S*_seed*_P.npy", "P")
# ATCNet saved probabilities for -0.5..4 s windows; labels are the same trials in the same order
a_acc, a_cm = report("ATCNet (honest, Keras-matched)", atc, ds="paper")
report("ATCNet (paper protocol)", atcp, ds="paper")
t_acc, t_cm = report("TCFormer (+aug)", tcf)
report("TCFormer (-aug)", tcf0)

print("\n### complementarity on 2a (TCFormer+aug vs ATCNet honest), per subject")
gain_ens = []
for s in sorted(tcf):
    y = y_test("tcf", s)
    pt, pa = np.mean(tcf[s], 0), np.mean(atc[s], 0)
    both_wrong = np.mean((pt.argmax(1) != y) & (pa.argmax(1) != y)) * 100
    oracle = np.mean((pt.argmax(1) == y) | (pa.argmax(1) == y)) * 100
    mix = np.mean(((pt + pa) / 2).argmax(1) == y) * 100
    gain_ens.append(mix)
    print(f"  S{s}: TCF {np.mean(pt.argmax(1)==y)*100:5.1f}  ATC {np.mean(pa.argmax(1)==y)*100:5.1f}  "
          f"both wrong {both_wrong:5.1f}  oracle {oracle:5.1f}  avg-ensemble {mix:5.1f}")
print(f"  mean avg-ensemble TCF+ATC: {np.mean(gain_ens):.2f}%")

print("\n### confidence of errors (TCFormer+aug, all runs): max softmax prob")
conf_ok, conf_bad = [], []
for s, Ps in tcf.items():
    y = y_test("tcf", s)
    for P in Ps:
        m = P.max(1); ok = P.argmax(1) == y
        conf_ok += list(m[ok]); conf_bad += list(m[~ok])
print(f"  correct: median conf {np.median(conf_ok):.2f} | wrong: median conf {np.median(conf_bad):.2f} | "
      f"wrong with conf>0.9: {np.mean(np.array(conf_bad) > 0.9)*100:.1f}% of errors")

print("\n### learning curves TCFormer+aug (test acc logged every 50 epochs, never used for selection)")
curves = {}
for f in glob.glob(f"{R}/tcformer_paper/2a_aug1_d2/S*_seed*.json"):
    h = json.load(open(f))["hist"]
    for r in h:
        if "test_acc_monitor" in r: curves.setdefault(r["ep"], []).append(r["test_acc_monitor"] * 100)
    for r in h:
        curves.setdefault(("train", r["ep"]), []).append(r["train_acc"] * 100)
for ep in [0, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 999]:
    if ep in curves:
        print(f"  epoch {ep:4d}: test {np.mean(curves[ep]):5.1f}%   train {np.mean(curves[('train', ep)]):5.1f}%")

print("\n### within-session vs cross-session (ATCNet honest: 20% of session T vs session E, same weights)")
v, t = [], []
for f in glob.glob(f"{R}/atcnet_paper/honest_keras/S*_run*.json"):
    j = json.load(open(f)); h = j["hist"]; b = j["res"]["best_ep"]
    v.append(h[b]["val_acc"] * 100); t.append(j["res"]["test_acc_selected"] * 100)
print(f"  within-session validation acc {np.mean(v):.1f}%  vs  next-session test acc {np.mean(t):.1f}%  -> shift costs {np.mean(v)-np.mean(t):.1f} points")

print("\n### HaLT TCFormer+aug confusion")
report("TCFormer HaLT (+aug)", load_runs(f"{R}/tcformer_paper/halt_aug1_d2/S*_seed*_P.npy", "P"), ds="halt")
