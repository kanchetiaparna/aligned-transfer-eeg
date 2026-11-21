"""Recompute every number used in the deck/report from saved result files (no training)."""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, json, os
import numpy as np
from scipy.stats import wilcoxon

R = f"{ROOT}/results"; D = f"{ROOT}/data"


def js(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(pat))]


def res(r):
    return r.get("res", r)


def single(tag, key="test_acc"):
    v = [res(r)[key] for r in js(f"{R}/{tag}/S*_*.json") if key in res(r)]
    return np.mean(v) * 100, len(v)


def ytest(folder, s):
    return np.load(f"{D}/{folder}/A{s:02d}.npz")["yte"]


def ens(tag, folder, pfile="P", n=5):
    per = {}
    for s in range(1, 12):
        fs = sorted(glob.glob(f"{R}/{tag}/S{s}_*[0-9]_{pfile}.npy"))[:n]
        if not fs or not os.path.exists(f"{D}/{folder}/A{s:02d}.npz"): continue
        y = ytest(folder, s); P = np.mean([np.load(f) for f in fs], 0)
        if len(P) != len(y): continue
        per[s] = (P.argmax(1) == y).mean() * 100
    return per


def show(name, claimed, got):
    flag = "OK " if abs(claimed - got) < 0.06 else "DIFF"
    print(f"  [{flag}] {name:58s} claimed {claimed:7.2f}   recomputed {got:7.2f}")


def pair(a, b):
    k = sorted(set(a) & set(b)); x = np.array([a[i] for i in k]); y = np.array([b[i] for i in k])
    return np.mean(x - y), int((x > y).sum()), len(k), wilcoxon(x, y).pvalue


print("== Baselines (single model, last epoch)")
show("TCFormer 2a (paper recipe, +aug)", 84.88, single("tcformer_paper/2a_aug1_d2")[0])
show("TCFormer HaLT", 61.99, single("tcformer_paper/halt_aug1_d2")[0])
show("ATCNet 2a honest (keras, 10 repeats, epoch chosen on 20% of day 1)", 74.57, np.mean([res(r)["test_acc_selected"] for r in js(f"{R}/atcnet_paper/honest_keras/*.json")]) * 100)
show("ATCNet HaLT honest (epoch chosen on 20% of day 1)", 49.17, np.mean([res(r)["test_acc_selected"] for r in js(f"{R}/atcnet_paper/halt_honest_keras/*.json")]) * 100)

print("== TCFormer-AT")
show("TCFormer-AT 2a single", 88.80, single("newmodel/V4_rt")[0])
show("TCFormer-AT HaLT single", 67.80, single("newmodel/H4_rt")[0])
for name, tag, folder, base_tag, c_off, c_on, c_base in [("2a", "newmodel/V4_rt", "tcf", "tcformer_paper/2a_aug1_d2", 89.97, 89.97, 86.50),
                                                         ("HaLT", "newmodel/H4_rt", "halt", "tcformer_paper/halt_aug1_d2", 68.15, 68.06, 62.43)]:
    off, on, base = ens(tag, folder), ens(tag, folder, "Pon"), ens(base_tag, folder)
    show(f"TCFormer-AT {name} 5-model ensemble", c_off, np.mean(list(off.values())))
    show(f"TCFormer-AT {name} real time ensemble", c_on, np.mean(list(on.values())))
    show(f"TCFormer {name} 5-seed ensemble", c_base, np.mean(list(base.values())))
    d, w, n, p = pair(off, base); print(f"         AT ens vs TCFormer ens: {d:+.2f}, better {w}/{n}, p={p:.4f}")

print("== Recipe on other networks (5-model ensembles)")
for name, b, a, folder, cb, ca in [("EEGNet 2a", "E_base", "E_AT", "tcf", 79.55, 85.49), ("ATCNet 2a", "A_base", "A_AT", "tcf", 84.88, 89.16),
                                   ("EEGNet HaLT", "EH_base", "EH_AT", "halt", 59.80, 66.26), ("ATCNet HaLT", "AH_base", "AH_AT", "halt", 58.07, 63.81)]:
    eb, ea_ = ens(f"newmodel/{b}", folder), ens(f"newmodel/{a}", folder)
    show(f"{name} plain", cb, np.mean(list(eb.values()))); show(f"{name} recipe", ca, np.mean(list(ea_.values())))
    d, w, n, p = pair(ea_, eb); print(f"         gain {d:+.2f}, better {w}/{n}, p={p:.4f}")

print("== Overfitting gap (train last - test)")
for name, tag, c in [("TCFormer 2a", "tcformer_paper/2a_aug1_d2", 15.1), ("TCFormer-AT 2a", "newmodel/V4_rt", 10.9),
                     ("TCFormer HaLT", "tcformer_paper/halt_aug1_d2", 35.9), ("TCFormer-AT HaLT", "newmodel/H4_rt", 27.3)]:
    show(name, c, single(tag, "train_acc_last")[0] - single(tag)[0])

print("== Phase 1 comparison (summary.csv, test_last)")
import csv
rows = {r["cfg"]: float(r["test_last"]) for r in csv.DictReader(open(f"{R}/summary.csv"))}
for name, cfg, c in [("EEG Conformer", "EEGConformer_within_ea1_sr1", 79.3), ("TCFormer", "TCFormer_within_ea1_sr1", 79.2), ("IFNet", "IFNet_within_ea1_sr1", 79.1),
                     ("CTNet", "CTNet_within_ea1_sr1", 79.1), ("MSVTNet", "MSVTNet_within_ea1_sr1", 78.4), ("ATCNet", "ATCNet_within_ea1_sr1", 77.7)]:
    show(name, c, rows[cfg])

print("== Phase 2 (pretraining on external datasets, ft_2a+ext vs ft_2a_only)")
for m in ["ATCNet", "CTNet", "MSVTNet", "TCFormer"]:
    print(f"         {m:9s} 2a only {rows[m+'_ft_2a_only']:.2f}  with external data {rows[m+'_ft_2a+ext']:.2f}  change {rows[m+'_ft_2a+ext']-rows[m+'_ft_2a_only']:+.2f}")

print("== Seed ensemble gain for plain TCFormer 2a")
show("5-seed ensemble minus single", 1.6, np.mean(list(ens("tcformer_paper/2a_aug1_d2", "tcf").values())) - single("tcformer_paper/2a_aug1_d2")[0])

print("== HyperTCF (5-model ensembles, 2a)")
for name, tag, c in [("HyperTCF", "newmodel/HY", 89.62), ("Control", "newmodel/HY_ctrl", 90.20)]:
    show(name, c, np.mean(list(ens(tag, "tcf").values())))
for v, c in [("P_descT", 90.24), ("P_descZero", 89.39), ("P_descOther", 85.80)]:
    show(f"HyperTCF fed {v}", c, np.mean(list(ens("newmodel/HY", "tcf", v).values())))
