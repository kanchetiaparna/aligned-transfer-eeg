"""Prepare external motor imagery datasets in the BCI IV 2a format, for cross-dataset pretraining.

For each dataset/subject:
  * keep only left hand / right hand / feet trials -> labels 0 / 1 / 2 (same codes as 2a; 2a tongue=3)
  * pick the 22 BCI IV 2a electrodes in 2a order (all present in these datasets)
  * band-pass 0.5-75 Hz (PhysioNet is sampled at 160 Hz, so 75 Hz is the common ceiling), resample to 250 Hz
  * 4 s window from cue onset -> 1000 samples, values in microvolts
Output: data/ext/<Dataset>_S<id>.npz with X (n, 22, 1000) float32, y (n,), session (n,)
usage: prep_external.py <DatasetName>
"""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import os, sys, numpy as np, mne
mne.set_config("MNE_DATA", f"{ROOT}/data/mne", set_env=True)
mne.set_log_level("ERROR")
import warnings; warnings.filterwarnings("ignore")
import moabb.datasets as D
from moabb.paradigms import MotorImagery

A2 = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
      "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"]
LAB = {"left_hand": 0, "right_hand": 1, "feet": 2}
OUT = f"{ROOT}/data/ext"; os.makedirs(OUT, exist_ok=True)

name = sys.argv[1]
ds = getattr(D, name)()
par = MotorImagery(events=list(LAB), n_classes=3, fmin=0.5, fmax=75, tmin=0.0, tmax=4.0 - 1 / 250, resample=250)
part, nparts = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
for s in ds.subject_list:
    if s % nparts != part: continue  # lets several processes split one dataset without overlap
    f = f"{OUT}/{name}_S{s:03d}.npz"
    if os.path.exists(f): continue
    try:
        ep, y, meta = par.get_data(ds, subjects=[s], return_epochs=True)
        low = {c.lower(): c for c in ep.ch_names}
        ep = ep.pick([low[c.lower()] for c in A2])
        X = (ep.get_data(copy=False) * 1e6).astype(np.float32)[:, :, :1000]
        y = np.array([LAB[v] for v in y])
        np.savez_compressed(f, X=X, y=y, session=meta["session"].astype(str).values)
        print(name, s, X.shape, np.bincount(y, minlength=3), flush=True)
    except Exception as e:
        print(name, s, "FAILED", type(e).__name__, str(e)[:150], flush=True)
