"""Download BCI Competition IV 2a via MOABB and cache epochs as numpy.
Protocol: 22 EEG channels, 250 Hz, cue onset 0..4 s (1000 samples), 4 classes.
Session '0train' = training session T, '1test' = evaluation session E."""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import os, numpy as np, mne
os.environ.setdefault("MNE_DATA", f"{ROOT}/data/mne")
mne.set_config("MNE_DATA", f"{ROOT}/data/mne", set_env=True)
mne.set_log_level("ERROR")
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

ds = BNCI2014_001()
par = MotorImagery(n_classes=4, fmin=0.5, fmax=100, tmin=0.0, tmax=4.0 - 1/250, resample=250)
lab = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
for s in ds.subject_list:
    X, y, meta = par.get_data(ds, subjects=[s])
    y = np.array([lab[v] for v in y])
    sess = meta["session"].values
    tr, te = sess == "0train", sess == "1test"
    np.savez_compressed(f"{ROOT}/data/A{s:02d}.npz", Xtr=X[tr].astype(np.float32), ytr=y[tr],
                        Xte=X[te].astype(np.float32), yte=y[te])
    print(s, X[tr].shape, X[te].shape, np.bincount(y[tr]), flush=True)
