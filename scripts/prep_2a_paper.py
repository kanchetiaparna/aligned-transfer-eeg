"""BCI IV 2a epochs exactly as in the official ATCNet code (preprocess.py: load_BCI2a_data, all_trials=True):
for every run with trials, take 7 s from each trial start index, keep samples 1.5 s .. 6 s
(= -0.5 s .. +4 s around the cue, 1125 samples at 250 Hz), first 22 columns (EEG, EOG dropped),
all trials kept (artifact-marked ones included), labels 1..4 -> 0..3, no extra filtering.
Reads the cached original .mat files (same files the ATCNet code reads).
Output: data/paper/A0{s}.npz with Xtr, ytr (session T) and Xte, yte (session E), in microvolts."""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import os, numpy as np
from scipy.io import loadmat

SRC = f"{ROOT}/data/mne/MNE-bnci-data/~bci/database/001-2014"
OUT = f"{ROOT}/data/paper"; os.makedirs(OUT, exist_ok=True)
fs, win = 250, 7 * 250
t1, t2 = int(1.5 * fs), int(6 * fs)


def load(path):
    runs = loadmat(path, struct_as_record=False, squeeze_me=True)["data"]
    X, y = [], []
    for r in runs:
        trial = np.atleast_1d(r.trial)
        if trial.size == 0:
            continue  # EOG / rest calibration runs have no trials
        for k, start in enumerate(trial):
            seg = r.X[int(start):int(start) + win, :22].T   # same indexing as the official code
            X.append(seg[:, t1:t2]); y.append(int(np.atleast_1d(r.y)[k]) - 1)
    return np.stack(X).astype(np.float32), np.array(y)


for s in range(1, 10):
    Xtr, ytr = load(f"{SRC}/A{s:02d}T.mat")
    Xte, yte = load(f"{SRC}/A{s:02d}E.mat")
    np.savez_compressed(f"{OUT}/A{s:02d}.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print(s, Xtr.shape, Xte.shape, np.bincount(ytr), np.bincount(yte), "nan:", int(np.isnan(Xtr).sum() + np.isnan(Xte).sum()), flush=True)
