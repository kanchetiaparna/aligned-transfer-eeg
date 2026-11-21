"""BCI IV 2a epochs as in the official TCFormer code (utils/load_bcic4.py with configs/tcformer.yaml):
braindecode windows from the cue events with start=0, stop=0 -> 0 .. 4 s after the cue (1000 samples at 250 Hz),
22 EEG channels in microvolts, no filtering (low_cut/high_cut = null), all 288 trials per session.
The cue is 2 s after each trial start, so the window is samples [start+500, start+1500).
Reads the same cached original .mat files as prep_2a_paper.py. Output: data/tcf/A0{s}.npz."""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import os, numpy as np
from scipy.io import loadmat

SRC = f"{ROOT}/data/mne/MNE-bnci-data/~bci/database/001-2014"
OUT = f"{ROOT}/data/tcf"; os.makedirs(OUT, exist_ok=True)


def load(path):
    X, y = [], []
    for r in loadmat(path, struct_as_record=False, squeeze_me=True)["data"]:
        trial = np.atleast_1d(r.trial)
        if trial.size == 0: continue
        for k, start in enumerate(trial):
            X.append(r.X[int(start) + 500:int(start) + 1500, :22].T); y.append(int(np.atleast_1d(r.y)[k]) - 1)
    return np.stack(X).astype(np.float32), np.array(y)


for s in range(1, 10):
    Xtr, ytr = load(f"{SRC}/A{s:02d}T.mat"); Xte, yte = load(f"{SRC}/A{s:02d}E.mat")
    np.savez_compressed(f"{OUT}/A{s:02d}.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print(s, Xtr.shape, Xte.shape, np.bincount(ytr), np.bincount(yte), flush=True)
