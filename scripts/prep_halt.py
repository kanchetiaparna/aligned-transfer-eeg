"""Kaya et al. 2018 HaLT paradigm -> 4-class set with the same classes as BCI IV 2a.
Markers: 1 left hand, 2 right hand, 3 passive, 4 left leg, 5 tongue, 6 right leg (cue ~1 s, 200 Hz).
Mapping: left hand->0, right hand->1, left/right leg->2 (feet), tongue->3; passive dropped.
Feet is subsampled at random (fixed seed) to the size of the other classes so chance stays 25%.
Epoch: 0 .. 1.5 s after cue onset, 19 EEG channels (A1, A2 references and X5 sync removed), resampled to 250 Hz (375 samples).
Split per subject (cross-session like 2a): all sessions but the last -> train, last session -> test.
Subject J has a single session and is skipped. Output: data/halt/A{01..11}.npz (+ subjects.json).
"""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, json, os, re, numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

SRC = f"{ROOT}/data/halt_raw"
OUT = f"{ROOT}/data/halt"; os.makedirs(OUT, exist_ok=True)
MAP = {1: 0, 2: 1, 4: 2, 6: 2, 5: 3}
DROP = {"A1", "A2", "X5"}
rng = np.random.RandomState(0)


def load_session(path):
    o = loadmat(path, struct_as_record=False, squeeze_me=True)["o"]
    fs = int(o.sampFreq); names = list(o.chnames)
    keep = [i for i, c in enumerate(names) if c not in DROP]
    mk = np.asarray(o.marker).astype(int); data = np.asarray(o.data, dtype=np.float64)[:, keep].T
    on = np.where((mk[1:] != mk[:-1]) & np.isin(mk[1:], list(MAP)))[0] + 1
    n = int(1.5 * fs)
    on = on[on + n <= data.shape[1]]
    X = np.stack([data[:, i:i + n] for i in on]); y = np.array([MAP[mk[i]] for i in on])
    X = resample_poly(X, 5, 4, axis=-1)                     # 200 Hz -> 250 Hz
    # balance: subsample merged feet class to the mean size of the other three classes
    tgt = int(np.mean([np.sum(y == c) for c in (0, 1, 3)]))
    feet = np.where(y == 2)[0]; drop = rng.choice(feet, max(0, len(feet) - tgt), replace=False)
    m = np.ones(len(y), bool); m[drop] = False
    return X[m].astype(np.float32), y[m], [names[i] for i in keep]


files = sorted(glob.glob(f"{SRC}/HaLT-Subject*-6St-*.mat"))
by_sub = {}
for f in files:
    sub, date = re.search(r"HaLT-Subject(\w)-(\d+)-", os.path.basename(f)).groups()
    by_sub.setdefault(sub, []).append((date, f))
meta, k = {}, 0
for sub in sorted(by_sub):
    sess = [f for _, f in sorted(by_sub[sub])]
    if len(sess) < 2:
        print(sub, "skipped (single session)"); continue
    k += 1
    tr = [load_session(f) for f in sess[:-1]]; te = load_session(sess[-1])
    Xtr = np.concatenate([t[0] for t in tr]); ytr = np.concatenate([t[1] for t in tr])
    np.savez_compressed(f"{OUT}/A{k:02d}.npz", Xtr=Xtr, ytr=ytr, Xte=te[0], yte=te[1])
    meta[k] = {"subject": sub, "train_sessions": [os.path.basename(f) for f in sess[:-1]],
               "test_session": os.path.basename(sess[-1]), "channels": te[2]}
    print(k, sub, Xtr.shape, te[0].shape, np.bincount(ytr), np.bincount(te[1]), flush=True)
json.dump(meta, open(f"{OUT}/subjects.json", "w"), indent=1)
