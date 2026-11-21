"""TCFormer-AT: ready-to-use predictor (5 separately pretrained + fine-tuned TCFormer copies, Euclidean Alignment).

Build the per-person bundles once (from checkpoints/newmodel/ft, no training):
    python predict.py build
Check that the bundles reproduce the reported accuracies:
    python predict.py verify

Use in your own code:
    from predict import TCFormerAT
    m = TCFormerAT.load("checkpoints/final/2a_S1.pt")
    labels, probs = m.predict(X)          # X: trials x channels x samples, raw EEG of one session
    # real time, one trial at a time (alignment uses only the trials seen so far):
    m.reset()
    for x in stream:                      # x: channels x samples
        label, p = m.predict_online(x)
"""; import os as _os; ROOT = _os.environ.get("NEUROAI_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import glob, os, sys
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tcformer import TCFormer

# ROOT (the repository root) is set on the docstring line at the top of this file
CLASSES = ["left hand", "right hand", "feet", "tongue"]
SETS = {"2a": ("tcf", "V4_rt"), "halt": ("halt", "H4_rt")}   # dataset name -> (data folder, fine-tune tag)


def _cov(x):
    return x @ x.T / x.shape[1]


def _inv_sqrt(C):
    w, V = np.linalg.eigh(C)
    return V @ np.diag(w ** -0.5) @ V.T


def _crop(X, n_times):
    return X[..., :n_times]


class TCFormerAT:
    WARMUP = 5  # first trials of a new session use the training-day reference, as in the reported real-time test

    def __init__(self, state_dicts, R_train_isq, n_channels, n_times, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_times, self.R_T = n_times, R_train_isq
        self.models = []
        for sd in state_dicts:
            m = TCFormer(n_channels=n_channels, n_classes=4, trans_depth=2)
            m.load_state_dict({k.removeprefix("tcf."): v for k, v in sd.items()})
            self.models.append(m.to(self.dev).eval())
        self.reset()

    @classmethod
    def load(cls, path, device=None):
        b = torch.load(path, map_location="cpu", weights_only=False)
        return cls(b["state_dicts"], b["R_train_isq"], b["n_channels"], b["n_times"], device)

    def reset(self):
        """Call at the start of every new session (new day / new cap placement)."""
        self._csum, self._n = None, 0

    @torch.no_grad()
    def _probs(self, X):
        x = torch.tensor(np.asarray(X, np.float32), device=self.dev)
        return np.mean([F.softmax(m(x), -1).cpu().numpy() for m in self.models], 0)

    def predict(self, X):
        """Offline: align with the whole session's reference, then average the 5 copies."""
        X = _crop(np.asarray(X, np.float64), self.n_times)
        R = _inv_sqrt(np.mean([_cov(x) for x in X], 0))
        P = self._probs(np.einsum("ij,njt->nit", R, X))
        return P.argmax(1), P

    def predict_online(self, x):
        """Real time: align this trial using only the trials seen so far in this session."""
        x = _crop(np.asarray(x, np.float64), self.n_times)
        c = _cov(x)
        self._csum = c if self._csum is None else self._csum + c
        self._n += 1
        R = self.R_T if self._n <= self.WARMUP else _inv_sqrt(self._csum / self._n)
        p = self._probs((R @ x)[None])[0]
        return int(p.argmax()), p


def build():
    out = f"{ROOT}/checkpoints/final"; os.makedirs(out, exist_ok=True)
    for name, (folder, tag) in SETS.items():
        for f in sorted(glob.glob(f"{ROOT}/data/{folder}/A*.npz")):
            s = int(os.path.basename(f)[1:3]); d = np.load(f)
            Xtr = d["Xtr"].astype(np.float64); T = Xtr.shape[-1]
            if T % 8 and T % 7: T -= T % 8                     # same crop as training (HaLT 375 -> 368)
            Xtr = Xtr[..., :T]
            sds = [torch.load(f"{ROOT}/checkpoints/newmodel/ft/{tag}_S{s}_seed{k}.pt", map_location="cpu") for k in range(5)]
            torch.save({"state_dicts": sds, "R_train_isq": _inv_sqrt(np.mean([_cov(x) for x in Xtr], 0)),
                        "n_channels": Xtr.shape[1], "n_times": T, "classes": CLASSES,
                        "info": f"TCFormer-AT {name} subject {s}: 5 copies, fine-tuned on session T"},
                       f"{out}/{name}_S{s}.pt")
            print("built", f"{out}/{name}_S{s}.pt")


def verify():
    for name, (folder, _) in SETS.items():
        off, on = [], []
        for f in sorted(glob.glob(f"{ROOT}/checkpoints/final/{name}_S*.pt"), key=lambda p: int(p.split("_S")[1][:-3])):
            s = int(f.split("_S")[1][:-3]); d = np.load(f"{ROOT}/data/{folder}/A{s:02d}.npz")
            m = TCFormerAT.load(f)
            off.append((m.predict(d["Xte"])[0] == d["yte"]).mean() * 100)
            m.reset(); pred = [m.predict_online(x)[0] for x in d["Xte"]]
            on.append((np.array(pred) == d["yte"]).mean() * 100)
            print(f"  {name} S{s}: offline {off[-1]:.1f}  real time {on[-1]:.1f}")
        print(f"{name}: offline {np.mean(off):.2f}%  real time {np.mean(on):.2f}%  ({len(off)} people)")


if __name__ == "__main__":
    {"build": build, "verify": verify}[sys.argv[1]]()
