# Data

The datasets are not stored in this repository. The preparation scripts in `scripts/` create the files below.

| Folder / file | Made by | Source |
|---|---|---|
| `data/mne/` | `prep_data.py` (MOABB download cache, about 20 GB with the external sets) | MOABB |
| `data/A01.npz` ... `A09.npz` | `prep_data.py` | BCI Competition IV 2a via MOABB (`BNCI2014_001`) |
| `data/tcf/` | `prep_2a_tcformer.py` | 2a, 0 to 4 s after the cue, 22 channels, no filter (TCFormer protocol) |
| `data/paper/` | `prep_2a_paper.py` | 2a, ATCNet paper protocol |
| `data/halt/` | `prep_halt.py` | HaLT paradigm of Kaya et al. 2018 |
| `data/ext/` | `prep_external.py <Dataset>` | PhysionetMI, Weibo2014, Schirrmeister2017 via MOABB |

Each `.npz` holds `Xtr` (trials x channels x samples), `ytr`, `Xte`, `yte`; labels are
0 left hand, 1 right hand, 2 feet, 3 tongue.

## BCI IV 2a

`python prep_data.py` downloads it automatically with MOABB. Run it before `prep_2a_tcformer.py` and
`prep_2a_paper.py`, which read the downloaded `.mat` files from `data/mne/`.

## HaLT

Download the HaLT `.mat` files (`HaLT-Subject*-6St-*.mat`) from the figshare collection linked in
Kaya et al. 2018, Scientific Data 5:180211 (https://doi.org/10.1038/sdata.2018.211), into `data/halt_raw/`.
`prep_halt.py` then:

- maps left hand to 0, right hand to 1, left leg and right leg to 2 (feet), tongue to 3, and drops "passive";
- removes the A1, A2 and X5 channels (19 EEG channels remain);
- cuts 0 to 1.5 s after the cue and resamples from 200 to 250 Hz;
- randomly reduces the merged feet class to the size of the other classes (fixed seed), so chance stays 25%;
- uses all sessions except the last for training and the last session for testing;
- skips subject J, who has only one session (11 people remain).

## External datasets

Used only in the external-data pretraining experiment. `prep_external.py` keeps left hand, right hand and
feet trials, picks the same 22 electrodes as 2a, band-passes 0.5 to 75 Hz, resamples to 250 Hz and cuts
4 s windows. It can split one dataset over several processes: `prep_external.py PhysionetMI 0 4` etc.
