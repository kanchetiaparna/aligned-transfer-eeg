# Aligned Transfer: cross-day EEG motor imagery decoding

A training method that makes motor imagery decoders work on a new recording day. It keeps the network
unchanged (TCFormer, 77,820 weights) and changes how it is trained:

1. **Align** every recording session with Euclidean Alignment (no labels needed).
2. **Pretrain** on all other people, then **fine-tune** on the target person's day 1.
3. **Average** 5 copies, each with its own pretraining seed.

Every number below uses an honest protocol: train on day 1, keep the last epoch, test once on day 2.

| Dataset | TCFormer | TCFormer-AT (5 copies) | Real time | Better for | Wilcoxon p |
|---|---|---|---|---|---|
| BCI IV 2a (9 people) | 84.88% | **89.97%** | 89.97% | 9 / 9 | 0.004 |
| HaLT (11 people) | 61.99% | **68.15%** | 68.06% | 10 / 11 | 0.003 |

p-values compare against an equal-size TCFormer ensemble (86.50% / 62.43%). The same recipe improves
EEGNet and ATCNet on both datasets (+3.5 to +6.5 points, all 6 significant after Holm correction).

## Repository layout

```
scripts/            all code (run from inside this folder)
scripts/jobs/       the exact job lists used for every experiment
results/            every run: <experiment>/S<person>_seed<k>.json and _P.npy / _Pon.npy
figures/            report figures; figures/deck and figures/deck2 = presentation figures
checkpoints/final/  trained TCFormer-AT bundles (5 copies per person) for predict.py
data/               not in git: see data/README.md
```

## Setup

Python 3.10 and a CUDA GPU for training (analysis and prediction run on CPU).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Paths are relative to the repository root. To keep data elsewhere, set `NEUROAI_ROOT` to a folder that
contains `data/`, `results/` and `checkpoints/`.

## 1. Prepare the data

See [data/README.md](data/README.md). In short:

```bash
cd scripts
python prep_data.py              # downloads BCI IV 2a with MOABB -> data/A01..A09.npz
python prep_2a_tcformer.py       # 2a epochs as in the TCFormer code -> data/tcf/
python prep_2a_paper.py          # 2a epochs as in the ATCNet code -> data/paper/
python prep_halt.py              # HaLT (download first, see data/README.md) -> data/halt/
python prep_external.py PhysionetMI   # only for the external-data experiment -> data/ext/
python prep_external.py Weibo2014
python prep_external.py Schirrmeister2017
```

## 2. Check the results without training

All saved results are in `results/`. With `data/tcf` and `data/halt` prepared (only the test labels are read):

```bash
cd scripts
python verify_deck_numbers.py    # recomputes every number in the report and slides (36 checks)
python ensemble_div.py           # 5-copy ensembles vs single models vs TCFormer
python ensemble_rt.py            # real-time vs offline alignment
python ensemble_backbones.py     # recipe on ATCNet and EEGNet
python compare_hyper.py          # session hypernetwork (negative result)
python make_deck_stats.py        # CIs, effect sizes, Holm correction, extra plots
python make_figures.py           # report figures
python make_deck_figures.py      # EEG and alignment figures
```

## 3. Use the trained model

```python
from predict import TCFormerAT
m = TCFormerAT.load("checkpoints/final/2a_S1.pt")
labels, probs = m.predict(X)             # X: trials x channels x samples, raw EEG of one session
m.reset()                                # real time, one trial at a time
for x in stream:
    label, p = m.predict_online(x)
```

`python scripts/predict.py verify` reloads every bundle and reproduces 89.97% / 68.15% (real time 89.97% / 68.06%).

## 4. Retrain from scratch

Each line of a job file is one run. `run_jobs.sh` spreads the lines over GPU workers and skips runs that
already have results. Training one fine-tuned copy takes 9 to 16 minutes on an L40S; pretraining about 30.

```bash
cd scripts
# SCRIPT = the training script from the table below; the two numbers = workers on GPU 0 and GPU 1
SCRIPT=train_new.py bash run_jobs.sh jobs/jobs_final_realtime.txt 2 2
```

Without `SCRIPT=...` the runner uses `train.py`. Logs go to `results/log_gpu0.txt` and `log_gpu1.txt`.
Set `PYTHON=/path/to/python` if `python` is not your environment's interpreter.

| Experiment | Script | Job file(s) | Results folder(s) |
|---|---|---|---|
| TCFormer paper reproduction | `train_tcformer_paper.py` | `jobs_tcf_paper.txt` | `tcformer_paper/` |
| ATCNet paper reproduction | `train_atcnet_paper.py` | `jobs_atc_paper.txt`, `jobs_atc_keras.txt`, `jobs_atc_halt.txt` | `atcnet_paper/` |
| Architecture survey (13 configs) | `train.py` | `jobs_p1.txt`, `jobs_tcf.txt`, `jobs_tcf_sorted.txt` | `<Model>_within_*`, `TCFormer_paperrecipe` |
| External-data pretraining | `train.py` | `jobs_x_pre_2a.txt`, `jobs_x_pre_ext.txt`, `jobs_x_ft.txt` | `<Model>_ft_2a_only`, `<Model>_ft_2a+ext` |
| Ablation V1 to V5 | `train_new.py` | `jobs_new.txt` | `newmodel/V1_ea` ... `V5_full` |
| Final model, 5 separate copies | `train_new.py` | `jobs_final_5copies_2a.txt`, `jobs_final_halt.txt` | `newmodel/V4_div`, `H4_div`, `V4_final`, `H4_final` |
| Final model + real time | `train_new.py` | `jobs_final_realtime.txt` | `newmodel/V4_rt`, `H4_rt` |
| Recipe on ATCNet and EEGNet | `train_new.py` | `jobs_recipe_atcnet_eegnet.txt` | `newmodel/A_*`, `E_*`, `AH_*`, `EH_*` |
| Session hypernetwork | `train_hyper.py` | `jobs_hypernetwork.txt` | `newmodel/HY`, `HY_ctrl` |

Pretraining lines (`--stage pre`) must finish before the matching fine-tuning lines; the fine-tuning
script waits for its checkpoint, so one job file can hold both.

## Script map

| Script | Purpose |
|---|---|
| `tcformer.py` | TCFormer network, ported from the authors' code (MIT licence, notice kept in the file) |
| `train_new.py` | Aligned Transfer: alignment, pretraining, fine-tuning, real-time test (main script) |
| `predict.py` | Ready-to-use 5-copy predictor, offline and real time |
| `train_tcformer_paper.py`, `train_atcnet_paper.py` | Faithful baseline reproductions |
| `train.py`, `common.py` | Earlier pipeline: architecture survey and external-data experiment |
| `train_hyper.py`, `compare_hyper.py` | Session hypernetwork and its control |
| `prep_*.py` | Data preparation |
| `ensemble_*.py`, `summarize*.py`, `analyze_baselines.py` | Analyses on saved predictions |
| `verify_deck_numbers.py`, `make_deck_stats.py`, `make_*figures.py` | Number checks, statistics, figures |
| `run_jobs.sh`, `add_workers.sh` | Multi-GPU job queue |

## References

Datasets
- Brunner, C., Leeb, R., Muller-Putz, G. R., Schlogl, A., Pfurtscheller, G. (2008). BCI Competition 2008, Graz data set A. Graz University of Technology. https://www.bbci.de/competition/iv/desc_2a.pdf
- Kaya, M., Binli, M. K., Ozbay, E., Yanar, H., Mishchenko, Y. (2018). A large electroencephalographic motor imagery dataset for electroencephalographic brain computer interfaces. Scientific Data, 5, 180211. https://doi.org/10.1038/sdata.2018.211
- Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. Human Brain Mapping, 38(11), 5391-5420. https://doi.org/10.1002/hbm.23730 (High Gamma dataset)

Networks
- Altaheri, H., Karray, F., Karimi, A.-H. (2025). Temporal convolutional transformer for EEG based motor imagery decoding (TCFormer). Scientific Reports, 15, 32959. https://doi.org/10.1038/s41598-025-16219-7
- Altaheri, H., Muhammad, G., Alsulaiman, M. (2023). Physics-informed attention temporal convolutional network for EEG-based motor imagery classification (ATCNet). IEEE Transactions on Industrial Informatics, 19(2), 2249-2258. https://doi.org/10.1109/TII.2022.3197419
- Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., Lance, B. J. (2018). EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. Journal of Neural Engineering, 15(5), 056013. https://doi.org/10.1088/1741-2552/aace8c

Alignment and transfer
- He, H., Wu, D. (2020). Transfer learning for brain-computer interfaces: A Euclidean space data alignment approach. IEEE Transactions on Biomedical Engineering, 67(2), 399-410. https://doi.org/10.1109/TBME.2019.2913914
- Junqueira, B., Aristimunha, B., Chevallier, S., de Camargo, R. Y. (2024). A systematic evaluation of Euclidean alignment with deep learning for EEG decoding. Journal of Neural Engineering, 21(3), 036038. https://doi.org/10.1088/1741-2552/ad4f18

Statistics
- Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1(6), 80-83. https://doi.org/10.2307/3001968
- Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1), 37-46. https://doi.org/10.1177/001316446002000104

