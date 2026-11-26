# PersonGuard

Code for the paper "Person, Not Window: Auditable Differentially Private Release of ECG Spectra under Public-Reference Mismatch".

PersonGuard releases the mean ECG power spectrum of a private cohort with differential privacy at the level of the person. Only the difference from a public reference spectrum is released, after projection onto a small smooth basis. A public gate and an optional certificate decide whether the release is worth publishing.

## Layout

```
src/smdp/                 spectra, bases, Gaussian mechanism, accounting, data readers
pg/run_pg.py              benchmark (utility, attacks, window audit)
pg/reliability.py         basis selection, public gate, certificate
pg/attacks.py             membership inference, canary audit, linkage
scripts/prepare_data.py   download, MIMIC labels, feature extraction
analysis/summarize.py     tables and figures
configs/pg.yaml           settings used in the paper
configs/pg_cert_replication.yaml   certificate replication (seed 7777)
```

## Setup

Python 3.10 or later.

```
pip install -r requirements.txt
```

Downloads also need `aria2c`.

## Data

All data are public on PhysioNet.

| Name | Dataset | Link |
|---|---|---|
| `ptbxl` | PTB-XL 1.0.3 | https://physionet.org/content/ptb-xl/1.0.3/ |
| `ecg_arrhythmia` | Chapman-Shaoxing and Ningbo 1.0.0 | https://physionet.org/content/ecg-arrhythmia/1.0.0/ |
| `cinc2021` | PhysioNet/CinC Challenge 2021, 1.0.3 (Georgia, CPSC 2018) | https://physionet.org/content/challenge-2021/1.0.3/ |
| `icentia11k` | Icentia11k 1.0 | https://physionet.org/content/icentia11k-continuous-ecg/1.0/ |
| `mimic_ecg` | MIMIC-IV-ECG 1.0 | https://physionet.org/content/mimic-iv-ecg/1.0/ |

```
python scripts/prepare_data.py download data ptbxl ecg_arrhythmia icentia11k cinc2021
python scripts/prepare_data.py download data mimic_ecg mimic_ecg_meta
python scripts/prepare_data.py labels --ecg data/mimic_ecg
```

MIMIC-IV-ECG is a 33.8 GB zip. Rerunning a download resumes it and checks the files against PhysioNet's checksums.

## Features

For each dataset:

```
python scripts/prepare_data.py featurize --dataset ptbxl --root data/ptbxl --workers 8
python scripts/prepare_data.py featurize --dataset ptbxl --merge
```

For MIMIC, pass the labels with `--iter-kwargs '{"labels_csv":"mimic_labels.csv"}'`. Large datasets can be split with `--shard i --num-shards N` and merged afterwards.

## Experiments

```
python pg/run_pg.py --config configs/pg.yaml --part utility
python pg/run_pg.py --config configs/pg.yaml --part attacks
python pg/run_pg.py --config configs/pg.yaml --part window_audit
python analysis/summarize.py --config configs/pg.yaml
```

One scenario or window cap can be run with `--scenario` and `--cap`. Results are written to `results/pg`, tables and figures to `results/pg/paper`. The certificate replication uses `configs/pg_cert_replication.yaml` with `--part utility --scenario in_dist_ptbxl --cap 64`.

The fixed seeds are for reproducing the experiments. A real release needs fresh randomness.
