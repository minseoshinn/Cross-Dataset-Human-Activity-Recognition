# Notebook and run logs for an anonymous submission

A model that recognizes activities from a phone's accelerometer and gyroscope loses accuracy when it is tested on a dataset other than the one it was trained on. Part of this loss comes from how the phone is carried. This repository contains the notebook of a study that measures how much of the gap between in-domain and cross-dataset macro-F1 is closed when device orientation is treated.
 
## Study design

- **Datasets:** HHAR, UCI-HAR, MotionSense and Shoaib, which give 12 directed source → target pairs.
- **Model families:** a LIMU-BERT encoder with a GRU head, XGBoost on engineered features, and a fine-tuned UniMTS model.
- **Orientation treatments:** gravity canonicalization; canonicalization with heading augmentation; canonicalization with PCA heading alignment; the vertical and horizontal decomposition of Mizell; the orientation-invariant transformation (OIT); SO(3) augmentation; learned SO(3) invariance.
- **Measure:** the fraction of the gap between in-domain and cross-dataset macro-F1 that a treatment closes, with 95% confidence intervals from a hierarchical bootstrap over datasets and seeds.
- **Runs:** 8 per pair for each encoder treatment (2 pretraining × 4 fine-tuning seeds) and 2 per pair for UniMTS.

All windows are 2.56 s long and resampled to 20 Hz, with acceleration in g, angular velocity in rad/s and the Android sign convention. Gravity canonicalization rotates each window by the smallest rotation that takes its mean acceleration to the z-axis. Heading augmentation rotates each window about that axis by a random angle during fine-tuning. Training uses no target data.

## Main results

Averages over the 12 pairs, as printed by `v5_make_all()` and `v5_report()` in the notebook.

| Model | Input | In-domain macro-F1 | Cross-dataset macro-F1 | Gap closed (95% CI) |
|---|---|---|---|---|
| LIMU-BERT encoder | Device frame | 0.905 | 0.344 | |
| | Gravity canonicalization | | 0.568 | 40% (13% to 56%) |
| | Canonicalization + heading augmentation | | 0.688 | 61% (47% to 84%) |
| XGBoost | Device frame | 0.925 | 0.570 | |
| | Gravity canonicalization | | 0.657 | 24% (−1% to 52%) |
| UniMTS | Device frame | not run | 0.480 | |
| | Gravity canonicalization | | 0.628 | |

The five other encoder treatments close between 50% and 62% of the gap. After treatment, a gap of 0.22 (encoder) to 0.27 (XGBoost) macro-F1 remains.

The datasets are public and are not distributed with the notebook.

## Sections of the notebook

| Section | Content |
|---|---|
| 1 | Setup |
| 2 | Base pipeline: loaders, LIMU-BERT encoder, GRU head, augmentation operators, engineered features |
| 3 | Experiments: datasets, orientation treatments, training, run log, stages, data checks |
| 4 | Fine-tuning check on the source, run logs with a tag, UniMTS joint |
| 5 | Stages with encoders pretrained per fold (log tag `strict`) and their runs |
| 6 | Runs of the stages of Section 3 |
| 7 | Orientation-invariant transformation: arm `I_oit2` |
| 8 | Stages of the final configuration (log tag `v4`) and their runs |
| 9 | Analysis: gap fraction, hierarchical bootstrap, equivalence tests, tables and figures |
| 10 | Additional runs (log tag `v5`): identity floor selected on the source, equal run budgets, OIT with settings selected on the source |
| 11 | Numerical checks of the formal statements |
| 12 | Revision analyses without training (tag `v6`): treated in-domain reference, order-averaged attribution, permutation reference of Kendall's W, verdicts of the reporting protocol, further numerical checks |

## Data

Download the four datasets from their sources:

| Dataset | Source |
|---|---|
| HHAR (Stisen et al., 2015) | <https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition> |
| UCI-HAR (Anguita et al., 2013) | <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones> |
| MotionSense (Malekzadeh et al., 2019) | <https://github.com/mmalekzadeh/motion-sense> |
| Shoaib (Shoaib et al., 2014) | <https://www.utwente.nl/en/eemcs/ps/research/dataset/> |

Place them under one data root:

```
<data root>/HHAR/Activity recognition exp/     Phones_accelerometer.csv, Phones_gyroscope.csv
<data root>/UCI HAR Dataset/                   train/ and test/ with "Inertial Signals"
<data root>/MotionSense/A_DeviceMotion_data/   <activity>_<trial>/sub_<k>.csv
<data root>/Shoaib/                            one CSV file per participant
```

On Google Colab the data root is `MyDrive`. Elsewhere it is taken from the environment variable `HAR_DATA_ROOT`.

## Requirements

Python 3 with NumPy, pandas, SciPy, scikit-learn, joblib, XGBoost, PyTorch and Matplotlib. The UniMTS stage clones the [UniMTS repository](https://github.com/xiyuanzh/UniMTS), installs CLIP and `huggingface_hub`, and downloads the released checkpoint.

The runs stored in the notebook were made on Google Colab with one NVIDIA T4 GPU.

## Running the notebook

1. Place the datasets under the data root.
2. On Google Colab, run the setup cell of Section 1. It mounts Google Drive, installs XGBoost and reports the GPU. Elsewhere, skip this cell and set `HAR_DATA_ROOT`.
3. Run the definition cells from top to bottom. These are the first code cells of Sections 2 to 5, 7 to 10 and 12. The order matters, because Sections 4, 5, 7 and 8 extend or replace functions of Section 3.
4. Run the stages. Each call trains the runs of one stage and appends one row per run to its log. A stage that is called again skips the runs already in its log, so an interrupted session can be continued. `RUN_STAGES` in Section 2 is `False` in this file: the stage calls then return without training, and the analysis reads the run logs that exist. Set it to `True` to train.
5. Run the analysis. It reads the logs and needs no GPU.

| Call | Content |
|---|---|
| `run_stage("audit")` | Data checks without training: MotionSense sign conversion, geometry identities, tables |
| `estimate_cost()` | Times one epoch and projects the hours of each stage |
| `run_stage("xgb")` | XGBoost on engineered features, device frame and canonicalized, 12 pairs (CPU) |
| `run_stage("ablation")`, `run_stage("site")`, `run_stage("baselines")` | Encoder treatments on the 12 pairs |
| `run_stage("ceiling")` | In-domain scores on the label set of each pair |
| `run_stage("rotation")` | Controlled rotations on the four datasets |
| `run_stage("unimts")` | UniMTS with device-frame and canonicalized input |
| `run_stage("posture")`, `run_stage("hparam")`, `run_stage("inference")` | Sitting, standing and lying; selection of settings on the source, with their sensitivity on the target; inference time |
| `run_strict(...)` | Within-dataset stages with encoders pretrained on the training subjects of each fold (Section 5) |
| `run_v4(...)`, `run_v4_all()` | Stages of the final configuration (Section 8) |
| `v5_run("floor")`, `v5_run("budget")`, `v5_run("oit")` | Additional runs (Section 10) |
| `make_all()`, `v5_make_all()` | Analysis: tables, figures and `manuscript_numbers.json` |
| `v6_all()` | Revision analyses of Section 12; no training, writes `revision/tables/v6/` |

`estimate_cost()` projected about 124 hours on the T4 GPU for the six stages it lists, of which 65 hours are for the matrix of 12 pairs. The stages of Sections 5, 8 and 10 add to this. `make_all(n_boot=300)` gives a quick check of the analysis with fewer bootstrap resamples.

## Output

All output is written to the folder `revision/`, which is created in `MyDrive` on Colab and in `<data root>/results/` elsewhere.

| Folder | Content |
|---|---|
| `revision/runs/` | Run logs, one row per training run |
| `revision/cache/` | Parsed datasets |
| `revision/enc/` | Pretrained encoders |
| `revision/tables/` | Tables and figures; `v13/` from `make_all()`, `v5/` from `v5_make_all()` |

| Log files | Stages |
|---|---|
| `xgb.csv`, `*_v2.csv` | `run_stage(...)`: cross-dataset matrix with encoders pretrained on the source dataset, UniMTS, and the within-dataset stages with encoders pretrained on all subjects of the dataset |
| `*_strict.csv` | `run_strict(...)`: within-dataset stages with encoders pretrained on the training subjects of each fold |
| `*_v4.csv` | `run_v4(...)`: stages of the final configuration |
| `*_v5.csv` | `v5_run(...)`: additional runs of Section 10 |

## Numerical checks

Section 11 writes and runs `verify_propositions.py`. The script checks Lemma 1, Corollary 1, Propositions 1 to 3 and 6 and Eqs. 3 and 9 of the manuscript numerically, together with further identities that the manuscript does not include. `v6_verify()` in Section 12 checks Corollary 2 and Propositions 3 to 5. The stored output reports that 17 of 17 checks passed.

## Notes
  
- The analysis uses the arm `I_oit2` for OIT and leaves out the runs of `I_oit`.
- The comment above `REV` in Section 3 says that none of the settings is chosen on target data. The fixed identity floor of 0.3 was first chosen with the target labels of two pairs. Section 10 selects the floor for each source dataset on held-out source subjects, and the main results use that selection (0.688 against 0.683 with the fixed floor).
