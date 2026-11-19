A model that recognizes activities from the accelerometer and gyroscope of a phone loses accuracy when it is tested on a dataset other than its training dataset. Treating the orientation of the device recovers part of this loss. This repository contains the notebook and the run logs of a study that measures the part of the gap between in-domain and cross-dataset macro-F1 that each treatment closes.

| Item | Content |
|---|---|
| `orientation_cross_dataset_har.ipynb` | Code and stored outputs |
| `revision/runs/` | Run logs, one row per training run |

## Study design

| Aspect | Setting |
|---|---|
| Datasets | HHAR, UCI-HAR, MotionSense and Shoaib, which give 12 directed source → target pairs |
| Label sets | The four or five classes that the two datasets of a pair share: static, walk, stairs-up, stairs-down, and bike or run. Static merges sitting and standing. The stage `posture` keeps sitting, standing and lying apart. |
| Model families | LIMU-BERT encoder with a GRU head; XGBoost on engineered features; fine-tuned UniMTS model |
| Orientation treatments | Gravity canonicalization; canonicalization with heading augmentation; canonicalization with PCA heading alignment; vertical and horizontal decomposition of Mizell; orientation-invariant transformation (OIT); SO(3) augmentation; learned SO(3) invariance |
| Control | Device-frame encoder on input with the window mean of each accelerometer axis removed and no rotation (gravity removed) |
| Measure | Fraction of the gap between in-domain and cross-dataset macro-F1 that a treatment closes, with 95% confidence intervals from a hierarchical bootstrap over datasets and seeds |
| Runs | 8 per pair for each encoder treatment (2 pretraining × 4 fine-tuning seeds), 3 per pair for XGBoost, 2 per pair for UniMTS |

All windows are 2.56 s long and resampled to 20 Hz, with acceleration in g, angular velocity in rad/s and the Android sign convention. Gravity canonicalization rotates each window by the smallest rotation that takes its mean acceleration to the z-axis. Heading augmentation rotates each window about that axis by a random angle during fine-tuning. Training uses no target data.

The fraction depends on the treatment. It equals the share of the gap caused by orientation only under the conditions stated in the manuscript.

## Main results

Averages over the 12 pairs, as printed in Sections 14 and 15 of the notebook.

| Model | Input | In-domain macro-F1 | Cross-dataset macro-F1 | Gap closed (95% CI) |
|---|---|---|---|---|
| LIMU-BERT encoder | Device frame | 0.905 | 0.344 | |
| | Gravity removed (control) | | 0.467 | 22% (2% to 36%) |
| | Gravity canonicalization | | 0.568 | 40% (13% to 56%) |
| | Canonicalization + heading augmentation | | 0.688 | 61% (46% to 84%) |
| XGBoost | Device frame | 0.925 | 0.570 | |
| | Gravity canonicalization | | 0.657 | 24% (−1% to 52%) |
| UniMTS | Device frame | not available | 0.480 | |
| | Gravity canonicalization | | 0.628 | |

- When the in-domain reference receives the same treatment, the two encoder fractions are 36% and 54%.
- The five other encoder treatments close between 48% and 62% of the gap. OIT reaches 0.690. Its difference from canonicalization with heading augmentation is inconclusive.
- After treatment, a gap of 0.22 (encoder) to 0.27 (XGBoost) macro-F1 remains.
- Canonicalization removes tilt information that separates static postures. With sitting, standing and lying as separate classes, canonicalization with heading augmentation reaches a mean in-domain macro-F1 of 0.677, and 0.873 when the estimated gravity direction is given to the classifier head.

## Sections of the notebook

| Section | Content |
|---|---|
| 1 | Setup |
| 2 | Base pipeline: loaders, LIMU-BERT encoder, GRU head, augmentation operators, engineered features |
| 3 | Experiments: datasets, orientation treatments, training, run log, stages, data checks |
| 4 | Fine-tuning check on the source, run logs, UniMTS joint |
| 5 | Stages with encoders pretrained per fold, and their runs |
| 6 | Runs of the stages of Section 3 |
| 7 | Orientation-invariant transformation: arm `I_oit2` |
| 8 | Stages of the final configuration, and their runs |
| 9 | Analysis: gap fraction, hierarchical bootstrap, equivalence tests, tables and figures |
| 10 | Additional runs: identity floor of the final arm selected on the source, equal run budgets, OIT with settings selected on the source, UniMTS |
| 11 | Numerical checks of the formal statements |
| 12 | Treated in-domain reference, order-averaged attribution, permutation reference of Kendall's W, verdicts of the reporting protocol, further numerical checks |
| 13 | SO(3) treatments with the identity floor selected on the source |
| 14 | Tables and figures of the manuscript |
| 15 | Gravity-removed control and tuning of the device-frame encoder |

The manuscript reports the values of Sections 14 and 15. The tables printed in Sections 9, 10 and 12 use the SO(3) runs with an identity floor of 0.3 on every pair.

## Data

The four datasets are public and are not distributed with the notebook.

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

On Google Colab the data root is `MyDrive`. Elsewhere it is taken from the environment variable `HAR_DATA_ROOT` (default `~/har_data`).

## Requirements

Python 3 with NumPy, pandas, SciPy, scikit-learn, joblib, XGBoost, PyTorch and Matplotlib. The UniMTS stage clones the [UniMTS repository](https://github.com/xiyuanzh/UniMTS), installs CLIP and `huggingface_hub`, and downloads the released checkpoint.

The runs in the logs were made on Google Colab with one NVIDIA T4 GPU.

## Output and run logs

All output is written to the folder `revision/`, which is created in `MyDrive` on Colab and in `<data root>/results/` elsewhere. Copy `revision/runs/` of this repository to that folder before running the analysis.

| Folder | Content |
|---|---|
| `revision/runs/` | Run logs |
| `revision/cache/` | Parsed datasets |
| `revision/enc/` | Pretrained encoders |
| `revision/tables/` | Tables and figures |

| Log files | Written by | Content |
|---|---|---|
| `xgb.csv`, `*_v2.csv` | `run_stage(...)` | Cross-dataset matrix with encoders pretrained on the source dataset, UniMTS, and the within-dataset stages with encoders pretrained on all subjects of the dataset |
| `*_strict.csv` | `run_strict(...)` | Within-dataset stages with encoders pretrained on the training subjects of each fold |
| `*_v4.csv` | `run_v4(...)` | Stages of the final configuration |
| `*_v5.csv` | `v5_run(...)` | Additional runs of Section 10 |
| `*_so3source.csv` | `so3_tune()`, `so3_run_final()` | Selection runs and 12-pair runs of the SO(3) treatments (Section 13) |
| `*_v7.csv` | `v7_run(...)` | Gravity-removed control and tuning of the device-frame encoder (Section 15) |

Other files in `revision/runs/` are not read by the analysis.

| Tables and figures | Written by |
|---|---|
| `revision/tables/so3source/paper_joint/` | `so3s_make_all()` (Section 14) |
| `revision/tables/so3source/` | `so3_tune()`, `so3_report()` (Section 13) |
| `revision/tables/v13/` | `make_all()` (Section 9) |
| `revision/tables/v5/` | `v5_make_all()` (Section 10) |
| `revision/tables/v6/` | `v6_all()` (Section 12) |

## Running the notebook

1. Place the datasets under the data root and the run logs in `revision/runs/`.
2. On Google Colab, run the setup cell of Section 1. It mounts Google Drive, installs XGBoost and reports the GPU. Elsewhere, skip this cell and set `HAR_DATA_ROOT`.
3. Run the cells from top to bottom. The first code cell of each of Sections 2 to 5, 7 to 10 and 12 to 15 contains definitions. Sections 4, 5, 7, 8, 14 and 15 extend or replace functions of earlier sections.

`RUN_STAGES` in Section 2 is `False`. The stage calls of Sections 5 to 10 then return at once, and the analysis reads the run logs. With `RUN_STAGES = True`, each call trains the runs of one stage and appends one row per run to its log. A stage that is called again skips the runs already in its log, so an interrupted session can be continued.

The stages of Sections 13 and 15 do not use `RUN_STAGES`. They need a GPU, skip the runs already in their logs and train the missing ones. Section 14 trains nothing. With the logs, it regenerates Tables 3 to 6, Fig. 4 and Supplementary Tables S1, S2 and S5 of the manuscript.

| Call | Content |
|---|---|
| `run_stage("audit")` | Data checks and dataset statistics |
| `estimate_cost()` | Times one epoch and projects the hours of each stage |
| `run_stage("xgb")` | XGBoost on engineered features, device frame and canonicalized, 12 pairs (CPU) |
| `run_stage("ablation")`, `run_stage("site")`, `run_stage("baselines")` | Encoder treatments on the 12 pairs |
| `run_stage("ceiling")` | In-domain scores on the label set of each pair |
| `run_stage("rotation")` | Controlled rotations on the four datasets |
| `run_stage("unimts")` | UniMTS with device-frame and canonicalized input |
| `run_stage("posture")` | Sitting, standing and lying as separate classes |
| `run_stage("hparam")` | Selection of settings on the source, and their sensitivity on the target |
| `run_stage("inference")` | Inference time |
| `run_strict(...)` | Stages `rotation`, `ceiling`, `positions` and `labeleff` with encoders pretrained per fold (Section 5) |
| `run_v4(...)`, `run_v4_all()` | Stages of the final configuration (Section 8) |
| `v5_run("floor")`, `v5_run("budget")`, `v5_run("oit")` | Additional runs (Section 10) |
| `v5_plan()`, `v5_lodo()`, `v5_fig3()` | Remaining runs with a time estimate; fractions with one dataset removed; residual-heading figure |
| `make_all()`, `v5_make_all()` | Tables and figures of Sections 9 and 10 |
| `v6_all()` | Analyses of Section 12 |
| `so3_tune()`, `so3_plan()`, `so3_run_final()`, `so3_report()` | Floor selection and 12-pair runs of the SO(3) treatments (Section 13) |
| `so3s_make_all(joint=True)` | Tables and figures of the manuscript (Section 14) |
| `v7_run("demean")` | Gravity-removed control on the 12 pairs (Section 15) |
| `v7_run("parity_val")`, `v7_parity_selected()`, `v7_run("parity")` | Tuning of the device-frame encoder on the source (Section 15) |
| `v7_summary()` | Results of Section 15 |

`make_all(n_boot=300)` runs the analysis of Section 9 with fewer bootstrap resamples. The stored outputs report 2.4 h and 7.3 h for the two training stages of Section 13, and 3.4 h and 1.6 h for the stages `demean` and `parity_val` of Section 15.

## Names of the treatments in the logs

| Name | Treatment |
|---|---|
| `A_ssl` | LIMU-BERT encoder, device frame |
| `A_demean` | Device frame, gravity removed (control) |
| `A_ssl_tuned` | Device frame, with input scaling and learning rate selected on the source |
| `B_grav` | Gravity canonicalization |
| `C_yawft` | Heading augmentation without canonicalization |
| `D_ft` | Canonicalization with heading augmentation |
| `E_so3aug` | SO(3) augmentation |
| `F_so3inv` | Learned SO(3) invariance |
| `G_pca` | Canonicalization with PCA heading alignment |
| `H_mizell` | Vertical and horizontal decomposition of Mizell |
| `I_oit2` | OIT |
| `XGB_raw`, `XGB_canon` | XGBoost, device frame and canonicalized |
| `UniMTS_raw`, `UniMTS_canon` | UniMTS, device frame and canonicalized |

## Numerical checks

Section 11 writes and runs `verify_propositions.py`. The script checks Lemma 1, Corollary 1, Propositions 1 to 3 and 6 and Eqs. 3 and 9 of the manuscript. `v6_verify()` in Section 12 checks Eq. 3, Corollary 2 and Propositions 3 to 5. Its stored output reports that 7 of 7 checks passed.

## Details of the analysis

**Bootstrap.** Pairs with the same source dataset and label set are evaluated with the same trained models. The bootstrap of Section 14 therefore draws the seeds once per source, label set and treatment in each resample.

**OIT.** The analysis uses the arm `I_oit2` and leaves out the runs of `I_oit`. The runs with settings selected on the source are logged as `I_oit3` and enter the analysis as `I_oit2`. The runs with the default settings are available as `I_oit2_default`.

**Identity floor.** During fine-tuning with rotation augmentation, a batch passes unrotated with a probability called the identity floor. For canonicalization with heading augmentation and for the two SO(3) treatments, the floor is selected for each source dataset on held-out source subjects (Sections 10 and 13). `D_ft`, `E_so3aug` and `F_so3inv` are the runs with the selected floor. `D_ft_fixed`, `E_so3aug_fixed` and `F_so3inv_fixed` are the runs with a floor of 0.3 on every pair. The floor of 0.3 was first chosen with the target labels of two pairs. It remains in the heading-only control, the consistency objective, the controlled rotations, the transfer between body positions, the posture stage, the augmentation-specificity stage and the treated in-domain model.

**Fine-tuning check.** A fine-tuning run whose macro-F1 on its own source training windows is below 0.5 is repeated with a new seed (seed + 1000), with at most three attempts. If all attempts stay below 0.5, the attempt with the highest source score is kept. The field `extra` of each log row records the number of attempts (`ft_attempts`) and the source score (`src_f1`, or `source_train_f1` in the logs of Section 13).

**UniMTS.** UniMTS has no in-domain score, so its gain is reported without a fraction. The in-domain stage of Section 10 has 19 of its 24 runs. Its log is in `revision/runs/partial/` and is not read by the analysis.
