A model that recognizes activities from a phone's accelerometer and gyroscope loses accuracy when it is tested on a dataset other than the one it was trained on. Part of this loss is recovered when the orientation of the device is treated. This repository contains the notebook and the run logs of a study that measures how much of the gap between in-domain and cross-dataset macro-F1 is closed by such a treatment.

## Study design

- **Datasets:** HHAR, UCI-HAR, MotionSense and Shoaib, which give 12 directed source → target pairs.
- **Label sets:** each pair uses the four or five classes that its two datasets share (static, walk, stairs-up, stairs-down, and bike or run). Static merges sitting and standing. The stage `posture` keeps sitting, standing and lying apart.
- **Model families:** a LIMU-BERT encoder with a GRU head, XGBoost on engineered features, and a fine-tuned UniMTS model.
- **Orientation treatments:** gravity canonicalization; canonicalization with heading augmentation; canonicalization with PCA heading alignment; the vertical and horizontal decomposition of Mizell; the orientation-invariant transformation (OIT); SO(3) augmentation; learned SO(3) invariance.
- **Measure:** the fraction of the gap between in-domain and cross-dataset macro-F1 that a treatment closes, with 95% confidence intervals from a hierarchical bootstrap over datasets and seeds. The fraction is a measured quantity that depends on the treatment. It equals the share of the gap caused by orientation only under the conditions stated in the manuscript.
- **Runs:** 8 per pair for each encoder treatment (2 pretraining × 4 fine-tuning seeds), 3 per pair for XGBoost and 2 per pair for UniMTS.

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

- When the in-domain reference receives the same treatment, the two encoder fractions are 36% and 54%.
- The five other encoder treatments close between 50% and 62% of the gap. OIT reaches 0.690, and its difference from canonicalization with heading augmentation is inconclusive.
- After treatment, a gap of 0.22 (encoder) to 0.27 (XGBoost) macro-F1 remains.
- Canonicalization removes tilt information that separates static postures. With sitting, standing and lying as separate classes, canonicalization with heading augmentation reaches a mean in-domain macro-F1 of 0.677, and 0.873 when the estimated gravity direction is given to the classifier head.

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
| 12 | Analyses from the run logs (tag `v6`): treated in-domain reference, order-averaged attribution, permutation reference of Kendall's W, verdicts of the reporting protocol, further numerical checks |

## Data

The four datasets are public and are not distributed with the notebook. Download them from their sources:

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

The runs stored in the logs were made on Google Colab with one NVIDIA T4 GPU.

## Output and run logs

All output is written to the folder `revision/`, which is created in `MyDrive` on Colab and in `<data root>/results/` elsewhere. The run logs of the study are in `revision/runs/` of this repository. Copy this folder to the same place in the output folder before running the analysis.

| Folder | Content |
|---|---|
| `revision/runs/` | Run logs, one row per training run |
| `revision/cache/` | Parsed datasets |
| `revision/enc/` | Pretrained encoders |
| `revision/tables/` | Tables and figures: `v13/` from `make_all()`, `v5/` from `v5_make_all()`, `v6/` from `v6_all()` |

| Log files | Stages |
|---|---|
| `xgb.csv`, `*_v2.csv` | `run_stage(...)`: cross-dataset matrix with encoders pretrained on the source dataset, UniMTS, and the within-dataset stages with encoders pretrained on all subjects of the dataset |
| `*_strict.csv` | `run_strict(...)`: within-dataset stages with encoders pretrained on the training subjects of each fold |
| `*_v4.csv` | `run_v4(...)`: stages of the final configuration |
| `*_v5.csv` | `v5_run(...)`: additional runs of Section 10 |

## Running the notebook

1. Place the datasets under the data root and the run logs in `revision/runs/`.
2. On Google Colab, run the setup cell of Section 1. It mounts Google Drive, installs XGBoost and reports the GPU. Elsewhere, skip this cell and set `HAR_DATA_ROOT`.
3. Run the definition cells from top to bottom. These are the first code cells of Sections 2 to 5, 7 to 10 and 12. The order matters, because Sections 4, 5, 7 and 8 extend or replace functions of Section 3.
4. Run the stages. `RUN_STAGES` in Section 2 is `False`: each stage call then returns at once, and the analysis reads the run logs. With `RUN_STAGES = True`, each call trains the runs of one stage and appends one row per run to its log. A stage that is called again skips the runs already in its log, so an interrupted session can be continued.
5. Run the analysis.

| Call | Content |
|---|---|
| `run_stage("audit")` | Data checks and dataset statistics: MotionSense sign conversion, geometry identities, tables |
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
| `v5_plan()`, `v5_lodo()`, `v5_fig3()` | Remaining runs with a time estimate; fractions with one dataset removed; residual-heading figure. `v5_plan(cost=False)` lists the runs without loading the datasets |
| `make_all()`, `v5_make_all()` | Analysis: tables, figures and `manuscript_numbers.json` |
| `v6_all()` | Analyses of Section 12 from the run logs; writes `revision/tables/v6/` |

`estimate_cost()` projected about 124 hours on the T4 GPU for the six stages it lists, of which 65 hours are for the matrix of 12 pairs. The stages of Sections 5, 8 and 10 add to this. `make_all(n_boot=300)` gives a quick check of the analysis with fewer bootstrap resamples.

## Results of the manuscript

The analysis uses fixed seeds for the bootstrap, so the same logs give the same tables.

| Result | Call | Output in `revision/tables/` | Reads |
|---|---|---|---|
| Table 3: gap, fraction closed, remaining gap | `v5_make_all()` | `v5/T1_gap_fraction_summary.csv` | run logs |
| Table 4 and Supplementary Table S1: differences and equivalence | `v5_make_all()`, `v6_contrasts()` | `v5/T3_differences_equivalence.csv`, `v6/T_contrasts_v6.csv` | run logs |
| Tables 5 and 6, Supplementary Table S5: macro-F1 and stairs-F1 per pair | `v5_make_all()` | `v5/T9_main_matrix.csv` | run logs |
| Supplementary Table S2: 2 × 2 ablation | `v5_make_all()` | `v5/T4_ablation_final.csv` | run logs |
| Fig. 4: agreement of the treatments | `v5_make_all()` | `v5/fig_agreement.png` | run logs |
| Repeated and failed fine-tuning runs | `v5_make_all()` | `v5/T10_failure_check.csv` | run logs |
| Fractions with one dataset removed | `v5_lodo()` | `v5/lodo_fraction.csv` | run logs |
| Treated in-domain reference | `v6_did()` | `v6/T_did_v6.csv` | run logs |
| Kendall's W and its permutation reference | `v6_agreement()` | `v6/agreement_v6.json` | run logs |
| Order-averaged attribution | `v6_shapley()` | `v6/T_shapley_v6.csv` | run logs |
| Fig. 3: residual heading | `v5_fig3()` | figure | rotation logs, or the datasets |
| Table 1: dataset statistics and device pose | `preprocessing_table()`, `antipode_report()`, `rotation_axis_decomposition()`, `heading_profile_v2()` | `c45_preprocessing.csv`, `c38_antipode.csv`, `rotation_axis_decomposition.csv`, `c32_heading_profile.csv` | datasets |
| Table 2: shared classes | `pair_label_table()` | `c44_pair_labels_merged.csv` | datasets |
| Supplementary Table S4: walking statistics | `cadence_energy_table()` | `c66_cadence_energy.csv` | datasets |

The functions for Table 1, Table 2 and Supplementary Table S4 belong to the stage `audit`. With `RUN_STAGES = False`, call them directly.

## Numerical checks

Section 11 writes and runs `verify_propositions.py`. The script checks Lemma 1, Corollary 1, Propositions 1 to 3 and 6 and Eqs. 3 and 9 of the manuscript numerically. `v6_verify()` in Section 12 checks Corollary 2 and Propositions 3 to 5. The stored output reports that 17 of 17 checks passed.

## Notes

- **OIT.** The analysis uses the arm `I_oit2` and leaves out the runs of `I_oit`.
- **Identity floor.** The fixed identity floor of 0.3 (`REV["floor"]`) was first chosen with the target labels of two pairs. Section 10 selects the floor for each source dataset on held-out source subjects, and the main results use that selection (0.688, against 0.683 with the fixed floor). In the analysis, `D_ft` is canonicalization with heading augmentation with the selected floor (runs `D_ft_sel` in `main_v5.csv` where the selected floor differs from 0.3), and `D_ft_fixed` is the same treatment with the fixed floor on every pair. The fixed floor remains in the SO(3) treatments, the heading-only control, the consistency objective, the controlled rotations, the transfer between body positions, the posture stage, the augmentation-specificity stage and the treated in-domain model.
- **Fine-tuning check.** A fine-tuning run whose macro-F1 on its own source training windows is below 0.5 is repeated with a new seed (seed + 1000), with at most three attempts. If all attempts stay below 0.5, the attempt with the highest source score is kept. The field `extra` of each log row records the number of attempts (`ft_attempts`) and the source score (`src_f1`).
