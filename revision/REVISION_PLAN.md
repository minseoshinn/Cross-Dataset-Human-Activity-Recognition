# Revision plan: draft 2 → draft 3

This plan covers the 71 comments in `GraviHAR_comments_2.docx`, the advisor's e-mail (missing baselines, unclear contribution, prior gravity-based work, page limit, references), and the research questions of the earlier e-mail (RQ1–RQ4). For each item it gives the experiment or text change that resolves it, the code that produces the evidence, and the order in which to run everything.

Code in this folder:

| File | Role |
|---|---|
| `pipeline_base.py` | Verbatim copy of notebook cell 20 (loaders, LIMU-BERT, GRU head, operators, XGBoost features). |
| `revision_experiments.py` | Corrected data layer, one pretraining and one fine-tuning path for all arms, every new experiment, and analyses that need no training. Each run is appended to `SAVE_PATH/revision/runs/*.csv`. |
| `revision_analysis.py` | Builds every table, figure, test and headline number from those CSVs. Runs without a GPU. |
| `GraviHAR_Revision_Colab.ipynb` | **Single file for Colab.** Contains the three modules above plus one cell per stage. All stages resume after a disconnect. |
| `build_colab_notebook.py` | Rebuilds the single notebook from the three modules after any edit. |
| `smoke_test.py` | Runs all 19 stages on synthetic data on a CPU in about 3 minutes, from the modules or (`--notebook`) from the single notebook. Both pass. |

---

## 1. Findings from the code review that change the paper before any new run

**1.1 MotionSense is loaded with the wrong sign (C33).**

- Core Motion reports gravity as (0, 0, −1) g for a phone lying face up. Android reports +9.81 m/s² on z.
- The loader (`load_motionsense_raw`) adds `gravity + userAcceleration` without negating it. All 6 MotionSense pairs therefore mix sign conventions.
- In the raw device frame, the whole accelerometer of MotionSense is negated relative to the other three datasets. This inflates the orientation gap that the raw-frame baselines face, and with it the measured gain of canonicalization.
- After canonicalization, the vertical channel is unaffected but the horizontal plane is mirrored relative to the gyroscope. This is an improper transform that no yaw rotation can undo. The smoke test confirms it: the determinant of the horizontal map is −1.
- The reported "65.1 % inverted" becomes about 34.9 % after conversion.

The new loader converts to the Android convention (`REV["ios_sign_fix"] = True`). `audit_motionsense_sign()` produces the before/after table for the response letter. Every MotionSense number in the paper must come from the re-run.

**1.2 "Pitch, roll and yaw" in the draft are rotations about device axes, not physical ones (C37, C50, C51).**

- The physical meaning of a device-axis rotation depends on the pose. On UCI-HAR, gravity lies along device x, so the draft's "pitch" (about device x) is a physical heading change. This is why 90° of "pitch" leaves 100.9° of heading.
- The yaw column of Fig. 7 and Table 1 is a mathematical identity, not a measurement (Lemma L1 below). The same holds for the "0.0° residual tilt" verification (C19, C36).
- Canonicalization does not make the signal invariant to pitch and roll. It reduces the 3-DOF nuisance to a 1-DOF heading nuisance (Proposition 1). A physical tilt is converted into a heading whose size depends on the device pose (Lemma L3). On synthetic poses, a pure 45° tilt leaves a median residual heading of 4–6° for poses within 30° of screen-up and 38–110° for poses beyond horizontal.

The corrected statement gives a cleaner reason why both components are needed than the interaction correlation (C58, C59, C68): yaw invariance covers the entire residual, whether it comes from a heading change or from a tilt. The new rotation experiment (E5) tests this directly.

**1.3 The consistency weight was chosen on a target pair (C40).**

- `run_consistency_stability(source="uci", target="hhar")` selects λ by target macro-F1. This contradicts the domain-generalization claim.
- The text describes a 10-epoch ramp, but the runner cell set `CONSISTENCY_SETTING = {"lambda": 0.1, "warmup": 0}` before E2. The cached encoder names (`_w0_` or `_w10_`) show which was used.
- E7 replaces the selection with a source-only criterion and reports the full grid on the targets.

**1.4 Facts that answer comments directly.**

- UCI-HAR has 8,355 windows because the 1,944 LAYING windows are excluded: 10,299 − 1,944 = 8,355 (C45).
- The draft's 66,968 parameters split into a 57,672-parameter LIMU-BERT encoder and a 9,296-parameter GRU head for 6 classes (9,274 for 4 classes). A 5,838-parameter decoder is used only during pretraining (C42).
- XGBoost stairs-F1 already exists for all 12 pairs in the notebook (cell 66). The averages are 0.459 for the raw frame and 0.602 for the canonicalized frame. The draft's stairs percentages compared 12 pairs with 2–3 pairs (C25, C30, C52, C54, C73).

**1.5 The current evidence against the strongest baseline is weak, and the paper should say so (C26, C62–C64).**

- Canonicalized XGBoost reaches 0.670 macro-F1 and 0.602 stairs-F1. GraviHAR reaches 0.697 and 0.616.
- The macro-F1 difference is +0.027, positive on 8 of 12 pairs, Wilcoxon p = 0.30.
- Canonicalized LIMU-BERT without yaw invariance (0.574) is below raw XGBoost (0.577).
- The defensible central result is the advisor's suggestion in C64: gravity canonicalization accounts for most of the cross-dataset gain in both model families. The encoder adds a smaller, uncertain margin.
- E3 (in-domain ceilings) turns this into the paper's main analysis: what share of the cross-dataset gap is due to orientation.

**1.6 Two numbers already disagree between the notebook and the draft (C41, C61).**

- The heading predictor gives ρ = 0.538, p = 0.035 in the notebook output but ρ = 0.518, p = 0.042 in the draft.
- The "no consistency" setting gives 0.644 in Sec. 3.4 and 0.647 in Fig. 11.

All numbers in draft 3 must come from `revision_analysis.py` (`tables/headline_numbers.json`, `tables/paper_tables.md`).

---

## 2. Positioning after the literature check (e-mail: prior gravity-based work, unclear contribution)

**Established prior work**

Gravity-based orientation normalization predates this work:

- Mizell (ISWC 2003) estimates gravity as the window mean and separates vertical and horizontal components.
- Yurtman & Barshan (Sensors 2017) remove absolute sensor orientation with orientation-invariant transformations.
- Yurtman, Barshan & Fidan (Sensors 2018) use differential rotations represented by quaternions.
- Gil-Martín et al. (Sensors 2023) estimate gravity and the forward direction to build a consistent reference frame. This is the closest prior method.
- TRI-HAR (Baek et al., ISWC 2026) builds full rotation invariance into a multi-IMU model.
- UniMTS (NeurIPS 2024) learns rotation invariance through augmentation.

The draft cannot claim gravity alignment, or the idea of treating orientation analytically, as new. C18's "first work" must go.

**Contributions that hold up, provided the experiments confirm them**

1. **A degree-of-freedom account of the orientation component of cross-dataset shift.** Canonicalization reduces the SO(3) nuisance to one pose-dependent heading angle (Proposition 1, Lemmas L1–L3). This gives testable predictions about which component repairs which perturbation. E5 tests them with physically defined tilt and heading rotations on all four datasets.
2. **A controlled comparison of how to divide the work between physics and learning.** The same encoder, data, budget and protocol are used for every arm (E1, E2):
   - Fully analytic invariance discards the horizontal direction (Mizell, OIT).
   - Analytic heading estimation is unreliable for static windows (gravity + PCA heading, as in Gil-Martín et al.).
   - Learned SO(3) invariance asks the network to learn two analytically available degrees of freedom (SO(3) augmentation or consistency without canonicalization; UniMTS).
   - Analytic tilt removal plus learned heading invariance is the proposed method.
3. **A measurement of how much of the cross-dataset gap is due to orientation** for two model families, under a strict no-target-data protocol, together with where it does not help: posture classes (E6) and device dynamics on MotionSense (E3, cadence and energy table).

The contribution should be written as these three findings, not as a new architecture. This matches both e-mails: the central question is scientific, and the model stays simple.

**Title candidates (C23)**

- "How Much of Cross-Dataset Shift in IMU-Based Activity Recognition Is Device Orientation? Physics-Guided Canonicalization and Learned Heading Invariance"
- "Canonicalize What Gravity Determines, Learn Only the Heading: Disentangling Device Orientation in Cross-Dataset IMU-Based Activity Recognition"

Use "cross-dataset" everywhere and drop "cross-domain".

---

## 3. Experiments

All runs use subject information, the corrected MotionSense data, the same 20 Hz / 2.56 s windows, no target data at any stage, and the same run budget for every source (C67). The ablation arms use 2 pretraining × 4 fine-tuning seeds; the added baselines use 2 × 2. Every run stores macro-F1, stairs-F1, per-class F1 and the confusion matrix.

### Arms (one code path; only the listed flags differ)

| Arm | Input | Pretraining | Fine-tuning op | Role |
|---|---|---|---|---|
| `A_ssl` | device frame | MLM | none | Baseline SSL (LIMU-BERT) |
| `B_grav` | canonicalized | MLM | none | + gravity canonicalization |
| `C_yaw` | device frame | MLM + yaw consistency | yaw | + yaw invariance only |
| `D_gravyaw` | canonicalized | MLM + yaw consistency | yaw | Proposed method |
| `D_pre`, `D_ft` | canonicalized | with / without yaw consistency | none / yaw | Where the yaw invariance must live (C39, C41) |
| `E_so3aug` | device frame | MLM | SO(3) | C46 (1): LIMU-BERT + rotation augmentation |
| `F_so3inv` | device frame | MLM + SO(3) consistency | SO(3) | Learn all three DOF (UniMTS-style) |
| `G_pca` | canonicalized + PCA heading | MLM | none | C46 (2): classical non-learned heading (Gil-Martín-style) |
| `H_mizell` | vertical / horizontal decomposition | MLM | none | Mizell 2003, fully invariant |
| `I_oit` | orientation-invariant transform | MLM | none | Yurtman & Barshan 2017, fully invariant |
| `XGB_raw`, `XGB_canon` | ~110 features | n/a | n/a | Engineered features in both frames |
| `UniMTS_raw`, `UniMTS_canon` | released checkpoint | n/a | authors' recipe | C46 (3): published method, with and without our canonicalization |

### Experiment list

| ID | Stage name | What it answers | Comments |
|---|---|---|---|
| E0 | `audit` | MotionSense sign, geometry identities, device-axis decomposition, antipode neighbourhood, heading definition, cadence/energy, preprocessing table, pair label sets, parameter split. No training. | C32 C33 C36 C37 C38 C42 C44 C45 C50 C51 C66 |
| E1 | `xgb`, `ablation`, `site` | 2×2 component ablation and yaw-site arms on 12 pairs; XGBoost in both frames; paired tests | RQ1 RQ2 RQ4, C25 C26 C30 C39 C41 C52–C54 C57 C62 C63 C73 |
| E2 | `baselines`, `unimts` | The five added baselines and UniMTS on 12 pairs | C46, e-mail |
| E3 | `ceiling` | In-domain ceiling of each target on the pair's own label set (subject-disjoint), gap, and share of gap closed | C28 C29 C64 C74 |
| E4 | analysis only | Hierarchical bootstrap (datasets, then seeds), leave-one-dataset-out, per-cell difference intervals | C49 C53 |
| E5 | `rotation` | Controlled rotations on all four datasets: device-axis (x, y, z) and physical (tilt about a horizontal axis, heading about gravity); ORS, worst-case drop, accuracy vs residual heading | RQ3, C37 C50 C51, single-dataset limitation |
| E6 | `posture` | Cost of canonicalization on sit / stand / lie, within and across datasets; side-channel arm that re-injects device pose | C44 |
| E7 | `hparam` | λ ∈ {0.03, 0.1, 0.3} × identity floor ∈ {0, 0.3, 0.5} chosen by a source-only criterion (held-out source subjects under random rotations); full grid on targets | C40 |
| E8 | `inference` | Latency with CPU model, threads and batch size recorded; TorchScript/ONNX export for an on-phone measurement | C27 C56 |
| E9 (optional) | `augspec`, `positions` | Axis specificity on 12 pairs; Shoaib body-position transfer | Fig. 11, Sec. 5.3 |
| E10 (optional) | `labeleff`, `autoaug` | Label efficiency re-run; AutoAugHAR on all 12 pairs | C47 C65 |

### How to read the results

These outcomes decide the story, so read them before writing.

- **`G_pca` ≈ `D_gravyaw`.** Heading can be estimated geometrically and the learned half is unnecessary. Report this as the finding: estimate both. If `D` wins, the margin is evidence for learning heading rather than estimating it, against the closest prior method rather than a straw man.
- **`F_so3inv` ≈ `D_gravyaw`.** Canonicalization adds nothing once full rotation invariance is learned. If `D` wins, this is the direct evidence for "canonicalize what physics can determine".
- **`H_mizell` or `I_oit` ≈ `D_gravyaw`.** The horizontal direction carries no class information on these label sets.
- **`D_gravyaw` − `XGB_canon` is not significant.** The paper is about orientation (E3), not about the encoder.
- **`D_ft` ≈ `D_gravyaw`.** The consistency term is optional (C41). Simplify the method to canonicalization plus one augmentation operator, and say so.
- **E5 accuracy-vs-heading curves.** For `B_grav` they should depend on the rotation only through the residual heading, and `D_gravyaw` should be flat. If not, the mechanism claim must be weakened.

### Run order and cost

1. Upload `GraviHAR_Revision_Colab.ipynb` to Colab, set the GPU runtime, and run the setup cell and Parts 1–3.
2. Run `audit` (minutes). Then run `estimate_cost()`, which times one epoch on the GPU and prints projected hours per stage.
3. Tier 1 (minimum for a resubmission): `xgb`, `ablation`, `baselines`, `ceiling`, `rotation`, `unimts`.
4. Tier 2: `site`, `posture`, `hparam`, `inference`.
5. Tier 3, if pages allow: `augspec`, `positions`, `labeleff`, `autoaug`.

Approximate run counts:

| Stage | Fine-tuning runs | Note |
|---|---|---|
| ablation | 384 | |
| site | 192 | |
| baselines | 240 | ablation, site and baselines share 64 pretrainings (8 input/objective combinations × 2 seeds × 4 sources) |
| rotation | 168 | |
| ceiling | 144 | 8 target/label-set combinations × 3 folds × 3 arms × 2, plus 48 XGBoost fits |
| posture | about 150 | |
| hparam | about 290 | |
| unimts | 48 | the slowest per run; reduce with `max_train` or `epochs` if needed |

Estimated Tier 1 time, computed from the real dataset sizes with typical step times: about 18 h on an A100 (xgb 0.7, ablation 6.3, baselines 4.1, ceiling 1.9, rotation 2.3, UniMTS 3.0) and about 40 h on a T4, where UniMTS alone takes about 12 h. `estimate_cost()` replaces these assumptions with step times measured on your runtime and subtracts finished runs, so it also reports the remaining time mid-way. Expect several Colab sessions. Every stage skips runs that are already in its CSV.

Re-run A–D rather than reusing E2, for three reasons:

- The MotionSense correction changes 6 of the 12 pairs.
- The old CSV has no per-run values, so the hierarchical bootstrap cannot use it.
- Canonicalization is now applied once, on the 20 Hz window, for every arm.

The old encoders cannot be reused by accident: every cached encoder name carries a fingerprint of its data.

---

## 4. Proofs for Sec. 3.2.3 (C36; replaces the "numerical verification")

Let R(v) be the rotation of Eq. 3, with R(v)v = z for a unit vector v ≠ −z. A mounting change Q acts linearly on the window, so the window mean becomes Qā and its direction becomes Qĝ.

**Proposition 1 (the residual is a heading).** The canonicalized window after the mounting change equals S(Q) applied to the canonicalized window before it, where S(Q) = R(Qĝ) Q R(ĝ)ᵀ. Then

S(Q) z = R(Qĝ) Q ĝ = z.

A rotation that fixes z is a rotation about z. Hence S(Q) is a rotation about z by some angle ψ(Q, ĝ), for every Q with Qĝ ≠ −z. ∎

**Lemma L1 (device-z rotations pass through unchanged).** The construction of R is equivariant under rotations U that fix z, because (Uv) × z = U(v × z) and (Uv)·z = v·z. Hence R(Uv) = U R(v) Uᵀ. For Q = U = R_z(θ) this gives S = U R(ĝ) Uᵀ U R(ĝ)ᵀ = R_z(θ) for every pose. The yaw column of Fig. 7 and Table 1 restates this identity and should not be presented as a finding.

**Lemma L2 (a rotation about gravity is pure heading).** If Q rotates about ĝ by θ, then Qĝ = ĝ and S = R(ĝ) Q R(ĝ)ᵀ. This is a rotation about R(ĝ)ĝ = z by θ, so ψ = θ.

**Lemma L3 (a tilt leaves a pose-dependent heading).** Let Q rotate about a horizontal axis h ⊥ ĝ.

- If ĝ = z, then R(Qz) = Q⁻¹ and S = I.
- In general S ≠ I. The heading reference of the canonical frame is set by the shortest rotation from the device pose to z, and that reference changes when the pose changes. ψ grows with the angle between ĝ and +z and is unbounded near −z (`geometry_identities()` tabulates it).

This explains C37: an applied pitch turns into heading for two reasons, both of which depend on the device pose:

- a device-axis "pitch" can be a physical heading change;
- even a true tilt leaves a residual heading through the canonical frame's heading reference.

**Singularity (C38).** At ĝ = −z exactly, the code returns diag(1, −1, −1), a 180° turn about x. Define the neighbourhood by the angle between ĝ and −z. `antipode_report()` gives the fraction of windows within 5°, 10°, 15° and 25°, and measures how much a 1° error in ĝ changes the canonical heading as a function of that angle.

**Residual heading (C32).** For a dynamic window after canonicalization, let h_t be the horizontal acceleration [a_x, a_y]ᵀ minus its window mean, and C = Σ_t h_t h_tᵀ. Define θ_h = atan2(v₂, v₁) mod 180°, where v is the leading eigenvector of C. θ_h is measured from the canonical x axis, which is fixed by Eq. 3. The concentration is R̄ = |mean exp(2iθ_h)|, between 0 (uniform) and 1 (identical). Values near 0.27 should be called weak concentration, not tight.

---

## 5. Comment-by-comment actions

Key: **E** = new run (stage name), **A** = analysis or table from runs, **W** = text change.

| # | Short form of the comment | Action |
|---|---|---|
| C2 | First sentence should not name datasets | W: open with deployment differences (devices, carrying positions, users). |
| C3 | Why can heading not be determined; why decompose; why gravity | W: gravity is one vector; any rotation about it leaves it unchanged, so it fixes two angles and leaves one free. Cite Proposition 1. |
| C5 | 3–4 application references | W: see Section 7. |
| C6 | Cite Fig. 1 before Fig. 2 | W: merge Figs. 1 and 2, or reorder. |
| C7 | Numbering font; why remove that rotation | W: larger panel numbers; the analytically resolvable part needs no labels, data or training. |
| C8 | Introduction should state others' limitations, not our superiority | W: move results out of the Fig. 2 caption; list limitations of learned-invariance and analytic-normalization work. |
| C10 | Prior works introduced abruptly | W: group them as (i) learned invariance (SSL, augmentation, foundation models) and (ii) analytic normalization (Mizell, Yurtman, Gil-Martín); state the gap: no controlled DOF-wise comparison under domain generalization. |
| C11 | "Not exploited" claim is wrong | W: delete; gravity has been exploited (Section 2 of this plan). |
| C12 | Hard to understand | W: rewrite as in C3. |
| C13 | "Removed exactly and for free" unclear | W: "computed from the window itself, without labels or training". |
| C14 | Cannot argue science with a dataset | W: argue from deployment (users carry phones differently); support with the pose audit (E0). |
| C15 | "Dataset" is not the real world | W: as C14. |
| C16 | Why a gravity-aligned frame; why not others | E2: device frame (A), gravity + estimated heading (G), fully invariant (H, I), learned SO(3) (F). W: an earth frame needs a magnetometer, which UCI-HAR and HHAR do not provide. |
| C17 | Fixed rate and duration reduce flexibility | W: they are an evaluation control. Canonicalization needs only a window long enough for body acceleration to average out; state this limitation. |
| C18 | Avoid "first" | W: delete. |
| C19 | This is an objective, not a result | W: remove from contributions (Proposition 1 is a proof). |
| C20 | "Two-stage" undefined | W: define the two components once and use the same name everywhere. |
| C21 | No results in contributions | W: remove numbers. |
| C22 | Same | W: "we evaluate on a 12-pair matrix with controlled rotations and ablations". |
| C23 | Cross-domain vs cross-dataset; title | W: "cross-dataset" throughout; title in Section 2. |
| C24 | Abstract length and tone | W: 200–250 words; numbers only from `headline_numbers.json`. |
| C25 | Three stairs numbers; 12 vs 3 pairs | E1 `xgb` + A: stairs-F1 for every pair; absolute points over the same pairs. |
| C26 | Compare with canonicalized XGBoost honestly, with a test | E1 + A: `D − XGB_canon` row in `T_tests.csv` (Wilcoxon, Holm, hierarchical bootstrap). |
| C27 | Latency not on a phone | E8: single-thread batch-1 CPU timing plus ONNX export; otherwise soften to "negligible compared with the encoder". |
| C28 | 30–50 point loss needs references | E3: replace with our measured gap (ceiling − cross) or cite; otherwise delete. |
| C29 | Within-dataset 0.870 uses the full label set | E3: ceilings on each pair's own label set. |
| C30 | Inconsistent 69.4 % | Same as C25. |
| C31 | A figure cannot prove | W: "Fig. 4 shows that the four datasets differ substantially in device pose." |
| C32 | Define residual heading; R̄ = 0.268 is not tight | A: definition in Section 4; `heading_profile_v2()`. |
| C33 | MotionSense iOS conventions | E0 + fixed loader (Section 1.1); describe the conversion in Sec. 4.1.2. |
| C34 | Notation | W: y ∈ {1,…,K}; C for channels; M for the mask; separate symbols for decoder and dataset; define Φ, E, D. |
| C35 | "E1.4" internal code | W: delete. |
| C36 | Provide the proof | W: Section 4. |
| C37 | Pitch turns into heading | A: Section 1.2, Lemma L3, `rotation_axis_decomposition()`, E5. |
| C38 | Neighbourhood; behaviour at s = 0 | A: `antipode_report()`; Section 4. |
| C39 | Sentence on removing the consistency term unclear | E1 `site`; rewrite with the measured numbers. |
| C40 | How hyperparameters were chosen; no target data | E7; disclose that the earlier choice used a target pair; masking parameters are LIMU-BERT defaults. |
| C41 | 0.941 vs 0.916 within noise; 0.644 vs 0.647 | E1 `site` (12 pairs, paired test); single source of numbers. |
| C42 | Parameter split | A: `param_breakdown()` (Section 1.4). |
| C43 | RQ names | W: RQ1 controlled robustness, RQ2 cross-dataset performance against baselines, RQ3 component ablation, RQ4 orientation share of the gap and failure cases. Efficiency becomes an implementation detail. |
| C44 | Class table; static merging hides a cost | A: `pair_label_table()`; E6 posture experiment and a limitation paragraph. |
| C45 | Preprocessing details; 8,355 windows | A: `preprocessing_table()` (native rates per HHAR phone model, units, conventions, overlap; LAYING exclusion). |
| C46 | Baselines insufficient | E2: SO(3) augmentation, learned SO(3), PCA heading, Mizell, OIT, UniMTS. |
| C47 | Why three AutoAugHAR pairs | E10 on all 12 pairs, or drop AutoAugHAR (first e-mail: policy search belongs to a later paper). Recommended: drop. |
| C48 | Early stopping on what data | W: pretraining stops on the plateau of the source reconstruction loss; fine-tuning uses a fixed 40 epochs; no validation data. |
| C49 | Pairs not independent; 2 pretraining seeds | E4: hierarchical bootstrap, leave-one-dataset-out, limitation paragraph. |
| C50 | Roll vs pitch explanation invalid | A: `rotation_axis_decomposition()` (energy per device axis, gravity displacement in the raw frame). |
| C51 | Why canonicalization improves yaw ORS | A: Lemma L1. A device-z rotation moves gravity by up to twice the pose tilt in the raw frame (about 39° on HHAR); canonicalization turns it into pure heading. |
| C52 | Explain every "—" | E1: none left; all pairs contain both stairs classes. |
| C53 | Which test supports "tied-best" | A: per-cell two-level bootstrap interval (marked "~" in the main table) or drop the claim. |
| C54 | 2-pair vs 3-pair averages | Same as C25. |
| C55 | "Well-known" needs references | W: cite or state as our observation (`T_family_frame.csv`). |
| C56 | CPU model, threads, batch | E8. |
| C57 | HHAR→UCI has the larger gain | A: `T_ablation.csv` column D − B; correct the sentence. |
| C58 | ρ = −0.937 is built into the mathematics | A: remove. Report the per-pair interaction with its bootstrap interval only. |
| C59 | Caption contradicts text | W: removed with C58. |
| C60 | Walking vs downstairs differ vertically | W: report the confusion change as an observation. |
| C61 | "Pre-registered"; outcome chosen afterwards; weak p | W: "a hypothesis formed before the analysis", reported as exploratory, or drop it. |
| C62 | Ordering reverses only with yaw | A: `T_family_frame.csv`; correct the sentence. |
| C63 | Wrong row in Table 5 | A: `T_family_frame.csv` (canonicalized LIMU-BERT row plus a separate GraviHAR row). |
| C64 | Build the paper around the orientation share of the shift | E3 + Section 2: the main analysis of draft 3. |
| C65 | Label-efficiency numbers inconsistent | E10 re-run or drop Sec. 5.2 (recommended for the page limit). |
| C66 | Show the cadence evidence | A: `cadence_energy_table()`; state the 0.39 Hz frequency resolution. |
| C67 | Seed mapping unclear | W: one budget for all sources. |
| C68 | Lesson depends on invalid ρ | W: remove. |
| C69 | Related Work too thin; cite Mizell, Yurtman, TRI-HAR | W: Section 7; explain the difference from each. |
| C70 | Cite every method named | W: add numbers. |
| C71 | Grammar | W: "GraviHAR uses no target-domain data, labelled or unlabelled." |
| C72 | Strong words | W: remove "proves", "strictly", "exactly", "perfectly", "precisely" unless a proof backs them. |
| C73 | Inconsistent stairs number | Same as C25. |
| C74 | In-domain ceiling only on one pair | E3: report per pair; limit the claim. |

E-mail items:

- Baselines: E2.
- Contribution: Section 2.
- Gravity-based prior work: Section 2 and arms G, H, I.
- References: Section 7.
- Page limit: Section 6.

---

## 6. Reaching 20 pages

| Section of draft 2 | Change | Saves |
|---|---|---|
| Figs. 1 and 2 | Merge into one figure (problem and measured gap) | ~0.7 page |
| Fig. 3 (stylised signals) | Remove; Fig. 1 carries the idea | ~0.6 page |
| 2.1 Background | Cut to one paragraph; keep the DOF argument | ~0.5 page |
| 3.5 and 4.4 Efficiency | One paragraph and one row of the implementation table | ~0.8 page |
| 4.6 Complementarity (ρ, KL) | Remove (C58–C61); keep the per-pair interaction column | ~1 page |
| 5.1 HHAR→UCI explanation | Shorten to one paragraph supported by E5 | ~0.8 page |
| 5.2 Label efficiency | Remove or one sentence | ~0.6 page |
| AutoAugHAR baseline | Remove (first e-mail) | ~0.4 page |
| 6.1 Lessons | Merge into the Conclusion | ~0.5 page |

Added material:

- Proofs: 0.5 page.
- Gap analysis: 1 page.
- Baselines table: 0.5 page.
- Rotation figure: 0.7 page.
- Related Work: 1 page.

The net change is about −3 pages.

---

## 7. References

**Found in this session.** The search confirmed each entry exists. Confirm the full bibliographic details before citing.

- D. Mizell. Using gravity to estimate accelerometer orientation. ISWC 2003.
- A. Yurtman, B. Barshan. Activity recognition invariant to sensor orientation with wearable motion sensors. Sensors 17(8):1838, 2017. doi:10.3390/s17081838
- A. Yurtman, B. Barshan, B. Fidan. Activity recognition invariant to wearable sensor unit orientation using differential rotational transformations represented by quaternions. Sensors 18(8):2725, 2018. doi:10.3390/s18082725
- M. Gil-Martín, J. López-Iniesta, F. Fernández-Martínez, R. San-Segundo. Reducing the impact of sensor orientation variability in human activity recognition using a consistent reference system. Sensors 23(13):5845, 2023. doi:10.3390/s23135845
- S. Baek, Y. Chai, Y. Lee, S. Choi, S. Suh. Rotation-invariant multi-IMU activity recognition under independent per-location orientation shifts (TRI-HAR). arXiv:2608.15621, ISWC 2026.
- K. Kunze, P. Lukowicz. Dealing with sensor displacement in motion-based onbody activity recognition systems. UbiComp 2008. doi:10.1145/1409635.1409639
- O. Banos et al. Dealing with the effects of sensor displacement in wearable activity recognition. Sensors 2014.
- S. Thiemjarus. A device-orientation independent method for activity recognition. BSN 2010.
- X. Zhang et al. UniMTS: Unified pre-training for motion time series. NeurIPS 2024. arXiv:2410.19818 (code and checkpoint: github.com/xiyuanzh/UniMTS)
- Z. Hong et al. CrossHAR. Proc. ACM IMWUT 8(2), 2024. doi:10.1145/3659597

**Well-known candidates to add (not verified in this session).** Check each before citing.

- Domain generalization: Gulrajani & Lopez-Paz, "In search of lost domain generalization" (ICLR 2021). This is the source for source-only model selection (C40). Also Wang et al., "Generalizing to unseen domains: a survey on domain generalization" (IEEE TKDE).
- Cross-dataset and unsupervised domain adaptation HAR: Chang et al., "A systematic study of unsupervised domain adaptation for robust human-activity recognition" (IMWUT 2020).
- Self-supervised HAR: Saeed et al., "Multi-task self-supervised learning for human activity detection" (IMWUT 2019); Tang et al., SelfHAR (IMWUT 2021); Haresamudram et al., "Assessing the state of self-supervised human activity recognition using wearables" (IMWUT 2022); Qian et al., "What makes good contrastive learning on small-scale wearable-based tasks?" (KDD 2022).
- Augmentation including rotation: Um et al., "Data augmentation of wearable sensor data for Parkinson's disease monitoring using convolutional neural networks" (ICMI 2017).
- Surveys and applications (C5): Bulling, Blanke & Schiele, "A tutorial on human activity recognition using body-worn inertial sensors" (ACM CSUR 2014); Lara & Labrador, "A survey on human activity recognition using wearable sensors" (IEEE COMST 2013); Chen et al., "Deep learning for sensor-based human activity recognition: overview, challenges, and opportunities" (ACM CSUR 2021); Ordóñez & Roggen, DeepConvLSTM (Sensors 2016).

With the current 10 references, these give about 30. Reach 40 by adding the device-heterogeneity work cited by HHAR, and the occupational-safety and healthcare applications the introduction names.
