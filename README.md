# Disentangling Device Orientation in Cross-Dataset HAR

Physics-guided canonicalization and self-supervised invariance for IMU-based human
activity recognition.
 

Device orientation has three rotational degrees of freedom. Two of them — pitch and
roll — are removable analytically once gravity is estimated from the window itself.
Only the third, heading (yaw), requires learning. This repository treats each degree
of freedom with a separate, individually ablatable mechanism, so the contribution of
each can be measured rather than assumed.
 

--- 

Macro-F1 over shared classes, mean of three seeds (42, 43, 44).

| Transfer | Engineered-feature baseline | Full stack |
|---|---|---|
| HHAR → UCI-HAR | 0.610 | **0.897** |
| 12-pair average | 0.577 | **0.638** |
| 12-pair average, stairs classes only | 0.459 | **0.557** | 

For ablation: 

Encoder, pretraining pool, and window geometry held identical; only the alignment
flag changes.

| Alignment | In-domain LOSO acc | Cross-UCI macro-F1 | Cross-UCI stairs-F1 |
|---|---|---|---|
| off | 0.930 ± 0.046 | 0.397 | 0.313 |
| on  | 0.940 ± 0.036 | **0.521** | **0.495** |
 

---
 
 

**Sampling-rate confound.** Benchmark datasets have native rates from 50 to 200 Hz
and different window durations. Resampling every window to a fixed *sequence length*
changes the effective sampling rate between source and target, so periodic gait
signatures land at different normalized frequencies and the displacement is then
counted as domain gap. This repository resamples each window to a true common rate
using its actual duration, `L = round(d · f_t)` with `f_t = 20 Hz` and `d = 2.56 s`
(`L = 51`). After the correction, the dominant walking cadence is identical at
1.96 Hz across all four datasets.

**Pretraining scope.** Pooling unlabeled target data into the pretraining corpus makes
a result unsupervised domain adaptation, not zero-shot transfer, and this is often
unstated. Three scopes are reported as separate claim levels and never merged:

| Scope | Pretraining pool | Claim |
|---|---|---|
| `source` | source dataset only | zero target access |
| `multi-src` | all datasets except the target, capped at 25k windows each | zero target access, source diversity |
| `pooled` | source + unlabeled target | unsupervised domain adaptation |
 

---

## Method

**Stage A — Signal conditioning.** Rate-consistent windowing (above), then gravity
canonicalization. Gravity is estimated from the mean accelerometer vector,
`ĝ = ā / ‖ā‖`, accepted when `‖ā‖ > 5 m/s²`. A Rodrigues rotation `R` satisfying
`Rĝ = [0,0,1]ᵀ` is applied to accelerometer *and* gyroscope channels, since a frame
change transforms every vector measured in that frame. This maps every window into a
common gravity-aligned frame where the vertical axis is physically meaningful.

**Stage B — Orientation-invariant representation learning.** A LIMU-BERT encoder
(~60k parameters, hidden width 72, four attention blocks with cross-layer parameter
sharing) is pretrained with span-masked reconstruction plus a rotation-consistency
term over two independently yaw-rotated views:

```
L_pre = L_mlm + λ · (1 − cos(e₁, e₂)),   λ = 0.1
```

The reconstruction term prevents the collapse a pure consistency objective permits.
The consistency term destabilizes small pools — on the 8,355-window UCI pool,
UCI → HHAR dropped from 0.507 to 0.372 ± 0.209 — so it is gated to pools of at least
15,000 windows, after which the result recovers exactly (0.507 ± 0.048).

**Stage C — Fine-tuning.** A GRU head (20-20-10, dropout 0.5) on the encoder output,
trained end-to-end with class-weighted cross-entropy, since several targets are more
than 50% static and macro-F1 is the metric.
 

---

## Datasets

| Dataset | Subjects | Native rate | Placement |
|---|---|---|---|
| HHAR (Stisen et al., 2015) | 9 | ~100–200 Hz | smartphone |
| UCI-HAR (Anguita et al., 2013) | 30 | 50 Hz | waist |
| MotionSense (Malekzadeh et al., 2019) | 24 | 50 Hz | front trouser pocket |
| Shoaib (Shoaib et al., 2014) | 10 | 50 Hz | 5 simultaneous positions |

All preprocessed to 20 Hz and 2.56 s windows. Labels harmonized to
`static / stairs-up / stairs-down / walk / run / bike`; each transfer pair is
restricted to the label intersection of its source and target. Shoaib contributes its
five body positions as independent streams.
 

---

## Reproducing

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt

# place raw datasets under data/ as described in data/README.md
python -m src.preprocess --rate 20 --duration 2.56

# single transfer pair
python -m src.run_transfer --source hhar --target uci --scope source --seed 42

# full 12-pair matrix, three seeds
python -m src.run_matrix --seeds 42 43 44
```

Pretraining: Adam, lr 1e-3, batch 256, ≤80 epochs with plateau early stopping.
Fine-tuning: Adam, lr 1e-3, batch 128, 40 epochs. All experiments run on a single
NVIDIA T4 (Google Colab).

To reproduce the alignment ablation specifically, run with `--align off` and
`--align on` holding `--scope pooled` and the encoder fixed. Note that the ablation
uses the full HHAR label set and 5.0 s in-domain windows, so its absolute values are
not comparable with the main matrix; the comparison is internally controlled.

---

## Repository structure

```
src/
  preprocess/     rate-consistent windowing, gravity canonicalization
  models/         LIMU-BERT encoder, GRU head, XGBoost feature tier
  pretrain/       span-masked reconstruction + rotation consistency
  augment/        operator space, bilevel policy search, guards
  eval/           transfer matrix, ablations, per-cell provenance logging
data/             download instructions and expected layout (no data committed)
results/          per-cell results, seeds, and which policy produced each cell
notebooks/        analysis and figures
```

Every reported cell records which mechanism produced it (searched policy or fixed
prior), in `results/provenance.csv`.

 
