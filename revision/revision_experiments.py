# =============================================================================
#  REVISION EXPERIMENTS  (draft 2 -> draft 3, answers to Bernardo's 71 comments)
#
#  Load order in Colab (GPU runtime, Drive mounted):
#      %run -i -n revision/pipeline_base.py          # loaders, LIMU-BERT, ops
#      %run -i -n revision/revision_experiments.py   # this file
#  then call the stages listed in REVISION_PLAN.md (or run_stage("...")).
#
#  Every training run, whatever the experiment, goes through ONE pretraining
#  function and ONE fine-tuning function and is appended as one row to a
#  long-format CSV under SAVE_PATH/revision/runs/. All paper numbers are then
#  produced by revision_analysis.py from those CSVs, so a number can no longer
#  differ between the abstract, a table and a figure (c25 c30 c41 c57 c63 c65 c73).
#
#  Comment IDs (cNN) refer to GraviHAR_comments_2.docx.
# =============================================================================
import os, re, json, math, time, glob, hashlib, platform, random, subprocess, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

_REQUIRED = ["LimuBERT", "LimuEncoder", "HARModel", "GRUClassifier", "span_mask", "op_jitter", "op_rot_z",
             "_rot_to_vertical", "set_seed", "DEVICE", "load_hhar_phone", "_uci_load",
             "_resample", "UCI_PATH", "MS_PATH", "SHOAIB_PATH", "SAVE_PATH", "make_xgb",
             "_features_df_from_raw", "LABEL_PRESERVING_OPS", "train_classifier",
             "PRETRAIN_EPOCHS", "PRETRAIN_BS", "PRETRAIN_LR", "FINETUNE_EPOCHS",
             "FINETUNE_BS", "FINETUNE_LR", "XDOMAIN_WINDOW_SEC", "XDOMAIN_SEQ_LEN",
             "TARGET_HZ", "STEP_SEC", "MIN_SAMPLES_HHAR", "H_DIM"]
_missing = [n for n in _REQUIRED if n not in globals()]
if _missing:
    raise RuntimeError("Load revision/pipeline_base.py first (%run -i -n). Missing: "
                       + ", ".join(_missing))

G_REV = 9.80665
REV_VERSION = "rev1"
# d2 = MotionSense converted from the iOS sign convention (c33) + subject IDs kept
DATA_VERSION = "d2"

REV_DIR = os.path.join(SAVE_PATH, "revision")
for _d in ("runs", "cache", "enc", "tables", "unimts"):
    os.makedirs(os.path.join(REV_DIR, _d), exist_ok=True)

# -----------------------------------------------------------------------------
#  Configuration. Everything a reviewer may ask about is here (c40, c48, c67).
#  None of these values is chosen on target data; exp_hparam() re-selects lam and
#  floor with a source-only criterion and reports the full grid on the targets.
# -----------------------------------------------------------------------------
REV = dict(
    lam=0.1,            # consistency weight (Eq. 6)
    warm=0,             # linear ramp length in epochs; 0 = no ramp. The draft text says
                        # a 10-epoch ramp but cell 29 set warmup=0 before E2 -- check the
                        # cached encoder names (_w0_ vs _w10_) and report what was used.
    floor=0.3,          # identity floor: probability that a fine-tuning batch is clean
    view_jitter=0.03,   # jitter added to both consistency views
    pre_epochs=PRETRAIN_EPOCHS, pre_bs=PRETRAIN_BS, pre_lr=PRETRAIN_LR,
    pre_min_epochs=30, pre_patience=8,   # early stop on the SOURCE MLM training loss (c48)
    ft_epochs=FINETUNE_EPOCHS, ft_bs=FINETUNE_BS, ft_lr=FINETUNE_LR,  # fixed, no early stop
    # one run budget for every source (c67): 2 pretraining seeds x 4 fine-tuning seeds
    # for the ablation arms the claims rest on; 2 x 2 for the added baselines
    budget={"pre": (42, 43), "ft": (42, 43, 44, 45)},
    budget_baselines={"pre": (42, 43), "ft": (42, 43)},
    ios_sign_fix=True,  # c33: Core Motion reports -specific force (flat, face up: z=-1 g)
    gate_g=5.0 / G_REV, # gravity gate in g units (acc stored in g)
)

PAIRS = [("hhar", "uci"), ("uci", "hhar"), ("hhar", "motion"), ("motion", "hhar"),
         ("hhar", "shoaib"), ("shoaib", "hhar"), ("uci", "motion"), ("motion", "uci"),
         ("uci", "shoaib"), ("shoaib", "uci"), ("motion", "shoaib"), ("shoaib", "motion")]
DATASETS_REV = ["hhar", "uci", "motion", "shoaib"]

CLS = {0: "static", 2: "up", 3: "down", 4: "walk", 5: "run", 6: "bike",
       8: "sit", 9: "stand", 10: "lie"}
STAIRS = (2, 3)


def _p(*a):
    print(*a, flush=True)


# =============================================================================
#  1. DATA LAYER v2  (c33 c44 c45)
# -----------------------------------------------------------------------------
#  Every loader returns UNALIGNED 20 Hz windows of 2.56 s (L=51), acc in g and
#  gyro in rad/s, in the ANDROID sign convention (flat, face up: acc_z = +1 g),
#  plus subject IDs, two label vectors and per-window metadata:
#      y_merged  : static / up / down / walk / run / bike   (the paper's label set)
#      y_posture : sit / stand / lie kept apart             (c44)
#  -1 marks a window that has no label in that mode (e.g. UCI LAYING in merged
#  mode -- this is why UCI has 8,355 rather than 10,299 windows, c45).
#  Canonicalization and every other input transform is applied afterwards by
#  prep(), so all arms see exactly the same windows.
# =============================================================================
HHAR_POSTURE = {"sit": 8, "stand": 9}
UCI_MERGED = {1: 4, 2: 2, 3: 3, 4: 0, 5: 0}
UCI_POSTURE = {1: 4, 2: 2, 3: 3, 4: 8, 5: 9, 6: 10}
MS_MERGED = {"dws": 3, "ups": 2, "wlk": 4, "jog": 5, "sit": 0, "std": 0}
MS_POSTURE = {"dws": 3, "ups": 2, "wlk": 4, "jog": 5, "sit": 8, "std": 9}
SHOAIB_MERGED = {"walking": 4, "jogging": 5, "running": 5, "sitting": 0, "standing": 0,
                 "biking": 6, "upstairs": 2, "downstairs": 3}
SHOAIB_POSTURE = dict(SHOAIB_MERGED, sitting=8, standing=9)
SHOAIB_POSITIONS = {0: "left_pocket", 1: "right_pocket", 2: "wrist", 3: "upper_arm", 4: "belt"}


def _win20(acc, gyr):
    """[n,3]+[n,3] native-rate arrays -> [51,6] at 20 Hz, acc in g. Same resampler as
    the pipeline (linear interpolation over the window's own duration)."""
    a = _resample(np.asarray(acc, np.float32), XDOMAIN_SEQ_LEN)
    g = _resample(np.asarray(gyr, np.float32), XDOMAIN_SEQ_LEN)
    w = np.concatenate([a, g], axis=1).astype(np.float32)
    w[:, 0:3] /= G_REV
    return w


def _mode_label(v):
    vals, cnt = np.unique(np.asarray(v), return_counts=True)
    return vals[int(np.argmax(cnt))]          # ties -> smallest, as pandas .mode().iloc[0]


def _load_hhar_v2():
    acc_raw, gyro_raw = load_hhar_phone()
    if acc_raw is None or gyro_raw is None:
        raise FileNotFoundError("HHAR phone CSVs not found under ACTIVITY_PATH")
    W, step, mn = XDOMAIN_WINDOW_SEC, STEP_SEC, MIN_SAMPLES_HHAR
    acc_raw = acc_raw.copy()
    acc_raw["posture"] = [HHAR_POSTURE.get(str(s).strip().lower(), -2) for s in acc_raw["gt"]]
    gyro_groups = {k: g for k, g in gyro_raw.groupby(["user", "device"])}
    X, ym, yp, grp, dev, mdl, rate = [], [], [], [], [], [], []
    for (user, device), ag in acc_raw.groupby(["user", "device"]):
        gg = gyro_groups.get((user, device))
        if gg is None or gg.empty:
            continue
        t0 = max(ag["time_sec"].min(), gg["time_sec"].min())
        t1 = min(ag["time_sec"].max(), gg["time_sec"].max())
        if t1 - t0 < W:
            continue
        at, gt = ag["time_sec"].values, gg["time_sec"].values
        axyz, gxyz = ag[["x", "y", "z"]].values, gg[["x", "y", "z"]].values
        lab, pos = ag["class_label"].values, ag["posture"].values
        model = str(ag["model"].iloc[0])
        t = t0
        while t + W <= t1:
            i0, i1 = np.searchsorted(at, t, "left"), np.searchsorted(at, t + W, "left")
            j0, j1 = np.searchsorted(gt, t, "left"), np.searchsorted(gt, t + W, "left")
            if (i1 - i0) >= mn and (j1 - j0) >= mn:
                lm = int(_mode_label(lab[i0:i1]))
                pm = int(_mode_label(pos[i0:i1]))
                X.append(_win20(axyz[i0:i1], gxyz[j0:j1]))
                ym.append(lm)
                yp.append(lm if lm != 0 else (pm if pm >= 0 else -1))
                grp.append(f"hhar_{user}"); dev.append(str(device)); mdl.append(model)
                rate.append((i1 - i0) / W)
            t += step
    X = np.asarray(X, np.float32)
    return dict(X=X, y_merged=np.asarray(ym), y_posture=np.asarray(yp), groups=np.asarray(grp),
                meta=dict(device=np.asarray(dev), model=np.asarray(mdl),
                          native_hz=np.asarray(rate, np.float32)),
                info=dict(native="~50-200 Hz (per phone model)", units="m/s^2, rad/s",
                          convention="Android", placement="smartphones (waist pouch)",
                          overlap=f"{1 - STEP_SEC / W:.1%} (step {STEP_SEC} s)"))


def _load_uci_v2():
    X, ym, yp, grp, split_ = [], [], [], [], []
    for split in ("train", "test"):
        ax, ay, az = (_uci_load(split, f"total_acc_{a}") * G_REV for a in "xyz")
        gx, gy, gz = (_uci_load(split, f"body_gyro_{a}") for a in "xyz")
        yv = np.loadtxt(os.path.join(UCI_PATH, split, f"y_{split}.txt")).astype(int)
        sv = np.loadtxt(os.path.join(UCI_PATH, split, f"subject_{split}.txt")).astype(int)
        for i in range(ax.shape[0]):
            acc = np.stack([ax[i], ay[i], az[i]], 1); gyr = np.stack([gx[i], gy[i], gz[i]], 1)
            X.append(_win20(acc, gyr))
            ym.append(UCI_MERGED.get(int(yv[i]), -1)); yp.append(UCI_POSTURE.get(int(yv[i]), -1))
            grp.append(f"uci_{sv[i]}"); split_.append(split)
    return dict(X=np.asarray(X, np.float32), y_merged=np.asarray(ym), y_posture=np.asarray(yp),
                groups=np.asarray(grp), meta=dict(split=np.asarray(split_)),
                info=dict(native="50 Hz (pre-segmented 128-sample windows)", units="g, rad/s",
                          convention="Android", placement="waist (belt)",
                          overlap="50% (dataset's own segmentation)",
                          note="LAYING excluded in merged mode: 10,299 - 1,944 = 8,355 windows"))


def _load_motion_v2(win=128, step=64):
    path = MS_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    sign = -1.0 if REV["ios_sign_fix"] else 1.0
    need = ["gravity.x", "gravity.y", "gravity.z", "userAcceleration.x", "userAcceleration.y",
            "userAcceleration.z", "rotationRate.x", "rotationRate.y", "rotationRate.z"]
    trial_dirs = []
    for root, dirs, _ in os.walk(path):
        dirs[:] = [d for d in dirs if d != "__MACOSX" and not d.startswith(".")]
        if "__MACOSX" in root:
            continue
        for d in dirs:
            act = d.split("_")[0].lower()
            if act in MS_MERGED:
                trial_dirs.append((os.path.join(root, d), act))
    X, ym, yp, grp, trial = [], [], [], [], []
    for tdir, act in sorted(trial_dirs):
        for fn in sorted(os.listdir(tdir)):
            if not fn.endswith(".csv") or fn.startswith("."):
                continue
            df = pd.read_csv(os.path.join(tdir, fn))
            if not all(c in df.columns for c in need):
                continue
            m = re.search(r"sub_(\d+)", fn)
            subj = f"ms_{m.group(1) if m else fn}"
            # Core Motion: total = gravity + userAcceleration, in g, with the opposite
            # sign to Android (device flat, face up -> gravity.z = -1). Negating all
            # three axes gives the Android specific-force convention. rotationRate is
            # right-handed rad/s on both platforms and is left unchanged.
            acc = sign * (df[need[0:3]].values + df[need[3:6]].values) * G_REV
            gyr = df[need[6:9]].values
            for s in range(0, len(acc) - win + 1, step):
                X.append(_win20(acc[s:s + win], gyr[s:s + win]))
                ym.append(MS_MERGED[act]); yp.append(MS_POSTURE[act])
                grp.append(subj); trial.append(os.path.basename(tdir))
    return dict(X=np.asarray(X, np.float32), y_merged=np.asarray(ym), y_posture=np.asarray(yp),
                groups=np.asarray(grp), meta=dict(trial=np.asarray(trial)),
                info=dict(native="50 Hz", units="g (Core Motion), rad/s",
                          convention="iOS -> converted to Android" if REV["ios_sign_fix"]
                          else "iOS (NOT converted)",
                          placement="front trouser pocket (iPhone 6s)",
                          overlap=f"{1 - step / win:.0%} (win {win}, step {step} @ 50 Hz)"))


def _shoaib_lab(s, table):
    s = str(s).strip().lower()
    for k, v in table.items():
        if k in s:
            return v
    return -1


def _load_shoaib_v2(win=128, step=64):
    path = SHOAIB_PATH
    files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(path)
    X, ym, yp, grp, pos = [], [], [], [], []
    for f_i, fp in enumerate(files):
        raw = pd.read_csv(fp, header=None, low_memory=False, dtype=str)
        hdr = next((r for r in range(min(5, len(raw)))
                    if (raw.iloc[r].astype(str).str.strip() == "Ax").any()), None)
        if hdr is None:
            continue
        names = raw.iloc[hdr].astype(str).str.strip().tolist()
        data = raw.iloc[hdr + 1:].reset_index(drop=True)
        ax_cols = [i for i, n in enumerate(names) if n == "Ax"]
        gx_cols = [i for i, n in enumerate(names) if n == "Gx"]
        lab_str = data.iloc[:, -1].values
        lm = np.array([_shoaib_lab(s, SHOAIB_MERGED) for s in lab_str])
        lp = np.array([_shoaib_lab(s, SHOAIB_POSTURE) for s in lab_str])
        m = re.search(r"(\d+)", os.path.basename(fp))
        subj = f"shoaib_{m.group(1) if m else f_i}"
        for p in range(min(len(ax_cols), len(gx_cols))):
            try:
                acc = data.iloc[:, ax_cols[p]:ax_cols[p] + 3].astype(float).values
                gyr = data.iloc[:, gx_cols[p]:gx_cols[p] + 3].astype(float).values
            except Exception:
                continue
            change = np.where(np.diff(lm) != 0)[0] + 1
            bounds = np.concatenate([[0], change, [len(lm)]])
            for b0, b1 in zip(bounds[:-1], bounds[1:]):
                if lm[b0] < 0 or (b1 - b0) < win:
                    continue
                a_seg, g_seg = acc[b0:b1], gyr[b0:b1]
                ok = np.isfinite(a_seg).all(1) & np.isfinite(g_seg).all(1)
                a_seg, g_seg = a_seg[ok], g_seg[ok]
                for s in range(0, len(a_seg) - win + 1, step):
                    X.append(_win20(a_seg[s:s + win], g_seg[s:s + win]))
                    ym.append(int(lm[b0])); yp.append(int(lp[b0])); grp.append(subj); pos.append(p)
    return dict(X=np.asarray(X, np.float32), y_merged=np.asarray(ym), y_posture=np.asarray(yp),
                groups=np.asarray(grp), meta=dict(position=np.asarray(pos)),
                info=dict(native="50 Hz", units="m/s^2, rad/s", convention="Android",
                          placement="5 simultaneous: L/R pocket, wrist, upper arm, belt",
                          overlap=f"{1 - step / win:.0%} (win {win}, step {step} @ 50 Hz)"))


# The smoke test replaces these with synthetic generators.
REV_LOADERS = {"hhar": _load_hhar_v2, "uci": _load_uci_v2,
               "motion": _load_motion_v2, "shoaib": _load_shoaib_v2}

_RAW_CACHE = {}


def _fingerprint(X):
    X = np.ascontiguousarray(X)
    step = max(1, len(X) // 64)
    h = hashlib.md5(X[::step].tobytes()).hexdigest()[:8]
    return f"n{len(X)}_{h}"


def load_raw_v2(name):
    """All windows of a dataset (both label modes), cached in memory and on disk."""
    key = (name, DATA_VERSION, REV["ios_sign_fix"])
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]
    tag = f"{name}_{DATA_VERSION}_ios{int(REV['ios_sign_fix'])}"
    fp = os.path.join(REV_DIR, "cache", f"data_{tag}.npz")
    if os.path.exists(fp) and REV_LOADERS.get(name) in (
            _load_hhar_v2, _load_uci_v2, _load_motion_v2, _load_shoaib_v2):
        z = np.load(fp, allow_pickle=True)
        d = dict(X=z["X"], y_merged=z["y_merged"], y_posture=z["y_posture"],
                 groups=z["groups"], meta=z["meta"].item(), info=z["info"].item())
    else:
        _p(f"[data] parsing {name} ...")
        d = REV_LOADERS[name]()
        if REV_LOADERS.get(name) in (_load_hhar_v2, _load_uci_v2, _load_motion_v2, _load_shoaib_v2):
            np.savez(fp, X=d["X"], y_merged=d["y_merged"], y_posture=d["y_posture"],
                     groups=d["groups"], meta=np.array(d["meta"], dtype=object),
                     info=np.array(d["info"], dtype=object))
    d["fp"] = _fingerprint(d["X"])
    _RAW_CACHE[key] = d
    _p(f"[data] {name}: {len(d['X'])} windows, {len(np.unique(d['groups']))} subjects")
    return d


def get_ds(name, label_mode="merged"):
    """-> X_raw [N,51,6] (unaligned, Android convention), y, groups, meta(dict), fp."""
    d = load_raw_v2(name)
    y = d["y_merged"] if label_mode == "merged" else d["y_posture"]
    k = y >= 0
    meta = {kk: np.asarray(v)[k] for kk, v in d["meta"].items()}
    X = d["X"][k]
    return X, y[k], d["groups"][k], meta, f"{d['fp']}_{label_mode}"


# =============================================================================
#  2. INPUT TRANSFORMS  (one implementation, used by every arm and analysis)
# =============================================================================
def gravity_dirs(X):
    """Unit gravity direction g_hat (device frame) and |a_bar| (g units) per window."""
    m = np.asarray(X)[:, :, 0:3].astype(np.float64).mean(1)
    n = np.linalg.norm(m, axis=1)
    return m / np.maximum(n, 1e-12)[:, None], n


def _skew(v):
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1),
                     np.stack([v[:, 2], z, -v[:, 0]], 1),
                     np.stack([-v[:, 1], v[:, 0], z], 1)], 1)


def rot_to_vertical_batch(g):
    """Vectorized Eq. 3: the minimal (Rodrigues) rotation with R g = z. Identical to
    the pipeline's _rot_to_vertical, including its antipodal convention: when g = -z
    exactly (s = 0, c < 0) it returns diag(1,-1,-1), a 180 deg turn about x (c38)."""
    g = np.asarray(g, np.float64)
    v = np.cross(g, np.array([0.0, 0.0, 1.0]))
    c = g[:, 2]; s = np.linalg.norm(v, axis=1)
    K = _skew(v)
    coef = np.where(s > 1e-8, (1 - c) / np.maximum(s * s, 1e-300), 0.0)
    R = np.eye(3)[None] + K + (K @ K) * coef[:, None, None]
    sing = s <= 1e-8
    if sing.any():
        R[sing] = np.where((c[sing] > 0)[:, None, None], np.eye(3)[None],
                           np.diag([1.0, -1.0, -1.0])[None])
    return R


def rotate_windows(X, R):
    """Apply a per-window rotation [N,3,3] (or one [3,3]) to acc and gyro."""
    X = np.asarray(X, np.float32)
    R = np.asarray(R, np.float64)
    if R.ndim == 2:
        R = np.broadcast_to(R, (len(X), 3, 3))
    out = X.copy()
    out[:, :, 0:3] = np.einsum("nij,ntj->nti", R, X[:, :, 0:3]).astype(np.float32)
    out[:, :, 3:6] = np.einsum("nij,ntj->nti", R, X[:, :, 3:6]).astype(np.float32)
    return out


def canonicalize(X, return_R=False):
    """Gravity canonicalization (Eqs. 2-4) on the 20 Hz window: g_hat from the window
    mean, Rodrigues rotation to +z, same R on acc and gyro. Windows failing the
    |a_bar| > 5 m/s^2 gate are passed through (fraction reported by audit)."""
    g, n = gravity_dirs(X)
    R = rot_to_vertical_batch(g)
    R[n < REV["gate_g"]] = np.eye(3)
    out = rotate_windows(X, R)
    return (out, R) if return_R else out


def pca_heading(Xc, return_ratio=False):
    """Classical non-learned heading alignment (closest prior work: Gil-Martin et
    al., Sensors 2023; Henpraserttae/Thiemjarus-style PCA): after canonicalization,
    rotate about z so the principal horizontal acceleration axis lands on +x. The
    axis sign is fixed by making the third moment of the projection positive
    (eigenvectors are only defined up to sign). ratio = lambda_min/lambda_max of
    the horizontal covariance; -> 1 means no dominant direction (static windows)."""
    Xc = np.asarray(Xc, np.float32)
    H = Xc[:, :, 0:2].astype(np.float64)
    H = H - H.mean(1, keepdims=True)
    C = np.einsum("nti,ntj->nij", H, H)
    w, V = np.linalg.eigh(C)
    p = V[:, :, 1]
    proj = np.einsum("nti,ni->nt", H, p)
    p = p * np.where((proj ** 3).mean(1) < 0, -1.0, 1.0)[:, None]
    th = np.arctan2(p[:, 1], p[:, 0])
    c, s = np.cos(-th), np.sin(-th)
    Rz = np.zeros((len(Xc), 3, 3)); Rz[:, 0, 0] = c; Rz[:, 0, 1] = -s
    Rz[:, 1, 0] = s; Rz[:, 1, 1] = c; Rz[:, 2, 2] = 1.0
    out = rotate_windows(Xc, Rz)
    ratio = w[:, 0] / np.maximum(w[:, 1], 1e-12)
    return (out, ratio) if return_ratio else out


def mizell_transform(X):
    """Mizell (ISWC 2003) vertical/horizontal decomposition, extended to the gyro:
    [a.g, |a - (a.g)g|, w.g, |w - (w.g)g|]. Invariant to ANY device rotation, but it
    discards the horizontal direction -- the information our method keeps."""
    X = np.asarray(X, np.float32)
    g, _ = gravity_dirs(X)
    out = []
    for sl in (slice(0, 3), slice(3, 6)):
        v = X[:, :, sl].astype(np.float64)
        par = np.einsum("nti,ni->nt", v, g)
        perp = np.linalg.norm(v - par[..., None] * g[:, None, :], axis=-1)
        out += [par, perp]
    return np.stack(out, -1).astype(np.float32)


def _angle(a, b):
    na = np.linalg.norm(a, axis=-1); nb = np.linalg.norm(b, axis=-1)
    cos = (a * b).sum(-1) / np.maximum(na * nb, 1e-12)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _pad_to(x, L):
    if x.shape[1] == L:
        return x
    return np.concatenate([x, np.repeat(x[:, -1:], L - x.shape[1], axis=1)], 1)


def oit_transform(X):
    """Heuristic orientation-invariant transformation of Yurtman & Barshan (Sensors
    2017), per sensor: norms of v, dv, d2v; angles between successive v, dv, d2v;
    angles between successive cross products of v, dv, d2v -> 9 sequences per sensor,
    18 channels. VERIFY against Sec. 3 of the paper before submission; the
    sequences shorter than L are edge-padded to L."""
    X = np.asarray(X, np.float64)
    L = X.shape[1]
    chans = []
    for sl in (slice(0, 3), slice(3, 6)):
        v = X[:, :, sl]
        d1 = v[:, 1:] - v[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        seqs = [np.linalg.norm(v, axis=-1), np.linalg.norm(d1, axis=-1), np.linalg.norm(d2, axis=-1),
                _angle(v[:, :-1], v[:, 1:]), _angle(d1[:, :-1], d1[:, 1:]), _angle(d2[:, :-1], d2[:, 1:])]
        for u in (v, d1, d2):
            cr = np.cross(u[:, :-1], u[:, 1:])
            seqs.append(_angle(cr[:, :-1], cr[:, 1:]))
        chans += [_pad_to(s, L) for s in seqs]
    return np.stack(chans, -1).astype(np.float32)


KIND_DIM = {"raw": 6, "canon": 6, "pca": 6, "mizell": 4, "oit": 18}


def prep(X_raw, kind):
    if kind == "raw":
        return np.asarray(X_raw, np.float32)
    if kind == "canon":
        return canonicalize(X_raw)
    if kind == "pca":
        return pca_heading(canonicalize(X_raw))
    if kind == "mizell":
        return mizell_transform(X_raw)
    if kind == "oit":
        return oit_transform(X_raw)
    raise ValueError(kind)


_PREP_CACHE = {}


def prep_cached(name, label_mode, kind):
    X, y, g, meta, fp = get_ds(name, label_mode)
    key = (fp, kind)
    if key not in _PREP_CACHE:
        if len(_PREP_CACHE) > 12:
            _PREP_CACHE.pop(next(iter(_PREP_CACHE)))
        _PREP_CACHE[key] = prep(X, kind)
    return _PREP_CACHE[key], y, g, meta, fp


# =============================================================================
#  3. TRAINING  (one pretraining and one fine-tuning path for every arm)
# =============================================================================
def _haar_rot_torch(B, device):
    q = torch.randn(B, 4, device=device)
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], 1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], 1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1)], 1)


def _apply_R_torch(x, R):
    out = x.clone()
    out[:, :, 0:3] = torch.einsum("bij,blj->bli", R, x[:, :, 0:3])
    out[:, :, 3:6] = torch.einsum("bij,blj->bli", R, x[:, :, 3:6])
    return out


def _axis_angle_torch(axis, th):
    a = axis / axis.norm(dim=1, keepdim=True).clamp(min=1e-9)
    z = torch.zeros_like(a[:, 0])
    K = torch.stack([torch.stack([z, -a[:, 2], a[:, 1]], 1),
                     torch.stack([a[:, 2], z, -a[:, 0]], 1),
                     torch.stack([-a[:, 1], a[:, 0], z], 1)], 1)
    I = torch.eye(3, device=a.device).expand(len(a), 3, 3)
    s = torch.sin(th).view(-1, 1, 1); c = torch.cos(th).view(-1, 1, 1)
    return I + s * K + (1 - c) * torch.bmm(K, K)


def op_yaw(x):
    """Rotation about the z axis, theta ~ U(-180,180): the canonical vertical after
    canonicalization (Eq. 7); the device z axis in the raw frame (arm C)."""
    return op_rot_z(x, 180.0)


def op_so3(x):
    """Haar-uniform random rotation per window (acc and gyro jointly)."""
    return _apply_R_torch(x, _haar_rot_torch(x.size(0), x.device))


def make_axis_op(axis=(1.0, 0.0, 0.0), max_deg=180.0):
    def op(x):
        B = x.size(0)
        th = (torch.rand(B, device=x.device) * 2 - 1) * math.radians(max_deg)
        ax = torch.tensor(axis, dtype=torch.float32, device=x.device).expand(B, 3)
        return _apply_R_torch(x, _axis_angle_torch(ax, th))
    return op


def make_so3_angle_op(max_deg=180.0):
    """Random axis, angle ~ U(-max,max) (the draft's 'SO(3)' control in Fig. 11)."""
    def op(x):
        B = x.size(0)
        th = (torch.rand(B, device=x.device) * 2 - 1) * math.radians(max_deg)
        return _apply_R_torch(x, _axis_angle_torch(torch.randn(B, 3, device=x.device), th))
    return op


AUG_OPS = {None: None, "yaw": op_yaw, "so3": op_so3}


def pretrain_rev(X, objective="mlm", seed=42, lam=None, warm=None, verbose=False):
    """Span-masked reconstruction (LIMU-BERT defaults, not tuned) + optional
    consistency between two independently rotated views: objective 'yaw' rotates
    about z, 'so3' draws Haar-uniform rotations. Early stopping monitors the
    plateau of the reconstruction loss on the SOURCE pool itself (c48)."""
    lam = REV["lam"] if lam is None else lam
    warm = REV["warm"] if warm is None else warm
    s_dim = X.shape[2]
    set_seed(seed)
    model = LimuBERT(s_dim=s_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=REV["pre_lr"])
    Xt = torch.tensor(X, dtype=torch.float32); n = len(Xt)
    op = {"yaw": op_yaw, "so3": op_so3}.get(objective)
    if op is not None and s_dim != 6:
        raise ValueError("rotation consistency needs the 6-channel acc+gyro input")
    best, stall, hist = float("inf"), 0, []
    for ep in range(REV["pre_epochs"]):
        lam_ep = 0.0 if op is None else (lam if warm <= 0 else lam * min(1.0, (ep + 1) / warm))
        model.train(); perm = torch.randperm(n); tm = tc = 0.0
        for i in range(0, n, REV["pre_bs"]):
            xb = Xt[perm[i:i + REV["pre_bs"]]].to(DEVICE)
            xm, mask = span_mask(xb)
            l_mlm = F.mse_loss(model(xm)[mask], xb[mask])
            loss = l_mlm
            if lam_ep > 0:
                e1 = model.encoder(op(op_jitter(xb, REV["view_jitter"]))).mean(1)
                e2 = model.encoder(op(op_jitter(xb, REV["view_jitter"]))).mean(1)
                l_c = 1.0 - F.cosine_similarity(e1, e2, dim=1).mean()
                loss = loss + lam_ep * l_c
                tc += float(l_c) * xb.size(0)
            opt.zero_grad(); loss.backward(); opt.step()
            tm += l_mlm.item() * xb.size(0)
        mlm = tm / n
        hist.append({"ep": ep, "lam": lam_ep, "mlm": mlm, "cons": tc / n})
        if verbose and (ep % 10 == 0):
            _p(f"    [pre] ep {ep:3d} lam {lam_ep:.3f} mlm {mlm:.4f} cons {tc / n:.4f}")
        if mlm < best * 0.995:
            best, stall = mlm, 0
        else:
            stall += 1
            if ep >= REV["pre_min_epochs"] and stall >= REV["pre_patience"]:
                break
    return {k: v.cpu() for k, v in model.encoder.state_dict().items()}, hist


def get_encoder_rev(name, kind, objective, seed, label_mode="merged", lam=None, warm=None,
                    X_override=None, tag_extra=""):
    """Source-scope encoder: pretrained on the full unlabeled pool of `name` (no
    target data). Cached on Drive; the tag carries the data fingerprint, so the
    MotionSense sign fix cannot silently reuse an old encoder."""
    lam = REV["lam"] if lam is None else lam
    warm = REV["warm"] if warm is None else warm
    if X_override is None:
        X, _, _, _, fp = prep_cached(name, label_mode, kind)
    else:
        X, fp = X_override, _fingerprint(X_override)
    lam_t = lam if objective != "mlm" else 0.0
    warm_t = warm if objective != "mlm" else 0
    tag = (f"{REV_VERSION}_{name}_{fp}_{kind}_{objective}_l{lam_t}_w{warm_t}_s{seed}"
           f"_e{REV['pre_epochs']}{tag_extra}")
    path = os.path.join(REV_DIR, "enc", tag + ".pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    t0 = time.time()
    enc, hist = pretrain_rev(X, objective, seed, lam=lam, warm=warm)
    torch.save(enc, path)
    pd.DataFrame(hist).to_csv(path.replace(".pt", "_log.csv"), index=False)
    _p(f"  [enc] {tag}  n={len(X)}  {len(hist)} ep  {(time.time() - t0) / 60:.1f} min")
    return enc


class GRUHeadAux(nn.Module):
    """The paper's GRU head (20-20-10) with an optional auxiliary vector joined
    before the last layers (used only by the posture side-channel arm, c44)."""
    def __init__(self, h_dim, n_classes, aux_dim=0):
        super().__init__()
        self.g1 = nn.GRU(h_dim, 20, batch_first=True)
        self.g2 = nn.GRU(20, 20, batch_first=True)
        self.g3 = nn.GRU(20, 10, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(10 + aux_dim, 10); self.fc2 = nn.Linear(10, n_classes)

    def forward(self, e, aux=None):
        h, _ = self.g1(e); h, _ = self.g2(h); h, _ = self.g3(h)
        h = self.drop(h[:, -1])
        if aux is not None:
            h = torch.cat([h, aux], 1)
        return self.fc2(F.relu(self.fc1(h)))


class HARModelRev(nn.Module):
    def __init__(self, n_classes, s_dim=6, aux_dim=0):
        super().__init__()
        self.encoder = LimuEncoder(s_dim=s_dim)
        self.head = GRUHeadAux(H_DIM, n_classes, aux_dim)

    def forward(self, x, aux=None):
        return self.head(self.encoder(x), aux)


def finetune_rev(enc_state, X, y, n_classes, aug=None, seed=42, floor=None, aux=None,
                 epochs=None):
    """Fine-tune encoder + GRU head end to end, class-weighted CE (Eq. 8). If `aug` is
    set, each batch passes clean with probability `floor`, otherwise through the op.
    Fixed epoch count, no early stopping, no validation data of any kind (c48)."""
    floor = REV["floor"] if floor is None else floor
    epochs = REV["ft_epochs"] if epochs is None else epochs
    set_seed(seed)
    s_dim = X.shape[2]
    aux_dim = 0 if aux is None else aux.shape[1]
    model = HARModelRev(n_classes, s_dim, aux_dim).to(DEVICE)
    if enc_state is not None:
        model.encoder.load_state_dict({k: v.to(DEVICE) for k, v in enc_state.items()})
    cnt = np.bincount(np.asarray(y), minlength=n_classes).astype(float); cnt[cnt == 0] = 1
    cw = torch.tensor(len(y) / (n_classes * cnt), dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=REV["ft_lr"])
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(np.asarray(y), dtype=torch.long)
    At = None if aux is None else torch.tensor(aux, dtype=torch.float32)
    op = aug if callable(aug) else AUG_OPS[aug]
    n, bs = len(Xt), REV["ft_bs"]
    for _ in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = Xt[idx].to(DEVICE); yb = yt[idx].to(DEVICE)
            ab = None if At is None else At[idx].to(DEVICE)
            if op is not None and random.random() >= floor:
                with torch.no_grad():
                    xb = op(xb)
            loss = F.cross_entropy(model(xb, ab), yb, weight=cw)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def predict_rev(model, X, aux=None, bs=1024):
    model.eval(); out = []
    Xt = torch.tensor(X, dtype=torch.float32)
    At = None if aux is None else torch.tensor(aux, dtype=torch.float32)
    for i in range(0, len(Xt), bs):
        ab = None if At is None else At[i:i + bs].to(DEVICE)
        out.append(model(Xt[i:i + bs].to(DEVICE), ab).argmax(1).cpu())
    return torch.cat(out).numpy()


def metrics_rev(y_true, y_pred, labels):
    """Macro-F1 over the pair's shared classes; stairs-F1 = macro-F1 over {up, down}
    computed from the SAME predictions, defined for every pair that has both
    stairs classes (all 12 here). Absolute F1, never relative % (c25)."""
    per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    st = (f1_score(y_true, y_pred, labels=list(STAIRS), average="macro", zero_division=0)
          if all(s in labels for s in STAIRS) else float("nan"))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    wad = float("nan")
    if 4 in labels and 3 in labels:
        r = cm[labels.index(4)]
        wad = float(r[labels.index(3)] / max(r.sum(), 1))
    return {"macro_f1": float(per.mean()), "stairs_f1": float(st),
            "acc": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
            "per_class": json.dumps({int(l): round(float(v), 4) for l, v in zip(labels, per)}),
            "confusion": json.dumps(cm.tolist()), "walk_as_down": wad}


# =============================================================================
#  4. RUN LOG  (one row per training run, resumable)
# =============================================================================
LOG_COLS = ["exp", "source", "target", "arm", "pre_seed", "ft_seed", "fold", "perturb",
            "angle", "labels", "n_train", "n_test", "macro_f1", "stairs_f1", "acc",
            "per_class", "confusion", "walk_as_down", "extra", "cfg", "data_fp", "time"]


class RunLog:
    def __init__(self, name):
        self.path = os.path.join(REV_DIR, "runs", f"{name}.csv")
        self.keys = set()
        if os.path.exists(self.path):
            prev = pd.read_csv(self.path)
            for _, r in prev.iterrows():
                self.keys.add(self._key(r))

    @staticmethod
    def _key(r):
        ang = r.get("angle", 0.0)
        ang = 0.0 if ang is None or (isinstance(ang, float) and np.isnan(ang)) else float(ang)
        return (str(r["exp"]), str(r["source"]), str(r["target"]), str(r["arm"]),
                int(r["pre_seed"]), int(r["ft_seed"]), int(r["fold"]), str(r["perturb"]),
                round(ang, 3))

    def has(self, **k):
        r = dict(fold=-1, perturb="none", angle=0.0, pre_seed=-1, ft_seed=-1)
        r.update(k)
        return self._key(r) in self.keys

    def add(self, row):
        r = {c: row.get(c, "") for c in LOG_COLS}
        for k, v in dict(fold=-1, perturb="none", angle=0.0, pre_seed=-1, ft_seed=-1).items():
            if r[k] == "":
                r[k] = v
        r["cfg"] = row.get("cfg", json.dumps({k: REV[k] for k in ("lam", "warm", "floor")}))
        r["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        pd.DataFrame([r], columns=LOG_COLS).to_csv(
            self.path, mode="a", header=not os.path.exists(self.path), index=False)
        self.keys.add(self._key(r))

    def df(self):
        return pd.read_csv(self.path) if os.path.exists(self.path) else pd.DataFrame(columns=LOG_COLS)


# =============================================================================
#  5. ARMS
# -----------------------------------------------------------------------------
#  kind      input transform (raw device frame / canonicalized / ...)
#  obj       pretraining objective (mlm / mlm+yaw consistency / mlm+SO(3) consistency)
#  aug       fine-tuning operator
# =============================================================================
ARMS_REV = {
    # 2x2 component ablation (advisor's email: Baseline SSL -> +Gravity -> +Yaw -> both)
    "A_ssl":     dict(kind="raw",   obj="mlm", aug=None,  label="Baseline SSL (LIMU-BERT)"),
    "B_grav":    dict(kind="canon", obj="mlm", aug=None,  label="+ Gravity canonicalization"),
    "C_yaw":     dict(kind="raw",   obj="yaw", aug="yaw", label="+ Yaw invariance only"),
    "D_gravyaw": dict(kind="canon", obj="yaw", aug="yaw", label="Gravity + yaw invariance (ours)"),
    # where the yaw invariance has to live (c39 c41)
    "D_pre":     dict(kind="canon", obj="yaw", aug=None,  label="Gravity + yaw consistency only"),
    "D_ft":      dict(kind="canon", obj="mlm", aug="yaw", label="Gravity + yaw augmentation only"),
    # baselines requested in c46 and by the e-mail ('lack of the baseline',
    # 'the gravity-based approach has existed before')
    "E_so3aug":  dict(kind="raw",   obj="mlm", aug="so3", label="LIMU-BERT + SO(3) augmentation"),
    "F_so3inv":  dict(kind="raw",   obj="so3", aug="so3", label="Learned SO(3) invariance (no canon.)"),
    "G_pca":     dict(kind="pca",   obj="mlm", aug=None,  label="Gravity + PCA heading alignment"),
    "H_mizell":  dict(kind="mizell", obj="mlm", aug=None, label="Vertical/horizontal decomposition [Mizell]"),
    "I_oit":     dict(kind="oit",   obj="mlm", aug=None,  label="Orientation-invariant transform [Yurtman]"),
}
ABLATION_ARMS = ["A_ssl", "B_grav", "C_yaw", "D_gravyaw"]
SITE_ARMS = ["D_pre", "D_ft"]
BASELINE_ARMS = ["E_so3aug", "F_so3inv", "G_pca", "H_mizell", "I_oit"]


def pair_data(src, tgt, kind, label_mode="merged"):
    Xs, ys, gs, ms, fps = prep_cached(src, label_mode, kind)
    Xt, yt, gt, mt, fpt = prep_cached(tgt, label_mode, kind)
    labels = sorted(int(c) for c in set(np.unique(ys)) & set(np.unique(yt)))
    a, b = np.isin(ys, labels), np.isin(yt, labels)
    lmap = {c: i for i, c in enumerate(labels)}
    return (Xs[a], np.array([lmap[v] for v in ys[a]]), Xt[b], yt[b], labels,
            f"{fps}|{fpt}", a, b)


def _run_cell(log, exp, src, tgt, arm, Xs, ys_idx, Xt, yt, labels, enc_src, label_mode,
              pre_seeds, ft_seeds, fold=-1, aux_s=None, aux_t=None, data_fp="", extra=None,
              enc_kw=None, ft_kw=None):
    a = ARMS_REV[arm] if isinstance(arm, str) else arm
    name = arm if isinstance(arm, str) else a["name"]
    for ps in pre_seeds:
        enc = None
        for fs in ft_seeds:
            if log.has(exp=exp, source=src, target=tgt, arm=name, pre_seed=ps, ft_seed=fs, fold=fold):
                continue
            if enc is None:
                enc = (None if a["obj"] == "none" else
                       get_encoder_rev(enc_src, a["kind"], a["obj"], ps, label_mode, **(enc_kw or {})))
            m = finetune_rev(enc, Xs, ys_idx, len(labels), aug=a["aug"], seed=fs, aux=aux_s,
                             **(ft_kw or {}))
            pred = np.array([labels[p] for p in predict_rev(m, Xt, aux_t)])
            r = metrics_rev(yt, pred, labels)
            log.add(dict(exp=exp, source=src, target=tgt, arm=name, pre_seed=ps, ft_seed=fs,
                         fold=fold, labels=json.dumps(labels), n_train=len(Xs), n_test=len(Xt),
                         data_fp=data_fp, extra=json.dumps(extra or {}), **r))
            _p(f"  [{exp}] {src}->{tgt} {name:10s} fold {fold:2d} pre {ps} ft {fs}  "
               f"F1 {r['macro_f1']:.3f}  stairs {r['stairs_f1']:.3f}")


# =============================================================================
#  6. EXPERIMENTS
# =============================================================================
#  E-MAIN  12-pair matrix: ablation + yaw site + baselines            [RQ1 RQ2 RQ4]
#  c26 c41 c46 c47 c49 c52 c53 c57 c62 c63 c67 + e-mail (baselines)
# -----------------------------------------------------------------------------
def exp_main(arms=None, pairs=None, budget=None, label_mode="merged", log_name="main"):
    arms = arms or (ABLATION_ARMS + SITE_ARMS + BASELINE_ARMS)
    pairs = pairs or PAIRS
    budget = budget or REV["budget"]
    log = RunLog(log_name)
    for src, tgt in pairs:
        for arm in arms:
            kind = ARMS_REV[arm]["kind"]
            Xs, ys, Xt, yt, labels, fp, _, _ = pair_data(src, tgt, kind, label_mode)
            _run_cell(log, log_name, src, tgt, arm, Xs, ys, Xt, yt, labels, src, label_mode,
                      budget["pre"], budget["ft"], data_fp=fp)
    return log.df()


# -----------------------------------------------------------------------------
#  E-XGB  engineered features, raw and canonicalized frame, all 12 pairs, both
#  metrics from the same predictions (c25 c26 c30 c52 c54 c62 c63 c73)
# -----------------------------------------------------------------------------
_XGB_FEAT = {}


def xgb_features(name, kind, label_mode="merged"):
    X, y, g, meta, fp = get_ds(name, label_mode)
    key = (fp, kind)
    if key in _XGB_FEAT:
        return _XGB_FEAT[key]
    path = os.path.join(REV_DIR, "cache", f"xgbfeat_{name}_{fp}_{kind}.pkl")
    if os.path.exists(path):
        df = pd.read_pickle(path)
    else:
        _p(f"  [xgb] extracting features {name}/{kind} ({len(X)} windows, CPU) ...")
        df = _features_df_from_raw(prep(X, kind)).reset_index(drop=True)
        df.to_pickle(path)
    _XGB_FEAT[key] = df
    return df


def _xgb_fit_predict(Fs, ys_idx, Ft, n_classes, seed):
    cnt = np.bincount(ys_idx, minlength=n_classes).astype(float); cnt[cnt == 0] = 1
    w = np.array([len(ys_idx) / (n_classes * cnt[c]) for c in ys_idx])
    mdl = make_xgb(n_classes); mdl.set_params(random_state=seed)
    mdl.fit(Fs, ys_idx, sample_weight=w)
    return mdl.predict(Ft)


def exp_xgb(pairs=None, kinds=("raw", "canon"), seeds=(42, 43, 44), label_mode="merged"):
    pairs = pairs or PAIRS
    log = RunLog("xgb")
    for src, tgt in pairs:
        for kind in kinds:
            arm = f"XGB_{kind}"
            if all(log.has(exp="xgb", source=src, target=tgt, arm=arm, pre_seed=0, ft_seed=s)
                   for s in seeds):
                continue
            _, ys, _, _, fps = get_ds(src, label_mode)
            _, yt, _, _, fpt = get_ds(tgt, label_mode)
            Fs, Ft = xgb_features(src, kind, label_mode), xgb_features(tgt, kind, label_mode)
            labels = sorted(int(c) for c in set(np.unique(ys)) & set(np.unique(yt)))
            a, b = np.isin(ys, labels), np.isin(yt, labels)
            cols = [c for c in Fs.columns if c in Ft.columns]
            lmap = {c: i for i, c in enumerate(labels)}
            ys_idx = np.array([lmap[v] for v in ys[a]])
            for s in seeds:
                if log.has(exp="xgb", source=src, target=tgt, arm=arm, pre_seed=0, ft_seed=s):
                    continue
                p = _xgb_fit_predict(Fs.loc[a, cols], ys_idx, Ft.loc[b, cols], len(labels), s)
                r = metrics_rev(yt[b], np.array([labels[i] for i in p]), labels)
                log.add(dict(exp="xgb", source=src, target=tgt, arm=arm, pre_seed=0, ft_seed=s,
                             labels=json.dumps(labels), n_train=int(a.sum()), n_test=int(b.sum()),
                             data_fp=f"{fps}|{fpt}", **r))
                _p(f"  [xgb] {src}->{tgt} {arm:9s} seed {s}  F1 {r['macro_f1']:.3f} "
                   f"stairs {r['stairs_f1']:.3f}")
    return log.df()


# -----------------------------------------------------------------------------
#  E-CEIL  in-domain ceiling on the SAME label set as each pair (c29 c64 c74)
#  Subject-disjoint 3-fold CV inside the target. Encoders are the target's own
#  source-scope encoders (pretrained on its whole unlabeled pool, i.e. the
#  ceiling is if anything optimistic -> the 'gap closed' fractions are
#  conservative). Gives: gap = ceiling - cross, orientation share of the gap.
# -----------------------------------------------------------------------------
def exp_ceiling(arms=("A_ssl", "B_grav", "D_gravyaw"), xgb_kinds=("raw", "canon"),
                pairs=None, n_folds=3, pre_seeds=(42,), ft_seeds=(42, 43), label_mode="merged"):
    pairs = pairs or PAIRS
    log = RunLog("ceiling")
    combos = {}
    for src, tgt in pairs:
        _, ys, _, _, _ = get_ds(src, label_mode)
        _, yt, _, _, _ = get_ds(tgt, label_mode)
        labels = tuple(sorted(int(c) for c in set(np.unique(ys)) & set(np.unique(yt))))
        combos.setdefault((tgt, labels), []).append(src)
    for (tgt, labels), srcs in combos.items():
        labels = list(labels)
        lab_tag = "-".join(map(str, labels))
        Xr, y, g, _, fp = get_ds(tgt, label_mode)
        keep = np.isin(y, labels)
        idx_all = np.where(keep)[0]
        folds = list(GroupKFold(n_splits=n_folds).split(idx_all, y[idx_all], g[idx_all]))
        lmap = {c: i for i, c in enumerate(labels)}
        for k, (tr, te) in enumerate(folds):
            itr, ite = idx_all[tr], idx_all[te]
            for arm in arms:
                Xk, _, _, _, _ = prep_cached(tgt, label_mode, ARMS_REV[arm]["kind"])
                _run_cell(log, "ceiling", f"in:{tgt}", f"{tgt}[{lab_tag}]", arm, Xk[itr],
                          np.array([lmap[v] for v in y[itr]]), Xk[ite], y[ite], labels, tgt,
                          label_mode, pre_seeds, ft_seeds, fold=k, data_fp=fp,
                          extra={"for_sources": srcs})
            for kind in xgb_kinds:
                arm = f"XGB_{kind}"
                if log.has(exp="ceiling", source=f"in:{tgt}", target=f"{tgt}[{lab_tag}]", arm=arm,
                           pre_seed=0, ft_seed=42, fold=k):
                    continue
                Fd = xgb_features(tgt, kind, label_mode)
                p = _xgb_fit_predict(Fd.iloc[itr], np.array([lmap[v] for v in y[itr]]),
                                     Fd.iloc[ite], len(labels), 42)
                r = metrics_rev(y[ite], np.array([labels[i] for i in p]), labels)
                log.add(dict(exp="ceiling", source=f"in:{tgt}", target=f"{tgt}[{lab_tag}]", arm=arm,
                             pre_seed=0, ft_seed=42, fold=k, labels=json.dumps(labels),
                             n_train=len(itr), n_test=len(ite), data_fp=fp,
                             extra=json.dumps({"for_sources": srcs}), **r))
    return log.df()


# -----------------------------------------------------------------------------
#  E-ROT  controlled rotations on ALL FOUR datasets, subject-disjoint   [RQ3]
#  advisor e-mail (controlled pitch/roll/yaw), c37 c50 c51, limitation 'single
#  dataset'. Two families of perturbation:
#    device_x/y/z  : rotation about a fixed DEVICE axis (what the draft called
#                    pitch/roll/yaw). Its physical meaning depends on the pose:
#                    on UCI-HAR gravity lies along device x, so 'pitch' about x
#                    is a physical heading change -- this is the answer to c37.
#    phys_tilt     : rotation about a HORIZONTAL axis (perpendicular to each
#                    window's own gravity, random azimuth) = physical pitch/roll.
#    phys_heading  : rotation about each window's own gravity = physical yaw.
#  For every test window we also log the residual heading psi left after
#  canonicalization (Prop. 1); accuracy binned by |psi| shows that a
#  canonicalized model depends on the rotation ONLY through psi.
# -----------------------------------------------------------------------------
ROT_PERTURB = {"device_x": (0, 30, 60, 90, 120, 150, 180),
               "device_y": (0, 30, 60, 90, 120, 150, 180),
               "device_z": (0, 30, 60, 90, 120, 150, 180),
               "phys_tilt": (0, 15, 30, 45, 60, 75, 90),
               "phys_heading": (0, 30, 60, 90, 120, 150, 180)}
PSI_BINS = (0, 10, 30, 60, 90, 120, 150, 180.01)


def axis_angle_np(axis, deg):
    axis = np.asarray(axis, np.float64)
    if axis.ndim == 1:
        axis = axis[None]
    a = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
    K = _skew(a); t = math.radians(deg)
    return np.eye(3)[None] + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)


def make_perturbation(X_raw, perturb, deg, seed=0):
    """Per-window rotation matrices Q [N,3,3] for one perturbation family/angle."""
    N = len(X_raw)
    if deg == 0:
        return np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    if perturb.startswith("device_"):
        ax = {"device_x": (1, 0, 0), "device_y": (0, 1, 0), "device_z": (0, 0, 1)}[perturb]
        return np.broadcast_to(axis_angle_np(ax, deg)[0], (N, 3, 3)).copy()
    g, _ = gravity_dirs(X_raw)
    if perturb == "phys_heading":
        return axis_angle_np(g, deg)
    if perturb == "phys_tilt":
        rng = np.random.default_rng(seed)
        u = rng.normal(size=(N, 3))
        h = np.cross(g, u)
        return axis_angle_np(h, deg)
    raise ValueError(perturb)


def residual_heading(g, Q):
    """psi (deg) of S(Q) = R(Q g) Q R(g)^T, the rotation left on the canonicalized
    window (Prop. 1: always a rotation about z)."""
    R0 = rot_to_vertical_batch(g)
    g1 = np.einsum("nij,nj->ni", Q, g)
    R1 = rot_to_vertical_batch(g1 / np.linalg.norm(g1, axis=1, keepdims=True))
    S = R1 @ Q @ np.transpose(R0, (0, 2, 1))
    return np.degrees(np.arctan2(S[:, 1, 0], S[:, 0, 0])), S


def exp_rotation(datasets=None, arms=None, perturbs=None, n_folds=3, pre_seeds=(42,),
                 ft_seeds=(42, 43), strict_pretrain=False):
    """strict_pretrain=False reuses the dataset's source-scope encoder (pretrained on
    unrotated windows of all subjects; labels and rotations never enter it). Set True
    to pretrain per fold on training subjects only (3x the encoders)."""
    datasets = datasets or DATASETS_REV
    arms = arms or (ABLATION_ARMS + ["E_so3aug", "G_pca", "H_mizell"])
    perturbs = perturbs or list(ROT_PERTURB)
    log = RunLog("rotation")
    for dname in datasets:
        Xr, y, g, _, fp = get_ds(dname)
        labels = sorted(int(c) for c in np.unique(y))
        lmap = {c: i for i, c in enumerate(labels)}
        folds = list(GroupKFold(n_splits=n_folds).split(Xr, y, g))
        gdir, _ = gravity_dirs(Xr)
        for k, (tr, te) in enumerate(folds):
            for arm in arms:
                a = ARMS_REV[arm]
                todo = [(pt, d) for pt in perturbs for d in ROT_PERTURB[pt]
                        if not all(log.has(exp="rotation", source=dname, target=dname, arm=arm,
                                           pre_seed=ps, ft_seed=fs, fold=k, perturb=pt, angle=d)
                                   for ps in pre_seeds for fs in ft_seeds)]
                if not todo:
                    continue
                Xtr = prep(Xr[tr], a["kind"])
                ytr = np.array([lmap[v] for v in y[tr]])
                for ps in pre_seeds:
                    if strict_pretrain:
                        enc = get_encoder_rev(dname, a["kind"], a["obj"], ps,
                                              X_override=prep(Xr[tr], a["kind"]),
                                              tag_extra=f"_fold{k}of{n_folds}")
                    else:
                        enc = get_encoder_rev(dname, a["kind"], a["obj"], ps)
                    for fs in ft_seeds:
                        m = None
                        for pt, d in todo:
                            if log.has(exp="rotation", source=dname, target=dname, arm=arm,
                                       pre_seed=ps, ft_seed=fs, fold=k, perturb=pt, angle=d):
                                continue
                            if m is None:
                                m = finetune_rev(enc, Xtr, ytr, len(labels), aug=a["aug"], seed=fs)
                            Q = make_perturbation(Xr[te], pt, d, seed=1000 * k + int(d))
                            Xte = prep(rotate_windows(Xr[te], Q), a["kind"])
                            pred = np.array([labels[p] for p in predict_rev(m, Xte)])
                            r = metrics_rev(y[te], pred, labels)
                            psi, _ = residual_heading(gdir[te], Q)
                            ok = pred == y[te]
                            b = np.digitize(np.abs(psi), PSI_BINS) - 1
                            bins = {str(PSI_BINS[i]): [int(ok[b == i].sum()), int((b == i).sum())]
                                    for i in range(len(PSI_BINS) - 1) if (b == i).any()}
                            log.add(dict(exp="rotation", source=dname, target=dname, arm=arm,
                                         pre_seed=ps, ft_seed=fs, fold=k, perturb=pt, angle=d,
                                         labels=json.dumps(labels), n_train=len(tr), n_test=len(te),
                                         data_fp=fp, extra=json.dumps(
                                             {"psi_bins": bins,
                                              "psi_median": float(np.median(np.abs(psi)))}), **r))
                        _p(f"  [rot] {dname} {arm:10s} fold {k} pre {ps} ft {fs} done")
    return log.df()


# -----------------------------------------------------------------------------
#  E-POSTURE  cost of canonicalization on posture classes (c44)
#  sit / stand / lie are separated by device pose, which canonicalization
#  removes. Within-dataset (subject-disjoint) and cross-dataset on the full label
#  set. D_pose re-injects the removed pose (g_hat in device coordinates) into the
#  classifier head as a 3-d side channel.
# -----------------------------------------------------------------------------
def _pose_aux(X_raw):
    g, _ = gravity_dirs(X_raw)
    return g.astype(np.float32)


def exp_posture(datasets=("uci", "motion", "shoaib", "hhar"),
                pairs=(("uci", "motion"), ("motion", "uci"), ("uci", "shoaib"),
                       ("shoaib", "uci"), ("motion", "shoaib"), ("shoaib", "motion")),
                arms=("A_ssl", "B_grav", "D_gravyaw", "D_pose"), n_folds=3,
                pre_seeds=(42,), ft_seeds=(42, 43)):
    log = RunLog("posture")
    lm = "posture"

    def _arm(a):
        return ARMS_REV["D_gravyaw"] if a == "D_pose" else ARMS_REV[a]

    for dname in datasets:
        Xr, y, g, _, fp = get_ds(dname, lm)
        labels = sorted(int(c) for c in np.unique(y))
        lmap = {c: i for i, c in enumerate(labels)}
        for k, (tr, te) in enumerate(GroupKFold(n_splits=n_folds).split(Xr, y, g)):
            for arm in arms:
                a = _arm(arm)
                Xk, _, _, _, _ = prep_cached(dname, lm, a["kind"])
                aux = _pose_aux(Xr) if arm == "D_pose" else None
                _run_cell(log, "posture", dname, dname, dict(a, name=arm), Xk[tr],
                          np.array([lmap[v] for v in y[tr]]), Xk[te], y[te], labels, dname, lm,
                          pre_seeds, ft_seeds, fold=k, data_fp=fp,
                          aux_s=None if aux is None else aux[tr],
                          aux_t=None if aux is None else aux[te])
    for src, tgt in pairs:
        for arm in arms:
            a = _arm(arm)
            Xs, ys, Xt, yt, labels, fp, ms, mt = pair_data(src, tgt, a["kind"], lm)
            aux_s = aux_t = None
            if arm == "D_pose":
                aux_s = _pose_aux(get_ds(src, lm)[0][ms]); aux_t = _pose_aux(get_ds(tgt, lm)[0][mt])
            _run_cell(log, "posture", src, tgt, dict(a, name=arm), Xs, ys, Xt, yt, labels, src, lm,
                      pre_seeds, ft_seeds, data_fp=fp, aux_s=aux_s, aux_t=aux_t)
    return log.df()


# -----------------------------------------------------------------------------
#  E-HPARAM  source-only hyperparameter selection + target sensitivity (c40)
#  Selection criterion uses ONLY the source: macro-F1 on held-out SOURCE subjects
#  after a random Haar rotation of their raw windows (a source-only proxy for an
#  orientation shift). The whole grid is then reported on the targets, so the
#  reader sees both the selected point and how much the choice matters.
#  (The earlier E4 run chose lambda on the UCI->HHAR TARGET; it is superseded.)
# -----------------------------------------------------------------------------
def exp_hparam(lams=(0.03, 0.1, 0.3), floors=(0.0, 0.3, 0.5), sources=None, pairs=None,
               pre_seed=42, ft_seeds=(42, 43), val_frac=0.2):
    sources = sources or DATASETS_REV
    pairs = pairs or PAIRS
    log = RunLog("hparam")
    for src in sources:
        Xr, y, g, _, fp = get_ds(src)
        labels = sorted(int(c) for c in np.unique(y))
        lmap = {c: i for i, c in enumerate(labels)}
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=0)
                      .split(Xr, y, g))
        rng = np.random.default_rng(0)
        Qv = np.stack([_haar_np(rng) for _ in range(len(va))])
        Xva_rot = canonicalize(rotate_windows(Xr[va], Qv))
        Xva = canonicalize(Xr[va]); Xtr = canonicalize(Xr[tr])
        for lam in lams:
            enc = get_encoder_rev(src, "canon", "yaw", pre_seed, lam=lam)
            for fl in floors:
                arm = f"D_l{lam}_f{fl}"
                for fs in ft_seeds:
                    if log.has(exp="hparam_val", source=src, target=src, arm=arm,
                               pre_seed=pre_seed, ft_seed=fs):
                        continue
                    m = finetune_rev(enc, Xtr, np.array([lmap[v] for v in y[tr]]), len(labels),
                                     aug="yaw", seed=fs, floor=fl)
                    r_clean = metrics_rev(y[va], np.array([labels[p] for p in predict_rev(m, Xva)]), labels)
                    r = metrics_rev(y[va], np.array([labels[p] for p in predict_rev(m, Xva_rot)]), labels)
                    log.add(dict(exp="hparam_val", source=src, target=src, arm=arm, pre_seed=pre_seed,
                                 ft_seed=fs, labels=json.dumps(labels), n_train=len(tr), n_test=len(va),
                                 data_fp=fp, cfg=json.dumps({"lam": lam, "floor": fl, "warm": REV["warm"]}),
                                 extra=json.dumps({"clean_f1": r_clean["macro_f1"]}), **r))
                    _p(f"  [hp-val] {src} lam {lam} floor {fl} ft {fs}: rotated-val F1 "
                       f"{r['macro_f1']:.3f} (clean {r_clean['macro_f1']:.3f})")
    for src, tgt in pairs:
        Xs, ys, Xt, yt, labels, fp, _, _ = pair_data(src, tgt, "canon")
        for lam in lams:
            for fl in floors:
                arm = f"D_l{lam}_f{fl}"
                _run_cell(log, "hparam", src, tgt, dict(ARMS_REV["D_gravyaw"], name=arm), Xs, ys, Xt,
                          yt, labels, src, "merged", (pre_seed,), ft_seeds, data_fp=fp,
                          enc_kw={"lam": lam}, ft_kw={"floor": fl})
    return log.df()


def _haar_np(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# -----------------------------------------------------------------------------
#  E-AUGSPEC  is it the AXIS or any augmentation? (draft Fig. 11, re-run on the
#  corrected data with the new logger; all 12 pairs instead of 3/6)
# -----------------------------------------------------------------------------
def _perturb_mag(X, op, n=2048, seed=0):
    idx = np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)
    xb = torch.tensor(X[idx], dtype=torch.float32, device=DEVICE)
    torch.manual_seed(seed)
    with torch.no_grad():
        d = (op(xb) - xb).flatten(1).norm(dim=1) / xb.flatten(1).norm(dim=1).clamp(min=1e-9)
    return float(d.mean())


def _bisect(fn, target, lo, hi, iters=22):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if fn(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


def exp_aug_specificity(pairs=None, pre_seeds=(42, 43), ft_seeds=(42, 43)):
    pairs = pairs or PAIRS
    log = RunLog("augspec")
    Xref, _, _, _, _ = prep_cached("hhar", "merged", "canon")
    target = _perturb_mag(Xref, op_yaw)
    d_ax = _bisect(lambda d: _perturb_mag(Xref, make_axis_op((1, 0, 0), d)), target, 0.5, 180)
    d_so3 = _bisect(lambda d: _perturb_mag(Xref, make_so3_angle_op(d)), target, 0.5, 180)
    sig = _bisect(lambda s: _perturb_mag(Xref, lambda z: op_jitter(z, s)), target, 1e-3, 3.0)
    ops = {"none": None, "yaw180": op_yaw,
           "hax180": make_axis_op((1, 0, 0), 180.0), "so3_180": make_so3_angle_op(180.0),
           "hax_matched": make_axis_op((1, 0, 0), d_ax), "so3_matched": make_so3_angle_op(d_so3),
           "jitter_matched": (lambda z: op_jitter(z, sig))}
    cal = {"target_mag": target, "hax_deg": d_ax, "so3_deg": d_so3, "jitter_sigma": sig}
    json.dump(cal, open(os.path.join(REV_DIR, "tables", "augspec_calibration.json"), "w"), indent=2)
    _p(f"  [augspec] calibration {cal}")
    for src, tgt in pairs:
        Xs, ys, Xt, yt, labels, fp, _, _ = pair_data(src, tgt, "canon")
        for name, op in ops.items():
            _run_cell(log, "augspec", src, tgt, dict(kind="canon", obj="yaw", aug=op, name=name),
                      Xs, ys, Xt, yt, labels, src, "merged", pre_seeds, ft_seeds, data_fp=fp,
                      extra=cal)
    return log.df()


# -----------------------------------------------------------------------------
#  E-POS  Shoaib body-position transfer (real mounting changes, same subjects)
# -----------------------------------------------------------------------------
def exp_positions(arms=("A_ssl", "B_grav", "C_yaw", "D_gravyaw", "G_pca", "H_mizell"),
                  pre_seeds=(42,), ft_seeds=(42, 43, 44)):
    log = RunLog("positions")
    Xr, y, g, meta, fp = get_ds("shoaib")
    pos = meta["position"]
    for arm in arms:
        Xk, _, _, _, _ = prep_cached("shoaib", "merged", ARMS_REV[arm]["kind"])
        for ps_ in sorted(np.unique(pos)):
            for pt_ in sorted(np.unique(pos)):
                if ps_ == pt_:
                    continue
                si, ti = pos == ps_, pos == pt_
                labels = sorted(int(c) for c in set(np.unique(y[si])) & set(np.unique(y[ti])))
                lmap = {c: i for i, c in enumerate(labels)}
                si &= np.isin(y, labels); ti &= np.isin(y, labels)
                _run_cell(log, "positions", f"shoaib:{SHOAIB_POSITIONS[ps_]}",
                          f"shoaib:{SHOAIB_POSITIONS[pt_]}", arm, Xk[si],
                          np.array([lmap[v] for v in y[si]]), Xk[ti], y[ti], labels, "shoaib",
                          "merged", pre_seeds, ft_seeds, data_fp=fp)
    return log.df()


# -----------------------------------------------------------------------------
#  E-LABEL  label efficiency (c65), optional: k labelled target windows per class
#  added to the source set; evaluated on the remaining target windows.
# -----------------------------------------------------------------------------
def exp_label_efficiency(pairs=(("hhar", "uci"), ("shoaib", "motion")), ks=(0, 5, 10, 25, 50, 100),
                         arms=("A_ssl", "D_gravyaw"), pre_seeds=(42,), ft_seeds=(42, 43, 44, 45)):
    log = RunLog("labeleff")
    for src, tgt in pairs:
        for arm in arms:
            a = ARMS_REV[arm]
            Xs, ys, Xt, yt, labels, fp, _, _ = pair_data(src, tgt, a["kind"])
            lmap = {c: i for i, c in enumerate(labels)}
            for k in ks:
                for ps in pre_seeds:
                    enc = None
                    for fs in ft_seeds:
                        if log.has(exp="labeleff", source=src, target=tgt, arm=arm, pre_seed=ps,
                                   ft_seed=fs, perturb="k", angle=k):
                            continue
                        enc = enc or get_encoder_rev(src, a["kind"], a["obj"], ps)
                        rng = np.random.default_rng(fs)
                        if k > 0:
                            sel = np.concatenate([rng.choice(np.where(yt == c)[0], min(k, int((yt == c).sum())),
                                                             replace=False) for c in labels])
                            Xtr = np.concatenate([Xs, Xt[sel]])
                            ytr = np.concatenate([ys, [lmap[v] for v in yt[sel]]])
                            hold = np.setdiff1d(np.arange(len(yt)), sel)
                        else:
                            Xtr, ytr, hold = Xs, ys, np.arange(len(yt))
                        m = finetune_rev(enc, Xtr, ytr, len(labels), aug=a["aug"], seed=fs)
                        pred = np.array([labels[p] for p in predict_rev(m, Xt[hold])])
                        r = metrics_rev(yt[hold], pred, labels)
                        log.add(dict(exp="labeleff", source=src, target=tgt, arm=arm, pre_seed=ps,
                                     ft_seed=fs, perturb="k", angle=k, labels=json.dumps(labels),
                                     n_train=len(Xtr), n_test=len(hold), data_fp=fp, **r))
    return log.df()


# -----------------------------------------------------------------------------
#  E-AUTOAUG  AutoAugHAR on ALL 12 pairs (c47), optional
#    autoaug_raw  : the published setting -- device-frame LIMU-BERT + bilevel search
#    autoaug_ours : the draft's setting -- our canonicalized yaw-consistent encoder
#  Uses the pipeline's own search implementation (train_classifier 'search').
# -----------------------------------------------------------------------------
def exp_autoaug(pairs=None, settings=(("autoaug_raw", "raw", "mlm"), ("autoaug_ours", "canon", "yaw")),
                pre_seed=42, ft_seeds=(42, 43, 44), search_epochs=15):
    pairs = pairs or PAIRS
    log = RunLog("autoaug")
    for src, tgt in pairs:
        for name, kind, obj in settings:
            if all(log.has(exp="autoaug", source=src, target=tgt, arm=name, pre_seed=pre_seed,
                           ft_seed=fs) for fs in ft_seeds):
                continue
            Xs, ys, Xt, yt, labels, fp, _, _ = pair_data(src, tgt, kind)
            enc = get_encoder_rev(src, kind, obj, pre_seed)
            t0 = time.time()
            _, pol = train_classifier(enc, Xs, ys, len(labels), aug_mode="search",
                                      epochs=search_epochs, freeze=False, return_policy=True,
                                      seed=pre_seed)
            mins = (time.time() - t0) / 60
            names = [n for n, _ in LABEL_PRESERVING_OPS]
            top = sorted(zip(names, pol), key=lambda t: -t[1])[:5]
            for fs in ft_seeds:
                if log.has(exp="autoaug", source=src, target=tgt, arm=name, pre_seed=pre_seed, ft_seed=fs):
                    continue
                m = train_classifier(enc, Xs, ys, len(labels), aug_mode="fixed", fixed_policy=pol, seed=fs)
                m.eval()
                with torch.no_grad():
                    pr = []
                    for i in range(0, len(Xt), 1024):
                        pr.append(m(torch.tensor(Xt[i:i + 1024], dtype=torch.float32, device=DEVICE))
                                  .argmax(1).cpu())
                pred = np.array([labels[p] for p in torch.cat(pr).numpy()])
                r = metrics_rev(yt, pred, labels)
                log.add(dict(exp="autoaug", source=src, target=tgt, arm=name, pre_seed=pre_seed,
                             ft_seed=fs, labels=json.dumps(labels), n_train=len(Xs), n_test=len(Xt),
                             data_fp=fp, extra=json.dumps({"search_min": mins, "peak": float(np.max(pol)),
                                                           "top5": [(n, round(float(v), 4)) for n, v in top]}),
                             **r))
    return log.df()


# -----------------------------------------------------------------------------
#  E-UNIMTS  a published cross-dataset method (c46 item 3)
#  UniMTS (NeurIPS 2024) released checkpoint, fine-tuned on the SOURCE with the
#  authors' recipe (Adam 1e-4, CE, best train loss), evaluated on the target. Run
#  with the raw device frame (as published) and with our canonicalization in
#  front of it (does physics canonicalization still help a foundation model?).
#  Joint placement follows the authors' data.py: waist -> 9 (UCI-HAR; HHAR waist
#  pouch assumed identical), MotionSense pocket -> joints 1 and 5, Shoaib
#  positions -> [1, 5, 21, 20, 0].
# -----------------------------------------------------------------------------
UNIMTS_DIR = os.environ.get("UNIMTS_DIR", "/content/UniMTS")
UNIMTS_JOINTS = {"hhar": [9], "uci": [9], "motion": [1, 5],
                 "shoaib": {0: [1], 1: [5], 2: [21], 3: [20], 4: [0]}}


def setup_unimts():
    """Clone the official repo, install CLIP, fetch the released checkpoint."""
    if not os.path.isdir(UNIMTS_DIR):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/xiyuanzh/UniMTS.git",
                        UNIMTS_DIR], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "git+https://github.com/openai/CLIP.git", "huggingface_hub", "ftfy", "regex"],
                   check=True)
    from huggingface_hub import hf_hub_download
    ck = os.path.join(UNIMTS_DIR, "checkpoint", "UniMTS.pth")
    if not os.path.exists(ck):
        hf_hub_download(repo_id="xiyuanz/UniMTS", filename="checkpoint/UniMTS.pth", local_dir=UNIMTS_DIR)
    if UNIMTS_DIR not in sys.path:
        sys.path.insert(0, UNIMTS_DIR)
    return ck


def unimts_batch(Xp, joints, gyro=False, pad_to=200):
    """[B,51,6] (acc g, gyro rad/s, 20 Hz) -> UniMTS input [B, C, 200, 22, 1].
    Our windows are already at 20 Hz, so the authors' resampling is the identity;
    padding is 'wrap' as in their load_custom_data."""
    B, L, _ = Xp.shape
    allX = np.zeros((B, L, 22, 6), np.float32)
    acc = Xp[:, :, 0:3] * G_REV; gyr = Xp[:, :, 3:6]
    for i in range(B):
        for j in (joints[i] if isinstance(joints[0], (list, tuple)) else joints):
            allX[i, :, j, 0:3] = acc[i]; allX[i, :, j, 3:6] = gyr[i]
    if L < pad_to:
        allX = np.pad(allX, ((0, 0), (0, pad_to - L), (0, 0), (0, 0)), mode="wrap")
    allX = allX[:, :pad_to]
    if not gyro:
        allX = allX[..., 0:3]
    return torch.from_numpy(allX).permute(0, 3, 1, 2).unsqueeze(-1).contiguous()


def exp_unimts(pairs=None, frames=("raw", "canon"), epochs=10, ft_seeds=(42, 43), gyro=False,
               bs=64, lr=1e-4, max_train=None):
    from types import SimpleNamespace
    ck = setup_unimts()
    from contrastive import ContrastiveModule
    pairs = pairs or PAIRS
    log = RunLog("unimts")
    for src, tgt in pairs:
        for frame in frames:
            arm = f"UniMTS_{frame}"
            Xs_r, ys_all, _, ms_meta, fps = get_ds(src)
            Xt_r, yt_all, _, mt_meta, fpt = get_ds(tgt)
            labels = sorted(int(c) for c in set(np.unique(ys_all)) & set(np.unique(yt_all)))
            a, b = np.isin(ys_all, labels), np.isin(yt_all, labels)
            Xs, Xt = prep(Xs_r[a], frame), prep(Xt_r[b], frame)
            js = UNIMTS_JOINTS[src]; jt = UNIMTS_JOINTS[tgt]
            js = [js[int(p)] for p in ms_meta["position"][a]] if isinstance(js, dict) else js
            jt = [jt[int(p)] for p in mt_meta["position"][b]] if isinstance(jt, dict) else jt
            lmap = {c: i for i, c in enumerate(labels)}
            ys = np.array([lmap[v] for v in ys_all[a]]); yt = yt_all[b]
            for fs in ft_seeds:
                if log.has(exp="unimts", source=src, target=tgt, arm=arm, pre_seed=0, ft_seed=fs):
                    continue
                set_seed(fs)
                args = SimpleNamespace(gyro=int(gyro), stft=0, stage="finetune", num_class=len(labels))
                model = ContrastiveModule(args).to(DEVICE)
                model.model.load_state_dict(torch.load(ck, map_location=DEVICE))
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                tr_idx = np.arange(len(Xs))
                if max_train and len(tr_idx) > max_train:
                    tr_idx = np.random.default_rng(fs).choice(tr_idx, max_train, replace=False)
                best, best_state = None, None
                for ep in range(epochs):
                    model.train(); perm = np.random.permutation(tr_idx); tot = 0.0
                    for i in range(0, len(perm), bs):
                        ii = perm[i:i + bs]
                        jj = [js[k] for k in ii] if isinstance(js, list) and isinstance(js[0], list) else js
                        xb = unimts_batch(Xs[ii], jj, gyro).to(DEVICE)
                        yb = torch.tensor(ys[ii], dtype=torch.long, device=DEVICE)
                        loss = F.cross_entropy(model.classifier(xb).float(), yb)
                        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(ii)
                    if best is None or tot < best:
                        best = tot
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    _p(f"  [unimts] {src}->{tgt} {frame} seed {fs} ep {ep} loss {tot / len(perm):.4f}")
                model.load_state_dict(best_state); model.eval()
                pr = []
                with torch.no_grad():
                    for i in range(0, len(Xt), 256):
                        jj = (jt[i:i + 256] if isinstance(jt, list) and isinstance(jt[0], list) else jt)
                        pr.append(model.classifier(unimts_batch(Xt[i:i + 256], jj, gyro).to(DEVICE))
                                  .argmax(1).cpu())
                pred = np.array([labels[p] for p in torch.cat(pr).numpy()])
                r = metrics_rev(yt, pred, labels)
                log.add(dict(exp="unimts", source=src, target=tgt, arm=arm, pre_seed=0, ft_seed=fs,
                             labels=json.dumps(labels), n_train=len(tr_idx), n_test=len(Xt),
                             data_fp=f"{fps}|{fpt}", extra=json.dumps({"epochs": epochs, "gyro": gyro}),
                             **r))
                _p(f"  [unimts] {src}->{tgt} {arm} seed {fs}  F1 {r['macro_f1']:.3f}")
                del model; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
    return log.df()


# =============================================================================
#  7. ANALYSES WITHOUT TRAINING (minutes on CPU)
# =============================================================================
def audit_motionsense_sign(n=4000):
    """c33. Pose statistics of MotionSense with and without the iOS->Android sign
    conversion, and what the error does after canonicalization: the vertical
    channel is identical, the horizontal plane is MIRRORED (an improper
    transform that no yaw rotation can undo), and in the raw device frame the
    whole accelerometer is negated relative to the other three datasets."""
    old = REV["ios_sign_fix"]
    try:
        REV["ios_sign_fix"] = True; Xf = get_ds("motion")[0]
        REV["ios_sign_fix"] = False; Xo, y = get_ds("motion")[0:2]
    finally:
        REV["ios_sign_fix"] = old
    idx = np.random.default_rng(0).choice(len(Xf), min(n, len(Xf)), replace=False)
    rows = []
    for tag, X in (("as loaded in draft 2 (iOS sign)", Xo), ("converted to Android sign", Xf)):
        g, nrm = gravity_dirs(X[idx])
        rows.append({"version": tag, "median_gz": float(np.median(g[:, 2])),
                     "frac_gz<0 ('inverted')": float((g[:, 2] < 0).mean()),
                     "frac_gz<-0.9": float((g[:, 2] < -0.9).mean()),
                     "median_|a_bar|_g": float(np.median(nrm))})
    ca, cb = canonicalize(Xo[idx]), canonicalize(Xf[idx])
    vert = float(np.abs(ca[:, :, 2] - cb[:, :, 2]).max())
    hm = float(np.abs(np.linalg.norm(ca[:, :, 0:2], axis=2) - np.linalg.norm(cb[:, :, 0:2], axis=2)).max())
    # det of the 2x2 map between the horizontal planes (least squares, per window)
    dets = []
    for i in range(min(500, len(idx))):
        A, B = cb[i, :, 0:2], ca[i, :, 0:2]
        M, *_ = np.linalg.lstsq(A, B, rcond=None)
        dets.append(np.linalg.det(M))
    df = pd.DataFrame(rows)
    _p("\n=== c33 MotionSense sign audit ===")
    _p(df.round(4).to_string(index=False))
    _p(f"  after canonicalization: max |vertical acc difference| = {vert:.2e} g, "
       f"max |horizontal norm difference| = {hm:.2e} g")
    _p(f"  median det of horizontal map old->fixed = {np.median(dets):+.3f} "
       f"(-1 = mirror image, +1 = rotation)")
    df.to_csv(os.path.join(REV_DIR, "tables", "c33_motionsense_sign.csv"), index=False)
    return df


def geometry_identities(n=2000, seed=0):
    """c36 c37 c51. Numerical check of four statements whose proofs go in the
    paper (Sec. 3.2.3):
      P1  S(Q) is a rotation about z for every Q and g      (Proposition 1)
      L1  a rotation about the DEVICE z axis passes through unchanged:
          S(Rz(t)) = Rz(t) for every g  (so 'yaw reproduces itself' is an identity)
      L2  a rotation about the window's own gravity by t leaves psi = t exactly
      L3  a physical tilt leaves psi = 0 only when g = +z; otherwise psi grows
          with the angle between g and +z and diverges near g = -z
          (the gauge of the minimal rotation)."""
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(n, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    Q = np.stack([_haar_np(rng) for _ in range(n)])
    psi, S = residual_heading(g, Q)
    tilt_axis = np.degrees(np.arccos(np.clip(np.abs(S[:, 2, 2]), 0, 1)))
    out = {"P1_max_axis_tilt_deg": float(tilt_axis.max())}
    for t in (30.0, 90.0, 150.0):
        psi_z, _ = residual_heading(g, np.broadcast_to(axis_angle_np((0, 0, 1), t)[0], (n, 3, 3)))
        psi_g, _ = residual_heading(g, axis_angle_np(g, t))
        out[f"L1_device_z_{int(t)}_max_err"] = float(np.abs(psi_z - t).max())
        out[f"L2_about_g_{int(t)}_max_err"] = float(np.abs(psi_g - t).max())
    ang_from_up = np.degrees(np.arccos(np.clip(g[:, 2], -1, 1)))
    h = np.cross(g, rng.normal(size=(n, 3)))
    psi_t, _ = residual_heading(g, axis_angle_np(h, 45.0))
    rows = []
    for lo, hi in ((0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 170), (170, 180.01)):
        m = (ang_from_up >= lo) & (ang_from_up < hi)
        if m.any():
            rows.append({"angle(g,+z) deg": f"{lo}-{int(hi)}", "n": int(m.sum()),
                         "median |psi| after 45 deg tilt": float(np.median(np.abs(psi_t[m]))),
                         "p95 |psi|": float(np.percentile(np.abs(psi_t[m]), 95))})
    _p("\n=== geometry identities (should be ~0 errors) ===")
    for k, v in out.items():
        _p(f"  {k:32s} {v:.2e}")
    _p("\n=== L3: residual heading left by a pure 45 deg physical tilt, by device pose ===")
    tab = pd.DataFrame(rows)
    _p(tab.round(2).to_string(index=False))
    tab.to_csv(os.path.join(REV_DIR, "tables", "geometry_L3_tilt_gauge.csv"), index=False)
    json.dump(out, open(os.path.join(REV_DIR, "tables", "geometry_identities.json"), "w"), indent=2)
    return out, tab


def rotation_axis_decomposition(datasets=None, degs=(30, 90), n=3000):
    """c37 c50 c51, data-driven. For each dataset and device-axis rotation:
    the angle between the rotation axis and the window's gravity (0 = the rotation
    is a physical heading change, 90 = a pure tilt), the gravity displacement it
    causes in the RAW frame (what the device-frame baseline sees), and the residual
    heading after canonicalization. Also the fraction of dynamic acc/gyro energy on
    each device axis: a rotation leaves the component along its own axis
    untouched, which is the valid explanation of why the raw baseline is more
    robust to one axis than another (c50)."""
    datasets = datasets or DATASETS_REV
    rows, energy = [], []
    for d in datasets:
        X = get_ds(d)[0]
        idx = np.random.default_rng(0).choice(len(X), min(n, len(X)), replace=False)
        X = X[idx]; g, _ = gravity_dirs(X)
        dyn = X - X.mean(1, keepdims=True)
        ea = (dyn[:, :, 0:3] ** 2).mean((0, 1)); eg = (dyn[:, :, 3:6] ** 2).mean((0, 1))
        energy.append({"dataset": d, **{f"acc_dyn_energy_{a}": float(v / ea.sum()) for a, v in zip("xyz", ea)},
                       **{f"gyro_energy_{a}": float(v / eg.sum()) for a, v in zip("xyz", eg)}})
        for pt in ("device_x", "device_y", "device_z"):
            ax = {"device_x": 0, "device_y": 1, "device_z": 2}[pt]
            ang = np.degrees(np.arccos(np.clip(np.abs(g[:, ax]), 0, 1)))
            for deg in degs:
                Q = make_perturbation(X, pt, deg)
                g1 = np.einsum("nij,nj->ni", Q, g)
                disp = np.degrees(np.arccos(np.clip((g * g1).sum(1), -1, 1)))
                psi, _ = residual_heading(g, Q)
                rows.append({"dataset": d, "rotation": pt, "deg": deg,
                             "median angle(axis, g)": float(np.median(ang)),
                             "median gravity displacement (raw frame)": float(np.median(disp)),
                             "median |psi| after canon.": float(np.median(np.abs(psi)))})
    df, en = pd.DataFrame(rows), pd.DataFrame(energy)
    _p("\n=== device-axis rotations, decomposed (c37 c50 c51) ===")
    _p(df.round(1).to_string(index=False))
    _p("\n=== share of dynamic energy per device axis (raw frame) ===")
    _p(en.round(3).to_string(index=False))
    df.to_csv(os.path.join(REV_DIR, "tables", "rotation_axis_decomposition.csv"), index=False)
    en.to_csv(os.path.join(REV_DIR, "tables", "axis_energy.csv"), index=False)
    return df, en


def antipode_report(datasets=None, deltas=(5, 10, 15, 25), noise_deg=1.0, n=3000):
    """c38. The neighbourhood of the singularity is defined by the angle between
    g_hat and -z. Within it, a small error in g_hat produces a large change of the
    canonical heading: we measure that amplification directly (heading change
    caused by a `noise_deg` perturbation of g_hat) as a function of the distance to
    the antipode. Exactly at g = -z the code returns diag(1,-1,-1)."""
    datasets = datasets or DATASETS_REV
    rows = []
    rng = np.random.default_rng(0)
    for d in datasets:
        X = get_ds(d)[0]
        g, nrm = gravity_dirs(X)
        dist = np.degrees(np.arccos(np.clip(-g[:, 2], -1, 1)))
        r = {"dataset": d, "n": len(g), "gate_fail_frac": float((nrm < REV["gate_g"]).mean())}
        for dl in deltas:
            r[f"within_{dl}deg_of_antipode"] = float((dist < dl).mean())
        idx = rng.choice(len(g), min(n, len(g)), replace=False)
        gi = g[idx]
        pert = gi + np.radians(noise_deg) * rng.normal(size=gi.shape)
        pert /= np.linalg.norm(pert, axis=1, keepdims=True)
        R0, R1 = rot_to_vertical_batch(gi), rot_to_vertical_batch(pert)
        D = R1 @ np.transpose(R0, (0, 2, 1))
        dh = np.abs(np.degrees(np.arctan2(D[:, 1, 0], D[:, 0, 0])))
        r["median heading jitter (deg) per 1 deg g error, far (>90 from antipode)"] = float(
            np.median(dh[dist[idx] > 90])) if (dist[idx] > 90).any() else np.nan
        r["... near (<15 from antipode)"] = float(np.median(dh[dist[idx] < 15])) if (dist[idx] < 15).any() else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    _p("\n=== c38 antipodal neighbourhood ===")
    _p(df.round(4).to_string(index=False))
    df.to_csv(os.path.join(REV_DIR, "tables", "c38_antipode.csv"), index=False)
    return df


def heading_profile_v2(datasets=None):
    """c32. theta_h = axis (mod 180) of the principal horizontal acceleration of a
    dynamic window after canonicalization, measured from the canonical x axis
    (x_c = R(g_hat) applied to the device frame, i.e. the gauge fixed by Eq. 3):
        C = sum_t h_t h_t^T, h_t = [a_x,t a_y,t]^T - mean,  theta_h = atan2(v2, v1) mod 180
    with v the leading eigenvector of C. Concentration R = |mean exp(2 i theta_h)| in
    [0,1] (0 uniform, 1 identical)."""
    datasets = datasets or DATASETS_REV
    rows = []
    for d in datasets:
        X, y = get_ds(d)[0:2]
        m = np.isin(y, (2, 3, 4, 5))
        Xc = canonicalize(X[m])
        H = Xc[:, :, 0:2].astype(np.float64); H = H - H.mean(1, keepdims=True)
        C = np.einsum("nti,ntj->nij", H, H)
        _, V = np.linalg.eigh(C)
        th = np.degrees(np.arctan2(V[:, 1, 1], V[:, 0, 1])) % 180.0
        z = np.exp(1j * np.radians(2 * th))
        rows.append({"dataset": d, "n_dynamic": int(m.sum()), "R_bar": float(abs(z.mean())),
                     "mean_theta_deg": float((np.degrees(np.angle(z.mean())) / 2) % 180)})
    df = pd.DataFrame(rows)
    _p("\n=== c32 residual-heading profile (dynamic windows) ===")
    _p(df.round(3).to_string(index=False))
    df.to_csv(os.path.join(REV_DIR, "tables", "c32_heading_profile.csv"), index=False)
    return df


def cadence_energy_table(datasets=None):
    """c66. Dominant walking cadence (FFT of the acc magnitude, walk windows) and
    acc/gyro energy per dataset, relative to HHAR."""
    datasets = datasets or DATASETS_REV
    rows = []
    for d in datasets:
        X, y = get_ds(d)[0:2]
        w = X[y == 4]
        if not len(w):
            continue
        mag = np.linalg.norm(w[:, :, 0:3], axis=2); mag = mag - mag.mean(1, keepdims=True)
        sp = np.abs(np.fft.rfft(mag, axis=1)); fr = np.fft.rfftfreq(mag.shape[1], d=1.0 / TARGET_HZ)
        dom = fr[1 + sp[:, 1:].argmax(1)]
        rows.append({"dataset": d, "n_walk": len(w), "cadence_median_hz": float(np.median(dom)),
                     "cadence_iqr": f"{np.percentile(dom, 25):.2f}-{np.percentile(dom, 75):.2f}",
                     "freq_resolution_hz": float(fr[1]),
                     "acc_dyn_rms_g": float(np.median(mag.std(1))),
                     "gyro_rms_rad_s": float(np.median(np.linalg.norm(w[:, :, 3:6], axis=2).std(1)))})
    df = pd.DataFrame(rows)
    if "hhar" in set(df.dataset):
        ref = df.set_index("dataset").loc["hhar"]
        df["acc_energy_vs_hhar"] = df["acc_dyn_rms_g"] / ref["acc_dyn_rms_g"]
        df["gyro_energy_vs_hhar"] = df["gyro_rms_rad_s"] / ref["gyro_rms_rad_s"]
    _p("\n=== c66 walking cadence and energy ===")
    _p(df.round(3).to_string(index=False))
    _p("  note: 2.56 s windows give a 0.39 Hz frequency resolution; report it with the cadence.")
    df.to_csv(os.path.join(REV_DIR, "tables", "c66_cadence_energy.csv"), index=False)
    return df


def preprocessing_table(datasets=None):
    """c45 (+c44): what each dataset contributes after preprocessing."""
    datasets = datasets or DATASETS_REV
    rows = []
    for d in datasets:
        raw = load_raw_v2(d)
        X, y, g, meta, _ = get_ds(d)
        _, yp, _, _, _ = get_ds(d, "posture")
        r = {"dataset": d, "subjects": len(np.unique(g)), "windows (merged labels)": len(X),
             "windows (posture labels)": len(yp), **raw["info"],
             "classes (merged)": ", ".join(f"{CLS[c]}:{int((y == c).sum())}" for c in np.unique(y)),
             "classes (posture)": ", ".join(f"{CLS[c]}:{int((yp == c).sum())}" for c in np.unique(yp))}
        if d == "hhar":
            mdl, hz = meta["model"], meta["native_hz"]
            r["devices"] = "; ".join(f"{m}: {np.median(hz[mdl == m]):.0f} Hz"
                                     for m in sorted(np.unique(mdl)))
        rows.append(r)
    df = pd.DataFrame(rows)
    _p(df.T.to_string())
    df.to_csv(os.path.join(REV_DIR, "tables", "c45_preprocessing.csv"), index=False)
    return df


def pair_label_table(pairs=None, label_mode="merged"):
    """c44: the classes each of the 12 pairs uses."""
    pairs = pairs or PAIRS
    rows = []
    for s, t in pairs:
        ys, yt = get_ds(s, label_mode)[1], get_ds(t, label_mode)[1]
        sh = sorted(int(c) for c in set(np.unique(ys)) & set(np.unique(yt)))
        rows.append({"pair": f"{s}->{t}", "classes": ", ".join(CLS[c] for c in sh),
                     "n_source": int(np.isin(ys, sh).sum()), "n_target": int(np.isin(yt, sh).sum())})
    df = pd.DataFrame(rows)
    _p(df.to_string(index=False))
    df.to_csv(os.path.join(REV_DIR, "tables", f"c44_pair_labels_{label_mode}.csv"), index=False)
    return df


def param_breakdown(n_classes=4):
    """c42."""
    rows = []
    for kind, s_dim in (("6-ch (raw/canon/pca)", 6), ("Mizell 4-ch", 4), ("OIT 18-ch", 18)):
        m = HARModelRev(n_classes, s_dim)
        e = sum(p.numel() for p in m.encoder.parameters())
        h = sum(p.numel() for p in m.head.parameters())
        rows.append({"input": kind, "LIMU-BERT encoder": e, "GRU head": h, "total": e + h})
    dec = sum(p.numel() for p in LimuBERT().decoder.parameters())
    df = pd.DataFrame(rows)
    _p(df.to_string(index=False))
    _p(f"  pretraining-only decoder (discarded after pretraining): {dec:,} parameters")
    df.to_csv(os.path.join(REV_DIR, "tables", "c42_parameters.csv"), index=False)
    return df


def inference_benchmark(n=2048, reps=20, export=True):
    """c27 c56. CPU model, threads and batch size are recorded with every number;
    single-thread batch-1 latency is the figure closest to a phone. Exports
    TorchScript/ONNX so the same model can be timed on a device."""
    X = get_ds("hhar")[0][:n]
    try:
        cpu = subprocess.run(["bash", "-c", "lscpu | grep 'Model name' | head -1"],
                             capture_output=True, text=True).stdout.split(":")[-1].strip()
    except Exception:
        cpu = platform.processor()
    rows = []
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); canonicalize(X); t.append(time.perf_counter() - t0)
    rows.append({"component": "canonicalization (numpy, vectorized)", "batch": n, "threads": "numpy default",
                 "ms_per_window": 1000 * np.median(t) / n})
    t = []
    for i in range(min(n, 500)):
        t0 = time.perf_counter(); canonicalize(X[i:i + 1]); t.append(time.perf_counter() - t0)
    rows.append({"component": "canonicalization (numpy)", "batch": 1, "threads": "numpy default",
                 "ms_per_window": 1000 * np.median(t)})
    enc = get_encoder_rev("hhar", "canon", "yaw", 42)
    model = HARModelRev(4).cpu(); model.encoder.load_state_dict(enc); model.eval()
    nt0 = torch.get_num_threads()
    for threads in (1, nt0):
        torch.set_num_threads(threads)
        for bsz in (1, 256):
            xb = torch.tensor(canonicalize(X[:bsz]), dtype=torch.float32)
            with torch.no_grad():
                for _ in range(3):
                    model(xb)
                t = []
                for _ in range(reps):
                    t0 = time.perf_counter(); model(xb); t.append(time.perf_counter() - t0)
            rows.append({"component": "encoder + GRU head (PyTorch CPU)", "batch": bsz, "threads": threads,
                         "ms_per_window": 1000 * np.median(t) / bsz})
    torch.set_num_threads(nt0)
    df = pd.DataFrame(rows)
    df["cpu"] = cpu; df["torch"] = torch.__version__
    df["fraction_of_2.56s_window_%"] = 100 * df["ms_per_window"] / 2560.0
    _p(df.to_string(index=False))
    df.to_csv(os.path.join(REV_DIR, "tables", "c56_inference.csv"), index=False)
    if export:
        xb = torch.zeros(1, XDOMAIN_SEQ_LEN, 6)
        ts = torch.jit.trace(model, (xb,))
        ts.save(os.path.join(REV_DIR, "tables", "gravihar_hhar_4cls.pt"))
        try:
            torch.onnx.export(model, (xb,), os.path.join(REV_DIR, "tables", "gravihar_hhar_4cls.onnx"),
                              input_names=["window"], output_names=["logits"], opset_version=17)
        except Exception as e:
            _p(f"  [onnx] export skipped: {e}")
    return df


# =============================================================================
#  8. COST ESTIMATE + DRIVER
# =============================================================================
def estimate_cost(source="shoaib", n_epochs=1):
    """Time one pretraining epoch and one fine-tuning epoch on the largest source and
    project the run budget of every stage."""
    X = prep_cached(source, "merged", "canon")[0]
    y = get_ds(source)[1]
    labels = sorted(np.unique(y)); lmap = {c: i for i, c in enumerate(labels)}
    old = (REV["pre_epochs"], REV["ft_epochs"], REV["pre_min_epochs"])
    REV["pre_epochs"], REV["ft_epochs"], REV["pre_min_epochs"] = n_epochs, n_epochs, 0
    try:
        t0 = time.time(); pretrain_rev(X, "yaw", 0); tp = (time.time() - t0) / n_epochs
        t0 = time.time(); finetune_rev(None, X, np.array([lmap[v] for v in y]), len(labels), "yaw", 0)
        tf = (time.time() - t0) / n_epochs
    finally:
        REV["pre_epochs"], REV["ft_epochs"], REV["pre_min_epochs"] = old
    ft_run = tf * REV["ft_epochs"]; pre_run = tp * 50   # ~50 epochs with early stopping
    nb = len(REV["budget"]["pre"]) * len(REV["budget"]["ft"])
    n_main_arms = len(ABLATION_ARMS + SITE_ARMS + BASELINE_ARMS)
    stages = {
        "main (11 arms x 12 pairs)": (12 * n_main_arms * nb, 4 * 8 * len(REV["budget"]["pre"])),
        "rotation (7 arms x 4 ds x 3 folds x 2)": (7 * 4 * 3 * 2, 0),
        "ceiling (3 arms, 8 combos x 3 folds x 2)": (3 * 8 * 3 * 2, 0),
        "posture (4 arms, 4 ds x 3 folds + 6 pairs, x2)": (4 * (12 + 6) * 2, 8),
        "hparam (9 configs x (4 val + 12 pairs) x 2)": (9 * 16 * 2, 8),
        "augspec (7 ops x 12 pairs x 4)": (7 * 12 * 4, 0),
    }
    rows = []
    for k, (nft, npre) in stages.items():
        h = (nft * ft_run + npre * pre_run) / 3600 * 0.7   # sources smaller than Shoaib on average
        rows.append({"stage": k, "fine-tunes": nft, "new pretrains": npre, "approx_hours": round(h, 1)})
    df = pd.DataFrame(rows)
    _p(f"[cost] {source}: pretrain {tp:.1f} s/epoch, fine-tune {tf:.1f} s/epoch on {DEVICE}")
    _p(df.to_string(index=False))
    return df


def run_stage(name):
    """Stages in priority order (see REVISION_PLAN.md)."""
    S = {
        "audit": lambda: (audit_motionsense_sign(), geometry_identities(), rotation_axis_decomposition(),
                          antipode_report(), heading_profile_v2(), cadence_energy_table(),
                          preprocessing_table(), pair_label_table(), pair_label_table(label_mode="posture"),
                          param_breakdown()),
        "xgb": exp_xgb,
        "ablation": lambda: exp_main(arms=ABLATION_ARMS),
        "site": lambda: exp_main(arms=SITE_ARMS),
        "baselines": lambda: exp_main(arms=BASELINE_ARMS, budget=REV["budget_baselines"]),
        "ceiling": exp_ceiling,
        "rotation": exp_rotation,
        "unimts": exp_unimts,
        "posture": exp_posture,
        "hparam": exp_hparam,
        "augspec": exp_aug_specificity,
        "positions": exp_positions,
        "inference": inference_benchmark,
        "labeleff": exp_label_efficiency,
        "autoaug": exp_autoaug,
    }
    if name not in S:
        raise ValueError(f"unknown stage {name}; choose from {list(S)}")
    _p(f"\n######## stage {name} ########")
    return S[name]()


_p(f"""
[revision] ready. results -> {REV_DIR}
  run_stage("audit")      no training: c33 sign fix, geometry identities, tables  (minutes)
  estimate_cost()         time one epoch, project hours per stage
  run_stage("xgb")        engineered-feature baselines, raw + canon, 12 pairs     (CPU)
  run_stage("ablation")   A/B/C/D on 12 pairs (re-run on corrected data, per-run log)
  run_stage("site")       where yaw invariance must live (c39 c41)
  run_stage("baselines")  SO(3) aug, learned SO(3), PCA heading, Mizell, OIT (c46, e-mail)
  run_stage("ceiling")    in-domain ceilings on each pair's label set (c29 c64)
  run_stage("rotation")   controlled device- and gravity-frame rotations, 4 datasets (RQ3)
  run_stage("unimts")     published method, raw vs canonicalized input (c46.3)
  run_stage("posture")    sit/stand/lie cost of canonicalization (c44)
  run_stage("hparam")     source-only selection + target sensitivity (c40)
  optional: "augspec" "positions" "inference" "labeleff" "autoaug"
Then: %run -i -n revision/revision_analysis.py ; make_all_tables()
""")
