# =============================================================================
#  pipeline_base.py -- VERBATIM copy of Gravity_Model.ipynb cell 20 (the pipeline
#  cell: loaders, LIMU-BERT, GRU head, operators, XGBoost features).
#  Kept as a file so the revision runner can load it with
#      %run -i -n revision/pipeline_base.py
#  (-n stops the __main__ block from running the old entry point).
#  Do not edit here; edit the notebook and re-export if the pipeline changes.
# =============================================================================
# =============================================================================
# HAR : Tier-1 (XGBoost, engineered features)  +  Tier-2 (LIMU-BERT + AutoAugHAR)
# =============================================================================

import os, glob, warnings, math, random
import numpy as np
import pandas as pd
import scipy.fftpack
import joblib

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call

warnings.filterwarnings("ignore")

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB  = True
    DATA_ROOT = '/content/drive/MyDrive'
except Exception:
    IN_COLAB  = False
    DATA_ROOT = os.environ.get('HAR_DATA_ROOT', os.path.expanduser('~/har_data'))

BASE_PATH         = os.path.join(DATA_ROOT, 'HHAR')
ACTIVITY_PATH     = os.path.join(BASE_PATH, 'Activity recognition exp')
ANDROID_DATA_PATH = os.path.join(DATA_ROOT, 'Android_Data')
UCI_PATH          = os.path.join(DATA_ROOT, 'UCI HAR Dataset')
SAVE_PATH         = DATA_ROOT if IN_COLAB else os.path.join(DATA_ROOT, 'results')
os.makedirs(SAVE_PATH, exist_ok=True)

MS_PATH = os.path.join(DATA_ROOT, 'MotionSense', 'A_DeviceMotion_data')
SHOAIB_PATH = os.path.join(DATA_ROOT, 'Shoaib')

# ------------------------------------------------------- window geometry -----
STEP_SEC   = 2.5
MIN_SAMPLES_HHAR = 30
TARGET_HZ  = 20.0                              # LIMU-BERT operating rate
N_CHANNELS = 6                                 # acc xyz + gyro xyz

WINDOW_SEC = 5.0                               # in-domain HHAR window
SEQ_LEN    = int(round(WINDOW_SEC * TARGET_HZ))   # L = 100 (5 s @ 20 Hz)

XDOMAIN_WINDOW_SEC = 2.56                       # UCI native window (128 @ 50 Hz)
XDOMAIN_SEQ_LEN    = int(round(XDOMAIN_WINDOW_SEC * TARGET_HZ))  # L = 51 @ 20 Hz

MAX_LEN = 128     # positional-embedding capacity; must be >= any L used above

USE_GRAVITY_ALIGN = False

QUICK  = False
RESUME = True
SEED   = 42
SEEDS  = (42,) if QUICK else (42, 43, 44)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
PRETRAIN_EPOCHS = 5  if QUICK else 80
FINETUNE_EPOCHS = 6  if QUICK else 40
POLICY_EPOCHS   = 4  if QUICK else 15
PRETRAIN_BS, PRETRAIN_LR = 256, 1e-3
FINETUNE_BS, FINETUNE_LR = 128, 1e-3
ALPHA_LR = 3e-3
FREEZE_ENCODER = False
PRETRAIN_CONSISTENCY = True
CONSISTENCY_LAMBDA   = 0.1
CONSISTENCY_MIN_N    = 15000

# Balanced multi-source pool: cap each dataset's contribution so one large set
MULTI_CAP = 25000

# Identity floor for fixed-policy augmentation: with this probability the batch
# passes through unaugmented
AUG_IDENTITY_FLOOR = 0.3

# LIMU-BERT hyper-params (paper defaults)
H_DIM, FF_DIM, N_LAYERS, N_HEADS = 72, 144, 4, 4
SHARE_LAYERS = True
MASK_RATIO, MASK_PROB, SPAN_P, SPAN_LMAX = 0.15, 0.80, 0.20, 10

LABEL_MAP = {
    'stand': 0, 'sit': 0, 'null': -1, 'bike': 6,
    'stairsup': 2, 'stairsdown': 3, 'walk': 4, 'elevator': 1,
    'running': 5, 'walking': 4,
}
CLASS_NAMES = {0:'static(sit/stand)',1:'elevator',2:'stairsUP',3:'stairsDOWN',
               4:'walk',5:'run',6:'bike'}
UCI2OURS = {1: 4, 2: 2, 3: 3, 4: 0, 5: 0}      # WALK,UP,DOWN,SIT,STAND
UCI_G, UCI_HZ = 9.80665, 50.0


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def map_gt_hhar(v):
    return LABEL_MAP.get(str(v).strip().lower(), -1)


# =============================================================================
# TIER-1 : gravity canonicalization + GenHAR/LLM4HAR feature extraction
def _rot_to_vertical(ghat):
    z = np.array([0.0, 0.0, 1.0]); v = np.cross(ghat, z)
    c = float(np.dot(ghat, z)); s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

def _apply_R(win_df, R):
    out = win_df.copy()
    out[['x', 'y', 'z']] = win_df[['x', 'y', 'z']].values @ R.T
    return out

def align_window(a_win, g_win):
    m = a_win[['x', 'y', 'z']].values.mean(0); n = np.linalg.norm(m)
    if n < 5.0:
        return a_win, g_win, False
    R = _rot_to_vertical(m / n)
    return _apply_R(a_win, R), _apply_R(g_win, R), True

def align_window_with_ref(a_win, g_win, grav_win):
    if grav_win is None or len(grav_win) == 0:
        return a_win, g_win, False
    m = grav_win[['x', 'y', 'z']].values.mean(0); n = np.linalg.norm(m)
    if n < 1e-6:
        return a_win, g_win, False
    R = _rot_to_vertical(m / n)
    return _apply_R(a_win, R), _apply_R(g_win, R), True

def instance_norm(win):
    w = win.copy()
    for c in "xyz":
        v = w[c].values
        w[c] = (v - v.mean()) / (v.std() + 1e-8)
    return w

def amp_features(win, prefix, hz):
    f = {}
    for ax in "xyz":
        sig = win[ax].values - win[ax].values.mean()
        amp = np.abs(np.fft.rfft(sig)); amp = amp / (amp.sum() + 1e-9)
        frq = np.fft.rfftfreq(len(sig), d=1.0 / hz)
        for lo, hi, nm in [(0,1,"b01"),(1,2,"b12"),(2,3,"b23"),(3,5,"b35")]:
            m = (frq >= lo) & (frq < hi)
            f[f"{prefix}_{ax}_amp_{nm}"] = float(amp[m].sum())
        f[f"{prefix}_{ax}_spec_centroid"] = float((frq * amp).sum())
        di = 1 + int(np.argmax(amp[1:])) if len(amp) > 1 else 0
        h2 = min(2 * di, len(amp) - 1)
        f[f"{prefix}_{ax}_harm_ratio"] = float(amp[h2] / (amp[di] + 1e-9))
    return f

def corr_features(a_win, g_win):
    def cc(u, v):
        n = min(len(u), len(v))
        if n < 2:
            return 0.0
        c = np.corrcoef(u[:n], v[:n])[0, 1]
        return float(c) if np.isfinite(c) else 0.0
    ax, ay, az = a_win.x.values, a_win.y.values, a_win.z.values
    gx, gy, gz = g_win.x.values, g_win.y.values, g_win.z.values
    return {"acc_xy_corr": cc(ax, ay), "acc_xz_corr": cc(ax, az), "acc_yz_corr": cc(ay, az),
            "accgyro_x_corr": cc(ax, gx), "accgyro_y_corr": cc(ay, gy), "accgyro_z_corr": cc(az, gz)}

def features_from_windows(a_win, g_win, sample_hz=100.0, grav_win=None,
                          add_amp=True, add_corr=True, use_instnorm=False):
    if USE_GRAVITY_ALIGN:
        if grav_win is not None:
            a_al, g_al, aligned = align_window_with_ref(a_win, g_win, grav_win)
        else:
            a_al, g_al, aligned = align_window(a_win, g_win)
    else:
        a_al, g_al, aligned = a_win, g_win, False
    a_use, g_use = (instance_norm(a_al), instance_norm(g_al)) if use_instnorm else (a_al, g_al)
    f = {**extract_sensor_features(a_use, "acc", sample_hz),
         **extract_sensor_features(g_use, "gyro", sample_hz)}
    f.pop('acc_z_drift', None); f.pop('acc_z_drift_abs', None)
    vz = a_al['z'].values - a_al['z'].values.mean()
    vel = np.cumsum(vz) * (1.0 / sample_hz)
    f['acc_vert_vel_range'] = float(vel.max() - vel.min())
    if add_amp:
        f.update(amp_features(a_use, "acc", sample_hz))
        f.update(amp_features(g_use, "gyro", sample_hz))
    if add_corr:
        f.update(corr_features(a_use, g_use))
    return f, aligned

def extract_sensor_features(window_df, sensor_prefix, sample_hz=100.0):
    feats = {}
    x = window_df['x'].values.astype(float); y = window_df['y'].values.astype(float)
    z = window_df['z'].values.astype(float); mag = np.sqrt(x*x + y*y + z*z)
    if sensor_prefix == 'acc' and mag.mean() > 5.0:
        x = x - x.mean(); y = y - y.mean(); z = z - z.mean()
        mag = np.sqrt(x*x + y*y + z*z)
    mag_centered = mag - mag.mean()
    feats[f'{sensor_prefix}_x_mean'] = float(np.mean(x)); feats[f'{sensor_prefix}_y_mean'] = float(np.mean(y))
    feats[f'{sensor_prefix}_z_mean'] = float(np.mean(z)); feats[f'{sensor_prefix}_x_std'] = float(np.std(x))
    feats[f'{sensor_prefix}_y_std'] = float(np.std(y));  feats[f'{sensor_prefix}_z_std'] = float(np.std(z))
    feats[f'{sensor_prefix}_mag_mean'] = float(np.mean(mag)); feats[f'{sensor_prefix}_mag_std'] = float(np.std(mag))
    feats[f'{sensor_prefix}_mag_p2p'] = float(np.ptp(mag))
    feats[f'{sensor_prefix}_mag_energy'] = float(np.sum(mag**2) / len(mag))
    feats[f'{sensor_prefix}_sma'] = float(np.sum(np.abs(x)+np.abs(y)+np.abs(z)) / len(mag))
    feats[f'{sensor_prefix}_mag_75th'] = float(np.percentile(mag, 75))
    feats[f'{sensor_prefix}_mag_25th'] = float(np.percentile(mag, 25))
    feats[f'{sensor_prefix}_zcr'] = float(((mag_centered[:-1] * mag_centered[1:]) < 0).sum() / len(mag))
    dt = 1.0 / sample_hz
    feats[f'{sensor_prefix}_z_drift'] = float(np.sum(z) * dt)
    feats[f'{sensor_prefix}_z_drift_abs'] = float(abs(np.sum(z) * dt))
    def _skew(a):
        m, s = a.mean(), a.std(); return float(((a-m)**3).mean() / (s**3 + 1e-9))
    def _kurt(a):
        m, s = a.mean(), a.std(); return float(((a-m)**4).mean() / (s**4 + 1e-9) - 3.0)
    feats[f'{sensor_prefix}_z_skew'] = _skew(z); feats[f'{sensor_prefix}_z_kurt'] = _kurt(z)
    feats[f'{sensor_prefix}_mag_skew'] = _skew(mag); feats[f'{sensor_prefix}_mag_kurt'] = _kurt(mag)
    z_pos = z[z > 0]; z_neg = z[z < 0]
    feats[f'{sensor_prefix}_z_pos_mean'] = float(z_pos.mean()) if len(z_pos) else 0.0
    feats[f'{sensor_prefix}_z_neg_mean'] = float(z_neg.mean()) if len(z_neg) else 0.0
    feats[f'{sensor_prefix}_z_peak_asym'] = feats[f'{sensor_prefix}_z_pos_mean'] + feats[f'{sensor_prefix}_z_neg_mean']
    fft_vals = np.abs(scipy.fftpack.fft(mag)); freqs = scipy.fftpack.fftfreq(len(mag), d=1.0/sample_hz)
    peak_idx = 1 + np.argmax(fft_vals[1:]) if len(fft_vals) > 1 else 0
    fs = fft_vals.sum() + 1e-9
    feats[f'{sensor_prefix}_dominant_freq'] = float(abs(freqs[peak_idx]))
    feats[f'{sensor_prefix}_spectral_entropy'] = float(-np.sum((fft_vals / fs) * np.log(fft_vals / fs + 1e-9)))
    fft_z = np.abs(scipy.fftpack.fft(z - z.mean()))
    z_pidx = 1 + np.argmax(fft_z[1:]) if len(fft_z) > 1 else 0
    feats[f'{sensor_prefix}_z_dominant_freq'] = float(abs(freqs[z_pidx]))
    return feats

def safe_read_csv(path):
    for enc in ['utf-8', 'latin1', 'cp1252']:
        try:    return pd.read_csv(path, encoding=enc)
        except Exception: continue
    return None

def standardize_columns(df):
    return df.rename(columns={c: c.strip().lower() for c in df.columns})


# =============================================================================
# TIER-1 : HHAR loading + windowing (features)  +  raw windowing (Tier-2)
# =============================================================================
def infer_sensor(path):
    n = os.path.basename(path).lower()
    return 'gyro' if 'gyro' in n else 'acc' if 'acc' in n else 'unknown'

def load_raw_phone_csv(file_path):
    df = safe_read_csv(file_path)
    if df is None: return None
    df = standardize_columns(df)
    req = ['creation_time', 'x', 'y', 'z', 'user', 'model', 'device', 'gt']
    if not all(c in df.columns for c in req): return None
    df = df[req].dropna().copy()
    for col in ['creation_time', 'x', 'y', 'z']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    df['class_label'] = df['gt'].apply(map_gt_hhar)
    df = df[df['class_label'] >= 0]
    df['time_sec'] = df['creation_time'] / 1e9
    return df.sort_values(['user', 'device', 'time_sec']).reset_index(drop=True)

def load_hhar_phone():
    all_csv = [f for f in glob.glob(os.path.join(ACTIVITY_PATH, '*.csv')) if 'readme' not in f.lower()]
    files_df = pd.DataFrame({'file_path': f, 'sensor_type': infer_sensor(f),
                             'is_phone': 'phone' in os.path.basename(f).lower()} for f in all_csv)
    files_df = files_df[files_df['is_phone']].copy()
    acc_p  = next((r.file_path for _, r in files_df.iterrows() if r.sensor_type == 'acc'),  None)
    gyro_p = next((r.file_path for _, r in files_df.iterrows() if r.sensor_type == 'gyro'), None)
    return load_raw_phone_csv(acc_p), load_raw_phone_csv(gyro_p)


# ---- feature windower (Tier-1) ----------------------------------------------
def window_hhar(acc_raw, gyro_raw, window_sec=WINDOW_SEC, step_sec=STEP_SEC,
                min_samples=MIN_SAMPLES_HHAR):
    results = []
    if acc_raw is None or gyro_raw is None: return results
    for (user, device), acc_grp in acc_raw.groupby(['user', 'device']):
        gyro_grp = gyro_raw[(gyro_raw['user'] == user) & (gyro_raw['device'] == device)]
        if gyro_grp.empty: continue
        t0 = max(acc_grp['time_sec'].min(), gyro_grp['time_sec'].min())
        t1 = min(acc_grp['time_sec'].max(), gyro_grp['time_sec'].max())
        if t1 - t0 < window_sec: continue
        at, gt = acc_grp['time_sec'].values, gyro_grp['time_sec'].values
        t, wid = t0, 0
        while t + window_sec <= t1:
            a_win = acc_grp[(at >= t) & (at < t + window_sec)]
            g_win = gyro_grp[(gt >= t) & (gt < t + window_sec)]
            if len(a_win) >= min_samples and len(g_win) >= min_samples:
                lbl = int(a_win['class_label'].mode().iloc[0])
                feats, _ = features_from_windows(a_win, g_win)
                results.append({'class_label': lbl, 'carrying_load': 0, 'user': user,
                                'device': device, 'source': 'hhar', 'window_id': wid, **feats})
            t += step_sec; wid += 1
    return results


# ---- raw windower (Tier-2), with array-based gravity alignment ---------------
def _resample(arr, L):
    n = arr.shape[0]
    if n == L: return arr.astype(np.float32)
    old = np.linspace(0.0, 1.0, n); new = np.linspace(0.0, 1.0, L)
    out = np.empty((L, arr.shape[1]), dtype=np.float32)
    for c in range(arr.shape[1]):
        out[:, c] = np.interp(new, old, arr[:, c])
    return out

def _align_arrays(acc, gyro):
    """Self-align from accelerometer gravity (array version of align_window)."""
    m = acc.mean(0); n = np.linalg.norm(m)
    if n < 5.0:
        return acc, gyro, False
    R = _rot_to_vertical(m / n)
    return acc @ R.T, gyro @ R.T, True

def raw_from_arrays(acc, gyro, align, has_gravity=True, seq_len=SEQ_LEN):
    """[n,3]+[n,3] -> normalized [L,6]. acc/9.80665 (LIMU-BERT Eq.1); gyro as-is."""
    if align and has_gravity:
        acc, gyro, _ = _align_arrays(acc, gyro)
    acc = _resample(acc, seq_len); gyro = _resample(gyro, seq_len)
    win = np.concatenate([acc, gyro], axis=1).astype(np.float32)
    win[:, 0:3] = win[:, 0:3] / 9.80665
    return win

def window_hhar_raw(acc_raw, gyro_raw, align, window_sec=WINDOW_SEC, step_sec=STEP_SEC,
                    min_samples=MIN_SAMPLES_HHAR, seq_len=SEQ_LEN):
    X, y, groups = [], [], []
    if acc_raw is None or gyro_raw is None:
        return np.empty((0, seq_len, N_CHANNELS), np.float32), np.array([]), np.array([])
    for (user, device), acc_grp in acc_raw.groupby(['user', 'device']):
        gyro_grp = gyro_raw[(gyro_raw['user'] == user) & (gyro_raw['device'] == device)]
        if gyro_grp.empty: continue
        t0 = max(acc_grp['time_sec'].min(), gyro_grp['time_sec'].min())
        t1 = min(acc_grp['time_sec'].max(), gyro_grp['time_sec'].max())
        if t1 - t0 < window_sec: continue
        at, gt = acc_grp['time_sec'].values, gyro_grp['time_sec'].values
        t = t0
        while t + window_sec <= t1:
            a_win = acc_grp[(at >= t) & (at < t + window_sec)]
            g_win = gyro_grp[(gt >= t) & (gt < t + window_sec)]
            if len(a_win) >= min_samples and len(g_win) >= min_samples:
                lbl = int(a_win['class_label'].mode().iloc[0])
                acc = a_win[['x', 'y', 'z']].values.astype(np.float32)
                gyr = g_win[['x', 'y', 'z']].values.astype(np.float32)
                X.append(raw_from_arrays(acc, gyr, align, has_gravity=True, seq_len=seq_len))
                y.append(lbl); groups.append(str(user))
            t += step_sec
    return (np.asarray(X, np.float32) if X else np.empty((0, seq_len, N_CHANNELS), np.float32),
            np.asarray(y), np.asarray(groups))


# =============================================================================
# TIER-1 : Phyphox (features only
def load_phyphox_csv(path):
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df.columns = ['time_sec', 'x', 'y', 'z'] + list(df.columns[4:])
    return df[['time_sec', 'x', 'y', 'z']].astype(np.float32)

def load_phyphox_gravity_ref(folder_path):
    for fn in ['Gravity.csv', 'Accelerometer.csv', 'Acceleration with g.csv', 'Raw Acceleration.csv']:
        df = load_phyphox_csv(os.path.join(folder_path, fn))
        if df is not None:
            return df, fn
    return None, None

def map_folder_to_label(folder_name):
    name = folder_name.lower(); label = -1
    if 'elevator' in name: label = 1
    elif 'stairsup' in name: label = 2
    elif 'stairsdown' in name: label = 3
    elif 'walking' in name: label = 4
    elif 'running' in name: label = 5
    return label, ('with_box' in name)

def process_phyphox_folder(folder_path, label, carrying_load,
                           window_sec=WINDOW_SEC, step_sec=2.0, min_samples=50):
    acc_df  = load_phyphox_csv(os.path.join(folder_path, 'Linear Acceleration.csv'))
    gyro_df = load_phyphox_csv(os.path.join(folder_path, 'Gyroscope.csv'))
    if acc_df is None or gyro_df is None: return []
    grav_df = None
    if USE_GRAVITY_ALIGN:
        grav_df, _ = load_phyphox_gravity_ref(folder_path)
    max_t = min(acc_df['time_sec'].max(), gyro_df['time_sec'].max())
    at, gt = acc_df['time_sec'].values, gyro_df['time_sec'].values
    grav_t = grav_df['time_sec'].values if grav_df is not None else None
    results, t = [], 0.0
    while t + window_sec <= max_t:
        a_win = acc_df.iloc[np.searchsorted(at, t):np.searchsorted(at, t + window_sec)]
        g_win = gyro_df.iloc[np.searchsorted(gt, t):np.searchsorted(gt, t + window_sec)]
        grav_win = None
        if grav_df is not None:
            grav_win = grav_df.iloc[np.searchsorted(grav_t, t):np.searchsorted(grav_t, t + window_sec)]
        if len(a_win) >= min_samples and len(g_win) >= min_samples:
            feats, _ = features_from_windows(a_win, g_win, grav_win=grav_win)
            results.append({'class_label': label, 'carrying_load': int(carrying_load),
                            'user': 'phyphox_team', 'device': 'phyphox', 'source': 'phyphox',
                            'window_id': int(t / step_sec), **feats})
        t += step_sec
    return results

def load_phyphox_features():
    rows = []
    if os.path.exists(ANDROID_DATA_PATH):
        for folder_name in os.listdir(ANDROID_DATA_PATH):
            fp = os.path.join(ANDROID_DATA_PATH, folder_name)
            if not os.path.isdir(fp): continue
            label, load = map_folder_to_label(folder_name)
            if label < 0: continue
            rows.extend(process_phyphox_folder(fp, label, load))
    return pd.DataFrame(rows).fillna(0) if rows else pd.DataFrame()


# =============================================================================
# TIER-1 : UCI cross-dataset builders (features for Tier-1, raw for Tier-2)
# =============================================================================
def _uci_load(split, sig):
    p = os.path.join(UCI_PATH, split, 'Inertial Signals', f'{sig}_{split}.txt')
    return np.loadtxt(p)

def build_uci_features(feature_cols):
    frames, ys = [], []
    for split in ['train', 'test']:
        ax, ay, az = (_uci_load(split, f'total_acc_{a}') * UCI_G for a in 'xyz')
        gx, gy, gz = (_uci_load(split, f'body_gyro_{a}')       for a in 'xyz')
        yv = np.loadtxt(os.path.join(UCI_PATH, split, f'y_{split}.txt')).astype(int)
        keep = np.isin(yv, list(UCI2OURS))
        rows = []
        for i in range(ax.shape[0]):
            a_df = pd.DataFrame({'x': ax[i], 'y': ay[i], 'z': az[i]})
            g_df = pd.DataFrame({'x': gx[i], 'y': gy[i], 'z': gz[i]})
            feats, _ = features_from_windows(a_df, g_df, sample_hz=UCI_HZ)
            feats['carrying_load'] = 0
            rows.append(feats)
        X = pd.DataFrame(rows).reindex(columns=feature_cols, fill_value=0.0).iloc[np.where(keep)[0]]
        frames.append(X); ys.append(np.array([UCI2OURS[v] for v in yv[keep]]))
    return pd.concat(frames, ignore_index=True), np.concatenate(ys)

def build_uci_raw(align, seq_len=SEQ_LEN):
    Xs, ys = [], []
    for split in ['train', 'test']:
        ax, ay, az = (_uci_load(split, f'total_acc_{a}') * UCI_G for a in 'xyz')
        gx, gy, gz = (_uci_load(split, f'body_gyro_{a}')       for a in 'xyz')
        yv = np.loadtxt(os.path.join(UCI_PATH, split, f'y_{split}.txt')).astype(int)
        for i in range(ax.shape[0]):
            if yv[i] not in UCI2OURS: continue
            acc = np.stack([ax[i], ay[i], az[i]], axis=1)
            gyr = np.stack([gx[i], gy[i], gz[i]], axis=1)
            Xs.append(raw_from_arrays(acc, gyr, align, has_gravity=True, seq_len=seq_len))
            ys.append(UCI2OURS[yv[i]])
    return np.asarray(Xs, np.float32), np.asarray(ys)


# =============================================================================
# TIER-1 : XGBoost evaluation (LOSO + cross-UCI), returns metrics
# =============================================================================
def make_xgb(n_classes):
    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
                         objective='multi:softprob', num_class=n_classes, eval_metric='mlogloss',
                         random_state=42, n_jobs=-1)

def tier1_loso(hhar_df, feature_cols):
    X = hhar_df[feature_cols].copy(); y = hhar_df['class_label'].astype(int).copy()
    groups = hhar_df['user'].astype(str)
    logo = LeaveOneGroupOut(); accs = []
    for tr_idx, te_idx in logo.split(X, y, groups=groups):
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        if (set(y_te.unique()) - set(y_tr.unique())) or len(y_te) < 10:
            continue
        classes = sorted(y_tr.unique())
        lmap = {o: n for n, o in enumerate(classes)}; inv = {n: o for o, n in lmap.items()}
        model = make_xgb(len(classes))
        model.fit(X.iloc[tr_idx], y_tr.map(lmap))
        pred = pd.Series(model.predict(X.iloc[te_idx])).map(inv).values
        accs.append(accuracy_score(y_te, pred))
    return float(np.mean(accs)) if accs else float('nan'), float(np.std(accs)) if accs else float('nan')

def tier1_cross_uci(hhar_df, feature_cols, X_uci, y_uci, seeds=(SEED,)):
    classes = sorted(hhar_df['class_label'].unique())
    lmap = {o: n for n, o in enumerate(classes)}; inv = {n: o for o, n in lmap.items()}
    f4s, fsts = [], []
    for sd in seeds:
        model = make_xgb(len(classes)); model.set_params(random_state=sd)
        model.fit(hhar_df[feature_cols], hhar_df['class_label'].map(lmap))
        pred = pd.Series(model.predict(X_uci[feature_cols])).map(inv).values
        f4s.append(f1_score(y_uci, pred, labels=[0, 2, 3, 4], average='macro'))
        fsts.append(f1_score(y_uci, pred, labels=[2, 3], average='macro'))
    return (float(np.mean(f4s)), float(np.std(f4s)),
            float(np.mean(fsts)), float(np.std(fsts)))


# =============================================================================
# TIER-2 : LIMU-BERT
# =============================================================================
class Embeddings(nn.Module):
    def __init__(self, s_dim, h_dim, max_len):
        super().__init__()
        self.proj = nn.Linear(s_dim, h_dim)
        self.norm1 = nn.LayerNorm(h_dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, h_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.norm2 = nn.LayerNorm(h_dim)
    def forward(self, x):
        h = self.norm1(self.proj(x)); h = h + self.pos[:, :x.size(1)]
        return self.norm2(h)

class MultiHeadAttention(nn.Module):
    """Plain matmul + softmax attention. Unlike the fused SDPA kernel, this
    supports double-backward (needed by the AutoAugHAR bilevel search) on GPU."""
    def __init__(self, h_dim, n_heads):
        super().__init__()
        assert h_dim % n_heads == 0
        self.h, self.d = n_heads, h_dim // n_heads
        self.qkv = nn.Linear(h_dim, 3 * h_dim)
        self.out = nn.Linear(h_dim, h_dim)
    def forward(self, x):                                   # x: [B, L, H]
        B, L, H = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                    # each [B, heads, L, d]
        att = (q @ k.transpose(-2, -1)) / (self.d ** 0.5)  # [B, heads, L, L]
        att = att.softmax(dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, L, H)      # [B, L, H]
        return self.out(o)

class EncoderBlock(nn.Module):
    def __init__(self, h_dim, ff_dim, n_heads):
        super().__init__()
        self.attn = MultiHeadAttention(h_dim, n_heads)
        self.norm_a = nn.LayerNorm(h_dim)
        self.proj = nn.Linear(h_dim, h_dim); self.norm_p = nn.LayerNorm(h_dim)
        self.ff = nn.Sequential(nn.Linear(h_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, h_dim))
        self.norm_h = nn.LayerNorm(h_dim)
    def forward(self, h):
        a = self.norm_a(self.attn(h) + h)
        p = self.norm_p(self.proj(a) + a)
        return self.norm_h(self.ff(p) + p)

class LimuEncoder(nn.Module):
    def __init__(self, s_dim=N_CHANNELS, h_dim=H_DIM, ff_dim=FF_DIM, n_layers=N_LAYERS,
                 n_heads=N_HEADS, max_len=MAX_LEN, share=SHARE_LAYERS):
        super().__init__()
        self.embed = Embeddings(s_dim, h_dim, max_len)
        self.share = share; self.n_layers = n_layers
        if share:
            self.block = EncoderBlock(h_dim, ff_dim, n_heads)
        else:
            self.blocks = nn.ModuleList([EncoderBlock(h_dim, ff_dim, n_heads) for _ in range(n_layers)])
    def forward(self, x):
        h = self.embed(x)
        if self.share:
            for _ in range(self.n_layers): h = self.block(h)
        else:
            for blk in self.blocks: h = blk(h)
        return h

class LimuDecoder(nn.Module):
    def __init__(self, h_dim=H_DIM, s_dim=N_CHANNELS):
        super().__init__()
        self.proj = nn.Linear(h_dim, h_dim); self.norm = nn.LayerNorm(h_dim)
        self.pred = nn.Linear(h_dim, s_dim)
    def forward(self, e):
        return self.pred(self.norm(F.gelu(self.proj(e))))

class LimuBERT(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.encoder = LimuEncoder(**kw)
        self.decoder = LimuDecoder(kw.get('h_dim', H_DIM), kw.get('s_dim', N_CHANNELS))
    def forward(self, x_masked):
        return self.decoder(self.encoder(x_masked))

def _span_mask_np(B, L, mask_ratio, p, l_max):
    """Build the [B, L] boolean mask on CPU (numpy) — cheap, no GPU sync."""
    mask = np.zeros((B, L), dtype=bool)
    n_mask = max(1, int(L * mask_ratio))
    for b in range(B):
        chosen = mask[b]; m, guard = 0, 0
        while m < n_mask and guard < 10 * L:
            guard += 1
            s = np.random.randint(0, L)
            if chosen[s]: continue
            l = min(int(np.random.geometric(p)), l_max, n_mask - m)
            e = min(s + l, L)
            for j in range(s, e):
                if not chosen[j]:
                    chosen[j] = True; m += 1
    return mask

def span_mask(x, mask_ratio=MASK_RATIO, p=SPAN_P, l_max=SPAN_LMAX, p_m=MASK_PROB):
    """Algorithm 1, vectorized. Returns (x_masked, mask[B,L]).
    Positions are always recorded for the loss; actual zeroing happens with
    per-sequence probability p_m. Mask is built on CPU then applied in ONE GPU
    op, so no per-element kernel launches stall the device."""
    B, L, C = x.shape
    mask = torch.from_numpy(_span_mask_np(B, L, mask_ratio, p, l_max)).to(x.device)
    do_zero = (torch.rand(B, device=x.device) < p_m).view(B, 1, 1)   # per-sequence
    x_masked = x.masked_fill(mask.unsqueeze(-1) & do_zero, 0.0)
    return x_masked, mask

class GRUClassifier(nn.Module):
    def __init__(self, h_dim=H_DIM, n_classes=7):
        super().__init__()
        self.g1 = nn.GRU(h_dim, 20, batch_first=True)
        self.g2 = nn.GRU(20, 20, batch_first=True)
        self.g3 = nn.GRU(20, 10, batch_first=True)
        self.drop = nn.Dropout(0.5); self.fc1 = nn.Linear(10, 10); self.fc2 = nn.Linear(10, n_classes)
    def forward(self, e):
        h, _ = self.g1(e); h, _ = self.g2(h); h, _ = self.g3(h)
        h = self.drop(h[:, -1]); return self.fc2(F.relu(self.fc1(h)))

class HARModel(nn.Module):
    def __init__(self, n_classes, freeze_encoder=FREEZE_ENCODER, **kw):
        super().__init__()
        self.encoder = LimuEncoder(**kw)
        self.head = GRUClassifier(kw.get('h_dim', H_DIM), n_classes)
        self.freeze = freeze_encoder
    def load_encoder(self, sd): self.encoder.load_state_dict(sd)
    def forward(self, x):
        if self.freeze:
            with torch.no_grad(): e = self.encoder(x)
        else:
            e = self.encoder(x)
        return self.head(e)


# =============================================================================
# TIER-2 : AutoAugHAR (label-preserving operator set + differentiable policy)
# =============================================================================
def op_identity(x): return x
def op_jitter(x, s):  return x + torch.randn_like(x) * s
def op_scaling(x, s):
    f = torch.randn(x.size(0), 1, x.size(2), device=x.device) * s + 1.0; return x * f
def op_moving_avg(x, ws):
    pad = ws // 2; xt = x.transpose(1, 2)
    w = torch.ones(x.size(2), 1, ws, device=x.device) / ws
    y = F.conv1d(F.pad(xt, (pad, pad), mode='replicate'), w, groups=x.size(2))
    return y[..., :x.size(1)].transpose(1, 2)
def op_slope(x, sr=0.1):
    B, L, C = x.shape
    s = (torch.rand(B, 1, C, device=x.device) * 2 - 1) * sr
    t = torch.linspace(0, 1, L, device=x.device).view(1, L, 1)
    return x + s * t
def _smooth_curve(B, L, C, sigma, n_knots, device):
    knots = torch.randn(B, n_knots, C, device=device) * sigma + 1.0
    return F.interpolate(knots.transpose(1, 2), size=L, mode='linear', align_corners=True).transpose(1, 2)
def op_mag_warp(x, s, n_knots=4):
    return x * _smooth_curve(x.size(0), x.size(1), x.size(2), s, n_knots, x.device)
def op_time_warp(x, s, n_knots=4):
    B, L, C = x.shape
    speed = _smooth_curve(B, L, 1, s, n_knots, x.device).clamp(min=0.1)
    cum = torch.cumsum(speed, dim=1); cum = (cum - cum[:, :1]) / (cum[:, -1:] - cum[:, :1] + 1e-6)
    xs = (cum.squeeze(-1) * 2 - 1).unsqueeze(1).expand(B, C, L)
    ys = torch.linspace(-1, 1, C, device=x.device).view(1, C, 1).expand(B, C, L)
    grid = torch.stack([xs, ys], dim=-1)
    img = x.permute(0, 2, 1).unsqueeze(1)
    out = F.grid_sample(img, grid, align_corners=True, padding_mode='border')
    return out.squeeze(1).transpose(1, 2)
def op_window_slice(x, lam_range=(0.7, 0.9)):
    B, L, C = x.shape; lam = random.uniform(*lam_range)
    w = max(2, int(L * lam)); s = random.randint(0, L - w)
    crop = x[:, s:s + w].transpose(1, 2)
    return F.interpolate(crop, size=L, mode='linear', align_corners=True).transpose(1, 2)

def op_rot_z(x, max_deg=180.0):
    """Random rotation about the VERTICAL axis, applied jointly to acc and gyro
    xy-components. After gravity canonicalization, z is vertical, so pitch/roll
    are already fixed and this op spans exactly the remaining free DOF (heading).
    Alignment (2 DOF, analytic) + yaw augmentation (1 DOF, learned invariance)
    = full orientation invariance. CrossHAR-style rotation aug, made principled
    by the canonicalized frame."""
    B = x.size(0)
    th = (torch.rand(B, 1, device=x.device) * 2 - 1) * math.radians(max_deg)
    c, s = torch.cos(th), torch.sin(th)                     # [B,1]
    out = x.clone()
    for i0 in (0, 3):                                       # acc xy, gyro xy
        xs_, ys_ = x[:, :, i0], x[:, :, i0 + 1]
        out[:, :, i0]     = c * xs_ - s * ys_
        out[:, :, i0 + 1] = s * xs_ + c * ys_
    return out

LABEL_PRESERVING_OPS = [
    ("identity", op_identity),
    ("jitter_0.05", lambda x: op_jitter(x, 0.05)), ("jitter_0.10", lambda x: op_jitter(x, 0.10)),
    ("jitter_0.15", lambda x: op_jitter(x, 0.15)),
    ("scale_0.1", lambda x: op_scaling(x, 0.1)), ("scale_0.2", lambda x: op_scaling(x, 0.2)),
    ("magwarp_0.2", lambda x: op_mag_warp(x, 0.2)), ("magwarp_0.4", lambda x: op_mag_warp(x, 0.4)),
    ("timewarp_0.1", lambda x: op_time_warp(x, 0.1)), ("timewarp_0.2", lambda x: op_time_warp(x, 0.2)),
    ("movavg_3", lambda x: op_moving_avg(x, 3)), ("movavg_5", lambda x: op_moving_avg(x, 5)),
    ("movavg_7", lambda x: op_moving_avg(x, 7)),
    ("winslice", lambda x: op_window_slice(x)), ("slope", lambda x: op_slope(x, 0.1)),
    ("rotz_30",  lambda x: op_rot_z(x, 30.0)), ("rotz_180", lambda x: op_rot_z(x, 180.0)),
]

def _default_policy():
    """Physics-informed fallback policy for when the bilevel search cannot be
    trusted (small source, or search converged to ~uniform). Emphasizes the ops
    that (a) the search consistently selected on large sources (jitter, magwarp)
    and (b) span the residual orientation DOF (yaw rotation)."""
    boost = {'rotz_180': 4.0, 'rotz_30': 2.0, 'jitter_0.10': 3.0, 'jitter_0.15': 3.0,
             'magwarp_0.4': 3.0, 'magwarp_0.2': 2.0, 'scale_0.1': 2.0}
    p = np.array([boost.get(n, 1.0) for n, _ in LABEL_PRESERVING_OPS], dtype=float)
    return p / p.sum()
DEFAULT_POLICY = _default_policy()

# Guards for the AutoAugHAR search
MIN_POLICY_N     = 15000
POLICY_MIN_PEAK  = 1.5

class AugPolicy(nn.Module):
    def __init__(self, n_ops=len(LABEL_PRESERVING_OPS), tau=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(n_ops)); self.tau = tau
        self.ops = [op for _, op in LABEL_PRESERVING_OPS]
    def probs(self): return F.softmax(self.alpha.detach(), dim=0)
    def forward(self, x, hard=False):
        g = F.gumbel_softmax(self.alpha, tau=self.tau, hard=hard)
        outs = torch.stack([op(x) for op in self.ops], dim=0)
        return (g.view(-1, 1, 1, 1) * outs).sum(0)

def mixup_cutmix(x, y, n_classes, a_mix=0.3, a_cut=0.8, p=0.5):
    B = x.size(0); y1 = F.one_hot(y, n_classes).float()
    if random.random() > p: return x, y1
    perm = torch.randperm(B, device=x.device)
    if random.random() < 0.5:
        lam = np.random.beta(a_mix, a_mix); x = lam * x + (1 - lam) * x[perm]
    else:
        lam = np.random.beta(a_cut, a_cut); L = x.size(1)
        w = int(L * (1 - lam)); s = random.randint(0, max(0, L - w))
        x = x.clone(); x[:, s:s + w] = x[perm, s:s + w]; lam = 1 - w / L
    return x, lam * y1 + (1 - lam) * y1[perm]


# =============================================================================
# TIER-2 : training / eval
# =============================================================================
def pretrain_limubert(X_unlabeled, epochs=None, bs=PRETRAIN_BS, lr=PRETRAIN_LR,
                      min_epochs=30, patience=8, verbose=True):
    """Span-masking MLM pretraining with plateau early stop, plus an optional
    UniMTS-style ROTATION-CONSISTENCY term (Zhang et al., NeurIPS'24): two random
    yaw-rotated views of the same window must produce similar pooled embeddings.
    MLM makes the representation informative; the consistency term makes it
    orientation-invariant BY CONSTRUCTION rather than hoping fine-tuning learns
    it. The reconstruction loss prevents the trivial collapse a pure consistency
    objective would allow."""
    if epochs is None:
        epochs = PRETRAIN_EPOCHS
    use_cons = PRETRAIN_CONSISTENCY and len(X_unlabeled) >= CONSISTENCY_MIN_N
    if PRETRAIN_CONSISTENCY and not use_cons and verbose:
        print(f"    [pretrain] pool={len(X_unlabeled)} < {CONSISTENCY_MIN_N} "
              f"-> consistency OFF (plain MLM) to avoid small-pool collapse")
    set_seed(); model = LimuBERT().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(X_unlabeled, dtype=torch.float32); n = len(X)
    best, stall = float('inf'), 0
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, bs):
            xb = X[perm[i:i + bs]].to(DEVICE)
            xm, mask = span_mask(xb)
            loss = F.mse_loss(model(xm)[mask], xb[mask])
            if use_cons:
                v1 = op_rot_z(op_jitter(xb, 0.03), 180.0)
                v2 = op_rot_z(op_jitter(xb, 0.03), 180.0)
                e1 = model.encoder(v1).mean(1)          # [B, H] pooled embeddings
                e2 = model.encoder(v2).mean(1)
                loss = loss + CONSISTENCY_LAMBDA * (1.0 - F.cosine_similarity(e1, e2, dim=1).mean())
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * xb.size(0)
        mse = tot / n
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"    [pretrain] ep {ep:3d}  loss {mse:.4f}")
        if mse < best * 0.995:
            best, stall = mse, 0
        else:
            stall += 1
            if ep >= min_epochs and stall >= patience:
                if verbose:
                    print(f"    [pretrain] early stop at ep {ep} (plateau, best {best:.4f})")
                break
    return {k: v.cpu() for k, v in model.encoder.state_dict().items()}

def _soft_ce(logits, soft): return -(soft * F.log_softmax(logits, dim=1)).sum(1).mean()

def train_classifier(encoder_state, X_tr, y_tr, n_classes, aug_mode="off",
                     epochs=FINETUNE_EPOCHS, bs=FINETUNE_BS, lr=FINETUNE_LR,
                     fixed_policy=None, return_policy=False, freeze=None,
                     seed=SEED, verbose=False):
    """aug_mode:
        'off'    - no augmentation
        'random' - MixUp/CutMix + one random label-preserving op
        'search' - AutoAugHAR bilevel policy search (expensive: double-backward)
        'fixed'  - apply a PRE-LEARNED policy (fixed_policy over the ops), cheap.
                   Use 'search' once via learn_policy(), then 'fixed' everywhere.
       freeze:   override FREEZE_ENCODER. The bilevel search REQUIRES freeze=False
                 (gradients must reach the augmentation params through the encoder).
       seed:     controls init + batch order. MUST be threaded from the caller for
                 multi-seed error bars; a hardcoded set_seed() here would silently
                 make every 'seed' identical (std = 0.000)."""
    set_seed(seed)
    fz = FREEZE_ENCODER if freeze is None else freeze
    model = HARModel(n_classes, freeze_encoder=fz).to(DEVICE)
    # class-weighted CE: static classes dominate several targets
    cnt = np.bincount(np.asarray(y_tr), minlength=n_classes).astype(float)
    cnt[cnt == 0] = 1.0
    cw = torch.tensor(len(y_tr) / (n_classes * cnt), dtype=torch.float32).to(DEVICE)
    if encoder_state is not None:
        model.load_encoder({k: v.to(DEVICE) for k, v in encoder_state.items()})
    w_params = ([p for p in model.parameters() if p.requires_grad]
                if not model.freeze else list(model.head.parameters()))
    w_opt = torch.optim.Adam(w_params, lr=lr)
    Xtr = torch.tensor(X_tr, dtype=torch.float32); ytr = torch.tensor(y_tr, dtype=torch.long)
    fp = np.asarray(fixed_policy, dtype=float) if fixed_policy is not None else None
    if fp is not None:
        fp = fp / fp.sum()

    aug = None; va_idx = []
    if aug_mode == "search":
        aug = AugPolicy().to(DEVICE); a_opt = torch.optim.Adam(aug.parameters(), lr=ALPHA_LR)
        idx = torch.randperm(len(Xtr)); cut = int(0.85 * len(Xtr))
        tr_idx, va_idx = idx[:cut], idx[cut:]
        Xva, yva = Xtr[va_idx], ytr[va_idx]; Xtr, ytr = Xtr[tr_idx], ytr[tr_idx]

    n = len(Xtr)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            xb = Xtr[perm[i:i + bs]].to(DEVICE); yb = ytr[perm[i:i + bs]].to(DEVICE)
            if aug_mode == "search" and len(va_idx) >= bs:
                vsel = torch.randint(0, len(Xva), (bs,))
                xv = Xva[vsel].to(DEVICE); yv = yva[vsel].to(DEVICE)
                with torch.backends.cudnn.flags(enabled=False):
                    params = {k: v for k, v in model.named_parameters() if v.requires_grad}
                    x_aug = aug(xb)
                    loss_tr = F.cross_entropy(
                        functional_call(model, {**dict(model.named_parameters()), **params}, (x_aug,)), yb,
                        weight=cw)
                    grads = torch.autograd.grad(loss_tr, params.values(), create_graph=True, allow_unused=True)
                    vparams = {k: (v - lr * g if g is not None else v)
                               for (k, v), g in zip(params.items(), grads)}
                    full = {**dict(model.named_parameters()), **vparams}
                    loss_val = F.cross_entropy(functional_call(model, full, (xv,)), yv, weight=cw)
                    a_opt.zero_grad(); loss_val.backward(); a_opt.step()
            if aug_mode == "search":
                xin = aug(xb).detach(); loss = F.cross_entropy(model(xin), yb, weight=cw)
            elif aug_mode == "fixed":                       # apply learned policy (cheap)
                if random.random() < AUG_IDENTITY_FLOOR:
                    xin = xb                                # clean-data anchor
                else:
                    op = LABEL_PRESERVING_OPS[int(np.random.choice(len(fp), p=fp))][1]
                    xin = op(xb).detach()
                loss = F.cross_entropy(model(xin), yb, weight=cw)
            elif aug_mode == "random":
                xin, soft = mixup_cutmix(xb, yb, n_classes)
                xin = random.choice(LABEL_PRESERVING_OPS)[1](xin).detach()
                loss = _soft_ce(model(xin), soft)
            else:
                loss = F.cross_entropy(model(xb), yb, weight=cw)
            w_opt.zero_grad(); loss.backward(); w_opt.step()

    policy_out = aug.probs().cpu().numpy() if aug is not None else None
    if aug is not None and verbose:
        print("    learned policy:", {LABEL_PRESERVING_OPS[j][0]: round(float(policy_out[j]), 3)
                                       for j in np.argsort(-policy_out)[:5]})
    return (model, policy_out) if return_policy else model


def learn_policy(encoder_state, X, y, n_classes, epochs=None, seed=SEED):
    """Run the AutoAugHAR bilevel search ONCE and return the learned op
    distribution (numpy). Apply it afterwards with aug_mode='fixed' -- this is
    the search-once/apply-many pattern, ~1 search instead of one per LOSO fold.
    The second-order step may lack kernels on Apple MPS; if it fails there,
    retry the search on CPU (slow but correct), then restore the device."""
    global DEVICE
    ep = epochs if epochs is not None else POLICY_EPOCHS
    try:
        _, policy = train_classifier(encoder_state, X, y, n_classes, aug_mode="search",
                                     epochs=ep, return_policy=True, freeze=False,
                                     seed=seed, verbose=True)
        return policy
    except RuntimeError as e:
        if DEVICE.type != "cpu":
            print(f"    [policy] search failed on {DEVICE} ({str(e)[:90]}...)")
            print("    [policy] retrying the policy search on CPU (slower, one-time) ...")
            old, DEVICE = DEVICE, torch.device("cpu")
            try:
                _, policy = train_classifier(encoder_state, X, y, n_classes, aug_mode="search",
                                             epochs=ep, return_policy=True, freeze=False,
                                             seed=seed, verbose=True)
            finally:
                DEVICE = old
            return policy
        raise


# ---- disk caches: pretraining
CACHE_VERSION = "v3"
def _cache_dir():
    d = os.path.join(SAVE_PATH, 'enc_cache'); os.makedirs(d, exist_ok=True); return d

def cached_pretrain(tag, X_pool, **kw):
    cons = PRETRAIN_CONSISTENCY and len(X_pool) >= CONSISTENCY_MIN_N
    tag = tag + ("" if cons else "_nc")
    fp = os.path.join(_cache_dir(), f"enc_{CACHE_VERSION}_{tag}.pt")
    if RESUME and os.path.exists(fp):
        print(f"    [cache] encoder '{tag}' loaded from disk (delete enc_cache to retrain)")
        return torch.load(fp, map_location='cpu')
    enc = pretrain_limubert(X_pool, **kw)
    try:
        torch.save(enc, fp)
    except Exception as e:
        print(f"    [warn] could not save encoder cache: {e}")
    return enc

def cached_policy(tag, enc, X, y, n_classes):
    fp = os.path.join(_cache_dir(), f"pol_{CACHE_VERSION}_{tag}.npy")
    if RESUME and os.path.exists(fp):
        pol = np.load(fp)
        if len(pol) == len(LABEL_PRESERVING_OPS):
            print(f"    [cache] policy '{tag}' loaded from disk")
            return pol
        print(f"    [cache] policy '{tag}' has stale op count ({len(pol)}) -> re-searching")
    pol = learn_policy(enc, X, y, n_classes)
    try:
        np.save(fp, pol)
    except Exception as e:
        print(f"    [warn] could not save policy cache: {e}")
    return pol

@torch.no_grad()
def predict(model, X, bs=512):
    model.eval(); X = torch.tensor(X, dtype=torch.float32); out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i:i + bs].to(DEVICE)).argmax(1).cpu())
    return torch.cat(out).numpy()

@torch.no_grad()
def predict_proba_t2(model, X, bs=512):
    model.eval(); X = torch.tensor(X, dtype=torch.float32); out = []
    for i in range(0, len(X), bs):
        out.append(F.softmax(model(X[i:i + bs].to(DEVICE)), dim=1).cpu())
    return torch.cat(out).numpy()

def tier2_loso(encoder_state, X, y, groups, aug_mode="off", fixed_policy=None):
    accs = []
    for subj in np.unique(groups):
        te = groups == subj; tr = ~te
        classes = sorted(np.unique(y[tr]))
        if (set(np.unique(y[te])) - set(classes)) or te.sum() < 10:
            continue
        lmap = {c: i for i, c in enumerate(classes)}; inv = {i: c for c, i in lmap.items()}
        y_tr = np.array([lmap[v] for v in y[tr]])
        model = train_classifier(encoder_state, X[tr], y_tr, len(classes),
                                 aug_mode=aug_mode, fixed_policy=fixed_policy)
        pred = np.array([inv[p] for p in predict(model, X[te])])
        accs.append(accuracy_score(y[te], pred))
    return float(np.mean(accs)) if accs else float('nan'), float(np.std(accs)) if accs else float('nan')

def tier2_cross_uci(encoder_state, X_hhar, y_hhar, X_uci, y_uci, aug_mode="off",
                    fixed_policy=None, seeds=(SEED,), verbose=False):
    """Repeat the fine-tune over `seeds` and return mean/std of each metric.
    The cross-domain metric is a SINGLE train/test split, so without repetition it
    carries no error bar at all -- the biggest weakness a reviewer will attack."""
    classes = sorted(np.unique(y_hhar))
    lmap = {c: i for i, c in enumerate(classes)}; inv = {i: c for c, i in lmap.items()}
    y_tr = np.array([lmap[v] for v in y_hhar])
    f4s, fsts = [], []
    for sd in seeds:
        model = train_classifier(encoder_state, X_hhar, y_tr, len(classes), aug_mode=aug_mode,
                                 fixed_policy=fixed_policy, seed=sd,
                                 verbose=(verbose and sd == seeds[0]))
        pred = np.array([inv[p] for p in predict(model, X_uci)])
        f4s.append(f1_score(y_uci, pred, labels=[0, 2, 3, 4], average='macro'))
        fsts.append(f1_score(y_uci, pred, labels=[2, 3], average='macro'))
    return (float(np.mean(f4s)), float(np.std(f4s)),
            float(np.mean(fsts)), float(np.std(fsts)))

G0 = 9.80665

# --- MotionSense --------------------------------------------------------------
# Layout: MS_PATH/<act>_<trial>/sub_<k>.csv with columns gravity.{x,y,z},
# userAcceleration.{x,y,z}, rotationRate.{x,y,z} (units of g / rad/s).
# Total acceleration = (gravity + userAcceleration) * 9.80665  -> gravity PRESENT
MS_LABEL = {'dws': 3, 'ups': 2, 'wlk': 4, 'jog': 5, 'sit': 0, 'std': 0}

def load_motionsense_raw(align, seq_len=XDOMAIN_SEQ_LEN, path=MS_PATH,
                         win=128, step=64):
    if not os.path.exists(path):
        print(f"  [MotionSense] path not found: {path}"); return None, None
    Xs, ys, n_files = [], [], 0
    trial_dirs = []
    for root, dirs, _ in os.walk(path):
        dirs[:] = [d for d in dirs if d != '__MACOSX' and not d.startswith('.')]
        if '__MACOSX' in root:
            continue
        for d in dirs:
            act = d.split('_')[0].lower()
            if act in MS_LABEL:
                trial_dirs.append((os.path.join(root, d), MS_LABEL[act]))
    for tdir, lbl in sorted(trial_dirs):
        for fn in sorted(os.listdir(tdir)):
            if not fn.endswith('.csv') or fn.startswith('.'):
                continue
            try:
                df = pd.read_csv(os.path.join(tdir, fn))
            except Exception as e:
                print(f"  [MotionSense] unreadable {fn}: {e} -- skipped")
                continue
            need = ['gravity.x', 'gravity.y', 'gravity.z',
                    'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z',
                    'rotationRate.x', 'rotationRate.y', 'rotationRate.z']
            if not all(c in df.columns for c in need):
                continue
            n_files += 1
            acc = (df[['gravity.x', 'gravity.y', 'gravity.z']].values
                   + df[['userAcceleration.x', 'userAcceleration.y',
                         'userAcceleration.z']].values) * G0
            gyr = df[['rotationRate.x', 'rotationRate.y', 'rotationRate.z']].values
            for s in range(0, len(acc) - win + 1, step):
                Xs.append(raw_from_arrays(acc[s:s+win].astype(np.float32),
                                          gyr[s:s+win].astype(np.float32),
                                          align, has_gravity=True, seq_len=seq_len))
                ys.append(lbl)
    if not Xs:
        print("  [MotionSense] no windows parsed -- check folder layout/columns.")
        return None, None
    X, y = np.asarray(Xs, np.float32), np.asarray(ys)
    print(f"  [MotionSense] files={n_files}  windows={len(X)}  "
          f"classes={dict(pd.Series(y).value_counts().sort_index())}")
    return X, y

# --- Shoaib 2014 ---------------------------------------------------------------
# Layout: Participant_1.csv .. Participant_10.csv. Each file: a header row
# containing repeated blocks (one per body position: left pocket, right pocket,
# wrist, upper arm, belt) of columns time_stamp, Ax,Ay,Az, Lx,Ly,Lz, Gx,Gy,Gz,
# Mx,My,Mz, and a trailing activity-label column. A=acc WITH gravity, G=gyro.
SHOAIB_LABEL = {'walking': 4, 'jogging': 5, 'running': 5, 'sitting': 0,
                'standing': 0, 'biking': 6, 'upstairs': 2, 'downstairs': 3}

_SHOAIB_META = {}   # align -> {"pos": [N], "subj": [N]}, filled by load_shoaib_raw

def _shoaib_label(s):
    s = str(s).strip().lower()
    for k, v in SHOAIB_LABEL.items():
        if k in s:
            return v
    return -1

def load_shoaib_raw(align, seq_len=XDOMAIN_SEQ_LEN, path=SHOAIB_PATH,
                    win=128, step=64):
    if not os.path.exists(path):
        print(f"  [Shoaib] path not found: {path}"); return None, None
    files = sorted(glob.glob(os.path.join(path, '**', '*.csv'), recursive=True))
    if not files:
        print("  [Shoaib] no CSV files found under path."); return None, None
    Xs, ys, ps, subj = [], [], [], []
    for f_i, fp in enumerate(files):
        raw = pd.read_csv(fp, header=None, low_memory=False, dtype=str)
        hdr_row = None
        for r in range(min(5, len(raw))):
            if (raw.iloc[r].astype(str).str.strip() == 'Ax').any():
                hdr_row = r; break
        if hdr_row is None:
            print(f"  [Shoaib] {os.path.basename(fp)}: no 'Ax' header found, skipped.")
            continue
        names = raw.iloc[hdr_row].astype(str).str.strip().tolist()
        data = raw.iloc[hdr_row + 1:].reset_index(drop=True)
        ax_cols = [i for i, n in enumerate(names) if n == 'Ax']
        gx_cols = [i for i, n in enumerate(names) if n == 'Gx']
        # activity label = last column
        lab = data.iloc[:, -1].map(_shoaib_label).values
        n_pos = min(len(ax_cols), len(gx_cols))
        if n_pos == 0:
            print(f"  [Shoaib] {os.path.basename(fp)}: no sensor blocks, skipped.")
            continue
        for p in range(n_pos):
            try:
                acc = data.iloc[:, ax_cols[p]:ax_cols[p]+3].astype(float).values
                gyr = data.iloc[:, gx_cols[p]:gx_cols[p]+3].astype(float).values
            except Exception:
                continue
            # run-length segments of constant, valid label
            change = np.where(np.diff(lab) != 0)[0] + 1
            bounds = np.concatenate([[0], change, [len(lab)]])
            for b0, b1 in zip(bounds[:-1], bounds[1:]):
                if lab[b0] < 0 or (b1 - b0) < win:
                    continue
                a_seg, g_seg = acc[b0:b1], gyr[b0:b1]
                ok = np.isfinite(a_seg).all(1) & np.isfinite(g_seg).all(1)
                a_seg, g_seg = a_seg[ok], g_seg[ok]
                for s in range(0, len(a_seg) - win + 1, step):
                    Xs.append(raw_from_arrays(a_seg[s:s+win].astype(np.float32),
                                              g_seg[s:s+win].astype(np.float32),
                                              align, has_gravity=True, seq_len=seq_len))
                    ys.append(int(lab[b0])); ps.append(p); subj.append(f_i)
    if not Xs:
        print("  [Shoaib] parsed 0 windows -- check the file format notes above.")
        return None, None
    X, y = np.asarray(Xs, np.float32), np.asarray(ys)
    # side-channel: body position + subject per window
    _SHOAIB_META[align] = {"pos": np.asarray(ps), "subj": np.asarray(subj)}
    print(f"  [Shoaib] files={len(files)}  windows={len(X)}  "
          f"positions={len(np.unique(ps))}  "
          f"classes={dict(pd.Series(y).value_counts().sort_index())}")
    return X, y

# --- HHAR / UCI raw at cross-domain geometry
_HHAR_RAW_CACHE = {}
def _hhar_xd_raw(align):
    if align not in _HHAR_RAW_CACHE:
        acc_raw, gyro_raw = load_hhar_phone()
        Xh, yh, _ = window_hhar_raw(acc_raw, gyro_raw, align,
                                    window_sec=XDOMAIN_WINDOW_SEC,
                                    seq_len=XDOMAIN_SEQ_LEN)
        _HHAR_RAW_CACHE[align] = (Xh, yh)
    return _HHAR_RAW_CACHE[align]

DATASETS = {
    'hhar':   lambda align: _hhar_xd_raw(align),
    'uci':    lambda align: build_uci_raw(align, seq_len=XDOMAIN_SEQ_LEN),
    'motion': lambda align: load_motionsense_raw(align),
    'shoaib': lambda align: load_shoaib_raw(align),
}

_DS_CACHE = {}
def get_dataset_raw(name, align):
    key = (name, align)
    if key not in _DS_CACHE:
        print(f"[data] loading {name} (align={align}) ...")
        _DS_CACHE[key] = DATASETS[name](align)
    return _DS_CACHE[key]


TRANSFER_PAIRS = [
    ("hhar", "uci"), ("uci", "hhar"),
    ("hhar", "motion"), ("motion", "hhar"),
    ("hhar", "shoaib"), ("shoaib", "hhar"),
    ("uci", "motion"), ("motion", "uci"),
    ("uci", "shoaib"), ("shoaib", "uci"),
    ("motion", "shoaib"), ("shoaib", "motion"),
]
ALL_DATASET_NAMES = ["hhar", "uci", "motion", "shoaib"]
TRANSFER_CONFIGS = [
    {"name": "T1 XGBoost",              "kind": "t1"},
    {"name": "T2 gravON [source]",      "kind": "t2", "align": True, "scope": "source", "aug": "off"},
    {"name": "T2 gravON [multi-src]",   "kind": "t2", "align": True, "scope": "multi",  "aug": "off"},
    {"name": "T2 gravON [pooled/UDA]",  "kind": "t2", "align": True, "scope": "pooled", "aug": "off"},
    {"name": "T2+Aug gravON [source]",  "kind": "t2", "align": True, "scope": "source", "aug": "search"},
    # Soft-voting
    {"name": "T1+T2 ensemble [source]", "kind": "ens", "align": True, "scope": "source", "aug": "search"},
]

def _features_df_from_raw(X_raw):
    """Tier-1 features from rate-consistent raw windows (acc re-scaled to m/s^2
    so the gravity-centering branch in extract_sensor_features behaves as in the
    original pipeline)."""
    rows = []
    for w in X_raw:
        a = pd.DataFrame(w[:, 0:3] * G0, columns=['x', 'y', 'z'])
        g = pd.DataFrame(w[:, 3:6],      columns=['x', 'y', 'z'])
        feats, _ = features_from_windows(a, g, sample_hz=TARGET_HZ)
        feats['carrying_load'] = 0
        rows.append(feats)
    return pd.DataFrame(rows).fillna(0)

_T1_FEAT_CACHE = {}
def _t1_features(name, X_raw):
    """Tier-1 features with a DISK cache: extraction is the slow CPU stage
    (~minutes for 48k Shoaib windows) and was previously recomputed every
    session. Keyed by dataset name + cache version; window definitions are
    deterministic per dataset, so the cache is stable across sessions."""
    if name in _T1_FEAT_CACHE:
        return _T1_FEAT_CACHE[name]
    fp = os.path.join(_cache_dir(), f"t1feat_{CACHE_VERSION}_{name}.joblib")
    if RESUME and os.path.exists(fp):
        try:
            df = joblib.load(fp)
            if len(df) == len(X_raw):
                print(f"    [cache] T1 features '{name}' loaded from disk ({len(df)} rows)")
                _T1_FEAT_CACHE[name] = df
                return df
            print(f"    [cache] T1 features '{name}' stale (rows {len(df)} != {len(X_raw)}) -> recompute")
        except Exception as e:
            print(f"    [cache] T1 features '{name}' unreadable ({e}) -> recompute")
    print(f"    [T1] extracting features for {name} ({len(X_raw)} windows) ...")
    df = _features_df_from_raw(X_raw)
    try:
        joblib.dump(df, fp)
    except Exception as e:
        print(f"    [warn] could not save T1 feature cache: {e}")
    _T1_FEAT_CACHE[name] = df
    return df

def _macro_f1(y_true, y_pred, labels):
    out = {"f1": f1_score(y_true, y_pred, labels=labels, average='macro')}
    if 2 in labels and 3 in labels:
        out["stairs"] = f1_score(y_true, y_pred, labels=[2, 3], average='macro')
    else:
        out["stairs"] = float('nan')
    return out

def transfer_eval_t2(enc, Xs, ys, Xt, yt, labels, aug_mode="off",
                     fixed_policy=None, seeds=SEEDS, report=False):
    lmap = {c: i for i, c in enumerate(labels)}; inv = {i: c for c, i in lmap.items()}
    y_tr = np.array([lmap[v] for v in ys])
    f1s, sts = [], []
    for si, sd in enumerate(seeds):
        m = train_classifier(enc, Xs, y_tr, len(labels), aug_mode=aug_mode,
                             fixed_policy=fixed_policy, seed=sd)
        pred = np.array([inv[p] for p in predict(m, Xt)])
        r = _macro_f1(yt, pred, labels)
        f1s.append(r["f1"]); sts.append(r["stairs"])
        if report and si == 0:
            per = f1_score(yt, pred, labels=labels, average=None)
            print("      per-class F1 (seed %d): %s" % (
                sd, {CLASS_NAMES.get(l, l): round(float(v), 3)
                     for l, v in zip(labels, per)}))
    return (float(np.mean(f1s)), float(np.std(f1s)),
            float(np.nanmean(sts)), float(np.nanstd(sts)))

def run_transfer_matrix(pairs=None, configs=None):
    pairs = pairs or TRANSFER_PAIRS
    configs = configs or TRANSFER_CONFIGS
    results, enc_cache, pol_cache, pol_src = [], {}, {}, {}
    tm_fp = os.path.join(SAVE_PATH, f"transfer_matrix_partial_{CACHE_VERSION}.csv")
    done = set()
    if RESUME and os.path.exists(tm_fp):
        prev = pd.read_csv(tm_fp)
        results = prev.to_dict('records')
        done = set(zip(prev['pair'], prev['config']))
        print(f"[resume] transfer matrix: {len(done)} cells already done, skipping them.")
    for (src, tgt) in pairs:
        pair_name = f"{src}->{tgt}"
        if all((pair_name, cfg["name"]) in done for cfg in configs):
            print(f"[resume] pair {pair_name}: all configs done, skipping data load.")
            continue
        Xsrc = get_dataset_raw(src, True);  Xtgt = get_dataset_raw(tgt, True)
        if Xsrc[0] is None or Xtgt[0] is None:
            print(f"[transfer] SKIP {src}->{tgt}: dataset missing."); continue
        Xs_a, ys_a = Xsrc; Xt_a, yt_a = Xtgt
        labels = sorted(int(c) for c in (set(np.unique(ys_a)) & set(np.unique(yt_a))))
        if len(labels) < 2:
            print(f"[transfer] SKIP {src}->{tgt}: <2 shared classes."); continue
        ms = np.isin(ys_a, labels); mt = np.isin(yt_a, labels)
        Xs, ys = Xs_a[ms], ys_a[ms]; Xt, yt = Xt_a[mt], yt_a[mt]
        print(f"\n[transfer] {src} -> {tgt} | shared={labels} | "
              f"train n={len(Xs)}  test n={len(Xt)}")
        # unaligned copies for T1
        Xs_u = get_dataset_raw(src, False)[0]; Xt_u = get_dataset_raw(tgt, False)[0]
        ys_u = get_dataset_raw(src, False)[1]; yt_u = get_dataset_raw(tgt, False)[1]
        msu = np.isin(ys_u, labels); mtu = np.isin(yt_u, labels)

        for cfg in configs:
            policy_used = "n/a"
            if (pair_name, cfg["name"]) in done:
                print(f"    [resume] skip {pair_name} / {cfg['name']}"); continue
            if cfg["kind"] == "t1":
                Ftr = _t1_features(f"{src}_u", Xs_u).iloc[np.where(msu)[0]]
                Fte = _t1_features(f"{tgt}_u", Xt_u).iloc[np.where(mtu)[0]]
                lmap = {c: i for i, c in enumerate(labels)}
                inv = {i: c for c, i in lmap.items()}
                y_map = pd.Series(ys_u[msu]).map(lmap)
                # class-balanced sample weights
                freq = y_map.value_counts()
                sw = y_map.map(lambda c: len(y_map) / (len(labels) * freq[c])).values
                f1s, sts = [], []
                for sd in SEEDS:
                    mdl = make_xgb(len(labels)); mdl.set_params(random_state=sd)
                    mdl.fit(Ftr, y_map, sample_weight=sw)
                    pred = pd.Series(mdl.predict(Fte)).map(inv).values
                    r = _macro_f1(yt_u[mtu], pred, labels)
                    f1s.append(r["f1"]); sts.append(r["stairs"])
                f1m, f1sd = float(np.mean(f1s)), float(np.std(f1s))
                stm, stsd = float(np.nanmean(sts)), float(np.nanstd(sts))
            elif cfg["kind"] == "ens":
                assert len(Xt) == int(mtu.sum()), "aligned/unaligned window mismatch"
                align, scope, aug = cfg["align"], cfg["scope"], cfg["aug"]
                ekey = ('src', src, align)
                enc = enc_cache.get(ekey)
                if enc is None:
                    print("    [ens] source encoder not in cache -> pretraining ...")
                    enc = cached_pretrain(f"tm_src_{src}_{align}", Xs, verbose=False)
                    enc_cache[ekey] = enc
                pol = pol_cache.get(ekey)
                if pol is None:
                    if len(Xs) < MIN_POLICY_N:
                        pol = DEFAULT_POLICY
                        pol_src[ekey] = "fixed-default(small-source)"
                    else:
                        lm0 = {c: i for i, c in enumerate(labels)}
                        pol = cached_policy(f"tm_src_{src}_{align}", enc, Xs,
                                            np.array([lm0[v] for v in ys]), len(labels))
                        if pol.max() < POLICY_MIN_PEAK / len(pol):
                            pol = DEFAULT_POLICY
                            pol_src[ekey] = "fixed-default(search~uniform)"
                        else:
                            pol_src[ekey] = "searched(AutoAugHAR)"
                    pol_cache[ekey] = pol
                policy_used = pol_src.get(ekey, "unknown")
                Ftr = _t1_features(f"{src}_u", Xs_u).iloc[np.where(msu)[0]]
                Fte = _t1_features(f"{tgt}_u", Xt_u).iloc[np.where(mtu)[0]]
                lmap = {c: i for i, c in enumerate(labels)}
                inv = {i: c for c, i in lmap.items()}
                y_map = pd.Series(ys_u[msu]).map(lmap)
                freq = y_map.value_counts()
                sw = y_map.map(lambda c: len(y_map) / (len(labels) * freq[c])).values
                y_tr2 = np.array([lmap[v] for v in ys])
                f1s, sts = [], []
                for sd in SEEDS:
                    mdl = make_xgb(len(labels)); mdl.set_params(random_state=sd)
                    mdl.fit(Ftr, y_map, sample_weight=sw)
                    p1 = mdl.predict_proba(Fte)              # [N, n_labels]
                    m2 = train_classifier(enc, Xs, y_tr2, len(labels), aug_mode="fixed",
                                          fixed_policy=pol, seed=sd)
                    p2 = predict_proba_t2(m2, Xt)            # [N, n_labels]
                    pred = np.array([inv[p] for p in (0.5 * p1 + 0.5 * p2).argmax(1)])
                    r = _macro_f1(yt, pred, labels)
                    f1s.append(r["f1"]); sts.append(r["stairs"])
                f1m, f1sd = float(np.mean(f1s)), float(np.std(f1s))
                stm, stsd = float(np.nanmean(sts)), float(np.nanstd(sts))
            else:
                align, scope, aug = cfg["align"], cfg["scope"], cfg["aug"]
                if scope == "source":
                    ekey = ('src', src, align);  etag = f"tm_src_{src}_{align}"
                elif scope == "pooled":
                    ekey = ('pool', frozenset({src, tgt}), align)
                    etag = f"tm_pool_{'-'.join(sorted({src, tgt}))}_{align}"
                else:                                    # 'multi': all datasets but target
                    ekey = ('multi', tgt, align);        etag = f"tm_multi_no-{tgt}_{align}"
                if ekey not in enc_cache:
                    if scope == "source":
                        pool = Xs
                    elif scope == "pooled":
                        pool = np.concatenate([Xs, Xt], 0)
                    else:
                        parts, rng = [], np.random.RandomState(SEED)
                        for nm in ALL_DATASET_NAMES:
                            if nm == tgt:
                                continue
                            d = get_dataset_raw(nm, align)
                            if d[0] is None:
                                continue
                            Xd = d[0]
                            if len(Xd) > MULTI_CAP:         # balance: cap each dataset
                                Xd = Xd[rng.choice(len(Xd), MULTI_CAP, replace=False)]
                            parts.append(Xd)
                        pool = np.concatenate(parts, 0)
                    print(f"    [T2] encoder {etag} (pool={len(pool)}) ...")
                    enc_cache[ekey] = cached_pretrain(etag, pool, verbose=False)
                enc = enc_cache[ekey]
                pol = None
                if aug == "search":
                    pkey = ekey
                    if pkey not in pol_cache:
                        if len(Xs) < MIN_POLICY_N:
                            print(f"    [T2] source too small for policy search "
                                  f"(n={len(Xs)} < {MIN_POLICY_N}) -> DEFAULT_POLICY")
                            pol_cache[pkey] = DEFAULT_POLICY
                            pol_src[pkey] = "fixed-default(small-source)"
                        else:
                            lm = {c: i for i, c in enumerate(labels)}
                            print(f"    [T2] aug policy for {etag} ...")
                            learned = cached_policy(
                                etag, enc, Xs, np.array([lm[v] for v in ys]), len(labels))
                            if learned.max() < POLICY_MIN_PEAK / len(learned):
                                print("    [T2] search stayed ~uniform -> DEFAULT_POLICY")
                                learned = DEFAULT_POLICY
                                pol_src[pkey] = "fixed-default(search~uniform)"
                            else:
                                pol_src[pkey] = "searched(AutoAugHAR)"
                            pol_cache[pkey] = learned
                    pol = pol_cache[pkey]
                    policy_used = pol_src.get(pkey, "unknown")
                eff = "fixed" if aug == "search" else aug
                f1m, f1sd, stm, stsd = transfer_eval_t2(
                    enc, Xs, ys, Xt, yt, labels, aug_mode=eff, fixed_policy=pol,
                    report=True)
            results.append({"pair": f"{src}->{tgt}", "config": cfg["name"],
                            "shared_classes": str(labels), "policy_used": policy_used,
                            "macroF1": f1m, "macroF1_std": f1sd,
                            "stairsF1": stm, "stairsF1_std": stsd})
            print(f"    {cfg['name']:28s} F1={f1m:.3f}±{f1sd:.3f}  "
                  f"stairs={stm:.3f}±{stsd:.3f}")
            try:
                pd.DataFrame(results).to_csv(tm_fp, index=False)
            except Exception as e:
                print(f"    [warn] save failed: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        disp = df.copy()
        disp["macro-F1"] = disp.apply(lambda r: f"{r.macroF1:.3f} ± {r.macroF1_std:.3f}", axis=1)
        disp["stairs-F1"] = disp.apply(
            lambda r: ("--" if np.isnan(r.stairsF1) else f"{r.stairsF1:.3f} ± {r.stairsF1_std:.3f}"), axis=1)
        print("\n" + "=" * 84)
        print("TRANSFER MATRIX  (rate-consistent 2.56 s / 20 Hz windows, shared-class macro-F1)")
        print("=" * 84)
        if "policy_used" not in disp.columns:
            disp["policy_used"] = "n/a"
        disp["policy_used"] = disp["policy_used"].fillna("n/a")
        print(disp[["pair", "config", "policy_used", "macro-F1", "stairs-F1"]]
              .to_string(index=False))
        print("=" * 84)
    return df


# =============================================================================
# PROBE : MotionSense walk collapse
# =============================================================================
def _walk_fingerprint(X, y, name, hz=TARGET_HZ):
    w = X[y == 4]
    if len(w) == 0:
        print(f"  [{name}] no walk windows"); return
    mag = np.linalg.norm(w[:, :, 0:3], axis=2)              # [N, L] acc magnitude (g)
    mag = mag - mag.mean(axis=1, keepdims=True)
    fft = np.abs(np.fft.rfft(mag, axis=1))
    frq = np.fft.rfftfreq(mag.shape[1], d=1.0 / hz)
    dom = frq[1 + fft[:, 1:].argmax(axis=1)]
    eng = mag.std(axis=1)
    gyr = np.linalg.norm(w[:, :, 3:6], axis=2).std(axis=1)
    print(f"  [{name:8s}] walk n={len(w):5d} | dom-freq {np.median(dom):.2f} Hz "
          f"(IQR {np.percentile(dom,25):.2f}-{np.percentile(dom,75):.2f}) | "
          f"acc-energy {np.median(eng):.3f} g (IQR {np.percentile(eng,25):.3f}-"
          f"{np.percentile(eng,75):.3f}) | gyro-energy {np.median(gyr):.3f}")

def apply_instance_norm_raw(X):
    """Per-window, per-channel z-score. Window-local statistics only ->
    zero target access. Destroys absolute magnitude; keeps shape/frequency."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-6
    return ((X - mu) / sd).astype(np.float32)

def probe_policy_confound(n_match=24208, align=True):
    """SUPERSEDED by probe_policy_datasize(retrain_encoder=True), which runs
    this same cut plus a size-matched single-position arm and multiple seeds.
    Kept so earlier results stay reproducible.

    Disentangle WHY the bilevel search converged on Shoaib but not elsewhere:
    data volume (34-41k windows) or placement diversity (5 body positions)?
    Design: subsample Shoaib (shared classes with UCI, i.e. [0,2,3,4]) down to
    n_match -- the size at which the HHAR search stayed ~uniform -- while keeping
    all five positions represented (positions are balanced by construction, so a
    uniform random subsample preserves their proportions in expectation). Then
    pretrain on the subsample and run the search end-to-end.
      * search CONVERGES (peak >= 1.5/17)  -> placement diversity drives it
      * search stays ~uniform              -> data volume is the requirement
    Caveat for the paper: subsampled Shoaib still differs from HHAR in device
    and subject population, so this is a strong probe, not a perfect isolation."""
    print("\n" + "=" * 78)
    print(f"PROBE: policy-search confound (size vs diversity), n_match={n_match}")
    print("=" * 78)
    X, y = get_dataset_raw('shoaib', align)
    if X is None:
        print("Shoaib not available."); return
    labels = [0, 2, 3, 4]                       # match the hhar->uci search setting
    m = np.isin(y, labels)
    Xs, ys = X[m], y[m]
    print(f"  Shoaib shared-class pool: {len(Xs)} windows "
          f"(full-size search peaked at 0.121 and CONVERGED)")
    if len(Xs) <= n_match:
        print("  pool already <= n_match; nothing to subsample."); return
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(Xs), n_match, replace=False)
    Xsub, ysub = Xs[idx], ys[idx]
    print(f"  subsampled to {len(Xsub)} windows "
          f"(size at which the HHAR search stayed ~uniform)")
    enc = cached_pretrain(f"probe_shoaib_sub{n_match}_{align}", Xsub, verbose=False)
    lm = {c: i for i, c in enumerate(labels)}
    pol = learn_policy(enc, Xsub, np.array([lm[v] for v in ysub]), len(labels))
    thr = POLICY_MIN_PEAK / len(pol)
    top = {LABEL_PRESERVING_OPS[j][0]: round(float(pol[j]), 3)
           for j in np.argsort(-pol)[:5]}
    print(f"  search peak = {pol.max():.3f}  (uniform = {1/len(pol):.3f}, "
          f"convergence threshold = {thr:.3f})")
    print(f"  top-5 ops: {top}")
    if pol.max() >= thr:
        print("  VERDICT: search CONVERGED at HHAR-matched size ->")
        print("           placement DIVERSITY drives search convergence.")
    else:
        print("  VERDICT: search stayed ~uniform at HHAR-matched size ->")
        print("           data VOLUME is the binding requirement.")
    print("=" * 78)


def probe_motion(align=True):
    """Run after the matrix. Prints P1 fingerprints for all datasets' walk
    windows, then P2: hhar->motion with and without instance norm, per-class."""
    print("\n" + "=" * 78)
    print("PROBE: MotionSense walk collapse")
    print("=" * 78)
    data = {nm: get_dataset_raw(nm, align) for nm in ALL_DATASET_NAMES}
    print("P1 -- walk fingerprints (dominant cadence vs signal energy):")
    for nm, (X, y) in data.items():
        if X is not None:
            _walk_fingerprint(X, y, nm)
    Xs, ys = data['hhar']; Xt, yt = data['motion']
    if Xs is None or Xt is None:
        print("P2 skipped: dataset missing."); return
    labels = sorted(int(c) for c in (set(np.unique(ys)) & set(np.unique(yt))))
    ms, mt = np.isin(ys, labels), np.isin(yt, labels)
    Xs, ys, Xt, yt = Xs[ms], ys[ms], Xt[mt], yt[mt]
    enc = cached_pretrain(f"tm_src_hhar_{align}", Xs, verbose=False)
    print("\nP2 -- instance-norm ablation (hhar->motion, T2+FixedAug source):")
    for tag, fn in [("raw", lambda a: a), ("instnorm", apply_instance_norm_raw)]:
        Xs_v, Xt_v = fn(Xs), fn(Xt)
        e = enc if tag == "raw" else cached_pretrain(f"probe_hhar_instnorm_{align}",
                                                     Xs_v, verbose=False)
        f1m, f1sd, stm, stsd = transfer_eval_t2(
            e, Xs_v, ys, Xt_v, yt, labels, aug_mode="fixed",
            fixed_policy=DEFAULT_POLICY, report=True)
        print(f"  {tag:9s} macro-F1={f1m:.3f}±{f1sd:.3f}  stairs={stm:.3f}±{stsd:.3f}")
    print("Reading: if instnorm recovers walk but dents static, the Motion gap is")
    print("device gain (dynamics), confirming the orientation-vs-dynamics boundary.")
    print("=" * 78)


# Subsampling is stratified over (position x class) cells
def _stratified_subsample(y, pos, n_target, rng, keep_positions=None):
    """Indices of an n_target-sized subset preserving the (position, class)
    composition of the input; optionally restricted to `keep_positions` first."""
    idx = np.arange(len(y))
    if keep_positions is not None:
        idx = idx[np.isin(pos[idx], keep_positions)]
    if n_target >= len(idx):
        return idx
    cells = {}
    for i in idx:
        cells.setdefault((int(pos[i]), int(y[i])), []).append(i)
    total, keep = len(idx), []
    for _, members in sorted(cells.items()):
        take = min(len(members), int(round(n_target * len(members) / total)))
        if take > 0:
            keep.extend(rng.choice(members, take, replace=False))
    keep = np.asarray(keep, dtype=int)
    if len(keep) > n_target:
        keep = rng.choice(keep, n_target, replace=False)
    return np.sort(keep)

def probe_policy_datasize(align=True, seeds=(42,), n_hhar=28914,
                          include_full=False, retrain_encoder=False,
                          downstream=False, downstream_target="motion"):
    """2x2 probe: does the augmentation search need volume or diversity?

    `retrain_encoder=False` (default) holds the encoder FIXED at the cached
    full-Shoaib source encoder, so only the labelled set the search runs on
    changes. This isolates the search itself: a failure cannot be blamed on a
    weaker encoder. `retrain_encoder=True` instead pretrains a fresh encoder per
    arm, which answers the different (also fair) question of what would happen
    if the dataset really were that small end-to-end -- at roughly double the
    compute. Reporting both is stronger than either alone; if time allows, run
    the fixed-encoder version first.

    `seeds` repeats each arm (the search is stochastic; a single run is thin
    evidence). `downstream=True` additionally fine-tunes with each arm's policy
    on Shoaib-><target> to check whether a converged policy still delivers its
    gain -- costs one fine-tune per arm per seed, so it is off by default."""
    print("\n" + "=" * 78)
    print("PROBE: policy-search convergence -- data volume vs placement diversity")
    print("=" * 78)
    X, y = get_dataset_raw('shoaib', align)
    meta = _SHOAIB_META.get(align)
    if X is None or meta is None:
        print("  Shoaib not loaded (or position metadata missing); probe skipped.")
        return None
    pos = meta["pos"]
    if len(pos) != len(y):
        print("  position metadata length mismatch; probe skipped."); return None

    labels = [0, 2, 3, 4, 6]
    m = np.isin(y, labels)
    X, y, pos = X[m], y[m], pos[m]
    lm = {c: i for i, c in enumerate(labels)}
    y_idx = np.array([lm[v] for v in y])

    enc = cached_pretrain(f"tm_src_shoaib_{align}", X, verbose=False)

    positions = np.unique(pos)
    per_pos = {int(p): int((pos == p).sum()) for p in positions}
    by_size = sorted(per_pos, key=per_pos.get, reverse=True)
    p_big = by_size[0]
    n_single = per_pos[p_big]
    p_mid = by_size[:3]
    n_mid = sum(per_pos[p] for p in p_mid)
    rng = np.random.RandomState(SEED)
    print(f"  source n={len(X)}  positions={len(positions)}  per-position={per_pos}")

    arms = [
        ("size-cut",   _stratified_subsample(y, pos, n_hhar,   rng)),          # 5 pos, HHAR volume
        ("mid-5pos",   _stratified_subsample(y, pos, n_mid,    rng)),          # 5 pos, mid volume
        ("mid-3pos",   _stratified_subsample(y, pos, n_mid,    rng, p_mid)),   # 3 pos, same volume
        ("div-cut",    _stratified_subsample(y, pos, n_single, rng, [p_big])), # 1 pos, low volume
    ]
    if include_full:
        arms.insert(0, ("full", np.arange(len(X))))
    print("  arms: " + ", ".join(f"{n}(n={len(i)})" for n, i in arms))

    uniform = 1.0 / len(LABEL_PRESERVING_OPS)
    thr = POLICY_MIN_PEAK / len(LABEL_PRESERVING_OPS)
    rows = []
    for name, idx in arms:
        got = {int(p): int((pos[idx] == p).sum()) for p in np.unique(pos[idx])}
        print(f"\n  --- arm '{name}': n={len(idx)}, positions={len(got)} {got}")
        if retrain_encoder and name != "full":
            enc_arm = cached_pretrain(f"probe_arm_{name}_{len(idx)}_{align}",
                                      X[idx], verbose=False)
        else:
            enc_arm = enc
        for sd in seeds:
            pol = learn_policy(enc_arm, X[idx], y_idx[idx], len(labels), seed=sd)
            peak = float(pol.max())
            top = [LABEL_PRESERVING_OPS[j][0] for j in np.argsort(-pol)[:3]]
            conv = peak >= thr
            print(f"      seed {sd}: peak={peak:.4f} ({peak/uniform:.2f}x uniform)  "
                  f"{'CONVERGED' if conv else 'near-uniform'}  top3={top}")
            row = {"arm": name, "n": len(idx), "positions": len(got), "seed": sd,
                   "encoder": "per-arm" if (retrain_encoder and name != "full") else "fixed",
                   "peak": peak, "peak_over_uniform": peak / uniform,
                   "converged": bool(conv), "top3": ",".join(top)}
            if downstream:
                Xt, yt = get_dataset_raw(downstream_target, align)
                if Xt is not None:
                    sh = sorted(int(c) for c in (set(labels) & set(np.unique(yt))))
                    a = np.isin(y[idx], sh); b = np.isin(yt, sh)
                    f1m, f1sd, _, _ = transfer_eval_t2(
                        enc_arm, X[idx][a], y[idx][a], Xt[b], yt[b], sh,
                        aug_mode="fixed", fixed_policy=pol, seeds=(sd,))
                    row["downstream_f1"] = f1m
                    print(f"                downstream shoaib->{downstream_target}: "
                          f"macro-F1={f1m:.3f}")
            rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("SUMMARY (guard threshold = %.4f; uniform = %.4f)" % (thr, uniform))
    agg = df.groupby("arm", sort=False).agg(
        n=("n", "first"), positions=("positions", "first"),
        peak_mean=("peak", "mean"), peak_min=("peak", "min"),
        converged=("converged", "all"))
    print(agg.to_string())
    print("\nReading:")
    print("  size-cut CONVERGED  -> HHAR-scale volume suffices when 5 positions")
    print("                         are present; placement DIVERSITY is the driver.")
    print("  size-cut ~uniform   -> data VOLUME is the binding requirement.")
    print("  mid-5pos vs mid-3pos-> volume matched, diversity cut: the direct test.")
    print("  reference: full Shoaib (41,005 / 5 pos) converged at peak 0.121.")
    print("=" * 78)
    try:
        df.to_csv(os.path.join(SAVE_PATH, f"policy_probe_{CACHE_VERSION}.csv"), index=False)
    except Exception as e:
        print(f"  [warn] could not save probe CSV: {e}")
    return df


TIER2_RUNS = [
    # --- gravity ablation (pooled) : grav-ON row is also the pooled member below
    {"name": "T2 LIMU-BERT (grav OFF)  [pooled/UDA]",  "align": False, "aug": "off",    "scope": "pooled"},
    {"name": "T2 LIMU-BERT (grav ON)   [pooled/UDA]",  "align": True,  "aug": "off",    "scope": "pooled"},
    # --- source counterpart of grav-ON LIMU-BERT (completes the scope pair)
    {"name": "T2 LIMU-BERT (grav ON)   [source-only]", "align": True,  "aug": "off",    "scope": "source"},
    # --- +AutoAugHAR : source-vs-pooled pair at best alignment
    {"name": "T2 +AutoAugHAR (grav ON) [pooled/UDA]",  "align": True,  "aug": "search", "scope": "pooled"},
    {"name": "T2 +AutoAugHAR (grav ON) [source-only]", "align": True,  "aug": "search", "scope": "source"},
]

def run_head_to_head(acc_raw, gyro_raw, hhar_df, feature_cols):
    set_seed()
    results, done = [], set()
    h2h_fp = os.path.join(SAVE_PATH, "head_to_head_partial.csv")
    if RESUME and os.path.exists(h2h_fp):
        prev = pd.read_csv(h2h_fp)
        results = prev.to_dict('records'); done = set(prev['model'])
        print(f"[resume] head-to-head: {len(done)} rows already done, skipping them.")

    if "T1 XGBoost (GenHAR+LLM4HAR)" not in done:
        print("\n[Tier-1] XGBoost LOSO + cross-UCI ...")
        t1_loso_m, t1_loso_s = tier1_loso(hhar_df, feature_cols)
        X_uci_feat, y_uci_feat = build_uci_features(feature_cols)
        t1_f4, t1_f4s, t1_fst, t1_fsts = tier1_cross_uci(hhar_df, feature_cols,
                                                         X_uci_feat, y_uci_feat, seeds=SEEDS)
        results.append({"model": "T1 XGBoost (GenHAR+LLM4HAR)", "align": "n/a", "aug": "n/a",
                        "scope": "n/a", "loso_acc": t1_loso_m, "loso_std": t1_loso_s,
                        "uci_4f1": t1_f4, "uci_4f1_std": t1_f4s,
                        "uci_stairs_f1": t1_fst, "uci_stairs_std": t1_fsts})
        pd.DataFrame(results).to_csv(h2h_fp, index=False)

    in_raw, in_enc = {}, {}       # in_enc keyed by align (HHAR-only pretrain)
    xd_raw, xd_enc = {}, {}       # xd_enc keyed by (align, scope)
    policy_cache = {}             # align -> learned AutoAugHAR policy (searched once)
    loso_cache = {}               # (align, eff_aug) -> (loso_m, loso_s)  [scope-independent]

    for spec in TIER2_RUNS:
        align, aug, scope = spec["align"], spec["aug"], spec["scope"]
        if spec["name"] in done:
            print(f"[resume] skip completed row: {spec['name']}"); continue

        # -- in-domain windows + HHAR-only encoder
        if align not in in_raw:
            print(f"\n[Tier-2] in-domain HHAR windows (align={align}, L={SEQ_LEN}) ...")
            in_raw[align] = window_hhar_raw(acc_raw, gyro_raw, align,
                                            window_sec=WINDOW_SEC, seq_len=SEQ_LEN)
            print(f"    HHAR raw: {len(in_raw[align][0])}")
        Xh, yh, gh = in_raw[align]
        if align not in in_enc:
            print(f"[Tier-2] pretraining in-domain encoder (align={align}, HHAR-only, "
                  f"pool={len(Xh)}) ...")
            in_enc[align] = cached_pretrain(f'in_{align}', Xh)

        # -- cross-domain windows
        if align not in xd_raw:
            print(f"[Tier-2] cross-domain windows (align={align}, L={XDOMAIN_SEQ_LEN}) ...")
            Xh2, yh2, _ = window_hhar_raw(acc_raw, gyro_raw, align,
                                          window_sec=XDOMAIN_WINDOW_SEC, seq_len=XDOMAIN_SEQ_LEN)
            Xu, yu = build_uci_raw(align, seq_len=XDOMAIN_SEQ_LEN)
            xd_raw[align] = (Xh2, yh2, Xu, yu)
            print(f"    HHAR raw: {len(Xh2)}  |  UCI raw: {len(Xu)}")
        Xh2, yh2, Xu, yu = xd_raw[align]
        key = (align, scope)
        if key not in xd_enc:
            pool = Xh2 if scope == "source" else np.concatenate([Xh2, Xu], axis=0)
            print(f"[Tier-2] pretraining cross-domain encoder (align={align}, scope={scope}, "
                  f"pool={len(pool)}) ...")
            xd_enc[key] = cached_pretrain(f'xd_{align}_{scope}', pool)

        # -- learn AutoAugHAR policy per (align, scope)
        if aug == "search" and key not in policy_cache:
            print(f"[Tier-2] learning AutoAugHAR policy (align={align}, scope={scope}) "
                  f"-> applied fixed across folds ...")
            cls_x = sorted(np.unique(yh2)); lm = {c: i for i, c in enumerate(cls_x)}
            policy_cache[key] = cached_policy(f'xd_{align}_{scope}', xd_enc[key], Xh2,
                                              np.array([lm[v] for v in yh2]), len(cls_x))
        eff_aug = "fixed" if aug == "search" else aug
        eff_pol = policy_cache.get(key) if aug == "search" else None

        # -- evaluate this row
        print(f"[Tier-2] evaluating: {spec['name']} ...")
        lkey = (align, eff_aug)
        if lkey not in loso_cache:
            loso_cache[lkey] = tier2_loso(in_enc[align], Xh, yh, gh,
                                          aug_mode=eff_aug, fixed_policy=eff_pol)
        loso_m, loso_s = loso_cache[lkey]
        f4, f4s, fst, fsts = tier2_cross_uci(xd_enc[key], Xh2, yh2, Xu, yu,
                                             aug_mode=eff_aug, fixed_policy=eff_pol,
                                             seeds=SEEDS, verbose=True)
        results.append({"model": spec["name"], "align": str(align), "aug": aug,
                        "scope": scope, "loso_acc": loso_m, "loso_std": loso_s,
                        "uci_4f1": f4, "uci_4f1_std": f4s,
                        "uci_stairs_f1": fst, "uci_stairs_std": fsts})
        try:
            pd.DataFrame(results).to_csv(os.path.join(SAVE_PATH, "head_to_head_partial.csv"),
                                         index=False)
            print(f"    [saved] {len(results)} rows -> head_to_head_partial.csv")
        except Exception as e:
            print(f"    [warn] could not save partial results: {e}")

    # ---- summary table -------------------------------------------------------
    df = pd.DataFrame(results)
    disp = df.copy()
    disp["in-domain LOSO acc"] = disp.apply(lambda r: f"{r.loso_acc:.3f} ± {r.loso_std:.3f}", axis=1)
    disp["cross-UCI 4-class F1"] = disp.apply(lambda r: f"{r.uci_4f1:.3f} ± {r.uci_4f1_std:.3f}", axis=1)
    disp["cross-UCI stairs F1"] = disp.apply(lambda r: f"{r.uci_stairs_f1:.3f} ± {r.uci_stairs_std:.3f}", axis=1)
    disp = disp[["model", "align", "aug", "scope", "in-domain LOSO acc",
                 "cross-UCI 4-class F1", "cross-UCI stairs F1"]]
    print("\n" + "=" * 92)
    print("HEAD-TO-HEAD SUMMARY  (rate-consistent: in-domain @5s/20Hz, cross-domain @2.56s/20Hz)")
    print("=" * 92)
    print(disp.to_string(index=False))
    print("=" * 92)
    print("Reading:")
    print("  * grav OFF vs ON     -> value of gravity canonicalization for the deep model")
    print("  * +AutoAugHAR vs off -> augmentation lift on top of the better alignment")
    print("  * pooled vs source   -> cross-UCI gap = value of unlabeled TARGET signal")
    print("  NOTE: LOSO uses an HHAR-only encoder, so it is identical within a source/")
    print("        pooled pair by construction -- scope moves the CROSS-UCI column only.")
    return df


# =============================================================================
# ENTRY POINT
# =============================================================================
def _device_banner():
    if DEVICE.type == "cuda":
        print(f"[device] CUDA active: {torch.cuda.get_device_name(0)}  (Tier-2 runs on GPU)")
    elif DEVICE.type == "mps":
        print("[device] Apple MPS active (Tier-2 runs on the Mac GPU).")
        print("         If the one-time AutoAugHAR policy search hits a missing MPS kernel,")
        print("         it automatically retries on CPU and then continues on MPS.")
    else:
        print("[device] WARNING: no GPU found -> Tier-2 runs on CPU (expect a long run).")
        print("         Colab: Runtime > Change runtime type > T4 GPU, then restart.")
    print(f"[env]    {'Colab' if IN_COLAB else 'local'} | DATA_ROOT={DATA_ROOT}")
    print(f"[resume] RESUME={RESUME} -> partial CSVs + enc_cache under {SAVE_PATH}")
    print("[note] Tier-1 (XGBoost + feature extraction) is CPU-bound by design;")
    print("       an idle GPU during the Tier-1 phase is expected, not a problem.")

RUN_ABLATION = False    # stage 1: the 6-row gravity/scope/aug ablation (HHAR<->UCI)
RUN_TRANSFER = False    # stage 2: quartet transfer matrix (full 12 directed pairs)
RUN_PROBE    = False    # stage 3: MotionSense walk-collapse diagnostic
RUN_POLICY_PROBE = False  # stage 4: search convergence -- volume vs diversity

def main():
    _device_banner()
    acc_raw, gyro_raw = load_hhar_phone()
    if acc_raw is None or gyro_raw is None:
        print("HHAR phone CSVs not found — check ACTIVITY_PATH."); return
    print(f"HHAR acc rows {len(acc_raw):,} | gyro rows {len(gyro_raw):,}")

    if RUN_ABLATION:
        print("Windowing HHAR (features) ...")
        hhar_df = pd.DataFrame(window_hhar(acc_raw, gyro_raw)).fillna(0)
        print(f"HHAR feature windows: {len(hhar_df):,}")
        phyphox_df = load_phyphox_features()
        combined_df = pd.concat([hhar_df, phyphox_df], ignore_index=True) if not phyphox_df.empty else hhar_df
        meta_cols = ['class_label', 'carrying_load', 'user', 'device', 'source', 'window_id']
        feature_cols = [c for c in combined_df.columns if c not in meta_cols] + ['carrying_load']
        run_head_to_head(acc_raw, gyro_raw, hhar_df, feature_cols)

    if RUN_TRANSFER:
        print("\n" + "#" * 84)
        print("# STAGE 2: QUARTET TRANSFER MATRIX")
        print("#" * 84)
        run_transfer_matrix()

    if RUN_PROBE:
        probe_motion()

    if RUN_POLICY_PROBE:
        probe_policy_datasize()


def _smoke_test():
    print("SMOKE TEST (synthetic)")
    set_seed()
    N, ncl = 300, 4
    Xh = np.random.randn(N, SEQ_LEN, N_CHANNELS).astype(np.float32)
    yh = np.random.randint(0, ncl, N)
    for c in range(ncl):
        Xh[yh == c, :, 0] += 0.4 * c
    yh = np.array([[0, 2, 3, 4][v] for v in yh])
    gh = np.array([f"s{ i%3 }" for i in range(N)])
    Xu = np.random.randn(90, SEQ_LEN, N_CHANNELS).astype(np.float32)
    yu = np.array([[0, 2, 3, 4][i % 4] for i in range(90)])
    enc = pretrain_limubert(Xh, epochs=2, bs=64)
    for aug in ["off", "random", "search"]:
        la, ls = tier2_loso(enc, Xh, yh, gh, aug_mode=aug)
        f4, f4s, fs, fss = tier2_cross_uci(enc, Xh, yh, Xu, yu, aug_mode=aug)
        print(f"  aug={aug:6s} loso={la:.3f}±{ls:.3f}  uci4F1={f4:.3f}±{f4s:.3f}  stairsF1={fs:.3f}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    if os.path.exists(ACTIVITY_PATH):
        main()
    else:
        _smoke_test()
