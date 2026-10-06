# =============================================================================
#  REVISION ANALYSIS -- every number in the paper comes from here.
#
#  Reads the long-format run logs written by revision_experiments.py
#  (SAVE_PATH/revision/runs/*.csv) and writes tables, figures and a JSON of the
#  headline numbers to SAVE_PATH/revision/tables/. Needs numpy/pandas/scipy only
#  (matplotlib for figures), so it also runs on a laptop with the CSVs copied over:
#      python revision/revision_analysis.py --rev-dir /path/to/revision
#  or in Colab after the experiments:  %run -i -n revision/revision_analysis.py
#                                      make_all_tables()
# =============================================================================
import os, json, math, argparse, itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

try:
    REV_DIR  # noqa: F821  (defined when run with %run -i after the experiments)
except NameError:
    REV_DIR = os.environ.get("REV_DIR", os.path.join(os.getcwd(), "revision_results"))

try:
    RUN_TAG  # noqa: F821  (set by revision_experiments.py: run-log version)
except NameError:
    RUN_TAG = os.environ.get("RUN_TAG", "v2")
UNTAGGED_LOGS = {"xgb"}          # no fine-tuning, so not re-run under the v2 check

ANA_DATASETS = ["hhar", "uci", "motion", "shoaib"]
DNAME = {"hhar": "HHAR", "uci": "UCI-HAR", "motion": "MotionSense", "shoaib": "Shoaib"}
ARM_LABEL = {
    "XGB_raw": "XGBoost (engineered features)", "XGB_canon": "XGBoost + gravity canon.",
    "A_ssl": "Baseline SSL (LIMU-BERT)", "B_grav": "+ Gravity canon.", "C_yaw": "+ Yaw inv. only",
    "D_gravyaw": "Gravity + yaw inv. (ours)", "D_pre": "Gravity + yaw consistency only",
    "D_ft": "Gravity + yaw augmentation only", "E_so3aug": "LIMU-BERT + SO(3) aug.",
    "F_so3inv": "Learned SO(3) invariance", "G_pca": "Gravity + PCA heading",
    "H_mizell": "Vertical/horizontal [Mizell]", "I_oit": "OIT [Yurtman & Barshan]",
    "UniMTS_raw": "UniMTS (fine-tuned)", "UniMTS_canon": "UniMTS + gravity canon.",
    "autoaug_raw": "AutoAugHAR", "autoaug_ours": "AutoAugHAR on our encoder",
}
MAIN_ORDER = ["XGB_raw", "A_ssl", "E_so3aug", "F_so3inv", "H_mizell", "I_oit", "G_pca",
              "UniMTS_raw", "UniMTS_canon", "XGB_canon", "D_gravyaw"]


def _p(*a):
    print(*a, flush=True)


def _tab_dir():
    d = os.path.join(REV_DIR, "tables"); os.makedirs(d, exist_ok=True); return d


def load_runs(*names):
    dfs = []
    for n in names:
        f = n if (n in UNTAGGED_LOGS or not RUN_TAG) else f"{n}_{RUN_TAG}"
        p = os.path.join(REV_DIR, "runs", f"{f}.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["pair"] = df["source"].astype(str) + "->" + df["target"].astype(str)
    return df


def cross_runs():
    """All cross-dataset runs of the main comparison in one frame."""
    return load_runs("main", "xgb", "unimts", "autoaug")


# =============================================================================
#  1. CELL STATISTICS
# =============================================================================
def cell_means(df, metric="macro_f1"):
    g = df.groupby(["pair", "arm"])[metric]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["ci95"] = [1.96 * s / math.sqrt(n) if n > 1 else np.nan for s, n in zip(out["std"], out["count"])]
    return out


def pivot(df, metric="macro_f1", arms=None):
    m = cell_means(df, metric).pivot(index="pair", columns="arm", values="mean")
    if arms:
        m = m[[a for a in arms if a in m.columns]]
    return m


def _runs_by_seed(df, pair, arm, metric):
    """{pre_seed: np.array(run values)} -- the two-level structure of a cell."""
    s = df[(df.pair == pair) & (df.arm == arm)]
    return {k: g[metric].dropna().values for k, g in s.groupby("pre_seed")}


def _resample_cell(cell, rng):
    keys = list(cell)
    if not keys:
        return np.nan
    pick = rng.choice(len(keys), len(keys), replace=True)
    vals = []
    for i in pick:
        v = cell[keys[i]]
        vals.append(v[rng.integers(0, len(v), len(v))].mean())
    return float(np.mean(vals))


def cell_difference_ci(df, pair, arm_hi, arm_lo, metric="macro_f1", n_boot=2000, seed=0):
    """Two-level bootstrap CI of (arm_hi - arm_lo) inside ONE pair (c53)."""
    a, b = _runs_by_seed(df, pair, arm_hi, metric), _runs_by_seed(df, pair, arm_lo, metric)
    if not a or not b:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    d = np.array([_resample_cell(a, rng) - _resample_cell(b, rng) for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# =============================================================================
#  2. TESTS ACROSS PAIRS  (c26 c49 c53)
# =============================================================================
def holm(p):
    p = np.asarray(p, float); k = len(p); order = np.argsort(p)
    adj = np.empty(k); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (k - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def hier_bootstrap(df, arm_hi, arm_lo, metric="macro_f1", n_boot=2000, seed=0):
    """c49. The 12 pairs share datasets (each appears in 6) and each cell has a
    pretraining x fine-tuning seed structure. Level 1 resamples the 4 ANA_DATASETS with
    replacement and keeps every directed pair between distinct draws (with
    multiplicity); level 2 resamples pretraining seeds, then fine-tuning seeds,
    inside each cell. Returns the mean difference, its 95% CI and a two-sided
    bootstrap p. With only four datasets this is deliberately conservative."""
    pairs = sorted(set(df[df.arm == arm_hi].pair) & set(df[df.arm == arm_lo].pair))
    cells = {p: (_runs_by_seed(df, p, arm_hi, metric), _runs_by_seed(df, p, arm_lo, metric))
             for p in pairs}
    rng = np.random.default_rng(seed)
    stats, skipped = [], 0
    while len(stats) < n_boot:
        draw = rng.choice(ANA_DATASETS, len(ANA_DATASETS), replace=True)
        sel = [f"{s}->{t}" for s, t in itertools.product(draw, draw) if s != t]
        sel = [p for p in sel if p in cells]
        if not sel:
            skipped += 1
            if skipped > 10 * n_boot:
                break
            continue
        stats.append(np.mean([_resample_cell(cells[p][0], rng) - _resample_cell(cells[p][1], rng)
                              for p in sel]))
    stats = np.array(stats)
    obs = np.mean([np.mean(np.concatenate(list(cells[p][0].values())))
                   - np.mean(np.concatenate(list(cells[p][1].values()))) for p in pairs]) if pairs else np.nan
    if len(stats) < 100:                     # too few pairs for a dataset-level bootstrap
        return {"mean_delta": float(obs), "ci_lo": np.nan, "ci_hi": np.nan, "p_boot": np.nan,
                "n_pairs": len(pairs), "degenerate_draws_skipped": skipped}
    p2 = 2 * min((stats <= 0).mean(), (stats >= 0).mean())
    return {"mean_delta": float(obs), "ci_lo": float(np.percentile(stats, 2.5)),
            "ci_hi": float(np.percentile(stats, 97.5)), "p_boot": float(min(1.0, p2)),
            "n_pairs": len(pairs), "degenerate_draws_skipped": skipped}


def lodo(df, arm_hi, arm_lo, metric="macro_f1"):
    """Leave-one-dataset-out: the mean difference recomputed without every pair
    that involves one dataset (6 pairs remain). Shows whether one dataset drives it."""
    P = pivot(df, metric, [arm_hi, arm_lo]).dropna()
    out = {}
    for d in ANA_DATASETS:
        keep = [p for p in P.index if d not in p.split("->")]
        dd = (P.loc[keep, arm_hi] - P.loc[keep, arm_lo])
        out[f"without_{d}"] = float(dd.mean()) if len(dd) else np.nan
    return out


def paired_report(df, contrasts, metric="macro_f1", family="main"):
    """Wilcoxon signed-rank over pairs (cell means), Holm across the declared family,
    plus the hierarchical bootstrap and LODO for every contrast."""
    P = pivot(df, metric)
    rows = []
    for hi, lo in contrasts:
        if hi not in P or lo not in P:
            continue
        j = P[[hi, lo]].dropna()
        d = j[hi] - j[lo]
        try:
            pw = wilcoxon(j[hi], j[lo])[1]
        except ValueError:
            pw = np.nan
        hb = hier_bootstrap(df, hi, lo, metric)
        rows.append({"family": family, "metric": metric, "contrast": f"{hi} - {lo}",
                     "n_pairs": len(j), "mean_hi": j[hi].mean(), "mean_lo": j[lo].mean(),
                     "mean_delta": d.mean(), "median_delta": d.median(),
                     "wins": f"{int((d > 0).sum())}/{len(d)}", "wilcoxon_p": pw,
                     "boot_ci": f"[{hb['ci_lo']:+.3f}, {hb['ci_hi']:+.3f}]", "boot_p": hb["p_boot"],
                     **{k: round(v, 3) for k, v in lodo(df, hi, lo, metric).items()}})
    out = pd.DataFrame(rows)
    if len(out):
        out["wilcoxon_p_holm"] = holm(out["wilcoxon_p"].fillna(1.0).values)
    return out


# =============================================================================
#  3. TABLES
# =============================================================================
def _fmt(v, nd=3):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}".lstrip("0").replace("-0.", "-.")


def main_table(df=None, arms=None, metric_pair=("macro_f1", "stairs_f1")):
    """Table 2 replacement (c46 c52 c53 c54): every method, every pair, "macro / stairs"
    from the same predictions, averages over the SAME pairs for every method. Bold = best cell
    mean; '~' marks methods whose two-level bootstrap CI of the difference to the
    best includes 0 (statistically indistinguishable from the best in that pair)."""
    df = cross_runs() if df is None else df
    if df.empty:
        _p("no cross-dataset runs yet"); return None
    arms = [a for a in (arms or MAIN_ORDER) if a in set(df.arm)]
    M = pivot(df, metric_pair[0], arms); S = pivot(df, metric_pair[1], arms)
    rows = []
    for pair in M.index:
        best = M.loc[pair].idxmax()
        r = {"pair": pair}
        for a in arms:
            m, s = M.loc[pair, a], S.loc[pair, a]
            tag = ""
            if a == best:
                tag = "**"
            elif not np.isnan(m):
                _, lo, hi = cell_difference_ci(df, pair, best, a)
                tag = "~" if (not np.isnan(lo) and lo <= 0 <= hi) else ""
            r[a] = f"{tag}{_fmt(m)}{tag if tag == '**' else ''} / {_fmt(s)}"
        rows.append(r)
    complete = M.dropna()
    avg = {"pair": f"mean over the {len(complete)} pairs every method has"}
    own = {"pair": "mean over the pairs each method has (n)"}
    for a in arms:
        avg[a] = f"{_fmt(complete[a].mean())} / {_fmt(S.loc[complete.index, a].mean())}"
        have = M[a].dropna()
        own[a] = f"{_fmt(have.mean())} / {_fmt(S.loc[have.index, a].mean())} (n={len(have)})"
    rows += [avg, own]
    out = pd.DataFrame(rows).rename(columns={a: ARM_LABEL.get(a, a) for a in arms})
    out.to_csv(os.path.join(_tab_dir(), "T_main.csv"), index=False)
    return out


def ablation_table(df=None):
    """Table 4 replacement (c41 c57 c62 c63): the 2x2 plus the yaw-site arms. The
    interaction column has a run-level bootstrap CI; the draft's rho=-0.937 analysis
    is removed (c58 c59 c68)."""
    df = load_runs("main") if df is None else df
    arms = ["A_ssl", "B_grav", "C_yaw", "D_gravyaw", "D_pre", "D_ft"]
    P = pivot(df, "macro_f1", arms)
    if not set(arms[:4]) <= set(P.columns):
        _p("ablation incomplete"); return None
    out = P.copy()
    out["dC (B-A)"] = P.B_grav - P.A_ssl
    out["dY (C-A)"] = P.C_yaw - P.A_ssl
    out["dBoth (D-A)"] = P.D_gravyaw - P.A_ssl
    out["yaw on top of canon (D-B)"] = P.D_gravyaw - P.B_grav
    inter = P.D_gravyaw - P.B_grav - P.C_yaw + P.A_ssl
    cis = []
    rng = np.random.default_rng(0)
    for pair in P.index:
        cells = [_runs_by_seed(df, pair, a, "macro_f1") for a in arms[:4]]
        bs = [(_resample_cell(cells[3], rng) - _resample_cell(cells[1], rng)
               - _resample_cell(cells[2], rng) + _resample_cell(cells[0], rng)) for _ in range(2000)]
        cis.append(f"{inter[pair]:+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    out["interaction [95% CI]"] = cis
    out.loc["mean"] = list(P.mean().reindex(P.columns)) + [out[c].mean() for c in
                                                           ["dC (B-A)", "dY (C-A)", "dBoth (D-A)",
                                                            "yaw on top of canon (D-B)"]] + [f"{inter.mean():+.3f}"]
    out.round(3).to_csv(os.path.join(_tab_dir(), "T_ablation.csv"))
    return out


def family_frame_table(df=None):
    """Table 5 replacement (c62 c63 c64): model family x input frame, with GraviHAR
    as its own row so both families are compared under the same treatment."""
    df = cross_runs() if df is None else df
    rows = []
    for fam, raw, canon in (("XGBoost (engineered features)", "XGB_raw", "XGB_canon"),
                            ("LIMU-BERT (sequence encoder)", "A_ssl", "B_grav")):
        for metric in ("macro_f1", "stairs_f1"):
            P = pivot(df, metric, [raw, canon]).dropna()
            if len(P.columns) < 2:
                continue
            rows.append({"family": fam, "metric": metric, "device frame": P[raw].mean(),
                         "gravity-aligned frame": P[canon].mean(), "gain": (P[canon] - P[raw]).mean(),
                         "n_pairs": len(P)})
    for metric in ("macro_f1", "stairs_f1"):
        P = pivot(df, metric, ["A_ssl", "D_gravyaw"]).dropna()
        if len(P.columns) == 2:
            rows.append({"family": "GraviHAR (gravity + yaw invariance)", "metric": metric,
                         "device frame": P["A_ssl"].mean(), "gravity-aligned frame": P["D_gravyaw"].mean(),
                         "gain": (P["D_gravyaw"] - P["A_ssl"]).mean(), "n_pairs": len(P)})
    out = pd.DataFrame(rows)
    out.round(3).to_csv(os.path.join(_tab_dir(), "T_family_frame.csv"), index=False)
    return out


def gap_table(df=None):
    """c29 c64 c74. For each pair: the in-domain ceiling of the target on the pair's
    own label set (same method, subject-disjoint CV), the cross-dataset score, the
    gap, and the share of the baseline's gap that orientation treatment closes:
        share = (cross_D - cross_A) / (ceil_D - cross_A)     (LIMU-BERT family)
        share = (XGB_canon - XGB_raw) / (ceil_XGBcanon - XGB_raw)  (feature family)"""
    df = cross_runs() if df is None else df
    ce = load_runs("ceiling")
    if ce.empty or df.empty:
        _p("ceiling runs missing"); return None
    ce["tgt"] = ce["target"].str.split("[").str[0]
    ce["labset"] = ce["labels"].astype(str)
    C = ce.groupby(["tgt", "labset", "arm"])["macro_f1"].mean()
    X = pivot(df, "macro_f1")
    lab = df.groupby("pair")["labels"].first().astype(str)
    rows = []
    for pair in X.index:
        s, t = pair.split("->")
        r = {"pair": pair}
        for arm in ("A_ssl", "B_grav", "D_gravyaw", "XGB_raw", "XGB_canon"):
            r[f"cross_{arm}"] = X.loc[pair].get(arm, np.nan)
            r[f"ceil_{arm}"] = C.get((t, lab.get(pair, ""), arm), np.nan)
        if not np.isnan(r["ceil_D_gravyaw"]):
            r["gap_A"] = r["ceil_D_gravyaw"] - r["cross_A_ssl"]
            r["gap_D"] = r["ceil_D_gravyaw"] - r["cross_D_gravyaw"]
            r["share_closed_ssl"] = (r["cross_D_gravyaw"] - r["cross_A_ssl"]) / max(r["gap_A"], 1e-9)
            r["share_closed_by_canon_only"] = (r["cross_B_grav"] - r["cross_A_ssl"]) / max(r["gap_A"], 1e-9)
        if not np.isnan(r.get("ceil_XGB_canon", np.nan)):
            g = r["ceil_XGB_canon"] - r["cross_XGB_raw"]
            r["share_closed_xgb"] = (r["cross_XGB_canon"] - r["cross_XGB_raw"]) / max(g, 1e-9)
        rows.append(r)
    out = pd.DataFrame(rows)
    num = out.select_dtypes("number")
    out = pd.concat([out, pd.DataFrame([{"pair": "mean", **num.mean().to_dict()}])], ignore_index=True)
    out.round(3).to_csv(os.path.join(_tab_dir(), "T_gap.csv"), index=False)
    return out


def rotation_tables(make_figs=True):
    """RQ3. Curves, ORS = mean_{theta>0} F1(theta)/F1(0), worst-case drop, per
    dataset, arm and perturbation family; and accuracy as a function of the
    residual heading |psi| pooled over every perturbation (canonicalized arms)."""
    df = load_runs("rotation")
    if df.empty:
        _p("no rotation runs"); return None
    cur = df.groupby(["source", "arm", "perturb", "angle"])["macro_f1"].mean().reset_index()
    rows = []
    for (d, a, pt), g in cur.groupby(["source", "arm", "perturb"]):
        g = g.set_index("angle")["macro_f1"]
        if 0.0 not in g.index:
            continue
        f0, rest = g.loc[0.0], g.drop(0.0)
        rows.append({"dataset": d, "arm": a, "perturb": pt, "F1(0)": f0,
                     "ORS": rest.mean() / max(f0, 1e-9), "worst_drop": f0 - rest.min()})
    ors = pd.DataFrame(rows)
    ors.round(3).to_csv(os.path.join(_tab_dir(), "T_rotation_ORS.csv"), index=False)
    piv = ors.pivot_table(index=["dataset", "arm"], columns="perturb", values="ORS")
    piv.round(3).to_csv(os.path.join(_tab_dir(), "T_rotation_ORS_wide.csv"))
    # accuracy vs |psi|
    acc = {}
    for _, r in df.iterrows():
        try:
            bins = json.loads(r["extra"]).get("psi_bins", {})
        except Exception:
            continue
        for b, (c, n) in bins.items():
            k = (r["source"], r["arm"], float(b))
            cc, nn = acc.get(k, (0, 0)); acc[k] = (cc + c, nn + n)
    psi = pd.DataFrame([{"dataset": k[0], "arm": k[1], "psi_bin_lo": k[2], "acc": v[0] / max(v[1], 1),
                         "n": v[1]} for k, v in acc.items()])
    psi.round(4).to_csv(os.path.join(_tab_dir(), "T_rotation_psi.csv"), index=False)
    if make_figs:
        _rotation_figs(cur, psi)
    return ors, piv, psi


def _rotation_figs(cur, psi):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        return
    arms = [a for a in ["A_ssl", "B_grav", "C_yaw", "D_gravyaw", "G_pca", "H_mizell", "E_so3aug"]
            if a in set(cur.arm)]
    perts = [p for p in ["device_x", "device_y", "device_z", "phys_tilt", "phys_heading"]
             if p in set(cur.perturb)]
    ds = [d for d in ANA_DATASETS if d in set(cur.source)]
    fig, ax = plt.subplots(len(ds), len(perts), figsize=(3.0 * len(perts), 2.4 * len(ds)),
                           squeeze=False, sharey=True)
    for i, d in enumerate(ds):
        for j, pt in enumerate(perts):
            for a in arms:
                g = cur[(cur.source == d) & (cur.arm == a) & (cur.perturb == pt)].sort_values("angle")
                if len(g):
                    ax[i, j].plot(g.angle, g.macro_f1, marker="o", ms=3, label=ARM_LABEL.get(a, a))
            ax[i, j].set_title(f"{DNAME[d]} / {pt}", fontsize=8); ax[i, j].set_ylim(0, 1)
    ax[0, -1].legend(fontsize=6, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(_tab_dir(), "F_rotation_curves.pdf")); plt.close(fig)
    if len(psi):
        fig, ax = plt.subplots(1, len(ds), figsize=(3.2 * len(ds), 2.6), squeeze=False, sharey=True)
        for j, d in enumerate(ds):
            for a in [x for x in ["B_grav", "D_gravyaw", "G_pca"] if x in set(psi.arm)]:
                g = psi[(psi.dataset == d) & (psi.arm == a)].sort_values("psi_bin_lo")
                ax[0, j].plot(g.psi_bin_lo, g.acc, marker="o", label=ARM_LABEL.get(a, a))
            ax[0, j].set_title(DNAME[d], fontsize=9); ax[0, j].set_xlabel("|residual heading| (deg)")
        ax[0, 0].set_ylabel("accuracy"); ax[0, -1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(_tab_dir(), "F_rotation_psi.pdf")); plt.close(fig)


def posture_table():
    """c44: per-class F1 of the posture classes by arm, within and across datasets."""
    df = load_runs("posture")
    if df.empty:
        _p("no posture runs"); return None
    df["setting"] = np.where(df.source == df.target, "within", "cross")
    rows = []
    for (st, pair, arm), g in df.groupby(["setting", "pair", "arm"]):
        pcs = pd.DataFrame([json.loads(x) for x in g.per_class]).mean()
        rows.append({"setting": st, "pair": pair, "arm": arm, "macro_f1": g.macro_f1.mean(),
                     **{f"F1_{n}": pcs.get(str(c), np.nan) for c, n in ((8, "sit"), (9, "stand"), (10, "lie"))}})
    out = pd.DataFrame(rows)
    out.round(3).to_csv(os.path.join(_tab_dir(), "T_posture.csv"), index=False)
    return out


def hparam_tables():
    """c40: which (lambda, floor) the SOURCE-ONLY criterion picks per source, the full
    grid on the targets, and the selected vs default configuration."""
    df = load_runs("hparam")
    if df.empty:
        _p("no hparam runs"); return None
    val = df[df.exp == "hparam_val"].groupby(["source", "arm"])["macro_f1"].mean().reset_index()
    sel = val.loc[val.groupby("source")["macro_f1"].idxmax()].rename(columns={"arm": "selected",
                                                                             "macro_f1": "rotated_val_f1"})
    tgt = df[df.exp == "hparam"].groupby(["pair", "arm"])["macro_f1"].mean().unstack()
    tgt["selected"] = [sel.set_index("source").loc[p.split("->")[0], "selected"]
                       if p.split("->")[0] in set(sel.source) else None for p in tgt.index]
    tgt["F1_selected"] = [tgt.loc[p, s] if s in tgt.columns else np.nan for p, s in zip(tgt.index, tgt.selected)]
    default = "D_l0.1_f0.3"
    if default in tgt.columns:
        tgt["F1_default"] = tgt[default]
    grid_range = tgt.drop(columns=[c for c in ["selected", "F1_selected", "F1_default"] if c in tgt]).agg(
        ["min", "max"], axis=1)
    tgt["grid_min"], tgt["grid_max"] = grid_range["min"], grid_range["max"]
    sel.round(4).to_csv(os.path.join(_tab_dir(), "T_hparam_selection.csv"), index=False)
    tgt.round(3).to_csv(os.path.join(_tab_dir(), "T_hparam_targets.csv"))
    return sel, tgt


def simple_table(log, index="pair", col="arm", metric="macro_f1", name=None):
    df = load_runs(log)
    if df.empty:
        return None
    t = df.groupby([index, col])[metric].mean().unstack()
    t.loc["mean"] = t.mean()
    t.round(3).to_csv(os.path.join(_tab_dir(), f"T_{name or log}.csv"))
    return t


# =============================================================================
#  4. HEADLINE NUMBERS (abstract / contributions / conclusion use ONLY these)
# =============================================================================
def headline_numbers(df=None):
    df = cross_runs() if df is None else df
    out = {}
    for metric in ("macro_f1", "stairs_f1"):
        P = pivot(df, metric).dropna(axis=1, how="all")
        full = P.dropna(axis=1)
        out[metric] = {a: round(float(full[a].mean()), 3) for a in full.columns}
        out[metric]["n_pairs"] = int(len(full))
        if "D_gravyaw" in full:
            base = [a for a in full.columns if a not in ("D_gravyaw", "D_pre", "D_ft", "B_grav", "C_yaw")]
            if base:
                best = max(base, key=lambda a: full[a].mean())
                out[metric]["best_baseline"] = best
                out[metric]["ours_minus_best_baseline_points"] = round(
                    float((full["D_gravyaw"] - full[best]).mean()), 3)
    json.dump(out, open(os.path.join(_tab_dir(), "headline_numbers.json"), "w"), indent=2)
    return out


# =============================================================================
#  5. ONE CALL FOR EVERYTHING
# =============================================================================
MAIN_CONTRASTS = [("B_grav", "A_ssl"), ("C_yaw", "A_ssl"), ("D_gravyaw", "A_ssl"),
                  ("D_gravyaw", "B_grav"), ("D_gravyaw", "C_yaw")]
SITE_CONTRASTS = [("D_gravyaw", "D_ft"), ("D_gravyaw", "D_pre"), ("D_ft", "B_grav")]
BASELINE_CONTRASTS = [("D_gravyaw", "XGB_canon"), ("D_gravyaw", "XGB_raw"), ("D_gravyaw", "E_so3aug"),
                      ("D_gravyaw", "F_so3inv"), ("D_gravyaw", "G_pca"), ("D_gravyaw", "H_mizell"),
                      ("D_gravyaw", "I_oit"), ("D_gravyaw", "UniMTS_raw"), ("UniMTS_canon", "UniMTS_raw"),
                      ("XGB_canon", "XGB_raw")]


def make_all_tables():
    df = cross_runs()
    md = ["# Revision tables (generated by revision_analysis.py -- do not edit by hand)\n"]

    def add(title, t):
        if t is None or (hasattr(t, "empty") and t.empty):
            md.append(f"\n## {title}\n\n_not available yet_\n"); return
        try:
            body = t.to_markdown(floatfmt=".3f")
        except ImportError:                      # tabulate not installed
            body = "```\n" + t.to_string() + "\n```"
        md.append(f"\n## {title}\n\n" + body + "\n")

    if not df.empty:
        add("Main cross-dataset results (macro-F1 / stairs-F1)", main_table(df))
        add("Component ablation", ablation_table(load_runs("main")))
        add("Model family x input frame", family_frame_table(df))
        tests = []
        for fam, con in (("ablation", MAIN_CONTRASTS), ("yaw site", SITE_CONTRASTS),
                         ("baselines", BASELINE_CONTRASTS)):
            for metric in ("macro_f1", "stairs_f1"):
                tests.append(paired_report(df, con, metric, fam))
        tests = pd.concat([t for t in tests if len(t)], ignore_index=True) if any(len(t) for t in tests) else None
        if tests is not None:
            tests.round(4).to_csv(os.path.join(_tab_dir(), "T_tests.csv"), index=False)
        add("Paired tests over pairs (Holm within family), hierarchical bootstrap, LODO", tests)
        add("Headline numbers", pd.DataFrame(headline_numbers(df)))
    add("In-domain ceiling and orientation share of the gap", gap_table(df) if not df.empty else None)
    r = rotation_tables()
    add("Controlled rotations: ORS", None if r is None else r[1])
    add("Posture classes (c44)", posture_table())
    h = hparam_tables()
    add("Hyperparameters: source-only selection (c40)", None if h is None else h[0])
    add("Hyperparameters: target sensitivity grid (c40)", None if h is None else h[1])
    add("Augmentation specificity", simple_table("augspec"))
    add("Shoaib body-position transfer", simple_table("positions"))
    add("Label efficiency", simple_table("labeleff", index="pair", col="angle", name="labeleff"))
    path = os.path.join(_tab_dir(), "paper_tables.md")
    open(path, "w").write("\n".join(md))
    _p(f"[analysis] wrote {path}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev-dir", default=REV_DIR)
    a = ap.parse_args()
    REV_DIR = a.rev_dir
    make_all_tables()
