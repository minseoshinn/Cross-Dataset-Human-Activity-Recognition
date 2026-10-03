# =============================================================================
#  smoke_test.py -- runs EVERY revision stage end to end on small synthetic data
#  (CPU, ~10 min). It checks that the code paths execute and that the geometry
#  identities hold; the F1 values it prints mean nothing.
#      python revision/smoke_test.py               # the three modules
#      python revision/smoke_test.py --notebook    # the single Colab notebook
# =============================================================================
import os, sys, math, json, types, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp(prefix="rev_smoke_")
os.environ["HAR_DATA_ROOT"] = TMP
os.environ["REV_DIR"] = os.path.join(TMP, "results", "revision")

# --notebook: take the code from the tagged cells of GraviHAR_Revision_Colab.ipynb and run
# all three in ONE namespace with __name__ == "__main__", exactly as Colab does.
NOTEBOOK = "--notebook" in sys.argv


def _code(module, tag):
    if not NOTEBOOK:
        return open(os.path.join(HERE, module)).read()
    nb = json.load(open(os.path.join(HERE, "GraviHAR_Revision_Colab.ipynb")))
    cells = [c for c in nb["cells"] if tag in c.get("metadata", {}).get("tags", [])]
    assert len(cells) == 1, (tag, len(cells))
    return "".join(cells[0]["source"])


NS = {"__name__": "__main__" if NOTEBOOK else "pipeline_base"}
exec(compile(_code("pipeline_base.py", "pipeline"), "pipeline", "exec"), NS)
NS["PRETRAIN_EPOCHS"], NS["FINETUNE_EPOCHS"] = 2, 1
if not NOTEBOOK:
    NS["__name__"] = "revision_experiments"
exec(compile(_code("revision_experiments.py", "experiments"), "experiments", "exec"), NS)
R = NS["REV"]
R.update(pre_epochs=2, pre_min_epochs=0, ft_epochs=1, ft_bs=64, pre_bs=64,
         budget={"pre": (42,), "ft": (42,)}, budget_baselines={"pre": (42,), "ft": (42,)})


# ----------------------------------------------------------------------------- synthetic IMU
def _rot(axis, deg):
    return NS["axis_angle_np"](axis, deg)[0]


def synth(name, n_subj=6, per=40, seed=0):
    rng = np.random.default_rng(abs(hash(name)) % 2 ** 31 + seed)
    classes = {"hhar": [0, 2, 3, 4, 6], "uci": [0, 2, 3, 4], "motion": [0, 2, 3, 4, 5],
               "shoaib": [0, 2, 3, 4, 5, 6]}[name]
    pose = {"hhar": (0, 0, 1), "uci": (1, 0, 0), "motion": (0, 1, 0), "shoaib": (0, -1, 0.2)}[name]
    t = np.arange(51) / 20.0
    X, ym, yp, grp, meta = [], [], [], [], {"position": [], "model": [], "native_hz": [],
                                           "split": [], "trial": [], "device": []}
    for s in range(n_subj):
        for c in classes:
            for k in range(per // len(classes) + 1):
                post = None
                if c == 0:
                    post = rng.choice([8, 9, 10] if name == "uci" else [8, 9])
                f = {0: 0.0, 2: 1.6, 3: 2.1, 4: 1.9, 5: 2.8, 6: 1.2}[c]
                A = {0: 0.0, 2: 0.25, 3: 0.35, 4: 0.3, 5: 0.8, 6: 0.15}[c]
                ph = rng.uniform(0, 2 * np.pi)
                body = np.zeros((51, 6))
                body[:, 2] = 1.0 + A * np.sin(2 * np.pi * f * t + ph)
                body[:, 0] = 0.6 * A * np.sin(2 * np.pi * f * t + ph + 0.8 + 0.3 * c)
                body[:, 4] = 2.0 * A * np.cos(2 * np.pi * f * t + ph)
                body += 0.03 * rng.normal(size=body.shape)
                if post == 8:
                    body = NS["rotate_windows"](body[None], _rot((0, 1, 0), 80))[0]
                if post == 10:
                    body = NS["rotate_windows"](body[None], _rot((1, 0, 0), 90))[0]
                # device pose: dataset tilt towards `pose`, subject-specific heading
                g_dev = np.array(pose, float); g_dev /= np.linalg.norm(g_dev)
                R_up = NS["rot_to_vertical_batch"](g_dev[None])[0]          # device -> canonical
                R = R_up.T @ _rot((0, 0, 1), rng.uniform(0, 360) if name != "hhar" else 10 * s)
                w = NS["rotate_windows"](body[None], R)[0]
                if name == "motion":                                         # stored in iOS sign
                    w[:, 0:3] *= -1.0
                    if R_["ios_sign_fix"]:
                        w[:, 0:3] *= -1.0
                X.append(w.astype(np.float32)); ym.append(c); grp.append(f"{name}_{s}")
                yp.append(c if c != 0 else int(post))
                meta["position"].append(k % 5); meta["model"].append(f"m{s % 2}")
                meta["native_hz"].append(100.0); meta["split"].append("train" if s < 4 else "test")
                meta["trial"].append(f"t{k}"); meta["device"].append(f"d{s}")
    X = np.asarray(X, np.float32)
    keep_meta = {"hhar": ["model", "native_hz", "device"], "uci": ["split"], "motion": ["trial"],
                 "shoaib": ["position"]}[name]
    return dict(X=X, y_merged=np.asarray(ym), y_posture=np.asarray(yp), groups=np.asarray(grp),
                meta={k: np.asarray(meta[k]) for k in keep_meta},
                info=dict(native="synthetic", units="g", convention="Android", placement="synthetic",
                          overlap="n/a"))


R_ = R
NS["REV_LOADERS"].update({d: (lambda d=d: synth(d)) for d in NS["DATASETS_REV"]})

t_start = time.time()
ok = []


def step(name, fn):
    t0 = time.time()
    fn()
    ok.append(name)
    print(f"\n>>> OK  {name}  ({time.time() - t0:.1f} s)\n", flush=True)


# ----------------------------------------------------------------------------- 1. geometry
def _geometry():
    out, tab = NS["geometry_identities"](n=500)
    assert out["P1_max_axis_tilt_deg"] < 1e-3, out
    for k, v in out.items():
        if k.startswith(("L1", "L2")):
            assert v < 1e-6, (k, v)
    # vectorized Rodrigues == pipeline implementation
    g = np.random.default_rng(1).normal(size=(200, 3)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    Rb = NS["rot_to_vertical_batch"](g)
    for i in range(200):
        assert np.allclose(Rb[i], NS["_rot_to_vertical"](g[i]), atol=1e-10)
    for gg in (np.array([[0, 0, 1.0]]), np.array([[0, 0, -1.0]])):
        assert np.allclose(NS["rot_to_vertical_batch"](gg)[0], NS["_rot_to_vertical"](gg[0]))
    # canonicalize() maps the mean acc onto +z for every window
    X = synth("uci")["X"]
    Xc = NS["canonicalize"](X)
    gc, _ = NS["gravity_dirs"](Xc)
    assert np.allclose(gc[:, 2], 1.0, atol=1e-5), gc[:5]
    # Mizell is invariant to any rotation; PCA heading invariant to heading
    Q = np.stack([NS["_haar_np"](np.random.default_rng(i)) for i in range(len(X))])
    assert np.allclose(NS["mizell_transform"](X), NS["mizell_transform"](NS["rotate_windows"](X, Q)), atol=1e-4)
    assert np.allclose(NS["oit_transform"](X), NS["oit_transform"](NS["rotate_windows"](X, Q)), atol=1e-3)
    gd, _ = NS["gravity_dirs"](X)
    Xh = NS["rotate_windows"](X, NS["axis_angle_np"](gd, 70.0))
    dyn = NS["get_ds"]("uci")[1] != 0
    assert np.allclose(NS["pca_heading"](NS["canonicalize"](X))[dyn],
                       NS["pca_heading"](NS["canonicalize"](Xh))[dyn], atol=1e-3)
step("geometry identities + transform invariances", _geometry)

step("audit (c33 c32 c37 c38 c42 c44 c45 c50 c51 c66)", lambda: NS["run_stage"]("audit"))


def _sign():
    df = NS["audit_motionsense_sign"]()
    a, b = df.iloc[0], df.iloc[1]
    assert abs(a["frac_gz<0 ('inverted')"] + b["frac_gz<0 ('inverted')"] - 1.0) < 0.05
step("c33 sign audit: inverted fractions are complementary", _sign)

PAIRS2 = [("hhar", "uci"), ("shoaib", "motion")]
step("xgb", lambda: NS["exp_xgb"](pairs=PAIRS2, seeds=(42,)))
step("main: all 11 arms", lambda: NS["exp_main"](pairs=PAIRS2))
step("main: resume skips finished runs", lambda: NS["exp_main"](pairs=PAIRS2))
step("ceiling", lambda: NS["exp_ceiling"](pairs=PAIRS2, ft_seeds=(42,)))
step("rotation", lambda: NS["exp_rotation"](datasets=["uci"], arms=["A_ssl", "B_grav", "D_gravyaw", "H_mizell"],
                                            perturbs=["device_x", "phys_tilt", "phys_heading"], ft_seeds=(42,)))
step("rotation strict pretrain", lambda: NS["exp_rotation"](datasets=["hhar"], arms=["D_gravyaw"],
                                                            perturbs=["phys_heading"], ft_seeds=(42,),
                                                            strict_pretrain=True))
step("posture", lambda: NS["exp_posture"](datasets=["uci"], pairs=(("uci", "motion"),), ft_seeds=(42,)))
step("hparam", lambda: NS["exp_hparam"](lams=(0.1, 0.3), floors=(0.0, 0.3), sources=["hhar"],
                                        pairs=[("hhar", "uci")], ft_seeds=(42,)))
step("augspec", lambda: NS["exp_aug_specificity"](pairs=[("hhar", "uci")], pre_seeds=(42,), ft_seeds=(42,)))
step("positions", lambda: NS["exp_positions"](arms=("A_ssl", "D_gravyaw"), ft_seeds=(42,)))
step("label efficiency", lambda: NS["exp_label_efficiency"](pairs=(("hhar", "uci"),), ks=(0, 5), ft_seeds=(42,)))
step("autoaug (pipeline search)", lambda: NS["exp_autoaug"](pairs=[("hhar", "uci")], ft_seeds=(42,),
                                                            search_epochs=1))
step("inference benchmark + export", lambda: NS["inference_benchmark"](n=64, reps=2))
step("cost estimate", lambda: NS["estimate_cost"]())


# ----------------------------------------------------------------------------- UniMTS with a stand-in model
def _unimts():
    import torch, torch.nn as nn
    fake = types.ModuleType("contrastive")

    class _Acc(nn.Module):
        def __init__(self, c):
            super().__init__(); self.lin = nn.Linear(c, 512)

        def forward(self, x):                   # x: [B, C, T, 22, 1]
            return self.lin(x.mean((2, 3, 4)))[:, :, None, None]

    class ContrastiveModule(nn.Module):
        def __init__(self, args):
            super().__init__()
            c = 6 if args.gyro else 3
            self.model = nn.Module(); self.model.acc = _Acc(c); self.fc = nn.Linear(512, args.num_class)

        def classifier(self, x):
            assert x.shape[1:] == (3, 200, 22, 1), x.shape
            return self.fc(self.model.acc(x).squeeze(-1).squeeze(-1))

    fake.ContrastiveModule = ContrastiveModule
    sys.modules["contrastive"] = fake
    ck = os.path.join(TMP, "fake_unimts.pth")
    torch.save(ContrastiveModule(types.SimpleNamespace(gyro=0, num_class=2)).model.state_dict(), ck)
    NS["setup_unimts"] = lambda: ck
    NS["exp_unimts"](pairs=[("hhar", "shoaib")], epochs=1, ft_seeds=(42,))
step("UniMTS adapter (stand-in model, real input formatting)", _unimts)

# ----------------------------------------------------------------------------- analysis
if NOTEBOOK:
    NA = NS                                   # same kernel namespace as in Colab
    exec(compile(_code("revision_analysis.py", "analysis"), "analysis", "exec"), NA)
    assert isinstance(NS["DATASETS"], dict), "analysis cell overwrote the pipeline's DATASETS"
else:
    NA = {"__name__": "revision_analysis", "REV_DIR": NS["REV_DIR"]}
    exec(compile(_code("revision_analysis.py", "analysis"), "analysis", "exec"), NA)


def _analysis():
    p = NA["make_all_tables"]()
    txt = open(p).read()
    assert "Main cross-dataset results" in txt
    print(txt[:3000])
step("analysis: all tables", _analysis)

print(f"\nSMOKE TEST PASSED: {len(ok)} steps in {(time.time() - t_start) / 60:.1f} min")
for s in ok:
    print("  -", s)
