# =============================================================================
#  build_colab_notebook.py -- packs pipeline_base.py, revision_experiments.py and
#  revision_analysis.py into ONE self-contained Colab notebook:
#      revision/GraviHAR_Revision_Colab.ipynb
#  Re-run after editing any of the three modules:
#      python revision/build_colab_notebook.py
# =============================================================================
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "GraviHAR_Revision_Colab.ipynb")


def _src(name):
    """Module source without its trailing `if __name__ == "__main__":` block, which
    would otherwise run the old entry point / argparse inside the notebook kernel."""
    s = open(os.path.join(HERE, name)).read()
    m = list(re.finditer(r'^if __name__ == "__main__":', s, flags=re.M))
    if m:
        s = s[:m[-1].start()].rstrip() + "\n"
    return s


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text, tag=None):
    meta = {"tags": [tag]} if tag else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


CELLS = [
    md("""
# GraviHAR revision experiments (draft 3), single notebook

This notebook contains everything needed for the revision: the original pipeline
(cell 20 of `Gravity_Model.ipynb`, unchanged), the revision experiments and the
analysis that produces every table and number. The plan and the comment-by-comment
mapping are in `revision/REVISION_PLAN.md`.

**How to use**
1. Runtime → Change runtime type → GPU.
2. Run the setup cell and Parts 1–3. They only define functions and take a few seconds.
3. Run the stage cells in order. Each stage appends one row per run to
   `MyDrive/revision/runs/*.csv` and skips runs that are already there, so after a
   disconnect re-run the setup cell and Parts 1–3, then the interrupted stage.
4. Run the last cell for tables, tests, figures and `headline_numbers.json`
   (written to `MyDrive/revision/tables/`).

**Expected data layout on Drive** (same as the original notebook)
- `MyDrive/HHAR/Activity recognition exp/Phones_accelerometer.csv`, `Phones_gyroscope.csv`
- `MyDrive/UCI HAR Dataset/{train,test}/...` (including `subject_*.txt`)
- `MyDrive/MotionSense/A_DeviceMotion_data/<activity>_<trial>/sub_<k>.csv`
- `MyDrive/Shoaib/*.csv` (Participant files)
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
!pip -q install xgboost tabulate
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE -- switch the runtime to GPU")
"""),
    md("## Part 1: pipeline (notebook cell 20, verbatim)\nLoaders, LIMU-BERT, GRU head, augmentation operators, XGBoost features."),
    code(_src("pipeline_base.py"), "pipeline"),
    md("## Part 2: revision experiments\nCorrected data layer (MotionSense iOS sign, subject IDs), one training path for every arm, all experiments and training-free audits."),
    code(_src("revision_experiments.py"), "experiments"),
    md("## Part 3: analysis\nTables, paired tests (Holm), hierarchical bootstrap, leave-one-dataset-out, gap decomposition, rotation summaries, headline numbers."),
    code(_src("revision_analysis.py"), "analysis"),
    md("---\n## Stage 0: audit (no training, minutes)\nMotionSense sign (c33), geometry identities (c36 c37 c51), device-axis decomposition (c37 c50 c51), antipode (c38), heading definition (c32), cadence and energy (c66), preprocessing and label tables (c44 c45), parameter split (c42)."),
    code('run_stage("audit")'),
    md("Time one epoch on this GPU and project the hours per stage."),
    code("estimate_cost()"),
    md("## Tier 1 (minimum for the resubmission)"),
    code('run_stage("xgb")        # engineered features, raw + canonicalized, 12 pairs (CPU; cached after the first run)'),
    code('run_stage("ablation")   # A/B/C/D on 12 pairs, 2 pretraining x 4 fine-tuning seeds'),
    code('run_stage("baselines")  # SO(3) aug, learned SO(3), gravity + PCA heading, Mizell, OIT'),
    code('run_stage("ceiling")    # in-domain ceilings on each pair\'s label set (c29 c64)'),
    code('run_stage("rotation")   # controlled device- and gravity-frame rotations, 4 datasets (RQ3)'),
    code("""
# UniMTS (c46.3): clones the official repo, installs CLIP, downloads the checkpoint.
# To reduce cost: exp_unimts(epochs=5, max_train=20000)
run_stage("unimts")
"""),
    md("## Tier 2"),
    code('run_stage("site")       # where the yaw invariance must live (c39 c41)'),
    code('run_stage("posture")    # sit / stand / lie (c44)'),
    code('run_stage("hparam")     # source-only selection + target sensitivity grid (c40)'),
    code('run_stage("inference")  # latency with CPU model, threads, batch; TorchScript/ONNX export (c27 c56)'),
    md("## Tier 3 (optional)"),
    code('# run_stage("augspec"); run_stage("positions"); run_stage("labeleff"); run_stage("autoaug")'),
    md("---\n## Tables, tests, figures, headline numbers\nCan be run at any point; tables for stages that have not run yet are marked as not available."),
    code("""
path = make_all_tables()
print(open(path).read()[:8000])
"""),
]


def build():
    nb = {"cells": CELLS,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    with open(OUT, "w") as f:
        json.dump(nb, f, indent=1)
    return OUT


if __name__ == "__main__":
    print("wrote", build())
