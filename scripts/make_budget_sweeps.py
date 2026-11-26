"""Compact experiment plan for a ~24 GPU-hour budget (DINOv3 ViT-L only).

    python scripts/make_budget_sweeps.py
    bash slurm/submit_sweep.sh slurm/sweeps/budget_tune.txt 4      # step 1: tune on FSS-1000 val
    # look at the tune results, fill in BEST below, re-run this script, then:
    bash slurm/submit_sweep.sh slurm/sweeps/budget_report.txt 4    # step 2: report on the test sets

Compared with the full plan (scripts/make_sweeps.py, ~550k episodes) this keeps only what the two
questions need:
  Q1  TabPFN vs kNN / logreg / proto on identical feature tables   -> 5 feature sets x {tabpfn, knn}
  Q2  best TabPFN system vs SegGPT / SAM3                          -> one row per method in the report
and cuts: 1 backbone, 300 tuning episodes, a single x2 resolution check, 1 seed, fewer test episodes,
no SAM3 refinement, no underwater sets.
"""
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "slurm", "sweeps")

TUNE_EPISODES = 300                       # FSS-1000 val, 1-shot (enough to rank settings)
REPORT_EPISODES = {"fss1000": 1000, "deepglobe": 600, "isic": 600, "chestx": 600}
REPORT_SHOTS = [1, 5]
BB = "--backbone dinov3_l --img_size 512"
FEATS = {
    "last": "layers=24",
    "ml6": "layers=4,8,12,16,20,24",
    "nb6": "layers=24;pools=1,3,5,9,17,g",
    "ml3nb2": "layers=12,18,24;pools=1,5",
    "ml2nb3": "layers=18,24;pools=1,5,g",
}
TUNE = f"--dataset fss1000 --split val --purpose tune --episodes 600 --limit {TUNE_EPISODES} --shots 1 --seed 0 {BB}"

# >>> EDIT after the tune step (choose on FSS-1000 val only) <<<
BEST = {  # chosen on FSS-1000 val, 1-shot, 300 episodes (budget_tune, 2026-11-26)
    "tabpfn": "--method tabpfn --n_estimators 8 --features 'layers=18,24;pools=1,5,g'",   # 79.35 (best TabPFN; est16 +0.2 only)
    "knn": "--method knn --features 'layers=24'",                                          # 81.39 (best kNN)
    "logreg": "--method logreg --logreg_C 0.01 --features 'layers=24;pools=1,3,5,9,17,g'", # 74.84 (best C)
    "proto": "--method proto --features 'layers=24'",                                      # 74.55
}


def write(name, lines):
    os.makedirs(OUT, exist_ok=True)
    lines = list(dict.fromkeys(lines))
    path = os.path.join(OUT, f"{name}.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{path}: {len(lines)} runs")


# ------------------------------------------------------------------ step 1: tune (FSS-1000 val)
t = []
# resolution ceilings (oracle, nearly free) on every dataset
for ds, eps in REPORT_EPISODES.items():
    for res in ["patch", "x2"]:
        if ds == "fss1000":
            t.append(f"{TUNE} --method oracle_grid --res {res}")
        else:  # the cross-domain sets have no val split; the oracle is a ceiling, not a tuned method
            t.append(f"--dataset {ds} --split test --episodes {eps} --shots 1 --seed 0 {BB} --method oracle_grid --res {res}")
# Q1 core: every feature set with TabPFN (8 estimators = all 6144 columns seen) and kNN
for f in FEATS.values():
    t.append(f"{TUNE} --method tabpfn --n_estimators 8 --features '{f}'")
    t.append(f"{TUNE} --method knn --features '{f}'")
t.append(f"{TUNE} --method proto --features '{FEATS['last']}'")
# fair linear baseline: tune C
for C in [0.01, 0.1, 1.0]:
    t.append(f"{TUNE} --method logreg --logreg_C {C} --features '{FEATS['nb6']}'")
# TabPFN knobs on nb6: estimators / PCA / soft-label regression
for est in [4, 16]:
    t.append(f"{TUNE} --method tabpfn --n_estimators {est} --features '{FEATS['nb6']}'")
t.append(f"{TUNE} --method tabpfn --n_estimators 1 --pca 768 --features '{FEATS['nb6']}'")
t.append(f"{TUNE} --method tabpfn_reg --n_estimators 8 --features '{FEATS['nb6']}'")
# one sub-patch check
for m in ["tabpfn --n_estimators 8", "knn"]:
    t.append(f"{TUNE} --method {m} --features '{FEATS['nb6']}' --res x2 --up jbu --sampling boundary --max_ctx 8000")
# dedicated segmenters on the same val episodes (Q2 reference + timing)
for m in ["seggpt", "sam3_stitch", "sam3_text"]:
    t.append(f"--dataset fss1000 --split val --purpose tune --episodes 600 --limit {TUNE_EPISODES} --shots 1 --seed 0 --method {m}")
# --- appended later (keep line numbers 1-31 stable for already-submitted arrays) ---
# position columns (y, x) and black/white anchor images as extra background context
for m in ["tabpfn --n_estimators 8", "knn"]:
    t.append(f"{TUNE} --method {m} --features '{FEATS['nb6']};pos=1'")
    t.append(f"{TUNE} --method {m} --features '{FEATS['nb6']}' --anchors bw")
write("budget_tune", t)

# ------------------------------------------------------------------ step 2: report (test sets)
r = []
for ds, eps in REPORT_EPISODES.items():
    for shots in REPORT_SHOTS:
        base = f"--dataset {ds} --split test --episodes {eps} --shots {shots} --seed 0"
        for cfg in BEST.values():
            r.append(f"{base} {BB} {cfg}")
        for m in ["seggpt", "sam3_stitch", "sam3_text"]:
            r.append(f"{base} --method {m}")
write("budget_report", r)
