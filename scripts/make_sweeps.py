"""Generate the experiment sweep files (one run per line) in slurm/sweeps/.

    python scripts/make_sweeps.py            # writes stage0/1/2 files
    bash slurm/submit_sweep.sh slurm/sweeps/stage1a_features.txt 8

Stage 0  sanity + resolution ceilings  (oracle_grid, cheap; and a proto baseline per dataset)
Stage 1  model selection on FSS-1000 *val* (--purpose tune; classes disjoint from test)
   1a  feature budget at ~6k dims: ViT-L last layer vs. multi-layer vs. patch+neighbourhood
   1b  working resolution: patch-level vs. sub-patch (x2/x4, bilinear vs JBU) and context sampling
   1c  TabPFN knobs: version, n_estimators, PCA, regression on soft labels, ambiguous-cell dropping
Stage 2  final report: frozen configs (edit FINAL below after looking at stage 1) on
         FSS-1000 test + Deepglobe + ISIC + Chest X-ray, 1/5-shot, several seeds,
         against SegGPT and SAM3 (+ SAM3 refinement applied to every method equally).
"""
import itertools
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "slurm", "sweeps")

BACKBONES = {  # name -> img_size giving a ~32-37 patch grid
    "dinov2r_l": 518,
    "dinov3_l": 512,
}
# ~6k-dim feature tables for a 1024-d ViT-L (the "6k features" comparison)
FEATS = {
    "last": "layers=24",                                  # 1024
    "ml6": "layers=4,8,12,16,20,24",                      # 6 layers x 1024 = 6144
    "nb6": "layers=24;pools=1,3,5,9,17,g",                # patch + 5 context scales = 6144
    "ml3nb2": "layers=12,18,24;pools=1,5",                # 3 x 2 x 1024 = 6144
    "ml2nb3": "layers=18,24;pools=1,5,g",                 # 2 x 3 x 1024 = 6144
}
HEADS = ["tabpfn", "logreg", "knn", "proto"]
CD = ["deepglobe", "isic", "chestx"]
TUNE = "--dataset fss1000 --split val --purpose tune --episodes 600 --shots 1 --seed 0"


def common(bb):
    return f"--backbone {bb} --img_size {BACKBONES[bb]}"


def write(name, lines):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.txt")
    lines = list(dict.fromkeys(lines))  # de-dup, keep order
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{path}: {len(lines)} runs")


# ----------------------------------------------------------------------------- stage 0
s0 = []
for ds in ["fss1000"] + CD:
    for bb in BACKBONES:
        for res in ["patch", "x2", "x4", "x8"]:
            s0.append(f"--dataset {ds} --method oracle_grid {common(bb)} --res {res} --shots 1 --seed 0")
        s0.append(f"--dataset {ds} --method proto {common(bb)} --features {FEATS['last']} --shots 1 --seed 0")
write("stage0_sanity", s0)

# ----------------------------------------------------------------------------- stage 1a
s1a = []
for bb, (fname, f), head in itertools.product(BACKBONES, FEATS.items(), HEADS):
    s1a.append(f"{TUNE} --method {head} {common(bb)} --features '{f}' --res patch --sampling all")
# same budget with a bigger backbone: DINOv2-g (1536-d) x 4 layers = 6144
for head in HEADS:
    s1a.append(f"{TUNE} --method {head} --backbone dinov2r_g --img_size 518 --features 'layers=10,20,30,40' --res patch")
write("stage1a_features", s1a)

# ----------------------------------------------------------------------------- stage 1b
s1b = []
for bb in BACKBONES:
    for fname in ["last", "nb6"]:
        for rgb in [0, 1]:
            f = FEATS[fname] + (";rgb=1" if rgb else "")
            for head in ["tabpfn", "logreg"]:
                base = f"{TUNE} --method {head} {common(bb)} --features '{f}'"
                s1b.append(f"{base} --res patch --sampling all")
                s1b.append(f"{base} --res patch --sampling all --ambig 0.2,0.8")
                for res, up, samp in itertools.product(["x2", "x4"], ["bilinear", "jbu"], ["uniform", "balanced", "boundary"]):
                    s1b.append(f"{base} --res {res} --up {up} --sampling {samp} --max_ctx 8000")
                for ctx in [2000, 4000, 16000, 32000]:
                    s1b.append(f"{base} --res x4 --up jbu --sampling boundary --max_ctx {ctx}")
write("stage1b_resolution", s1b)

# ----------------------------------------------------------------------------- stage 1c
s1c = []
bb = "dinov2r_l"
for fname in ["ml6", "nb6"]:
    base = f"{TUNE} {common(bb)} --features '{FEATS[fname]}' --res patch"
    for ver in ["v3.5", "v3.5-fast", "v3", "v2.5"]:
        for est in [1, 4, 8]:
            s1c.append(f"{base} --method tabpfn --tabpfn_version {ver} --n_estimators {est}")
    for pca in [64, 256, 1024]:
        s1c.append(f"{base} --method tabpfn --pca {pca}")
        s1c.append(f"{base} --method logreg --pca {pca}")
    s1c.append(f"{base} --method tabpfn_reg")
    for C in [0.01, 0.1, 1.0]:
        s1c.append(f"{base} --method logreg --logreg_C {C}")
    for k in [4, 16, 64]:
        s1c.append(f"{base} --method knn --knn_k {k}")
write("stage1c_tabpfn_knobs", s1c)

# ----------------------------------------------------------------------------- stage 2
# >>> EDIT after stage 1: the winning configurations (chosen on FSS-1000 val ONLY) <<<
FINAL = {
    "tabpfn_patch": "--method tabpfn --backbone dinov2r_l --img_size 518 --features 'layers=24;pools=1,3,5,9,17,g' --res patch",
    "tabpfn_x4": "--method tabpfn --backbone dinov2r_l --img_size 518 --features 'layers=24;pools=1,3,5,9,17,g;rgb=1' "
                 "--res x4 --up jbu --sampling boundary --max_ctx 16000",
    "logreg_patch": "--method logreg --backbone dinov2r_l --img_size 518 --features 'layers=24;pools=1,3,5,9,17,g' --res patch",
    "knn_patch": "--method knn --backbone dinov2r_l --img_size 518 --features 'layers=24;pools=1,3,5,9,17,g' --res patch",
    "proto_patch": "--method proto --backbone dinov2r_l --img_size 518 --features 'layers=24' --res patch",
    "seggpt": "--method seggpt",
    "sam3_stitch": "--method sam3_stitch",
    "sam3_text": "--method sam3_text",  # reference only: uses class names, not in-context
}
REFINE = ["tabpfn_patch", "proto_patch", "seggpt"]
SEEDS = [0, 1, 2]

s2 = []
for ds, shots, seed in itertools.product(["fss1000"] + CD, [1, 5], SEEDS):
    split = "--split test"
    for key, cfg in FINAL.items():
        s2.append(f"--dataset {ds} {split} --shots {shots} --seed {seed} {cfg}")
        if key in REFINE:
            s2.append(f"--dataset {ds} {split} --shots {shots} --seed {seed} {cfg} --refine sam3")
write("stage2_report", s2)
