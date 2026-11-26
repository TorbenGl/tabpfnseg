"""Download all model weights on a node WITH internet, then sanity-check each model on one image.

    export HF_TOKEN=...        # dinov3 + sam3 are gated: accept the licences on huggingface.co first
    export TABPFN_TOKEN=...    # https://ux.priorlabs.ai -> License tab (TabPFN-3.5 weights, non-commercial)
    python scripts/prefetch_models.py --models dinov3_l dinov3_h seggpt sam3 tabpfn

Afterwards compute nodes can run fully offline with HF_HUB_OFFLINE=1 (set in slurm/common.sh).
"""
import argparse
import traceback

import numpy as np
import torch

from tabpfnseg.data.datasets import SyntheticDataset

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["dinov3_l", "dinov3_h", "seggpt", "sam3", "tabpfn"])
ap.add_argument("--tabpfn_versions", nargs="+", default=["v3.5", "v3.5-fast"])
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

ds = SyntheticDataset(size=224)
(s_img, s_mask), (q_img, q_mask) = ds.load("cls0/0", "cls0/0"), ds.load("cls0/1", "cls0/1")
ok = {}
for m in a.models:
    try:
        if m.startswith("dino"):
            from tabpfnseg.features.backbone import build_backbone
            bb = build_backbone(m, a.device)
            ps = bb.patch_size
            f = bb.forward_layers(bb.preprocess(q_img, ps * 16), [6, -1])
            print(m, {k: tuple(v.shape) for k, v in f.items()})
        elif m == "seggpt":
            from tabpfnseg.methods.seggpt import SegGPTSegmenter
            p, _ = SegGPTSegmenter(a.device).predict([s_img], [s_mask], q_img)
            print(m, "fg frac", float((p >= .5).mean()), "gt frac", float(q_mask.mean()))
        elif m == "sam3":
            from tabpfnseg.methods.sam3 import Sam3Refiner, Sam3StitchSegmenter, Sam3TextSegmenter
            p, _ = Sam3TextSegmenter(device=a.device).predict([s_img], [s_mask], q_img, class_name="circle")
            print("sam3_text fg frac", float((p >= .5).mean()))
            p, st = Sam3StitchSegmenter(device=a.device).predict([s_img], [s_mask], q_img)
            print("sam3_stitch fg frac", float((p >= .5).mean()), st)
            r = Sam3Refiner(device=a.device)(q_img, q_mask.astype(np.float32))
            print("sam3 refine fg frac", float((r >= .5).mean()))
        elif m == "tabpfn":
            from tabpfnseg.methods.heads import TabPFNHead
            for v in a.tabpfn_versions:
                h = TabPFNHead(v, 1, a.device)
                rng = np.random.default_rng(0)
                X = rng.normal(size=(200, 64)).astype(np.float32)
                y = (X[:, 0] > 0).astype(np.float32)
                p = h.fit_predict(X[:150], y[:150], X[150:])
                print("tabpfn", v, "acc", float(((p > .5) == (y[150:] > .5)).mean()), h.info())
        ok[m] = True
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        ok[m] = False
print("\nSTATUS:", ok)
