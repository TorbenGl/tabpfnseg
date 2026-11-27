"""Evaluate one method on one dataset/shot/seed.

    python -m tabpfnseg.run --dataset fss1000 --split test --shots 1 --method tabpfn \
        --backbone dinov3_l --features "layers=24" --res patch

Writes <out>/<run_name>/{config.json, episodes.jsonl, summary.json}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data.datasets import DEFAULT_EPISODES, build_dataset
from .data.episodes import get_or_create_episodes, load_episodes
from .eval.metrics import Meter

FEATURE_HEADS = {"tabpfn", "tabpfn_reg", "logreg", "knn", "proto"}
IMAGE_METHODS = {"seggpt", "sam3_text", "sam3_stitch"}
CD_DATASETS = {"deepglobe", "isic", "chestx"}


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("data / protocol")
    g.add_argument("--dataset", default="fss1000")
    g.add_argument("--root", default=os.environ.get("TABPFNSEG_DATA", "datasets"))
    g.add_argument("--split", default="test", help="fss1000: trn|val|test. Others: test only")
    g.add_argument("--purpose", default="report", choices=["tune", "report"],
                   help="'tune' refuses test splits -> hyper-parameters are chosen on FSS-1000 val only")
    g.add_argument("--shots", type=int, default=1)
    g.add_argument("--episodes", type=int, default=None, help="default: protocol value (FSS 2400, CD 1200)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--episode_dir", default="episodes")
    g.add_argument("--episode_file", default=None)
    g.add_argument("--dedup_hamming", type=int, default=2, help="reject supports near-identical to the query (-1 = off)")
    g.add_argument("--eval_size", type=int, default=0, help="0 = evaluate at original mask resolution")
    g.add_argument("--start", type=int, default=0)
    g.add_argument("--limit", type=int, default=None)

    g = ap.add_argument_group("method")
    g.add_argument("--method", default="tabpfn",
                   help="tabpfn | tabpfn_reg | logreg | knn | proto | oracle_grid | seggpt | sam3_text | sam3_stitch")
    g.add_argument("--refine", default="none", choices=["none", "sam3", "sam3band"],
                   help="sam3: replace the mask by SAM3's; sam3band: accept SAM3 only within a band around the coarse boundary")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    g = ap.add_argument_group("features (feature-table methods)")
    g.add_argument("--backbone", default="dinov3_l")
    g.add_argument("--img_size", type=int, default=512, help="518 for /14 backbones (37x37), 512 for /16 (32x32)")
    g.add_argument("--features", default="layers=-1", help="see FeatureSpec, e.g. 'layers=24;pools=1,3,5,9,17,g'")
    g.add_argument("--res", default="patch", help="patch | x2 | x4 ...")
    g.add_argument("--up", default="bilinear", choices=["bilinear", "nearest", "jbu"])
    g.add_argument("--sampling", default="all", choices=["all", "uniform", "balanced", "boundary"])
    g.add_argument("--max_ctx", type=int, default=10000, help="max context rows over all supports")
    g.add_argument("--ambig", default="", help="e.g. '0.2,0.8': drop support cells with mixed fg fraction")
    g.add_argument("--prior_correct", type=int, default=1)
    g.add_argument("--debias", type=int, default=0, help="FROST/INSID3 positional debiasing rank (0 = off; FROST: 250)")
    g.add_argument("--whiten", type=float, default=0.0,
                   help="FROST within-class shrinkage whitening, intensity lambda (0 = off; FROST: 0.95)")
    g.add_argument("--support_flip", type=int, default=0, help="add horizontally flipped supports (FROST)")
    g.add_argument("--frost", type=int, default=0, help="shortcut: --debias 250 --whiten 0.95 --support_flip 1")
    g.add_argument("--anchors", default="none", choices=["none", "black", "white", "bw"],
                   help="add the rows of an all-black / all-white image to the context, labelled background")
    g.add_argument("--pca", type=int, default=0, help="PCA dim fitted on context rows only (0 = off)")

    g = ap.add_argument_group("heads")
    g.add_argument("--tabpfn_version", default="v3.5")
    g.add_argument("--tabpfn_model_path", default=None)
    g.add_argument("--tabpfn_fit_mode", default="fit_preprocessors")
    g.add_argument("--n_estimators", default="4")
    g.add_argument("--logreg_C", type=float, default=1.0)
    g.add_argument("--knn_k", type=int, default=16)

    g = ap.add_argument_group("output")
    g.add_argument("--out", default="results")
    g.add_argument("--name", default=None)
    g.add_argument("--save_vis", type=int, default=0, help="save overlays for the first N episodes")
    g.add_argument("--skip_done", type=int, default=0, help="exit if <out>/<name>/summary.json exists")
    a = ap.parse_args(argv)
    a.n_estimators = int(a.n_estimators) if a.n_estimators.isdigit() else a.n_estimators
    if a.frost:
        a.debias = a.debias or 250
        a.whiten = a.whiten or 0.95
        a.support_flip = 1
    return a


def check_protocol(a):
    """Guards against the most common few-shot-segmentation leaks."""
    if a.dataset in CD_DATASETS and a.purpose == "tune":
        sys.exit("[leakage-guard] CD-FSS target datasets have no validation split: never tune on them. "
                 "Tune on --dataset fss1000 --split val.")
    if a.purpose == "tune" and a.split == "test":
        sys.exit("[leakage-guard] --purpose tune on a test split is not allowed.")
    if a.dataset == "fss1000" and a.split == "trn":
        print("[warn] FSS-1000 trn split: only meaningful for debugging.")
    if a.method == "oracle_grid":
        print("[info] oracle_grid uses the query ground truth: it is an upper bound, not a method.")
    if a.method == "sam3_text":
        print("[info] sam3_text uses the class NAME (privileged info) and is not in-context learning.")


def build_method(a):
    from .features.backbone import build_backbone
    from .features.builder import FeatureBuilder, FeatureSpec, ResSpec

    if a.method in FEATURE_HEADS or a.method == "oracle_grid":
        bb = build_backbone(a.backbone, a.device)
        fb = FeatureBuilder(bb, FeatureSpec.parse(a.features), ResSpec.parse(a.res, a.up), a.img_size, debias=a.debias)
        if a.method == "oracle_grid":
            from .methods.feature_segmenter import OracleGrid

            return OracleGrid(fb), {"dim": fb.dim, "grid": fb.out_hw}
        from .methods.feature_segmenter import FeatureSegmenter
        from .methods.heads import build_head

        head = build_head(a.method, a)
        return FeatureSegmenter(fb, head, a), {"dim": fb.dim, "grid": fb.out_hw, "head": head}
    if a.method == "seggpt":
        from .methods.seggpt import SegGPTSegmenter

        return SegGPTSegmenter(a.device), {}
    if a.method == "sam3_text":
        from .methods.sam3 import Sam3TextSegmenter

        return Sam3TextSegmenter(device=a.device), {}
    if a.method == "sam3_stitch":
        from .methods.sam3 import Sam3StitchSegmenter

        return Sam3StitchSegmenter(device=a.device), {}
    raise ValueError(a.method)


def run_name(a) -> str:
    if a.name:
        return a.name
    parts = [a.dataset, a.split, f"{a.shots}shot", f"s{a.seed}", a.method]
    if a.method in FEATURE_HEADS or a.method == "oracle_grid":
        feat = a.features.replace(";", "_").replace("=", "").replace(",", "-")
        parts += [a.backbone, feat, a.res, a.up if a.res != "patch" else "", a.sampling, f"ctx{a.max_ctx}"]
        if a.img_size != 512:  # non-default input size (e.g. 1024) must not collide with the 512 run
            parts.append(f"px{a.img_size}")
        if a.pca:
            parts.append(f"pca{a.pca}")
        if a.ambig:
            parts.append("ambig" + a.ambig.replace(",", "-"))
        if a.method.startswith("tabpfn"):
            parts += [a.tabpfn_version, f"est{a.n_estimators}"]
        if a.method == "logreg" and a.logreg_C != 1.0:   # non-default only: keeps existing run names valid
            parts.append(f"C{a.logreg_C:g}")
        if a.method == "knn" and a.knn_k != 16:
            parts.append(f"k{a.knn_k}")
        if getattr(a, "anchors", "none") != "none":
            parts.append(f"anchor-{a.anchors}")
        if a.debias:
            parts.append(f"deb{a.debias}")
        if a.whiten:
            parts.append(f"wh{a.whiten:g}")
        if a.support_flip:
            parts.append("flip")
    if a.refine != "none":
        parts.append(f"refine-{a.refine}")
    return "_".join(p for p in parts if p)


def overlay(img, gt, pred):
    v = img.astype(np.float32).copy()
    v[pred & gt] = 0.5 * v[pred & gt] + 0.5 * np.array([0, 255, 0])
    v[pred & ~gt] = 0.5 * v[pred & ~gt] + 0.5 * np.array([255, 0, 0])
    v[~pred & gt] = 0.5 * v[~pred & gt] + 0.5 * np.array([0, 0, 255])
    return v.astype(np.uint8)


def main(argv=None):
    a = get_args(argv)
    check_protocol(a)
    if a.skip_done and os.path.exists(os.path.join(a.out, run_name(a), "summary.json")):
        print(f"[skip] {run_name(a)} already done")
        return None
    torch.manual_seed(a.seed)
    ds = build_dataset(a.dataset, a.root, a.split)
    if a.episode_file:
        episodes, _ = load_episodes(a.episode_file)
        ep_path = a.episode_file
    else:
        n = a.episodes if a.episodes is not None else DEFAULT_EPISODES.get(a.dataset)
        if a.dataset == "fss1000" and n == ds.num_images():
            n = None
        episodes, ep_path = get_or_create_episodes(ds, a.shots, n, a.seed, a.episode_dir, a.dedup_hamming)
    episodes = episodes[a.start : (a.start + a.limit) if a.limit else None]

    method, minfo = build_method(a)
    refiner = None
    if a.refine in ("sam3", "sam3band"):
        from .methods.sam3 import Sam3Refiner

        refiner = Sam3Refiner(device=a.device, band=(a.refine == "sam3band"))

    name = run_name(a)
    out_dir = os.path.join(a.out, name)
    os.makedirs(out_dir, exist_ok=True)
    cfg = {k: v for k, v in vars(a).items()}
    cfg.update({"run_name": name, "episode_file": ep_path, "n_episodes_run": len(episodes),
                "dim": minfo.get("dim"), "grid": minfo.get("grid")})
    print(json.dumps({k: cfg[k] for k in ("run_name", "dim", "grid", "n_episodes_run")}))

    meter = Meter()
    t_start = time.time()
    with open(os.path.join(out_dir, "episodes.jsonl"), "w") as fout:
        for k, ep in enumerate(episodes):
            q_img, q_mask = ds.load(*ep.query)
            sups = [ds.load(*s) for s in ep.supports]
            t0 = time.time()
            prob, stats = method.predict(
                [s[0] for s in sups], [s[1] for s in sups], q_img,
                sup_keys=[s[0] for s in ep.supports], q_key=ep.query[0],
                q_mask_oracle=q_mask if getattr(method, "needs_query_mask", False) else None,
                class_name=ds.prompt_name(ep.cls),
            )
            if refiner is not None:
                prob = refiner(q_img, prob)
            dt = time.time() - t0
            gt = q_mask
            if a.eval_size:
                s = a.eval_size
                gt = np.asarray(Image.fromarray(q_mask * 255).resize((s, s), Image.NEAREST)) > 127
                prob = F.interpolate(torch.from_numpy(prob)[None, None], size=(s, s), mode="bilinear",
                                     align_corners=False)[0, 0].numpy()
            pred = prob >= 0.5
            iou = meter.add(ep.cls, pred, gt)
            rec = {"idx": ep.idx, "cls": ep.cls, "iou": iou, "time": dt, **{kk: vv for kk, vv in stats.items()
                                                                          if isinstance(vv, (int, float, str))}}
            fout.write(json.dumps(rec) + "\n")
            if k < a.save_vis:
                os.makedirs(os.path.join(out_dir, "vis"), exist_ok=True)
                Image.fromarray(overlay(q_img if not a.eval_size else np.asarray(Image.fromarray(q_img).resize(
                    (a.eval_size, a.eval_size))), gt.astype(bool), pred)).save(os.path.join(out_dir, "vis", f"{ep.idx}.jpg"))
            if (k + 1) % 50 == 0 or k + 1 == len(episodes):
                s = meter.summary()
                print(f"[{k + 1}/{len(episodes)}] mIoU {s['mIoU']:.2f} FB-IoU {s['FB-IoU']:.2f} "
                      f"({(time.time() - t_start) / (k + 1):.2f}s/ep)", flush=True)

    summ = meter.summary()
    summ["seconds_per_episode"] = (time.time() - t_start) / max(len(episodes), 1)
    head = minfo.get("head")
    if head is not None:
        cfg["head_info"] = head.info()
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1, default=str)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(f"DONE {name}: mIoU {summ['mIoU']:.2f}  FB-IoU {summ['FB-IoU']:.2f}")
    return summ


if __name__ == "__main__":
    main()
