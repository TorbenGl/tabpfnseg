"""Data-leakage audit. Run once per dataset after download (and after generating episodes).

Checks
  1. FSS-1000 trn / val / test class lists are pairwise disjoint (tune on val, report on test).
  2. No exact / near-duplicate images across FSS-1000 val and test. Candidates come from a perceptual
     dHash (8x8, many false alarms on plain backgrounds / X-rays); each candidate is then CONFIRMED by
     the pixel correlation of 64x64 thumbnails (>= --confirm_corr). Only confirmed pairs count.
     Side-by-side pictures of every candidate are saved to --pairs_dir for a visual check.
  3. For every episode file: query never in its own support set; reports supports that are
     near-duplicates of the query (should be 0 with --dedup_hamming >= 0 at generation).
  4. Within-dataset duplicate images (e.g. same chest X-ray exported twice) -> reported.
  5. Prints the known PRE-TRAINING overlap of every compared model with every benchmark.

    python scripts/check_leakage.py --root datasets --episodes episodes
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from tabpfnseg.data.datasets import SPLIT_DIR, build_dataset
from tabpfnseg.data.episodes import dhash, hamming

PRETRAIN_OVERLAP = """
Known pre-training data overlap (from the papers; 'yes' = images or the same domain were seen):

| model            | pre-training data                                        | FSS-1000 | Deepglobe            | ISIC | Chest X-ray | COCO-20i / PASCAL-5i | underwater (SUIM/UWS/UIIS) |
|------------------|----------------------------------------------------------|----------|----------------------|------|-------------|----------------------|----------------------------|
| TabPFN-3.5       | synthetic tabular priors only                            | no       | no                   | no   | no          | no                   | no |
| DINOv3 (LVD-1.7B)| Instagram-scale web images (+ separate SAT-493M model)   | unclear* | unclear*             | no** | no**        | unclear*             | unclear* (web) |
| SegGPT           | COCO, ADE20K, VOC, Cityscapes, LIP, PACO, iSAID, loveDA, retina sets | no | DOMAIN (loveDA/iSAID aerial land cover) | no | no | YES (COCO/VOC train) | no |
| SAM3             | SA-Co (incl. medical, aerial, ...), SA-1B, 5.4B image-text pairs | unclear | likely domain | likely domain | likely domain | YES (COCO, LVIS) | DOMAIN (FathomNet in SA-Co) |

*  web-scale corpora; individual benchmark images cannot be ruled out, but no segmentation labels.
** frozen backbone: no masks of these domains seen; same for TabPFN heads on top.
Headline numbers should therefore be reported on FSS-1000 test + ISIC + Chest X-ray (clean for all
in-context methods); Deepglobe is domain-adjacent for SegGPT; COCO-20i/PASCAL-5i only as a flagged
'seen-data' reference.
"""

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="datasets")
ap.add_argument("--episodes", default="episodes")
ap.add_argument("--hamming", type=int, default=2)
ap.add_argument("--confirm_corr", type=float, default=0.95,
                help="64x64 grayscale pixel correlation needed to confirm a dHash candidate as a real duplicate")
ap.add_argument("--pairs_dir", default="logs/leakage_pairs", help="side-by-side pictures of flagged pairs")
ap.add_argument("--datasets", nargs="+", default=["fss1000", "deepglobe", "isic", "chestx", "suim", "uws"])
a = ap.parse_args()

problems = 0
# 1. class split disjointness
splits = {s: set((SPLIT_DIR / "fss" / f"{s}.txt").read_text().split()) for s in ("trn", "val", "test")}
for x, y in (("trn", "val"), ("trn", "test"), ("val", "test")):
    inter = splits[x] & splits[y]
    print(f"[1] FSS-1000 {x} ∩ {y}: {len(inter)} classes", "OK" if not inter else f"LEAK {sorted(inter)[:5]}")
    problems += bool(inter)


THUMB = {}   # image path -> 64x64 grayscale thumbnail (float32, zero-mean, unit-norm)


def hashes(ds):
    h = {}
    for c in ds.classes:
        for it in ds.items[c]:
            img = ds.load(*it)[0]
            h[it[0]] = dhash(img)
            t = np.asarray(Image.fromarray(img).convert("L").resize((64, 64), Image.BILINEAR), np.float32).ravel()
            t -= t.mean()
            THUMB[it[0]] = t / (np.linalg.norm(t) + 1e-6)
    return h


def corr(k1, k2):
    return float(THUMB[k1] @ THUMB[k2]) if k1 in THUMB and k2 in THUMB else 0.0


def confirmed(pairs):
    return [(k1, k2) for k1, k2 in pairs if corr(k1, k2) >= a.confirm_corr]


def save_pairs(pairs, tag, root, limit=50):
    os.makedirs(a.pairs_dir, exist_ok=True)
    for i, (k1, k2) in enumerate(pairs[:limit]):
        ims = [Image.open(os.path.join(root, k)).convert("RGB") for k in (k1, k2)]
        ims = [im.resize((256, round(256 * im.height / im.width))) for im in ims]
        out = Image.new("RGB", (512 + 8, max(im.height for im in ims)), "white")
        out.paste(ims[0], (0, 0)); out.paste(ims[1], (264, 0))
        out.save(os.path.join(a.pairs_dir, f"{tag}_{i:02d}_corr{corr(k1, k2):.2f}.jpg"))


def near_dups(h1, h2=None, thr=2):
    pairs = []
    items1 = list(h1.items())
    items2 = list((h2 or h1).items())
    by_bucket = defaultdict(list)
    for k, v in items2:
        by_bucket[v >> 48].append((k, v))  # coarse bucket on top bits, then exact hamming
    for k, v in items1:
        for k2, v2 in by_bucket.get(v >> 48, []):
            if k2 != k and hamming(v, v2) <= thr and (h2 is not None or k < k2):
                pairs.append((k, k2))
    return pairs


# 2./4. duplicates
cache = {}
for name in a.datasets:
    try:
        ds = build_dataset(name, a.root, "test")
    except FileNotFoundError as e:
        print(f"[skip] {e}")
        continue
    cache[name] = hashes(ds)
    d = near_dups(cache[name], thr=a.hamming)
    c = confirmed(d)
    print(f"[4] {name}: {len(d)} dHash candidates, {len(c)} confirmed near-duplicate pairs inside the test set"
          + (f", e.g. {c[:3]}" if c else ""))
if "fss1000" in cache:
    try:
        hv = hashes(build_dataset("fss1000", a.root, "val"))
        d = near_dups(hv, cache["fss1000"], a.hamming)
        c = confirmed(d)
        if d:
            save_pairs(d, "fss_val_test", a.root)
        print(f"[2] FSS-1000 val vs test: {len(d)} dHash candidates, {len(c)} confirmed duplicates",
              "OK" if not c else f"CHECK {c[:5]}", f"(pictures: {a.pairs_dir}/)" if d else "")
        for k1, k2 in d:
            print(f"      corr={corr(k1, k2):.3f}  {k1}  <->  {k2}")
        problems += bool(c)
    except FileNotFoundError:
        pass

# 3. episode files
for f in sorted(glob.glob(os.path.join(a.episodes, "*.json"))):
    d = json.load(open(f))
    ds_name = d["meta"]["dataset"]
    bad = sum(e["query"][0] in [s[0] for s in e["supports"]] for e in d["episodes"])
    nd = 0
    if ds_name in cache:
        h = cache[ds_name]
        nd = sum(any(hamming(h[e["query"][0]], h[s[0]]) <= a.hamming and corr(e["query"][0], s[0]) >= a.confirm_corr
                     for s in e["supports"] if s[0] in h and e["query"][0] in h) for e in d["episodes"])
    print(f"[3] {os.path.basename(f)}: query-in-support={bad}, confirmed near-dup support/query={nd}")
    problems += bad
print(PRETRAIN_OVERLAP)
print("LEAKAGE CHECK:", "PASSED" if problems == 0 else f"{problems} PROBLEM(S)")
