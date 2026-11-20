"""Collect results/*/summary.json into one table (mean +- std over seeds).

    python scripts/aggregate.py results --out results/table.md
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?", default="results")
ap.add_argument("--out", default=None)
ap.add_argument("--metric", default="mIoU")
a = ap.parse_args()

groups = defaultdict(list)
for f in sorted(glob.glob(os.path.join(a.root, "*", "summary.json"))):
    run = os.path.basename(os.path.dirname(f))
    cfg_f = os.path.join(os.path.dirname(f), "config.json")
    cfg = json.load(open(cfg_f)) if os.path.exists(cfg_f) else {}
    s = json.load(open(f))
    key = re.sub(r"_s\d+_", "_", run)  # merge seeds
    groups[key].append((s[a.metric], s.get("FB-IoU"), s.get("seconds_per_episode"), s.get("n_episodes"), cfg))

rows = []
for key, vals in sorted(groups.items()):
    m = np.array([v[0] for v in vals])
    fb = np.array([v[1] for v in vals])
    rows.append((key, len(vals), m.mean(), m.std(), fb.mean(), np.mean([v[2] for v in vals]), vals[0][3]))

lines = [f"| run | seeds | {a.metric} | std | FB-IoU | s/episode | episodes |", "|---|---|---|---|---|---|---|"]
for r in rows:
    lines.append(f"| {r[0]} | {r[1]} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]} |")
txt = "\n".join(lines)
print(txt)
if a.out:
    open(a.out, "w").write(txt + "\n")
