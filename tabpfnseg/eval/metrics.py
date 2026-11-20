"""Few-shot segmentation metrics (HSNet / PATNet conventions).

mIoU  : for every class c accumulate intersection and union of the foreground over all of
        its episodes, IoU_c = sum I / sum U, mIoU = mean_c IoU_c.   (the headline metric)
FB-IoU: mean of the (dataset-accumulated) foreground IoU and background IoU.
We also keep the per-episode foreground IoU to compute bootstrap CIs / paired tests.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def inter_union(pred: np.ndarray, gt: np.ndarray):
    p, g = pred.astype(bool), gt.astype(bool)
    return (
        float((p & g).sum()), float((p | g).sum()),      # foreground
        float((~p & ~g).sum()), float((~p | ~g).sum()),  # background
    )


class Meter:
    def __init__(self):
        self.cls_i = defaultdict(float)
        self.cls_u = defaultdict(float)
        self.bg_i = self.bg_u = self.fg_i = self.fg_u = 0.0
        self.ep_iou = []

    def add(self, cls: str, pred: np.ndarray, gt: np.ndarray) -> float:
        fi, fu, bi, bu = inter_union(pred, gt)
        self.cls_i[cls] += fi
        self.cls_u[cls] += fu
        self.fg_i += fi
        self.fg_u += fu
        self.bg_i += bi
        self.bg_u += bu
        iou = fi / fu if fu > 0 else 1.0
        self.ep_iou.append(iou)
        return iou

    def summary(self) -> dict:
        per_cls = {c: self.cls_i[c] / max(self.cls_u[c], 1.0) for c in self.cls_i}
        miou = 100 * float(np.mean(list(per_cls.values()))) if per_cls else float("nan")
        fb = 100 * 0.5 * (self.fg_i / max(self.fg_u, 1) + self.bg_i / max(self.bg_u, 1))
        ep = np.array(self.ep_iou)
        return {
            "mIoU": miou,
            "FB-IoU": fb,
            "episode_mean_IoU": 100 * float(ep.mean()) if len(ep) else float("nan"),
            "n_episodes": len(ep),
            "n_classes": len(per_cls),
            "per_class_IoU": {c: 100 * v for c, v in per_cls.items()},
        }
