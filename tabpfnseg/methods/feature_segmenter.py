"""Feature-table segmenter: backbone features -> table -> in-context head -> mask.

Pipeline per episode (identical for TabPFN and all feature baselines):
  1. features for each support and the query  (FeatureBuilder; frozen backbone)
  2. support cell labels = mask area fraction on the working grid
  3. context selection (all / uniform / balanced / boundary, capped at max_ctx)
  4. optional PCA, fit on the CONTEXT rows only (never on query rows / dataset statistics)
  5. head.fit_predict(context, query rows) -> p(fg) per query cell
  6. optional prior correction if the context class balance was altered by sampling
  7. bilinear upsampling of p(fg) to the query's original resolution, threshold 0.5
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from ..features.builder import FeatureBuilder, select_context


def pca_fit_transform(Xc: np.ndarray, Xq: np.ndarray, k: int, device: str):
    """PCA fitted on context rows only."""
    if k <= 0 or k >= min(Xc.shape):
        return Xc, Xq
    A = torch.from_numpy(Xc).float().to(device)
    mu = A.mean(0, keepdim=True)
    U, S, V = torch.pca_lowrank(A - mu, q=k, center=False, niter=4)
    P = V[:, :k]
    Q = torch.from_numpy(Xq).float().to(device)
    return ((A - mu) @ P).cpu().numpy(), ((Q - mu) @ P).cpu().numpy()


def prior_correct(p: np.ndarray, sampled_rate: float, true_rate: float, eps: float = 1e-4) -> np.ndarray:
    """Bayes prior shift: the head saw class prior `sampled_rate`; the support images have `true_rate`."""
    if abs(sampled_rate - true_rate) < 1e-6 or not (0 < sampled_rate < 1) or not (0 < true_rate < 1):
        return p
    p = np.clip(p, eps, 1 - eps)
    odds = p / (1 - p) * (true_rate / (1 - true_rate)) / (sampled_rate / (1 - sampled_rate))
    return odds / (1 + odds)


def within_class_whiten(Xc: np.ndarray, yc: np.ndarray, Xq: np.ndarray, lam: float, device: str):
    """FROST-style shrinkage whitening, fit on the CONTEXT rows only.
    Sigma_W = within-class scatter of fg and bg (each centred on its own mean, so the fg-bg mean difference
    survives), shrunk towards mu*I:  Sigma = (1-lam) Sigma_W + lam mu I.  Returns X Sigma^{-1/2}."""
    y = yc >= 0.5
    if y.all() or (~y).all():
        return Xc, Xq
    dev = device if torch.cuda.is_available() or device == "cpu" else "cpu"
    Xc_t = torch.from_numpy(Xc).float().to(dev)
    Xq_t = torch.from_numpy(Xq).float().to(dev)
    yt = torch.from_numpy(y).to(dev)
    A = torch.cat([Xc_t[yt] - Xc_t[yt].mean(0), Xc_t[~yt] - Xc_t[~yt].mean(0)])  # N x D
    N, D = A.shape
    denom = max(N - 2, 1)
    mu = float((A ** 2).sum()) / (denom * D)
    if N <= D:  # eigendecomposition of the small N x N Gram matrix
        e, Ug = torch.linalg.eigh(A @ A.T / denom)
        keep = e > 1e-8 * e.max()
        e, Ug = e[keep], Ug[:, keep]
        V = (A.T @ Ug) / torch.sqrt(e * denom)[None, :]       # D x k orthonormal
    else:
        e, V = torch.linalg.eigh(A.T @ A / denom)
    c0 = (lam * mu) ** -0.5
    d = ((1 - lam) * e + lam * mu) ** -0.5

    def apply(X):
        P = X @ V
        return c0 * X + (P * (d - c0)[None, :]) @ V.T

    return apply(Xc_t).cpu().numpy(), apply(Xq_t).cpu().numpy()


class FeatureSegmenter:
    def __init__(self, builder: FeatureBuilder, head, args):
        self.fb, self.head, self.args = builder, head, args
        self.rng = np.random.default_rng(args.seed)
        ab = args.ambig
        self.ambig = tuple(float(x) for x in ab.split(",")) if ab else None

    @property
    def name(self):
        return self.head.name

    def predict(self, sup_imgs, sup_masks, q_img, sup_keys=None, q_key=None, q_mask_oracle=None, **_):
        a = self.args
        t0 = time.time()
        if getattr(a, "support_flip", 0):  # FROST: horizontal flip of every support doubles the context
            keys = list(sup_keys) if sup_keys else [None] * len(sup_imgs)
            sup_imgs = list(sup_imgs) + [np.ascontiguousarray(im[:, ::-1]) for im in sup_imgs]
            sup_masks = list(sup_masks) + [np.ascontiguousarray(m[:, ::-1]) for m in sup_masks]
            sup_keys = keys + [None if k is None else f"{k}#hflip" for k in keys]
        feats = [self.fb.features(im, k) for im, k in zip(sup_imgs, sup_keys or [None] * len(sup_imgs))]
        fq = self.fb.features(q_img, q_key)
        fracs = [self.fb.mask_fraction(m).numpy() for m in sup_masks]
        D, H, W = fq.shape

        idx, sampled_rate, true_rate = select_context(fracs, a.sampling, a.max_ctx, self.rng, self.ambig)
        Xc = torch.cat([f.reshape(D, -1)[:, torch.from_numpy(i).to(f.device)].T for f, i in zip(feats, idx)]).cpu().numpy()
        soft = np.concatenate([fr.reshape(-1)[i] for fr, i in zip(fracs, idx)]).astype(np.float32)
        yc = soft if getattr(self.head, "uses_soft_labels", False) else (soft >= 0.5).astype(np.float32)
        Xq = fq.reshape(D, -1).T.cpu().numpy()
        # optional synthetic "anchor" images (all black / all white) added to the context as background
        n_anchor = 0
        if getattr(a, "anchors", "none") != "none":
            vals = {"black": [0], "white": [255], "bw": [0, 255]}[a.anchors]
            for v in vals:
                img = np.full((a.img_size, a.img_size, 3), v, np.uint8)
                fa = self.fb.features(img, f"__anchor_{v}__").reshape(D, -1).T.cpu().numpy()
                Xc = np.concatenate([Xc, fa])
                yc = np.concatenate([yc, np.zeros(len(fa), np.float32)])
                n_anchor += len(fa)
            sampled_rate = float((yc >= 0.5).mean())  # prior correction maps back to the support's own rate
        if getattr(a, "whiten", 0):
            Xc, Xq = within_class_whiten(Xc, yc, Xq, a.whiten, a.device)
        t_feat = time.time() - t0

        Xc, Xq = pca_fit_transform(Xc, Xq, a.pca, a.device)
        t1 = time.time()
        p = self.head.fit_predict(Xc.astype(np.float32), yc, Xq.astype(np.float32))
        t_head = time.time() - t1
        if a.prior_correct:
            p = prior_correct(p, sampled_rate, true_rate)
        prob = self.to_mask_res(p.reshape(H, W), q_img.shape[:2])
        stats = {"n_ctx": int(len(yc)), "n_anchor": n_anchor, "ctx_fg_rate": float(sampled_rate), "dim": int(Xc.shape[1]),
                 "n_query_rows": int(len(Xq)), "t_feat": t_feat, "t_head": t_head}
        return prob, stats

    @staticmethod
    def to_mask_res(p_grid: np.ndarray, hw) -> np.ndarray:
        t = torch.from_numpy(np.ascontiguousarray(p_grid)).float()[None, None]
        return F.interpolate(t, size=hw, mode="bilinear", align_corners=False)[0, 0].numpy()


class OracleGrid:
    """Upper bound of a given working resolution: the *ground-truth* query mask, area-pooled
    to the working grid and bilinearly upsampled back. Not a method -- it tells you how much
    IoU is lost purely by predicting on a patch / sub-patch grid. Uses the query mask (by design)."""

    name = "oracle_grid"
    needs_query_mask = True

    def __init__(self, builder: FeatureBuilder):
        self.fb = builder

    def predict(self, sup_imgs, sup_masks, q_img, sup_keys=None, q_key=None, q_mask_oracle=None, **_):
        frac = self.fb.mask_fraction(q_mask_oracle).numpy()
        return FeatureSegmenter.to_mask_res(frac, q_img.shape[:2]), {}
