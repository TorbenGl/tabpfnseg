"""Turn an image (+ mask) into a table: one row per spatial location, one column per feature.

This is the core design space of the "is TabPFN an in-context segmenter?" question:

  WHICH features per row  (FeatureSpec)
    layers : which ViT blocks to concatenate, e.g. "24" or "4,8,12,16,20,24" (6 x 1024 = 6144 for ViT-L)
    pools  : local-context gathering around each patch, applied to every layer:
             1 = the patch itself, k = k x k mean pool (stride 1) centred on the patch,
             g = global mean of the image's own tokens (image context)
             e.g. layers=24 pools=1,3,5,9,17,g  -> 6 x 1024 = 6144 ("patch + area around it")
    rgb    : append mean Lab colour of the cell (+ its local std) -- cheap high-res cue
    l2     : L2-normalise every (layer,pool) block (keeps blocks on the same scale)

  AT WHICH RESOLUTION rows live  (ResSpec)
    patch  : one row per ViT patch (e.g. 37x37 for 518px / 14). Labels = mask area fraction >= .5
    xN     : features upsampled to N x the patch grid (bilinear | nearest | jbu = joint bilateral
             upsampling guided by the RGB image). Labels = mask area fraction on that grid.

  WHICH rows go into the context  (sampling)
    all | uniform | balanced | boundary  with a cap `max_ctx` (rows over all K supports).
    For non-uniform sampling the class prior seen by the in-context learner differs from the
    true prior -> `prior_correct` re-weights the predicted odds back (see methods/feature_methods).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- specs
@dataclass
class FeatureSpec:
    layers: list[int] = field(default_factory=lambda: [-1])
    pools: list[str] = field(default_factory=lambda: ["1"])
    rgb: bool = False
    l2: bool = True

    @classmethod
    def parse(cls, s: str) -> "FeatureSpec":
        """'layers=4,8,12,16,20,24;pools=1,3;rgb=1;l2=1'"""
        spec = cls()
        if not s:
            return spec
        for part in s.split(";"):
            if not part.strip():
                continue
            k, v = part.split("=")
            k = k.strip()
            if k == "layers":
                spec.layers = [int(x) for x in v.split(",")]
            elif k == "pools":
                spec.pools = [x.strip() for x in v.split(",")]
            elif k == "rgb":
                spec.rgb = bool(int(v))
            elif k == "l2":
                spec.l2 = bool(int(v))
            else:
                raise ValueError(f"unknown feature key {k}")
        return spec

    def dim(self, backbone_dim: int) -> int:
        return len(self.layers) * len(self.pools) * backbone_dim + (6 if self.rgb else 0)


@dataclass
class ResSpec:
    mode: str = "patch"  # patch | xN
    factor: int = 1
    up: str = "bilinear"  # bilinear | nearest | jbu

    @classmethod
    def parse(cls, res: str, up: str = "bilinear") -> "ResSpec":
        if res == "patch":
            return cls("patch", 1, up)
        if res.startswith("x"):
            return cls("up", int(res[1:]), up)
        raise ValueError(res)


# --------------------------------------------------------------------------- colour
def rgb_to_lab(x: torch.Tensor) -> torch.Tensor:
    """x: 3xHxW in [0,1] -> Lab scaled to roughly [0,1]."""
    x = torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)
    M = torch.tensor([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], device=x.device)
    xyz = torch.einsum("ij,jhw->ihw", M, x) / torch.tensor([0.9505, 1.0, 1.089], device=x.device).view(3, 1, 1)
    f = torch.where(xyz > 0.008856, xyz.clamp_min(1e-8) ** (1 / 3), 7.787 * xyz + 16 / 116)
    L = 116 * f[1] - 16
    a = 500 * (f[0] - f[1])
    b = 200 * (f[1] - f[2])
    return torch.stack([L / 100, (a + 128) / 255, (b + 128) / 255])


# --------------------------------------------------------------------------- upsampling
def jbu_upsample(feat: torch.Tensor, guide_hr: torch.Tensor, out_hw: tuple[int, int],
                 radius: int = 1, sigma_spatial: float = 0.75, sigma_range: float = 0.12) -> torch.Tensor:
    """Joint bilateral upsampling (Kopf et al. 2007), training-free.

    feat     : C x h x w low-res features
    guide_hr : 3 x H x W guide image (Lab, [0,1]) at the output resolution
    Each output pixel = normalised sum over the (2r+1)^2 nearest low-res cells of
        exp(-d_spatial^2 / 2 s_s^2) * exp(-||guide(p) - guide_lr(q)||^2 / 2 s_r^2) * feat(q)
    so features "snap" to colour edges instead of being blurred across them.
    """
    C, h, w = feat.shape
    H, W = out_hw
    dev = feat.device
    guide_hr = F.interpolate(guide_hr[None], size=(H, W), mode="bilinear", align_corners=False, antialias=True)[0]
    guide_lr = F.interpolate(guide_hr[None], size=(h, w), mode="area")[0]
    # continuous low-res coordinates of output pixel centres
    uy = (torch.arange(H, device=dev) + 0.5) * h / H - 0.5
    ux = (torch.arange(W, device=dev) + 0.5) * w / W - 0.5
    cy, cx = uy.round().long(), ux.round().long()
    # log-weights first, then subtract the per-pixel max before exp: with a sharp colour edge all
    # (2r+1)^2 range weights can underflow to 0 in float32, which would zero the feature there.
    logw, idx = [], []
    for dy in range(-radius, radius + 1):
        qy = (cy + dy).clamp(0, h - 1)
        ly = -((uy - qy) ** 2) / (2 * sigma_spatial**2)
        for dx in range(-radius, radius + 1):
            qx = (cx + dx).clamp(0, w - 1)
            lx = -((ux - qx) ** 2) / (2 * sigma_spatial**2)
            g_q = guide_lr[:, qy][:, :, qx]  # 3xHxW
            lr = -((guide_hr - g_q) ** 2).sum(0) / (2 * sigma_range**2)
            logw.append(ly[:, None] + lx[None, :] + lr)
            idx.append((qy, qx))
    m = torch.stack(logw).amax(0)
    num = torch.zeros(C, H, W, device=dev, dtype=feat.dtype)
    den = torch.zeros(1, H, W, device=dev, dtype=feat.dtype)
    for lw, (qy, qx) in zip(logw, idx):
        wgt = torch.exp(lw - m)[None].to(feat.dtype)  # max weight per pixel is exactly 1 -> den >= 1
        num += wgt * feat[:, qy][:, :, qx]
        den += wgt
    return num / den


def upsample(feat: torch.Tensor, out_hw, mode: str, guide: torch.Tensor | None = None) -> torch.Tensor:
    if feat.shape[-2:] == tuple(out_hw):
        return feat
    if mode == "jbu":
        return jbu_upsample(feat, guide, out_hw)
    if mode == "nearest":
        return F.interpolate(feat[None], size=out_hw, mode="nearest")[0]
    return F.interpolate(feat[None], size=out_hw, mode="bilinear", align_corners=False)[0]


# --------------------------------------------------------------------------- builder
class FeatureBuilder:
    def __init__(self, backbone, spec: FeatureSpec, res: ResSpec, img_size: int, cache_size: int = 64):
        self.bb, self.spec, self.res, self.img_size = backbone, spec, res, img_size
        assert img_size % backbone.patch_size == 0, "img_size must be a multiple of the patch size"
        self.grid = img_size // backbone.patch_size
        self.out_hw = (self.grid * res.factor, self.grid * res.factor)
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_order: list[str] = []
        self.cache_size = cache_size

    @property
    def dim(self) -> int:
        return self.spec.dim(self.bb.dim)

    @torch.no_grad()
    def features(self, img: np.ndarray, key: str | None = None) -> torch.Tensor:
        """-> D x H' x W' feature grid (on the backbone device)."""
        if key is not None and key in self._cache:
            return self._cache[key]
        x = self.bb.preprocess(img, self.img_size)
        layer_maps = self.bb.forward_layers(x, self.spec.layers)
        img01 = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(self.bb.device)
        guide = rgb_to_lab(F.interpolate(img01[None], size=(self.img_size, self.img_size), mode="bilinear",
                                         align_corners=False, antialias=True)[0])
        blocks = []
        for l in self.spec.layers:
            f = layer_maps[l]  # C x g x g
            for p in self.spec.pools:
                if p == "1":
                    b = f
                elif p == "g":
                    b = f.mean(dim=(1, 2), keepdim=True).expand_as(f)
                else:
                    k = int(p)
                    b = F.avg_pool2d(f[None], k, stride=1, padding=k // 2, count_include_pad=False)[0]
                if self.spec.l2:
                    b = F.normalize(b, dim=0)
                blocks.append(b)
        feat = torch.cat(blocks, 0)
        feat = upsample(feat, self.out_hw, self.res.up, guide)
        if self.spec.rgb:
            lab = F.interpolate(guide[None], size=self.out_hw, mode="area")[0]
            lab2 = F.interpolate((guide**2)[None], size=self.out_hw, mode="area")[0]
            std = (lab2 - lab**2).clamp_min(0).sqrt()
            feat = torch.cat([feat, lab, std], 0)
        if key is not None:
            self._cache[key] = feat
            self._cache_order.append(key)
            if len(self._cache_order) > self.cache_size:
                self._cache.pop(self._cache_order.pop(0), None)
        return feat

    def mask_fraction(self, mask: np.ndarray) -> torch.Tensor:
        """Fraction of foreground pixels inside each grid cell (H' x W'), via area pooling."""
        m = torch.from_numpy(mask).float()[None, None]
        m = F.interpolate(m, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return F.adaptive_avg_pool2d(m, self.out_hw)[0, 0]


# --------------------------------------------------------------------------- context sampling
def boundary_weight(frac: np.ndarray, sigma: float = 2.0, lam: float = 4.0) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    hard = frac >= 0.5
    if hard.all() or (~hard).all():
        return np.ones_like(frac)
    d = np.minimum(distance_transform_edt(hard), distance_transform_edt(~hard))
    return 1.0 + lam * np.exp(-(d**2) / (2 * sigma**2))


def select_context(fracs: list[np.ndarray], strategy: str, max_ctx: int, rng: np.random.Generator,
                   ambig: tuple[float, float] | None = None):
    """Choose which support cells become context rows.

    Returns (list of flat index arrays per support, sampled_fg_rate, true_fg_rate)
    """
    n_sup = len(fracs)
    flat = [f.reshape(-1) for f in fracs]
    labels = [(f >= 0.5) for f in flat]
    valid = []
    for f in flat:
        v = np.ones_like(f, dtype=bool)
        if ambig is not None:  # drop mixed cells (e.g. 0.2 < frac < 0.8): cleaner labels, fewer boundary rows
            v = (f <= ambig[0]) | (f >= ambig[1])
        valid.append(v)
    total = sum(int(v.sum()) for v in valid)
    true_rate = float(np.concatenate([l[v] for l, v in zip(labels, valid)]).mean()) if total else 0.0
    if total <= max_ctx and strategy in ("all", "uniform"):
        idx = [np.nonzero(v)[0] for v in valid]
        return idx, true_rate, true_rate
    if strategy == "all":  # cap exceeded -> uniform subsample keeps the true prior
        strategy = "uniform"

    per = max_ctx // n_sup
    out = []
    for f, l, v, fr in zip(flat, labels, valid, fracs):
        cand = np.nonzero(v)[0]
        if len(cand) <= per:
            out.append(cand)
            continue
        if strategy == "uniform":
            out.append(rng.choice(cand, per, replace=False))
        elif strategy == "balanced":
            fg, bg = cand[l[cand]], cand[~l[cand]]
            n_fg = min(len(fg), per // 2)
            n_bg = min(len(bg), per - n_fg)
            n_fg = min(len(fg), per - n_bg)
            out.append(np.concatenate([rng.choice(fg, n_fg, replace=False) if n_fg else fg[:0],
                                       rng.choice(bg, n_bg, replace=False) if n_bg else bg[:0]]))
        elif strategy == "boundary":
            w = boundary_weight(fr).reshape(-1)[cand]
            out.append(rng.choice(cand, per, replace=False, p=w / w.sum()))
        else:
            raise ValueError(strategy)
    sampled = np.concatenate([l[i] for l, i in zip(labels, out)])
    return out, float(sampled.mean()) if len(sampled) else 0.0, true_rate
