"""Frozen ViT backbones returning per-layer patch-token grids.

Backbones are loaded through HF transformers so that intermediate layers are available
via `output_hidden_states`. Prefix tokens (CLS + registers) are stripped.

Aliases (all frozen, no fine-tuning anywhere in this repo):
  dinov2_l    facebook/dinov2-large                    ViT-L/14, 24 blocks, 1024-d
  dinov2r_l   facebook/dinov2-with-registers-large     ViT-L/14 + 4 registers (cleaner dense maps)
  dinov2_g    facebook/dinov2-giant                    ViT-g/14, 40 blocks, 1536-d
  dinov2_b    facebook/dinov2-base                     ViT-B/14, 12 blocks, 768-d
  dinov3_l    facebook/dinov3-vitl16-pretrain-lvd1689m ViT-L/16, 24 blocks, 1024-d (gated on HF)
  dinov3_b    facebook/dinov3-vitb16-pretrain-lvd1689m ViT-B/16 (gated on HF)
  dinov3_h    facebook/dinov3-vith16plus-pretrain-lvd1689m ViT-H+/16, 1280-d (gated on HF)
All experiments use DINOv3 (dinov3_l default, dinov3_h as the bigger-backbone check); DINOv2 aliases are kept
only for comparison with older work.
  dummy       random patch embedding + pixel statistics, for CPU smoke tests only
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALIASES = {
    "dinov2_s": "facebook/dinov2-small",
    "dinov2_b": "facebook/dinov2-base",
    "dinov2_l": "facebook/dinov2-large",
    "dinov2_g": "facebook/dinov2-giant",
    "dinov2r_l": "facebook/dinov2-with-registers-large",
    "dinov2r_g": "facebook/dinov2-with-registers-giant",
    "dinov3_b": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3_h": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Backbone(nn.Module):
    patch_size: int
    num_layers: int
    dim: int

    def preprocess(self, img: np.ndarray, size: int) -> torch.Tensor:
        """uint8 HxWx3 -> normalized 1x3xSxS (square resize, the standard FSS protocol)."""
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        return (x - IMAGENET_MEAN) / IMAGENET_STD

    def forward_layers(self, x: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
        raise NotImplementedError


class HFViT(Backbone):
    def __init__(self, name: str, device: str = "cuda", dtype=torch.bfloat16, apply_norm: bool = True):
        super().__init__()
        from transformers import AutoModel

        repo = ALIASES.get(name, name)
        self.model = AutoModel.from_pretrained(repo, torch_dtype=dtype).to(device).eval()
        cfg = self.model.config
        self.patch_size = cfg.patch_size
        self.num_layers = cfg.num_hidden_layers
        self.dim = cfg.hidden_size
        self.n_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.device, self.dtype, self.apply_norm = device, dtype, apply_norm
        self.norm = getattr(self.model, "layernorm", None) or getattr(self.model, "norm", None)

    @torch.no_grad()
    def forward_layers(self, x, layers):
        x = x.to(self.device, self.dtype)
        h, w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        out = self.model(pixel_values=x, output_hidden_states=True)
        hs = out.hidden_states  # (embeddings, block1, ..., blockN)
        res = {}
        for l in layers:
            l_abs = l if l > 0 else self.num_layers + 1 + l  # allow negative indexing (-1 = last)
            t = hs[l_abs][:, self.n_prefix :]
            if self.apply_norm and self.norm is not None:
                t = self.norm(t)
            assert t.shape[1] == h * w, (t.shape, h, w)
            res[l] = t[0].transpose(0, 1).reshape(-1, h, w).float()
        return res


class DummyBackbone(Backbone):
    """Deterministic random-projection 'backbone' (no download). Smoke tests only."""

    def __init__(self, device="cpu", dim=64, patch_size=8, num_layers=4):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.patch_size, self.num_layers, self.dim, self.device = patch_size, num_layers, dim, device
        self.projs = [torch.randn(3 * patch_size * patch_size, dim, generator=g) / patch_size for _ in range(num_layers)]

    @torch.no_grad()
    def forward_layers(self, x, layers):
        p = self.patch_size
        h, w = x.shape[-2] // p, x.shape[-1] // p
        patches = F.unfold(x.float(), kernel_size=p, stride=p)[0].T  # (hw, 3pp)
        res = {}
        for l in layers:
            l_abs = l if l > 0 else self.num_layers + 1 + l
            f = torch.tanh(patches @ self.projs[l_abs - 1])
            res[l] = f.T.reshape(self.dim, h, w).to(self.device)
        return res


def build_backbone(name: str, device: str = "cuda") -> Backbone:
    if name == "dummy":
        return DummyBackbone(device=device)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    return HFViT(name, device=device, dtype=dtype)
