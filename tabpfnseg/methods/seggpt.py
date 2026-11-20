"""SegGPT (Wang et al. 2023) via HF transformers, BAAI/seggpt-vit-large.

K-shot uses the paper's *feature ensemble* (query features averaged over the K prompt
pairs after each attention layer): HF implements it as `feature_ensemble=True` with the
query image repeated K times in the batch.

LEAKAGE NOTE: SegGPT was trained on COCO, ADE20K, PASCAL VOC, Cityscapes, LIP, PACO,
iSAID + loveDA (aerial) and CHASE/DRIVE/HRF/STARE (retina). COCO-20i / PASCAL-5i are
therefore NOT held-out for SegGPT, and Deepglobe is domain-adjacent (loveDA has similar
land-cover classes). FSS-1000, ISIC and Chest X-ray are not in its training data.
"""
from __future__ import annotations

import numpy as np
import torch


class SegGPTSegmenter:
    name = "seggpt"

    def __init__(self, device="cuda", repo="BAAI/seggpt-vit-large"):
        from transformers import SegGptForImageSegmentation, SegGptImageProcessor

        self.proc = SegGptImageProcessor.from_pretrained(repo)
        dtype = torch.float32
        self.model = SegGptForImageSegmentation.from_pretrained(repo, torch_dtype=dtype).to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict(self, sup_imgs, sup_masks, q_img, **_):
        k = len(sup_imgs)
        inputs = self.proc(
            images=[q_img] * k,
            prompt_images=list(sup_imgs),
            prompt_masks=[m.astype(np.uint8) for m in sup_masks],
            num_labels=1,
            return_tensors="pt",
        )
        inputs = {kk: v.to(self.device) for kk, v in inputs.items()}
        out = self.model(**inputs, feature_ensemble=k > 1)
        res = self.proc.post_process_semantic_segmentation(
            out, target_sizes=[q_img.shape[:2]] * k, num_labels=1, return_segmentation_scores=True
        )[0]
        s = res.segmentation_scores  # (2,H,W) negative squared distance to palette colours (bg, fg)
        # convert to a pseudo-probability; threshold 0.5 == argmin distance (the official decoding)
        prob = torch.sigmoid((s[1] - s[0]) / max(float((s[1] - s[0]).abs().std()), 1e-6)).float().cpu().numpy()
        return prob, {}
