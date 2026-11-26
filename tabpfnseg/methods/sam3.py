"""SAM3 (Carion et al. 2025) baselines via HF transformers (facebook/sam3, gated on HF).

SAM3 is NOT natively a cross-image in-context segmenter: its image exemplars are boxes
*inside the target image*. We therefore evaluate three clearly separated usages:

  sam3_text    text prompt = class name. Zero-shot open-vocabulary reference, NOT in-context.
               Uses privileged information (the class name); also note SAM3's training data
               (SA-Co, 5.4B-pair PE pretraining) is far larger than any other method's and
               cannot be checked for overlap with the test images.
  sam3_stitch  in-context via stitching: canvas = [support_1 | ... | support_K | query];
               positive exemplar boxes are drawn around the support masks' connected
               components; SAM3 then finds 'all instances of that concept' on the canvas and
               we keep what falls in the query tile. This mirrors how SegGPT/Painter turn
               in-context learning into an image-completion problem.
  refine=sam3  post-processing for ANY method (incl. TabPFN): the coarse mask gives a box
               (+ peak point) prompt to SAM3's PVS tracker head; the refined mask replaces the
               coarse one. Applied equally to every method if enabled.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage


def mask_boxes(mask: np.ndarray, max_boxes: int = 5, min_area_frac: float = 0.002):
    lab, n = ndimage.label(mask > 0)
    if n == 0:
        return []
    areas = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    order = np.argsort(-areas)
    H, W = mask.shape
    boxes = []
    for o in order[:max_boxes]:
        if areas[o] < min_area_frac * H * W and boxes:
            break
        ys, xs = np.nonzero(lab == o + 1)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
    return boxes


def _cast_floats(inputs, dtype):
    """bf16 model on GPU: every floating input (pixels, boxes, points) must match the weights' dtype.
    original_sizes / reshaped_input_sizes are integer tensors and stay untouched."""
    for k, v in inputs.items():
        if torch.is_tensor(v) and v.is_floating_point():
            inputs[k] = v.to(dtype)
    return inputs


class _Sam3Base:
    def __init__(self, device="cuda", repo="facebook/sam3", threshold=0.4, mask_threshold=0.5):
        from transformers import Sam3Model, Sam3Processor

        self.proc = Sam3Processor.from_pretrained(repo)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = Sam3Model.from_pretrained(repo, torch_dtype=dtype).to(device).eval()
        self.device, self.dtype = device, dtype
        self.threshold, self.mask_threshold = threshold, mask_threshold

    def _run(self, image: np.ndarray, text=None, boxes=None):
        kw = {}
        if text is not None:
            kw["text"] = text
        if boxes:
            kw["input_boxes"] = [boxes]
            kw["input_boxes_labels"] = [[1] * len(boxes)]
        inputs = self.proc(images=Image.fromarray(image), return_tensors="pt", **kw).to(self.device)
        inputs = _cast_floats(inputs, self.dtype)
        with torch.no_grad():
            out = self.model(**inputs)
        res = self.proc.post_process_instance_segmentation(
            out, threshold=self.threshold, mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        H, W = image.shape[:2]
        prob = np.zeros((H, W), dtype=np.float32)
        masks, scores = res.get("masks"), res.get("scores")
        if masks is not None and len(masks):
            m = masks.float().cpu().numpy()
            s = scores.float().cpu().numpy()
            prob = (m * s[:, None, None]).max(0)  # score-weighted union of instances
        return prob


class Sam3TextSegmenter(_Sam3Base):
    name = "sam3_text"
    uses_class_name = True

    def predict(self, sup_imgs, sup_masks, q_img, class_name=None, **_):
        p = self._run(q_img, text=class_name)
        return (p > 0).astype(np.float32) * np.maximum(p, 0.5 + 1e-3), {"prompt": class_name}


class Sam3StitchSegmenter(_Sam3Base):
    name = "sam3_stitch"

    def __init__(self, tile: int = 504, **kw):
        super().__init__(**kw)
        self.tile = tile

    def predict(self, sup_imgs, sup_masks, q_img, **_):
        t = self.tile
        tiles, boxes = [], []
        for i, (im, m) in enumerate(zip(sup_imgs, sup_masks)):
            im_r = np.asarray(Image.fromarray(im).resize((t, t), Image.BILINEAR))
            m_r = np.asarray(Image.fromarray(m * 255).resize((t, t), Image.NEAREST)) > 127
            tiles.append(im_r)
            boxes += [[b[0] + i * t, b[1], b[2] + i * t, b[3]] for b in mask_boxes(m_r)]
        tiles.append(np.asarray(Image.fromarray(q_img).resize((t, t), Image.BILINEAR)))
        canvas = np.concatenate(tiles, axis=1)
        if not boxes:
            return np.zeros(q_img.shape[:2], np.float32), {"n_boxes": 0}
        p = self._run(canvas, boxes=boxes)[:, -t:]
        p = np.asarray(Image.fromarray(p.astype(np.float32)).resize(q_img.shape[1::-1], Image.BILINEAR))
        return np.where(p > 0, np.maximum(p, 0.5 + 1e-3), 0.0).astype(np.float32), {"n_boxes": len(boxes)}


class Sam3Refiner:
    """Coarse probability map -> box (+ peak point) prompt -> SAM3 tracker (PVS) mask."""

    def __init__(self, device="cuda", repo="facebook/sam3", band=False, band_frac=0.02):
        """band=True: SAM3 may only change pixels within band_frac * image size of the coarse boundary,
        so it sharpens contours but cannot add/remove whole objects (region from the in-context learner,
        boundary from SAM3)."""
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        self.band, self.band_frac = band, band_frac

        self.proc = Sam3TrackerProcessor.from_pretrained(repo)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = Sam3TrackerModel.from_pretrained(repo, torch_dtype=dtype).to(device).eval()
        self.device, self.dtype = device, dtype

    def __call__(self, q_img: np.ndarray, prob: np.ndarray) -> np.ndarray:
        coarse = prob >= 0.5
        boxes = mask_boxes(coarse, max_boxes=8)
        if not boxes:
            return prob
        inputs = self.proc(images=Image.fromarray(q_img), input_boxes=[boxes], return_tensors="pt").to(self.device)
        inputs = _cast_floats(inputs, self.dtype)
        with torch.no_grad():
            out = self.model(**inputs, multimask_output=False)
        masks = self.proc.post_process_masks(out.pred_masks.float().cpu(), inputs["original_sizes"].cpu())[0]
        refined = masks.reshape(-1, *masks.shape[-2:]).any(0).numpy()
        if self.band:
            r = max(1, round(self.band_frac * max(coarse.shape)))
            c = torch.from_numpy(coarse.astype(np.float32))[None, None]
            dil = F.max_pool2d(c, 2 * r + 1, stride=1, padding=r)[0, 0].numpy() > 0.5
            ero = -F.max_pool2d(-c, 2 * r + 1, stride=1, padding=r)[0, 0].numpy() > 0.5
            ring = dil & ~ero
            refined = np.where(ring, refined, coarse)
        # keep the coarse probability inside the refined mask, push outside below threshold
        return np.where(refined, np.maximum(prob, 0.5 + 1e-3), np.minimum(prob, 0.5 - 1e-3)).astype(np.float32)
