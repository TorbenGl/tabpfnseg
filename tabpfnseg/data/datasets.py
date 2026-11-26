"""Few-shot segmentation datasets.

Every dataset exposes the same minimal interface:

    ds.name                      -> str
    ds.classes                   -> list[str]               (classes of the requested split)
    ds.items[cls]                -> list[(img_rel, mask_rel)] (paths relative to ds.root)
    ds.load(img_rel, mask_rel)   -> (image uint8 HxWx3, mask uint8 HxW in {0,1})
    ds.prompt_name(cls)          -> human readable class name (ONLY used by text-prompted
                                    reference methods such as SAM3-text; never by in-context methods)

Layouts follow the PATNet / HSNet conventions used by the cross-domain few-shot
segmentation (CD-FSS) literature, so that the public preprocessed archives can be
used unchanged (see scripts/download_data.sh).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = REPO_ROOT / "data" / "splits"


def _load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_binary_mask(path: str) -> np.ndarray:
    m = np.asarray(Image.open(path).convert("L"))
    return (m > 127).astype(np.uint8)


@dataclass
class FewShotDataset:
    name: str
    root: str
    split: str
    classes: list[str] = field(default_factory=list)
    items: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    prompt_names: dict[str, str] = field(default_factory=dict)

    def load(self, img_rel: str, mask_rel: str) -> tuple[np.ndarray, np.ndarray]:
        img = _load_rgb(os.path.join(self.root, img_rel))
        mask = _load_binary_mask(os.path.join(self.root, mask_rel))
        if mask.shape != img.shape[:2]:  # a handful of FSS-1000 masks differ in size
            mask = np.asarray(
                Image.fromarray(mask * 255).resize((img.shape[1], img.shape[0]), Image.NEAREST)
            )
            mask = (mask > 127).astype(np.uint8)
        return img, mask

    def prompt_name(self, cls: str) -> str:
        return self.prompt_names.get(cls, cls.replace("_", " "))

    def num_images(self) -> int:
        return sum(len(v) for v in self.items.values())

    def _rel(self, p: str) -> str:
        return os.path.relpath(p, self.root)


# --------------------------------------------------------------------------- FSS-1000
def build_fss1000(root: str, split: str = "test") -> FewShotDataset:
    """FSS-1000 (Li et al. 2020) with the HSNet class split (520 trn / 240 val / 240 test).

    Expected layout: <root>/FSS-1000/<class>/<n>.jpg + <n>.png
    `val` classes are disjoint from `test` classes -> use `val` for ALL hyper-parameter
    selection, report on `test`.
    """
    base = os.path.join(root, "FSS-1000")
    split_file = SPLIT_DIR / "fss" / f"{split}.txt"
    classes = [c.strip() for c in split_file.read_text().splitlines() if c.strip()]
    ds = FewShotDataset("fss1000", root, split)
    excl_file = SPLIT_DIR / "fss" / "exclude.txt"
    exclude = set()
    if excl_file.exists():
        exclude = {ln.split("#")[0].strip() for ln in excl_file.read_text().splitlines()} - {""}
    for c in classes:
        imgs = sorted(p for p in glob.glob(os.path.join(base, c, "*.jpg"))
                      if f"{c}/{os.path.basename(p)}" not in exclude)
        pairs = [(ds._rel(p), ds._rel(p[:-4] + ".png")) for p in imgs if os.path.exists(p[:-4] + ".png")]
        if pairs:
            ds.classes.append(c)
            ds.items[c] = pairs
    return ds


# --------------------------------------------------------------------------- Deepglobe
DEEPGLOBE_NAMES = {
    "1": "urban land",
    "2": "agriculture land",
    "3": "rangeland",
    "4": "forest",
    "5": "water",
    "6": "barren land",
}


def build_deepglobe(root: str, split: str = "test") -> FewShotDataset:
    """Deepglobe land cover, PATNet preprocessing (2448^2 tiles cut into sub-images,
    one binary mask per class).

    Expected layout: <root>/Deepglobe/04_train_cat/<1..6>/test/origin/*.jpg
                                                   .../test/groundtruth/*.png
    """
    base = os.path.join(root, "Deepglobe", "04_train_cat")
    ds = FewShotDataset("deepglobe", root, split, prompt_names=DEEPGLOBE_NAMES)
    for c in sorted(DEEPGLOBE_NAMES):
        imgs = sorted(glob.glob(os.path.join(base, c, "test", "origin", "*")))
        pairs = []
        for p in imgs:
            m = p.replace("origin", "groundtruth")
            m = os.path.splitext(m)[0] + ".png"
            if os.path.exists(m):
                pairs.append((ds._rel(p), ds._rel(m)))
        if pairs:
            ds.classes.append(c)
            ds.items[c] = pairs
    return ds


# --------------------------------------------------------------------------- ISIC 2018
ISIC_NAMES = {"1": "skin lesion", "2": "skin lesion", "3": "skin lesion"}


def build_isic(root: str, split: str = "test") -> FewShotDataset:
    """ISIC-2018 task 1, PATNet layout (images grouped into 3 diagnosis classes).

    Expected layout: <root>/ISIC/ISIC2018_Task1-2_Training_Input/<1|2|3>/*.jpg
                     <root>/ISIC/ISIC2018_Task1_Training_GroundTruth/<name>_segmentation.png
    """
    base = os.path.join(root, "ISIC")
    inp = os.path.join(base, "ISIC2018_Task1-2_Training_Input")
    ds = FewShotDataset("isic", root, split, prompt_names=ISIC_NAMES)
    by_cls: dict[str, list[str]] = {}
    if os.path.isdir(os.path.join(inp, "1")):
        for c in ["1", "2", "3"]:
            by_cls[c] = sorted(glob.glob(os.path.join(inp, c, "*.jpg")))
    else:
        # flat official download: group with PATNet's data/isic/class_id.csv (copied to data/splits/isic)
        name2id = {"nevus": "1", "seborrheic_keratosis": "2", "melanoma": "3"}
        for line in (SPLIT_DIR / "isic" / "class_id.csv").read_text().splitlines()[1:]:
            img_id, name = line.strip().split(",")
            p = os.path.join(inp, img_id + ".jpg")
            if os.path.exists(p):
                by_cls.setdefault(name2id[name], []).append(p)
    for c in sorted(by_cls):
        imgs = sorted(by_cls[c])
        pairs = []
        for p in imgs:
            stem = os.path.splitext(os.path.basename(p))[0]
            m = os.path.join(base, "ISIC2018_Task1_Training_GroundTruth", stem + "_segmentation.png")
            if os.path.exists(m):
                pairs.append((ds._rel(p), ds._rel(m)))
        if pairs:
            ds.classes.append(c)
            ds.items[c] = pairs
    return ds


# --------------------------------------------------------------------------- Chest X-ray
def build_chestx(root: str, split: str = "test") -> FewShotDataset:
    """Chest X-ray lung segmentation (Shenzhen + Montgomery), PATNet layout.

    Expected layout: <root>/LungSegmentation/CXR_png/<name>.png
                     <root>/LungSegmentation/masks/<name>_mask.png  (or <name>.png)
    """
    base = os.path.join(root, "LungSegmentation")
    ds = FewShotDataset("chestx", root, split, prompt_names={"lung": "lung"})
    pairs = []
    for m in sorted(glob.glob(os.path.join(base, "masks", "*.png"))):
        stem = os.path.basename(m)[:-4]
        img_stem = stem[:-5] if stem.endswith("_mask") else stem
        p = os.path.join(base, "CXR_png", img_stem + ".png")
        if os.path.exists(p):
            pairs.append((ds._rel(p), ds._rel(m)))
    if pairs:
        ds.classes = ["lung"]
        ds.items["lung"] = pairs
    return ds


# --------------------------------------------------------------------------- underwater: SUIM-FSS
SUIM_COLORS = {  # SUIM colour code (R,G,B bits); BW (waterbody, 000) is background and excluded
    "HD": (0, 0, 1), "PF": (0, 1, 0), "WR": (0, 1, 1), "RO": (1, 0, 0),
    "RI": (1, 0, 1), "FV": (1, 1, 0), "SR": (1, 1, 1),
}
SUIM_NAMES = {"HD": "human diver", "PF": "aquatic plants and sea-grass", "WR": "wreck or ruin",
              "RO": "underwater robot", "RI": "reef and invertebrates", "FV": "fish and vertebrates",
              "SR": "sea-floor and rocks"}


class SuimDataset(FewShotDataset):
    """SUIM (Islam et al., IROS'20) as few-shot benchmark, following SUIM-FSS (FSSUWNet, 2025):
    7 classes (waterbody = background), one binary mask per (image, class), instances whose class
    covers < 10% of the image are dropped. Nothing is trained here, so both SUIM-FSS folds are test
    classes; images from train_val and TEST are pooled.

    Expected layout: <root>/SUIM/{train_val,TEST}/images/*.jpg  and  .../masks/<stem>.bmp
    """

    def __init__(self, root, split="test", min_frac=0.10):
        super().__init__("suim", root, split, prompt_names=SUIM_NAMES)
        index_file = os.path.join(root, "SUIM", f".tabpfnseg_index_{min_frac}.txt")
        if os.path.exists(index_file):
            rows = [l.split("\t") for l in open(index_file).read().splitlines() if l]
        else:
            rows = []
            for sub in ("train_val", "TEST"):
                for p in sorted(glob.glob(os.path.join(root, "SUIM", sub, "images", "*"))):
                    stem = os.path.splitext(os.path.basename(p))[0]
                    mp = os.path.join(root, "SUIM", sub, "masks", stem + ".bmp")
                    if not os.path.exists(mp):
                        continue
                    codes = self._codes(mp)
                    for c, bits in SUIM_COLORS.items():
                        if (codes == self._bits2code(bits)).mean() >= min_frac:
                            rows.append([c, self._rel(p), self._rel(mp)])
            if rows:
                with open(index_file, "w") as f:
                    f.write("\n".join("\t".join(r) for r in rows))
        for c, ip, mp in rows:
            self.items.setdefault(c, []).append((ip, f"{mp}::{c}"))
        self.classes = sorted(self.items)

    @staticmethod
    def _bits2code(b):
        return b[0] * 4 + b[1] * 2 + b[2]

    @staticmethod
    def _codes(path):
        m = np.asarray(Image.open(path).convert("RGB")) > 127
        return m[..., 0] * 4 + m[..., 1] * 2 + m[..., 2]

    def load(self, img_rel, mask_rel):
        mp, c = mask_rel.split("::")
        img = _load_rgb(os.path.join(self.root, img_rel))
        mask = (self._codes(os.path.join(self.root, mp)) == self._bits2code(SUIM_COLORS[c])).astype(np.uint8)
        if mask.shape != img.shape[:2]:
            mask = (np.asarray(Image.fromarray(mask * 255).resize(img.shape[1::-1], Image.NEAREST)) > 127).astype(np.uint8)
        return img, mask


# --------------------------------------------------------------------------- COCO-json (UIIS, UIIS10K, USIS10K, ...)
class CocoSemanticDataset(FewShotDataset):
    """Instance datasets in COCO json format -> one binary mask per (image, category) = union of
    that category's instances. Covers the underwater instance sets UIIS, UIIS10K, USIS10K.

    Expected layout: <root>/<name>/annotations.json (or set TABPFNSEG_COCO_ANN) and images under
    <root>/<name>/images/ (file_name relative to that folder).
    Pairs whose category covers < min_frac of the image are dropped (tiny targets are ill-posed
    for 1-shot and dominate noise).
    """

    def __init__(self, name, root, split="test", ann=None, img_dir=None, min_frac=0.01):
        import json

        super().__init__(name, root, split)
        base = os.path.join(root, name)
        ann = ann or os.environ.get("TABPFNSEG_COCO_ANN") or os.path.join(base, "annotations.json")
        self.img_dir = img_dir or os.path.join(base, "images")
        d = json.load(open(ann))
        self.cats = {c["id"]: c["name"] for c in d["categories"]}
        self.imgs = {im["id"]: im for im in d["images"]}
        self.anns: dict[tuple[int, int], list] = {}
        for a in d["annotations"]:
            if a.get("iscrowd", 0) and not a.get("segmentation"):
                continue
            self.anns.setdefault((a["image_id"], a["category_id"]), []).append(a)
        for (iid, cid), anns in sorted(self.anns.items()):
            im = self.imgs[iid]
            area = sum(a.get("area", 0) for a in anns)
            if area < min_frac * im["width"] * im["height"]:
                continue
            cname = self.cats[cid]
            self.items.setdefault(cname, []).append(
                (os.path.relpath(os.path.join(self.img_dir, im["file_name"]), root), f"coco::{iid}::{cid}"))
        self.classes = sorted(c for c, v in self.items.items() if len(v) >= 2)
        self.prompt_names = {c: c.replace("_", " ") for c in self.classes}

    def load(self, img_rel, mask_rel):
        from PIL import ImageDraw

        _, iid, cid = mask_rel.split("::")
        im = self.imgs[int(iid)]
        img = _load_rgb(os.path.join(self.root, img_rel))
        H, W = im["height"], im["width"]
        canvas = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(canvas)
        mask = np.zeros((H, W), bool)
        for a in self.anns[(int(iid), int(cid))]:
            seg = a["segmentation"]
            if isinstance(seg, list):
                for poly in seg:
                    if len(poly) >= 6:
                        draw.polygon([tuple(poly[i:i + 2]) for i in range(0, len(poly), 2)], fill=1)
            else:  # RLE
                from pycocotools import mask as mask_utils

                rle = mask_utils.frPyObjects(seg, H, W) if isinstance(seg["counts"], list) else seg
                mask |= mask_utils.decode(rle).astype(bool)
        mask |= np.asarray(canvas, bool)
        mask = mask.astype(np.uint8)
        if mask.shape != img.shape[:2]:
            mask = (np.asarray(Image.fromarray(mask * 255).resize(img.shape[1::-1], Image.NEAREST)) > 127).astype(np.uint8)
        return img, mask


# --------------------------------------------------------------------------- generic folder
def build_folder(root: str, split: str = "test") -> FewShotDataset:
    """Generic layout for your own data: <root>/<class>/images/*.{jpg,png} + <root>/<class>/masks/<same stem>.png"""
    ds = FewShotDataset("folder", root, split)
    for c in sorted(os.listdir(root)):
        idir, mdir = os.path.join(root, c, "images"), os.path.join(root, c, "masks")
        if not os.path.isdir(idir):
            continue
        pairs = []
        for p in sorted(glob.glob(os.path.join(idir, "*"))):
            m = os.path.join(mdir, os.path.splitext(os.path.basename(p))[0] + ".png")
            if os.path.exists(m):
                pairs.append((ds._rel(p), ds._rel(m)))
        if pairs:
            ds.classes.append(c)
            ds.items[c] = pairs
    return ds


# --------------------------------------------------------------------------- synthetic (smoke tests)
class SyntheticDataset(FewShotDataset):
    """Procedurally generated textured shapes. No files on disk; used for CPU smoke tests.

    Class k = a shape with a class-specific texture on a random textured background,
    so that the task is solvable from local appearance features.
    """

    def __init__(self, n_classes: int = 4, per_class: int = 6, size: int = 160, seed: int = 0, split: str = "test"):
        super().__init__("synthetic", "<memory>", split)
        self.size, self.seed = size, seed
        self.classes = [f"cls{k}" for k in range(n_classes)]
        self.items = {c: [(f"{c}/{i}", f"{c}/{i}") for i in range(per_class)] for c in self.classes}

    def load(self, img_rel: str, mask_rel: str):
        c, i = img_rel.split("/")
        k, i = int(c[3:]), int(i)
        rng = np.random.default_rng([self.seed, k, i])
        s = self.size
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        bg = rng.uniform(0, 1, 3)[None, None] * 0.5 + 0.25 * np.sin(xx / rng.uniform(3, 9) + yy / rng.uniform(3, 9))[..., None]
        cy, cx = rng.uniform(0.3, 0.7, 2) * s
        r = rng.uniform(0.15, 0.3) * s
        if k % 2 == 0:
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r**2
        else:
            mask = (np.abs(yy - cy) < r) & (np.abs(xx - cx) < r * 0.7)
        freq = 1.5 + k
        tex = 0.5 + 0.5 * np.sign(np.sin(freq * xx / 4.0) * np.sin(freq * yy / 4.0))
        col = np.array([(k * 0.37) % 1, (k * 0.61 + 0.3) % 1, (k * 0.13 + 0.6) % 1])
        fg = col[None, None] * (0.6 + 0.4 * tex[..., None])
        img = np.where(mask[..., None], fg, bg) + rng.normal(0, 0.04, (s, s, 3))
        return (np.clip(img, 0, 1) * 255).astype(np.uint8), mask.astype(np.uint8)


DATASETS = {
    "fss1000": build_fss1000,
    "deepglobe": build_deepglobe,
    "isic": build_isic,
    "chestx": build_chestx,
    "folder": build_folder,
    "suim": lambda root, split="test": SuimDataset(root, split),
    # UWS (Kabir et al. IROS'23, 21 underwater animals): use the generic folder layout under <root>/UWS
    "uws": lambda root, split="test": _named(build_folder(os.path.join(root, "UWS"), split), "uws", root),
    "uiis": lambda root, split="test": CocoSemanticDataset("UIIS", root, split),
    "uiis10k": lambda root, split="test": CocoSemanticDataset("UIIS10K", root, split),
    "usis10k": lambda root, split="test": CocoSemanticDataset("USIS10K", root, split),
}


def _named(ds, name, root):
    """re-root a folder dataset so paths stay relative to the global data root"""
    ds.items = {c: [(os.path.relpath(os.path.join(ds.root, i), root), os.path.relpath(os.path.join(ds.root, m), root))
                    for i, m in v] for c, v in ds.items.items()}
    ds.root, ds.name = root, name
    return ds

# Default number of episodes per the CD-FSS protocol (PATNet and follow-ups).
# FSS-1000: every test image is used as query exactly once (240 classes x 10 = 2400).
DEFAULT_EPISODES = {"fss1000": 2400, "deepglobe": 1200, "isic": 1200, "chestx": 1200, "synthetic": 40,
                    "suim": 1200, "uws": 1200, "uiis": 1200, "uiis10k": 1200, "usis10k": 1200}


def build_dataset(name: str, root: str = "datasets", split: str = "test") -> FewShotDataset:
    if name == "synthetic":
        return SyntheticDataset(split=split)
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name}; choose from {sorted(DATASETS) + ['synthetic']}")
    ds = DATASETS[name](root, split)
    if not ds.classes:
        raise FileNotFoundError(f"No images found for {name} under {root}. See scripts/download_data.sh")
    return ds
