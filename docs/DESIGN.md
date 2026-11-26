# Is TabPFN‑3.5 an in‑context segmenter? Design, protocol and discussion

## 1. The question, made testable

An **in‑context segmenter** takes K annotated examples (image, mask) plus a query image and
outputs the query mask **without any parameter update and without having been trained for this
task on the target classes**. SegGPT is one (it was meta-trained to do this on segmentation data).
SAM3 is *not* one across images, because its exemplars are boxes inside the target image (see §4).

TabPFN‑3.5 is an in‑context learner for **tables**: it was meta‑trained only on synthetic tabular
tasks and solves a new task in one forward pass, conditioned on labelled rows. We turn
segmentation into a table:

```
row     = one spatial location (a ViT patch, or a sub-patch cell)
columns = frozen-backbone features of that location (+ optional context / colour features)
label   = foreground / background from the support mask
context = rows of the K support images     query rows = all locations of the query image
```

So "TabPFN as a segmenter" = **frozen ViT → table → TabPFN in context → mask**. Nothing is trained.
The question splits into two sub-questions, which the experiments keep apart:

| question | comparison (same episodes, same features) |
|---|---|
| **Q1** Does TabPFN add anything *as the in-context learner* over simple per-episode learners? | `tabpfn` vs `logreg` / `knn` / `proto` on the **identical** feature table |
| **Q2** Is the full system competitive with dedicated in-context segmenters? | best `tabpfn` system vs `seggpt`, `sam3_stitch` (and `sam3_text` as a non-ICL reference) |

If Q1 is "no", TabPFN is just an expensive classifier on top of DINO. If Q1 is "yes" but Q2 is "no",
the bottleneck is the representation or the resolution (see §3 and `oracle_grid`).

## 2. Evaluation protocol (what the others do) and data leakage

### How in-context / few-shot segmentation is evaluated
* **Episodes**: 1-way K-shot. For each episode we pick a class, a query image and K other support
  images of the same class (HSNet/PATNet sampling). K ∈ {1, 5}.
* **Metric**: class-wise **mIoU**. Foreground intersection and union are summed over all episodes of a
  class, giving IoU_c, and mIoU is the mean over classes. **FB-IoU** (mean of fg and bg IoU) is secondary.
  SegGPT reports both on FSS-1000.
* **Benchmarks** in the literature:
  * COCO‑20i / PASCAL‑5i: 4 folds, 1000 episodes per fold. These are **not usable here**, because SegGPT
    trained on COCO and VOC (its own paper marks this as in-domain) and SAM3 trained on COCO and LVIS.
  * FSS‑1000: 240 test classes × 10 images gives 2400 episodes. SegGPT reports 85.6 (1-shot) and 89.3
    (few-shot) mIoU here, so it is the main common ground.
  * **CD‑FSS** (PATNet, ECCV'22, the standard for "unseen domain"): Deepglobe (satellite, 6 classes),
    ISIC‑2018 (skin lesions, 3 classes), Chest X‑ray (lungs). 1200 episodes each, 1- and 5-shot, seeds
    averaged. Recent training-free DINOv3 methods reach about 68 mIoU on the 4-dataset average
    (1-shot), and Matcher (DINOv2 + SAM) about 62.
* **Underwater** (added): **SUIM‑FSS**, following FSSUWNet (2025). SUIM has 7 classes (waterbody is
  background), and class instances covering less than 10% of the image are dropped. We run no training,
  so all 7 classes are test classes. **UWS** (Kabir et al., IROS'23) has 576 images of 21 underwater animals;
  its 4-fold cross-validation reduces to "all classes are test" here as well.
  **UIIS / UIIS10K / USIS10K** use a COCO‑json loader that unions the instances of one category into a binary mask.
  Underwater imagery is hard for frozen backbones because of colour cast, haze and low contrast. That makes
  it a good stress test of whether multi-layer and low-level features (`ml6`, `rgb=1`) help TabPFN.
  Published SUIM‑FSS numbers are about 43/49 mIoU (1/5-shot, trained FSSUWNet).
* **Resolution**: images are resized to a square input. We evaluate at the **original mask resolution**
  by default; `--eval_size 400` reproduces the common "resize GT to 400×400" variant.

### Our protocol
1. **Development** happens only on **FSS‑1000 val**, whose 240 classes are disjoint from the 240 test
   classes (HSNet split, `data/splits/fss`). The runner refuses `--purpose tune` on any test split
   and on the CD datasets, which have no val split.
2. **Reporting** uses the frozen configurations on FSS‑1000 test + Deepglobe + ISIC + Chest X‑ray + underwater
   (SUIM‑FSS, UWS), 1/5-shot,
   with three seeds (`stage2_report`).
3. **Shared episodes**: each (dataset, split, K, seed) episode file is generated once and reused by
   **every** method. This gives a paired comparison: the per-episode IoU is stored in `episodes.jsonl`.

### Leakage checklist (enforced or audited by code)
| risk | guard |
|---|---|
| query appears in its own support set | asserted at episode generation and re-checked by `scripts/check_leakage.py` |
| duplicate / near-duplicate images (support ≈ query) | dHash rejection at generation (`--dedup_hamming 2`) + audit |
| tuning on test classes | `--purpose tune` is blocked on test splits; tuning only on FSS‑1000 val |
| FSS val/test overlap | class lists checked for disjointness, images checked for near-duplicates |
| transductive statistics (PCA / standardisation over the whole dataset or the query) | PCA and standardisation are fit on **context rows only** |
| threshold tuned per dataset | a fixed 0.5 on p(fg) everywhere; the prior correction uses only support statistics |
| privileged information | `sam3_text` uses the class name, so it is reported separately and is not ICL. `oracle_grid` uses the query GT and is a diagnostic upper bound |
| **pre-training overlap** | cannot be removed, only documented and avoided (table below). Headline = FSS‑1000 test, ISIC, Chest X‑ray |

| model | FSS‑1000 | Deepglobe | ISIC | Chest X‑ray | COCO/PASCAL | underwater |
|---|---|---|---|---|---|---|
| TabPFN‑3.5 (synthetic tables only) | clean | clean | clean | clean | clean | clean |
| DINOv2 / v3 backbone (self-supervised, no masks) | web images, no labels | idem | idem | idem | idem | idem |
| SegGPT | clean | **domain-adjacent** (loveDA/iSAID aerial land-cover training) | clean | clean | **seen** | clean |
| SAM3 | unclear (SA‑Co is huge) | likely domain-seen | likely domain-seen | likely domain-seen | **seen** | **domain-seen** (FathomNet is in SA‑Co) |

The frozen backbone is shared by TabPFN and the feature baselines, so Q1 is leak-free by construction.

## 3. Where should the rows live? Patch level vs. "pixel level"

### 3a. Patch level (what Matcher, PerSAM, prototype methods and DINO linear probes do)
* rows = ViT tokens (DINOv3 ViT‑L/16 at 512 px gives 32×32 = 1024 rows per image). Label = mask area fraction ≥ 0.5.
* p(fg) is predicted per query patch, upsampled bilinearly to the image, and thresholded at 0.5.
* **Pros:** every row is a distinct, information-carrying token. 1-shot = 1.4k context rows and 5-shot
  = 6.8k, which TabPFN handles comfortably. All feature heads see exactly the same data.
* **Cons:** a resolution ceiling. We **measure** that ceiling with `oracle_grid`, which puts the true query mask
  on the working grid and upsamples it back (stage 0). If the patch ceiling is about 95 and every method sits
  at 60–80, resolution is not the bottleneck yet.

### 3b. "Upscale the features, then classify pixels": what it does and does not buy
Upsampling the features to the target resolution and sampling fg/bg pixels with weights is the right
instinct, with one catch:

> **Bilinear feature upsampling adds no information.** Every upsampled vector is a convex combination
> of 4 patch tokens. For a *linear* head, classifying bilinear features equals bilinearly upsampling the
> patch logits, so the result is identical to 3a. For TabPFN or kNN it is not exactly identical, but the
> context fills with highly correlated, interpolated near-duplicate rows. That costs context budget and
> gives little new signal.

Going below patch level therefore helps only if the extra rows carry **new high-resolution
evidence**. The repo implements the two training-free ways to get it:
1. **Guided upsampling (`--up jbu`)**: a joint bilateral upsampler steered by the RGB image. Features snap
   to colour edges instead of being blurred across object boundaries. Learned upsamplers (FeatUp, LoftUp,
   AnyUp, JAFAR) could be dropped in at `features/builder.py::upsample`, but check their training data first.
2. **Per-cell pixel cues (`rgb=1`)**: mean Lab colour + local std of the cell, appended as 6 extra columns.
   Here TabPFN has a structural advantage over linear heads: it can learn non-linear interactions between
   "DINO says object" and "colour says edge" from the context alone.

### 3c. Sampling the context (the "weighted random sample from mask and background" idea)
A sub-patch grid has far more support cells than are useful (x4 → 148² ≈ 22k per image), so we sample
up to `--max_ctx` rows:
* `uniform`: keeps the true class prior.
* `balanced`: 50/50 fg/bg. Helps small objects, but the in-context learner then believes p(fg) = 0.5 and
  **over-segments**. `--prior_correct 1` (on by default) maps predictions back with a Bayes prior shift
  that uses only support statistics.
* `boundary`: sampling weight 1 + 4·exp(−d²/2σ²), where d is the distance to the support mask edge.
  Interior rows are redundant, while boundary rows carry the information that sub-patch prediction is for.
* Mixed cells: either drop them (`--ambig 0.2,0.8`) or keep the soft fraction as a regression target
  (`tabpfn_reg`, TabPFN regressor).

**Recommendation.** Do not go to full pixel level. A 512² query is 262k rows, and with JBU the features are
still patch-limited. Use **x2–x4** (cells of 3–7 px) with JBU + rgb + boundary sampling. Compare it against
patch level and against **patch level + SAM3 refinement** (`--refine sam3`), the Matcher/PerSAM recipe:
the in-context learner handles semantics and SAM handles boundaries. The refinement is applied
identically to every method, so it does not bias Q1.

## 4. SAM3 in this comparison
SAM3's "promptable concept segmentation" accepts a noun phrase and/or **exemplar boxes in the same image**.
It reports 1-exemplar results on COCO/LVIS/ODinW, but has no cross-image few-shot protocol. We use it in
three clearly labelled ways:
* `sam3_stitch` (**in-context**): the canvas is `[support₁ | … | support_K | query]`. Positive exemplar boxes
  are placed around the support masks' connected components, and we keep the detections in the query tile.
  This is the same "visual prompting by image completion" trick that Painter and SegGPT use.
* `sam3_text` (**not ICL**): class-name prompt. A zero-shot open-vocabulary reference that uses privileged
  information.
* `--refine sam3`: a SAM3 tracker (PVS) refinement of any coarse mask, using a box prompt from the coarse
  mask's components.

## 5. The "6k features" experiment (stage 1a)
ViT‑L has 1024 dims per layer. At an equal budget of about 6144 dims we compare:

| name | spec | idea |
|---|---|---|
| `last` | `layers=24` (1024) | standard: last layer only |
| `ml6` | `layers=4,8,12,16,20,24` | multi-layer: low-level texture/edges + semantics. Cross-domain FSS papers ("the devil is in low-level features") suggest this helps ISIC/Chest |
| `nb6` | `layers=24;pools=1,3,5,9,17,g` | **the patch and the area around it**: the patch plus mean-pooled 3×3, 5×5, 9×9 and 17×17 neighbourhoods and the image mean. A cheap "gathering" of context that disambiguates parts from wholes |
| `ml3nb2`, `ml2nb3` | mixed | both |
| DINOv3‑H+ | `layers=-25,-19,-13,-7,-1` (5×1280 = 6400) | about the same budget, bigger backbone |

Each block is L2-normalised, so blocks are on a comparable scale.

**Caveat specific to TabPFN:** TabPFN may **subsample features per ensemble member** when D exceeds its
per-estimator cap. TabPFN‑2 caps at 500 and 2.5 at 2000. For 3.5 the cap is stored in the checkpoint.
The runner records the effective config in `config.json → head_info.effective`. If the cap is below D,
"6k features" means "each estimator sees a random subset". Stage 1c therefore sweeps `n_estimators` and
PCA (64/256/1024, fit on the context only) to separate "more information" from "more columns than the
model uses".

## 6. Expected costs on one H200 (verify with `slurm/smoke_gpu.sbatch`; the runner logs s/episode)
* Backbone: DINOv3 ViT‑L at 512² takes a few ms per image (features are cached within an episode).
* Feature heads (proto/kNN/logreg): well under 1 s per episode.
* TabPFN, patch level, 1-shot: about 1.4k × 6k context, 1.4k query rows. 5-shot is about 6.8k context.
* TabPFN at x4: up to `max_ctx` context rows and about 22k query rows (predicted in chunks of 32k).
  This is the expensive arm; use `v3.5-fast`, PCA or fewer estimators while exploring.

## 7. Reading the results
* **Stage 0**: `oracle_grid` gives the resolution ceiling per dataset and grid. `proto` gives the floor.
* **Q1**: compare `tabpfn` against the best of `logreg`/`knn` per feature set, with a paired comparison on
  the shared episodes. A consistent gain across all four domains, especially 1-shot cross-domain, supports
  "TabPFN is a good in-context *learner* for segmentation".
* **Q2**: TabPFN system vs SegGPT on FSS‑1000 test, ISIC and Chest X‑ray, which are clean for both
  (report Deepglobe with the aerial caveat), and vs `sam3_stitch`. Report `sam3_text` separately.
