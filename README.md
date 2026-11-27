# tabpfnseg: Is TabPFN‑3.5 an in‑context segmenter?

Frozen ViT features → one table row per location → **TabPFN‑3.5 in context** (support rows = labelled
context, query rows = prediction) → mask. No training anywhere. Compared under one leakage-controlled
few-shot protocol against
* the **same features** with per-episode logistic regression / kNN / prototypes (*does TabPFN add anything as the learner?*)
* **SegGPT** (a dedicated in-context segmenter) and **SAM3** (stitched-canvas exemplars, plus text and refinement references).

Full rationale, protocol, leakage analysis and the patch-level vs pixel-level discussion:
**[docs/DESIGN.md](docs/DESIGN.md)**.

What was run, why, with which code, and what we decided: **[docs/PROVENANCE.md](docs/PROVENANCE.md)**
(results registry: `python scripts/provenance.py` → `results/REGISTRY.md`).

## Layout
```
tabpfnseg/data/        datasets (FSS-1000, Deepglobe, ISIC, Chest X-ray, SUIM, UWS, COCO-json underwater sets, folder, synthetic) + shared episodes
tabpfnseg/features/    HF ViT backbones (DINOv3 used in all experiments; DINOv2 aliases kept), intermediate layers, feature tables, JBU upsampling, context sampling
tabpfnseg/methods/     heads (tabpfn, tabpfn_reg, logreg, knn, proto), oracle_grid, seggpt, sam3 (stitch/text/refine)
tabpfnseg/eval/        class-wise mIoU, FB-IoU
tabpfnseg/run.py       one evaluation run (CLI)
scripts/               download_data.sh, prefetch_models.py, check_leakage.py, make_sweeps.py, aggregate.py
slurm/                 setup_env.sh, smoke_gpu.sbatch, run_one.sbatch, sweep.sbatch, submit_sweep.sh, sweeps/*.txt
tests/                 CPU smoke tests (no downloads)
```

## Local quick check (CPU, no downloads)
```bash
pip install -e ".[dev]"
```
```bash
pytest -q tests
```
```bash
python -m tabpfnseg.run --dataset synthetic --backbone dummy --img_size 160 --method knn --device cpu
```

## On the cluster (H200, SLURM)
1. Build the **uv** environment in `.venv` on the login node (needs internet). The script installs uv to
   `~/.local/bin` if it is missing and downloads its own Python 3.12 if needed:
   ```bash
   bash slurm/setup_env.sh
   ```
   It defaults to CUDA 12.8 wheels. On older drivers use `TORCH_CUDA=cu126` (check `nvidia-smi`, "CUDA Version").
   If the home quota is small, set `UV_CACHE_DIR` to a scratch filesystem.
2. Accept the licences (DINOv3 and SAM3 on Hugging Face; TabPFN‑3.5 at ux.priorlabs.ai), then export the tokens:
   ```bash
   export HF_TOKEN=... TABPFN_TOKEN=...
   ```
3. Download the weights and sanity-check every model once (compute nodes then run offline):
   ```bash
   source slurm/common.sh && HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python scripts/prefetch_models.py --device cpu
   ```
4. Kaggle credentials (only for Chest X‑ray): on kaggle.com go to **Settings → API → Create New Token**.
   This downloads `kaggle.json`, which you copy to the server:
   ```bash
   mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
   Alternatively export `KAGGLE_API_TOKEN=...` (newer token type) or `KAGGLE_USERNAME` + `KAGGLE_KEY`.
5. Download the data:
   ```bash
   bash scripts/download_data.sh
   ```
6. Run the leakage audit:
   ```bash
   source slurm/common.sh && python scripts/check_leakage.py
   ```
7. Interactive GPU work through **srun** (environment, caches and paths already set):
   ```bash
   bash slurm/srun.sh
   ```
   You can also run one command directly:
   ```bash
   bash slurm/srun.sh python -m tabpfnseg.run --dataset fss1000 --split val --purpose tune --episodes 600 --limit 20 --method tabpfn
   ```
   Resources and cluster flags are set via environment variables:
   ```bash
   TIME=04:00:00 GPUS=h200:1 MEM=128G SRUN_EXTRA="--partition=gpu --account=myproj" bash slurm/srun.sh
   ```
8. Run the smoke job (every method, 10 episodes):
   ```bash
   sbatch slurm/smoke_gpu.sbatch
   ```
9. Run the stages:
   ```bash
   python scripts/make_sweeps.py
   ```
   ```bash
   bash slurm/submit_sweep.sh slurm/sweeps/stage0_sanity.txt 8
   ```
   ```bash
   bash slurm/submit_sweep.sh slurm/sweeps/stage1a_features.txt 8
   ```
   ```bash
   bash slurm/submit_sweep.sh slurm/sweeps/stage1b_resolution.txt 8
   ```
   ```bash
   bash slurm/submit_sweep.sh slurm/sweeps/stage1c_tabpfn_knobs.txt 8
   ```
   Then edit `FINAL` in `scripts/make_sweeps.py`, choosing from **FSS‑1000 val results only**, and:
   ```bash
   python scripts/make_sweeps.py && bash slurm/submit_sweep.sh slurm/sweeps/stage2_report.txt 8
   ```
   **Without sbatch (srun only)**: run the smoke job and the sweeps through srun on a given partition.
   `srun_sweep.sh` starts one srun job per sweep line, with N in parallel, and blocks, so start it in tmux:
   ```bash
   PARTITION=compute-node bash slurm/srun.sh bash slurm/smoke_gpu.sbatch
   ```
   ```bash
   PARTITION=compute-node bash slurm/srun_sweep.sh slurm/sweeps/stage0_sanity.txt 8
   ```
10. Build the results table:
    ```bash
    python scripts/aggregate.py results --out results/table.md
    ```

Adapt the `#SBATCH --gres/--partition/--account` lines and the `module load` line in `slurm/common.sh`
to your cluster. Re-submitting a sweep skips finished runs (`--skip_done`).

## Single run examples
```bash
# TabPFN-3.5 on patch tokens, "patch + neighbourhood" 6k features, FSS-1000 val (development)
python -m tabpfnseg.run --dataset fss1000 --split val --purpose tune --episodes 600 \
  --method tabpfn --backbone dinov3_l --img_size 512 --features "layers=24;pools=1,3,5,9,17,g"
```
```bash
# sub-patch rows: x4 grid, joint-bilateral upsampling, colour cues, boundary-weighted context
python -m tabpfnseg.run --dataset isic --shots 1 --method tabpfn --features "layers=4,8,12,16,20,24;rgb=1" \
  --res x4 --up jbu --sampling boundary --max_ctx 16000
```
```bash
# baselines on the same episodes
python -m tabpfnseg.run --dataset chestx --shots 5 --method seggpt
```
```bash
python -m tabpfnseg.run --dataset suim --shots 1 --method sam3_stitch
```

## Status
* Tested here (CPU): data and episode logic, feature builder, JBU, every feature head, the TabPFN code path
  (with the openly licensed v2 weights), and SegGPT (with a tiny random model).
* Not yet run with real weights: TabPFN‑3.5 (needs your token), DINOv3 / SAM3 (gated). `slurm/smoke_gpu.sbatch`
  and `scripts/prefetch_models.py` are the first things to run on the H200.
