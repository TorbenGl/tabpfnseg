#!/usr/bin/env bash
# Run ONCE on the login node (needs internet):
#   bash slurm/setup_env.sh
#   export HF_TOKEN=... TABPFN_TOKEN=...   (accept DINOv3 / SAM3 licences on HF, TabPFN licence on ux.priorlabs.ai)
#   source slurm/common.sh && HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python scripts/prefetch_models.py --device cpu
#   bash scripts/download_data.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
python -c "import torch, transformers, tabpfn; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
