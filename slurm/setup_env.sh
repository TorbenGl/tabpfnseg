#!/usr/bin/env bash
# Build the uv virtual environment in <repo>/.venv. Run ONCE on the login node (needs internet):
#   bash slurm/setup_env.sh                       # CUDA 12.8 PyTorch wheels (H200 / Hopper)
#   TORCH_CUDA=cu126 bash slurm/setup_env.sh      # if the cluster driver is older (nvidia-smi -> "CUDA Version")
#   TORCH_CUDA=cpu   bash slurm/setup_env.sh      # CPU-only (laptop / CI)
# Then:
#   export HF_TOKEN=... TABPFN_TOKEN=...
#   source slurm/common.sh && HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python scripts/prefetch_models.py --device cpu
#   bash scripts/download_data.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
PYTHON_VERSION=${PYTHON_VERSION:-3.12}
TORCH_CUDA=${TORCH_CUDA:-cu128}

# uv: use an existing install, else install it to ~/.local/bin (no root needed)
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "[setup] installing uv into ~/.local/bin"
    python3 -m pip install --user --quiet uv || curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
uv --version

# uv downloads its own Python if the cluster's is too old; cache lives in ~/.cache/uv
# (set UV_CACHE_DIR to a scratch/work filesystem if your home quota is small)
uv venv --python "$PYTHON_VERSION" --seed .venv
uv pip install --python .venv/bin/python torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
uv pip install --python .venv/bin/python -e ".[dev,data]"

.venv/bin/python - <<'PY'
import torch, transformers, tabpfn
print("torch", torch.__version__, "| cuda build", torch.version.cuda, "| cuda available here:", torch.cuda.is_available())
print("transformers", transformers.__version__, "| tabpfn", tabpfn.__version__)
PY
echo "[setup] done. Activate with: source .venv/bin/activate   (or: source slurm/common.sh)"
