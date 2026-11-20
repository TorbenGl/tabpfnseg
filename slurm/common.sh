# Sourced by every sbatch script. Adapt the module/conda lines to your cluster.
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$REPO"
# module load cuda/12.4  # <- cluster specific, uncomment / adapt
VENV=${VENV:-$REPO/.venv}
source "$VENV/bin/activate"

export TABPFNSEG_DATA=${TABPFNSEG_DATA:-$REPO/datasets}
export HF_HOME=${HF_HOME:-$REPO/checkpoints/hf}
export TABPFN_MODEL_CACHE_DIR=${TABPFN_MODEL_CACHE_DIR:-$REPO/checkpoints/tabpfn}
export TABPFN_NO_BROWSER=1
# compute nodes: everything must be prefetched (scripts/prefetch_models.py on the login node)
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTHONUNBUFFERED=1
mkdir -p logs results episodes
echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
