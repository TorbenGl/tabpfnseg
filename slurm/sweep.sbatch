#!/usr/bin/env bash
#SBATCH --job-name=tabpfnseg-sweep
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
# One line of the sweep file = one run (arguments for tabpfnseg.run). Submit with
#   bash slurm/submit_sweep.sh slurm/sweeps/stage1_tune.txt [max_parallel]
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
SWEEP=$1
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SWEEP")
echo "ARGS: $LINE"
# skip finished runs (re-submitting a sweep only runs what is missing)
eval "python -m tabpfnseg.run $LINE --skip_done 1"
