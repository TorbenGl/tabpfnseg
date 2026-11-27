#!/usr/bin/env bash
#SBATCH --job-name=tabpfnseg-sweep
#SBATCH --partition=gpu-node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
# One line of the sweep file = one run (arguments for tabpfnseg.run). Submit with
#   bash slurm/submit_sweep.sh slurm/sweeps/stage1_tune.txt [max_parallel]
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
SWEEP=$1
export TABPFNSEG_SWEEP="$SWEEP"   # recorded in results/<run>/config.json -> provenance
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SWEEP")
echo "ARGS: $LINE"
# skip finished runs (re-submitting a sweep only runs what is missing)
eval "python -m tabpfnseg.run $LINE --skip_done 1"
