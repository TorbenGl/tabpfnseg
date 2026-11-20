#!/usr/bin/env bash
#SBATCH --job-name=tabpfnseg
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch slurm/run_one.sbatch --dataset fss1000 --split val --purpose tune --method tabpfn ...
# (adjust --gres / --partition / --account to your cluster)
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
python -m tabpfnseg.run "$@"
