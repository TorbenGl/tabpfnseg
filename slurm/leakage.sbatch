#!/usr/bin/env bash
#SBATCH --job-name=tabpfnseg-leakage
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
# CPU only: full leakage audit (all datasets + every episode file written so far).
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
python scripts/check_leakage.py
