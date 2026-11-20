#!/usr/bin/env bash
# bash slurm/submit_sweep.sh slurm/sweeps/stage1_tune.txt 8
set -euo pipefail
SWEEP=$1; PAR=${2:-8}
N=$(grep -c . "$SWEEP")
mkdir -p logs
# episode files are written atomically and deterministically, so parallel tasks may safely create them
sbatch --array=1-"$N"%"$PAR" slurm/sweep.sbatch "$SWEEP"
