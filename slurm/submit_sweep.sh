#!/usr/bin/env bash
# Submit a sweep file (one run per line) as a SLURM job array.
#   bash slurm/submit_sweep.sh slurm/sweeps/budget_tune.txt 4          # all lines, 4 at a time
#   bash slurm/submit_sweep.sh slurm/sweeps/budget_tune.txt 4 1-8      # only lines 1-8 (a chunk)
#   bash slurm/submit_sweep.sh slurm/sweeps/budget_tune.txt 4 9-16     # next chunk later
# Finished runs are skipped (--skip_done), so re-submitting a range never redoes work.
# GPU hours used so far: bash slurm/gpu_hours.sh
set -euo pipefail
SWEEP=$1; PAR=${2:-4}
N=$(grep -c . "$SWEEP")
RANGE=${3:-1-$N}
mkdir -p logs
echo "$SWEEP: $N runs in file, submitting lines $RANGE, $PAR in parallel"
# episode files are written atomically and deterministically, so parallel tasks may safely create them
sbatch --array="$RANGE"%"$PAR" slurm/sweep.sbatch "$SWEEP"
