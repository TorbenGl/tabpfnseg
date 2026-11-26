#!/usr/bin/env bash
# GPU hours used by your jobs (allocated GPUs x elapsed time, including running jobs).
#   bash slurm/gpu_hours.sh                 # since 2026-11-26 (start of the project budget)
#   bash slurm/gpu_hours.sh 2026-12-01      # since another date
#   BUDGET=24 bash slurm/gpu_hours.sh       # budget to compare against (default 24)
SINCE=${1:-2026-11-26}
BUDGET=${BUDGET:-24}
sacct -u "$USER" -S "$SINCE" -X -n -P --format=JobID,JobName,State,ElapsedRaw,AllocTRES |
awk -F'|' -v budget="$BUDGET" '
  { g = 0
    if (match($5, /gres\/gpu=[0-9]+/)) { split(substr($5, RSTART, RLENGTH), a, "="); g = a[2] }
    if (g > 0) { h = g * $4 / 3600; tot += h; n++
                 by[$2] += h } }
  END { for (k in by) printf "  %-22s %6.2f GPU-h\n", k, by[k]
        printf "TOTAL %.2f of %d GPU-h used (%d GPU jobs), %.2f left\n", tot, budget, n, budget - tot }'
