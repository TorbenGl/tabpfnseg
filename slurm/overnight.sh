#!/usr/bin/env bash
# Unattended: GPU smoke test (every method, 10 episodes), then the full leakage audit on CPU
# (after the smoke job, so it also checks the episode files the smoke job wrote).
#   bash slurm/overnight.sh
#   GPU_PARTITION=gpu CPU_PARTITION=compute-node bash slurm/overnight.sh   # override auto-detection
# Results:  logs/tabpfnseg-smoke-<id>.out   logs/tabpfnseg-leakage-<id>.out   logs/leakage_pairs/
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
CPU_PARTITION=${CPU_PARTITION:-compute-node}
# GPU partition: first one whose nodes advertise an H200 (else any GPU)
if [ -z "${GPU_PARTITION:-}" ]; then
  GPU_PARTITION=$(sinfo -h -o "%P %G" | grep -i h200 | head -1 | awk '{print $1}' | tr -d '*' || true)
  [ -z "$GPU_PARTITION" ] && GPU_PARTITION=$(sinfo -h -o "%P %G" | grep -i "gpu" | head -1 | awk '{print $1}' | tr -d '*' || true)
fi
[ -z "$GPU_PARTITION" ] && { echo "no GPU partition found - set GPU_PARTITION=..."; sinfo -o "%P %G"; exit 1; }
GRES=$(sinfo -h -p "$GPU_PARTITION" -o "%G" | grep -qi h200 && echo "gpu:h200:1" || echo "gpu:1")
echo "GPU partition: $GPU_PARTITION ($GRES)   CPU partition: $CPU_PARTITION"

SMOKE=$(sbatch --parsable -p "$GPU_PARTITION" --gres="$GRES" slurm/smoke_gpu.sbatch)
LEAK=$(sbatch --parsable -p "$CPU_PARTITION" --dependency=afterany:"$SMOKE" slurm/leakage.sbatch)
echo "submitted: smoke=$SMOKE  leakage=$LEAK (starts after smoke)"
echo "check:  squeue -u \$USER     logs: logs/tabpfnseg-smoke-$SMOKE.out  logs/tabpfnseg-leakage-$LEAK.out"
