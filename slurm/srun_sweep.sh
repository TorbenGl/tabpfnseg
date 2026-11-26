#!/usr/bin/env bash
# Run a sweep file with srun instead of an sbatch job array: one srun job per line,
# PAR workers in parallel (each worker takes every PAR-th line). Finished runs are skipped.
#   PARTITION=compute-node bash slurm/srun_sweep.sh slurm/sweeps/stage0_sanity.txt 4
# Blocks until all lines are done -> start it inside tmux/screen so it survives logout.
# Resources as in slurm/srun.sh: PARTITION, GPUS, CPUS, MEM, TIME (per line), SRUN_EXTRA.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
SWEEP=$1; PAR=${2:-4}
export MEM=${MEM:-128G} TIME=${TIME:-12:00:00}
NAME=$(basename "$SWEEP" .txt)
mkdir -p logs
mapfile -t LINES < <(grep . "$SWEEP")
N=${#LINES[@]}
echo "[srun_sweep] $SWEEP: $N runs, $PAR parallel, logs in logs/${NAME}-<line>.out"

worker() {
  local k=$1 i
  for ((i = k; i < N; i += PAR)); do
    local line=${LINES[$i]} log="logs/${NAME}-$((i + 1)).out"
    echo "[worker $k] line $((i + 1))/$N: $line"
    if bash slurm/srun.sh bash -c "echo \"ARGS: \$1\"; eval \"python -m tabpfnseg.run \$1 --skip_done 1\"" _ "$line" >"$log" 2>&1; then
      echo "[worker $k] line $((i + 1)) OK"
    else
      echo "[worker $k] line $((i + 1)) FAILED (see $log)"
    fi
  done
}

for ((k = 0; k < PAR; k++)); do worker "$k" & done
wait
echo "[srun_sweep] done. Failed lines: grep -l Traceback logs/${NAME}-*.out"
