#!/usr/bin/env bash
# Interactive / one-off GPU work through srun, with the uv environment and all paths/caches set.
#   bash slurm/srun.sh                                   # interactive shell on a GPU node (env active)
#   bash slurm/srun.sh python -m tabpfnseg.run --dataset fss1000 --split val --purpose tune --limit 20 --method tabpfn
#   bash slurm/srun.sh pytest -q tests
# Resources via env vars (defaults below), e.g.  TIME=04:00:00 GPUS=h200:1 bash slurm/srun.sh
# Extra srun flags (partition, account, ...):    SRUN_EXTRA="--partition=gpu --account=myproj" bash slurm/srun.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
GPUS=${GPUS:-h200:1}
CPUS=${CPUS:-16}
MEM=${MEM:-64G}
TIME=${TIME:-02:00:00}
ARGS=(--gres="gpu:${GPUS}" --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" --job-name=tabpfnseg-srun)
# shellcheck disable=SC2206
[ -n "${SRUN_EXTRA:-}" ] && ARGS+=($SRUN_EXTRA)

if [ $# -eq 0 ]; then
  exec srun "${ARGS[@]}" --pty bash -c "source '$REPO/slurm/common.sh' && set +euo pipefail && exec bash -i"
else
  exec srun "${ARGS[@]}" bash -c "source '$REPO/slurm/common.sh' && exec \"\$@\"" _ "$@"
fi
