#!/usr/bin/env bash
#SBATCH --job-name=tabpfnseg-smoke
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# First job to run on the cluster: every method on 10 real FSS-1000 val episodes.
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
pytest -q tests || echo "FAILED unit tests (continuing with the GPU runs)"
python scripts/check_leakage.py --datasets fss1000 || echo "FAILED leakage check (continuing)"
C="--dataset fss1000 --split val --purpose tune --episodes 600 --limit 10 --shots 1 --out results_smoke"
F="--backbone dinov2r_l --img_size 518 --features layers=24;pools=1,3,5,9,17,g"
for m in oracle_grid proto knn logreg tabpfn tabpfn_reg; do
  python -m tabpfnseg.run $C $F --method $m || echo "FAILED $m"
done
python -m tabpfnseg.run $C $F --method tabpfn --res x4 --up jbu --sampling boundary --max_ctx 16000 || echo "FAILED tabpfn x4"
python -m tabpfnseg.run $C --backbone dinov3_l --img_size 512 --features "layers=4,8,12,16,20,24" --method tabpfn || echo "FAILED dinov3"
for m in seggpt sam3_stitch sam3_text; do
  python -m tabpfnseg.run $C --method $m --save_vis 10 || echo "FAILED $m"
done
python -m tabpfnseg.run $C $F --method tabpfn --refine sam3 || echo "FAILED refine"
python scripts/aggregate.py results_smoke
