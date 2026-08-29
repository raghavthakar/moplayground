#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-migrate-100m
#SBATCH --output=MORLAX-migrate-100m_%A_%a.out
# One array task per (variant, seed) cell.
# Matrix = {baseline-100m + 7 splits e20..e50} x 10 seeds = 80 runs -> 0..79.
#SBATCH --array=0-79

# 100M budget sweep (sparse MOHopper, 50x50):
#   baseline-100m : plain MORLAX (cold) for 100M
#   e20-f80 .. e50-f50 : explore -> BC -> finetune (E+F=100)
#
# BC head-start logged to W&B (bc/cold/eval/*, bc/eval/*) and <run>/bc_eval.json.
#
# Submit:
#   sbatch scripts/slurm/migration_sweep_100m.sh
# Preview matrix:
#   python -m scripts.migration_sweep --total-m 100 --seeds 0,1,2,3,4,5,6,7,8,9 --list

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
BASE=config/morlax/mohopper_sparse_migration_100m.yaml
SAVE_DIR=/nfs/hpc/share/thakarr/SMORL/results/migration_budget_sweep_100m
GROUP=mohopper-thr50-budget-sweep-100m
TOTAL_M=100
SPLITS="20,80;25,75;30,70;35,65;40,60;45,55;50,50"
SEEDS="0,1,2,3,4,5,6,7,8,9"

INDEX="${SLURM_ARRAY_TASK_ID:-0}"

module load conda
source activate base
conda activate "${ENV_DIR}"

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset WANDB_MODE

echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID:-local} (array ${SLURM_ARRAY_JOB_ID:-NA} task ${SLURM_ARRAY_TASK_ID:-NA})"
echo "Base: ${BASE}  SaveDir: ${SAVE_DIR}  Group: ${GROUP}"
echo "Budget: ${TOTAL_M}M  Splits: ${SPLITS}  Seeds: ${SEEDS}  Index: ${INDEX}"
nvidia-smi -L || true

"${ENV_DIR}/bin/python" -m scripts.migration_sweep \
    --base "${BASE}" \
    --save-dir "${SAVE_DIR}" \
    --group "${GROUP}" \
    --total-m "${TOTAL_M}" \
    --splits "${SPLITS}" \
    --seeds "${SEEDS}" \
    --index "${INDEX}" \
    --skip-existing
