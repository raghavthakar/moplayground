#!/bin/bash
#SBATCH --time=0-04:00:00
# jax[cuda13] dropped Volta (sm_70); target H100 (dgxh) and A100 (ampere, sm_80).
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-migrate-sweep
#SBATCH --output=MORLAX-migrate-sweep_%A_%a.out
# One array task per (variant, seed) cell.
# Matrix = {baseline + 5 splits} x 3 seeds = 18 runs -> indices 0..17.
# If you change SEEDS or SPLITS_M, update this range (see: --list below).
#SBATCH --array=0-17

# End-to-end migration budget sweep (sparse MOHopper, 50x50).
#   baseline-50m : plain MORLAX (cold) for 50M
#   e{E}-f{F}    : E M explore -> BC -> F M MORLAX finetune (E+F=50)
# All seeds/variants land in one W&B group so they aggregate cleanly.
#
# Submit:
#   sbatch scripts/slurm/migration_sweep.sh
# Preview the matrix (which index is which run):
#   python -m scripts.migration_sweep --list
# One cell interactively (debug):
#   SLURM_ARRAY_TASK_ID=0 bash scripts/slurm/migration_sweep.sh

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
BASE=config/morlax/mohopper_sparse_migration.yaml
SAVE_DIR=/nfs/hpc/share/thakarr/SMORL/results/migration_budget_sweep
GROUP=mohopper-thr50-budget-sweep
SEEDS="0,1,2"

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
echo "Base: ${BASE}  SaveDir: ${SAVE_DIR}  Group: ${GROUP}  Seeds: ${SEEDS}  Index: ${INDEX}"
nvidia-smi -L || true

"${ENV_DIR}/bin/python" -m scripts.migration_sweep \
    --base "${BASE}" \
    --save-dir "${SAVE_DIR}" \
    --group "${GROUP}" \
    --seeds "${SEEDS}" \
    --index "${INDEX}" \
    --skip-existing
