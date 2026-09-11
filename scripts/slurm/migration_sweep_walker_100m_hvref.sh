#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-walker-migrate-hvref
#SBATCH --output=MORLAX-walker-migrate-hvref_%A_%a.out
# 5 thresholds x {baseline-100m + 7 splits} x 10 seeds = 400 cells.
#SBATCH --array=0-399

# 100M budget sweep on sparse MOWalker with nadir-margin HV reference.
#
# Thresholds (two easier + previous cliff band):
#   0-79    100x-50  easy
#   80-159  150x0    mid-easy
#   160-239 200x0    previous easiest
#   240-319 250x10   at cliff
#   320-399 300x50   past cliff
#
# HV ref = [-180, -440] (dense nadir - 5% range). New W&B groups:
#   mowalker-thr{TAG}-budget-sweep-100m-hvref
#
# Submit:
#   sbatch scripts/slurm/migration_sweep_walker_100m_hvref.sh

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
BASE=config/morlax/mowalker_sparse_migration_100m_hvref.yaml
HV_REF="-180,-440"
TOTAL_M=100
SPLITS="20,80;25,75;30,70;35,65;40,60;45,55;50,50"
SEEDS="0,1,2,3,4,5,6,7,8,9"
N_CELLS=80

THRESHOLDS=("100,-50" "150,0" "200,0" "250,10" "300,50")

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
THR_IDX=$((INDEX / N_CELLS))
CELL_IDX=$((INDEX % N_CELLS))

if (( THR_IDX < 0 || THR_IDX >= ${#THRESHOLDS[@]} )); then
    echo "Index ${INDEX} out of range (need 0-$(( ${#THRESHOLDS[@]} * N_CELLS - 1 )))"
    exit 1
fi

THRESHOLD="${THRESHOLDS[$THR_IDX]}"
TAG="${THRESHOLD//,/x}"
GROUP="mowalker-thr${TAG}-budget-sweep-100m-hvref"
SAVE_DIR="/nfs/hpc/share/thakarr/SMORL/results/migration_budget_sweep_walker_100m_hvref/thr${TAG}"

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
echo "Threshold: [${THRESHOLD}]  Tag: ${TAG}  Group: ${GROUP}"
echo "HV ref: [${HV_REF}]  SaveDir: ${SAVE_DIR}  Cell: ${CELL_IDX}"
echo "Budget: ${TOTAL_M}M  Splits: ${SPLITS}  Seeds: ${SEEDS}"
nvidia-smi -L || true

"${ENV_DIR}/bin/python" -m scripts.migration_sweep \
    --base "${BASE}" \
    --save-dir "${SAVE_DIR}" \
    --group "${GROUP}" \
    --threshold "${THRESHOLD}" \
    --hv-ref="${HV_REF}" \
    --total-m "${TOTAL_M}" \
    --splits "${SPLITS}" \
    --seeds "${SEEDS}" \
    --index "${CELL_IDX}" \
    --skip-existing
