#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-ant-migrate-100m
#SBATCH --output=MORLAX-ant-migrate-100m_%A_%a.out
# 3 thresholds x {baseline-100m + 7 splits} x 10 seeds = 240 cells.
#SBATCH --array=0-239

# 100M budget sweep on sparse MOAnt (vx vs vy).
#
# Thresholds (multiples of 50) sit around the automated cliff (~[435, 610]):
#   0-79    400x600  onset (just below cliff)
#   80-159  450x650  past cliff
#   160-239 550x750  harder
#
# W&B: one group per threshold so you can group-by `variant` the same way
# as Hopper. Filter job_type in {baseline, finetune}. Config also has
# `threshold_tag` (e.g. 400x600) and `variant`.
#   moant-thr400x600-budget-sweep-100m
#   moant-thr450x650-budget-sweep-100m
#   moant-thr550x750-budget-sweep-100m
#
# Submit:
#   sbatch scripts/slurm/migration_sweep_ant_100m.sh
# Preview one threshold:
#   python -m scripts.migration_sweep \
#       --base config/morlax/moant_sparse_migration_100m.yaml \
#       --group moant-thr400x600-budget-sweep-100m \
#       --threshold 400,600 --total-m 100 --seeds 0,1,2,3,4,5,6,7,8,9 --list

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
BASE=config/morlax/moant_sparse_migration_100m.yaml
TOTAL_M=100
SPLITS="20,80;25,75;30,70;35,65;40,60;45,55;50,50"
SEEDS="0,1,2,3,4,5,6,7,8,9"
N_CELLS=80

THRESHOLDS=("400,600" "450,650" "550,750")

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
THR_IDX=$((INDEX / N_CELLS))
CELL_IDX=$((INDEX % N_CELLS))

if (( THR_IDX < 0 || THR_IDX >= ${#THRESHOLDS[@]} )); then
    echo "Index ${INDEX} out of range (need 0-$(( ${#THRESHOLDS[@]} * N_CELLS - 1 )))"
    exit 1
fi

THRESHOLD="${THRESHOLDS[$THR_IDX]}"
TAG="${THRESHOLD//,/x}"
GROUP="moant-thr${TAG}-budget-sweep-100m"
SAVE_DIR="/nfs/hpc/share/thakarr/SMORL/results/migration_budget_sweep_ant_100m/thr${TAG}"

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
echo "SaveDir: ${SAVE_DIR}  Cell: ${CELL_IDX}"
echo "Budget: ${TOTAL_M}M  Splits: ${SPLITS}  Seeds: ${SEEDS}"
nvidia-smi -L || true

"${ENV_DIR}/bin/python" -m scripts.migration_sweep \
    --base "${BASE}" \
    --save-dir "${SAVE_DIR}" \
    --group "${GROUP}" \
    --threshold "${THRESHOLD}" \
    --total-m "${TOTAL_M}" \
    --splits "${SPLITS}" \
    --seeds "${SEEDS}" \
    --index "${CELL_IDX}" \
    --skip-existing
