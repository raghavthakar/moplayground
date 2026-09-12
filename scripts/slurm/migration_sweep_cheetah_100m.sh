#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-cheetah-migrate-100m
#SBATCH --output=MORLAX-cheetah-migrate-100m_%A_%a.out
# {baseline-100m + 7 splits e20..e50} x 10 seeds = 80 runs -> 0..79.
#SBATCH --array=0-79

# 100M budget sweep on sparse MOCheetah (run vs energy) at 90x240.
# That is the round onset from relaxed cliff-search (alive T ~[90, 237]).
#
# W&B group: mocheetah-thr90x240-budget-sweep-100m
# Group-by `variant`; filter job_type in {baseline, finetune}.
#
# Submit:
#   sbatch scripts/slurm/migration_sweep_cheetah_100m.sh
# Preview:
#   python -m scripts.migration_sweep \
#       --base config/morlax/mocheetah_sparse_migration_100m.yaml \
#       --group mocheetah-thr90x240-budget-sweep-100m \
#       --threshold 90,240 --total-m 100 --seeds 0,1,2,3,4,5,6,7,8,9 --list

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
BASE=config/morlax/mocheetah_sparse_migration_100m.yaml
THRESHOLD="90,240"
TAG="90x240"
GROUP="mocheetah-thr${TAG}-budget-sweep-100m"
SAVE_DIR="/nfs/hpc/share/thakarr/SMORL/results/migration_budget_sweep_cheetah_100m/thr${TAG}"
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

# W&B stages artifacts and caches them under $HOME by default, which fills the
# 25G NFS home quota and then kills every later job with ENOSPC. Keep that
# scratch on the group share instead.
export WANDB_DATA_DIR=/nfs/hpc/share/thakarr/SMORL/wandb/data
export WANDB_CACHE_DIR=/nfs/hpc/share/thakarr/SMORL/wandb/cache
export WANDB_DIR=/nfs/hpc/share/thakarr/SMORL/wandb/runs
mkdir -p "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DIR}"

echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID:-local} (array ${SLURM_ARRAY_JOB_ID:-NA} task ${SLURM_ARRAY_TASK_ID:-NA})"
echo "Threshold: [${THRESHOLD}]  Tag: ${TAG}  Group: ${GROUP}"
echo "SaveDir: ${SAVE_DIR}  Index: ${INDEX}"
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
    --index "${INDEX}" \
    --skip-existing
