#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=humanoid-fix-100m
#SBATCH --output=humanoid-fix-100m_%A_%a.out
# 2 thresholds x 80 cells = 160 jobs.
#   0-79    70x40
#   80-159  80x50
#SBATCH --array=0-159

# 100M baseline vs explore->BC->finetune on MOHumanoid, re-run on the FIXED
# environment. The humanoid env now terminates on a non-finite (diverged) MJX
# state and zeros any non-finite reward, so divergent exploration episodes
# autoreset instead of leaking NaN into the eval returns (which previously
# crashed the explore phase before BC/finetune could start).
#
# Each threshold is baseline-100m + e20-f80 .. e50-f50, seeds 0-9.
#
# W&B project SMORL-TMLR. The groups carry a "-fixedenv" suffix and the results
# live under a separate "humanoid_fixedenv" tree, so these runs do NOT mix with
# or resume the earlier (broken-env) humanoid runs:
#   tmlr-humanoid-thr70x40-100m-fixedenv
#   tmlr-humanoid-thr80x50-100m-fixedenv
# Group by variant. Filter job_type to {baseline, finetune} for the
# comparison. Explore is a separate run and its archive is ungated.
#
# Submit from the moplayground checkout on the login node:
#   sbatch scripts/slurm/humanoid_fixedenv_100m.sh

set -euo pipefail

ENV_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL
CODE_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL/moplayground
BASE=config/morlax/mohumanoid_sparse_migration_100m.yaml
TOTAL_M=100
SPLITS="20,80;25,75;30,70;35,65;40,60;45,55;50,50"
SEEDS="0,1,2,3,4,5,6,7,8,9"
N_CELLS=80
ROOT=/nfs/stak/users/thakarr/hpc-share/SMORL/results/tmlr_migration_100m/humanoid_fixedenv

THRESHOLDS=("70,40" "80,50")

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
THR_IDX=$((INDEX / N_CELLS))
CELL_IDX=$((INDEX % N_CELLS))
LAST=$(( ${#THRESHOLDS[@]} * N_CELLS - 1 ))

if (( THR_IDX < 0 || THR_IDX >= ${#THRESHOLDS[@]} )); then
    echo "Index ${INDEX} out of range (need 0-${LAST})"
    exit 1
fi

THRESHOLD="${THRESHOLDS[$THR_IDX]}"
TAG="${THRESHOLD//,/x}"
GROUP="tmlr-humanoid-thr${TAG}-100m-fixedenv"
SAVE_DIR="${ROOT}/thr${TAG}"

module load conda
source activate base
conda activate "${ENV_DIR}"

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset WANDB_MODE

export WANDB_DATA_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL/wandb/data
export WANDB_CACHE_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL/wandb/cache
export WANDB_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL/wandb/runs
mkdir -p "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DIR}" "${SAVE_DIR}"

echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID:-local} (array ${SLURM_ARRAY_JOB_ID:-NA} task ${INDEX})"
echo "Humanoid (fixed env) threshold: [${THRESHOLD}]  Group: ${GROUP}"
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
