#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=humanoid-fix-300m-hi
#SBATCH --output=humanoid-fix-300m-hi_%A_%a.out
# 3 thresholds x 80 cells = 240 jobs.
#   0-79    90x60
#   80-159  100x70
#   160-239 110x80
#SBATCH --array=0-239

# 300M baseline vs explore->BC->finetune on the FIXED MOHumanoid env, at
# higher sparsity gates than the 70x40 / 80x50 300M sweep. Same budget,
# splits, eval cadence, and W&B group naming as
# scripts/slurm/humanoid_fixedenv_300m.sh.
#
# Each threshold is baseline-300m + e60-f240 .. e150-f150, seeds 0-9.
#
# W&B project SMORL-TMLR. Groups:
#   tmlr-humanoid-thr90x60-300m-fixedenv
#   tmlr-humanoid-thr100x70-300m-fixedenv
#   tmlr-humanoid-thr110x80-300m-fixedenv
# Group by variant. Filter job_type to {baseline, finetune} for the
# comparison. Explore is a separate run and its archive is ungated.
#
# Submit from the moplayground checkout on the login node:
#   sbatch scripts/slurm/humanoid_fixedenv_300m_higher_thr.sh

set -euo pipefail

ENV_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL
CODE_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL/moplayground
BASE=config/morlax/mohumanoid_sparse_migration_100m.yaml
TOTAL_M=300
SPLITS="60,240;75,225;90,210;105,195;120,180;135,165;150,150"
SEEDS="0,1,2,3,4,5,6,7,8,9"
NUM_EVALS=61
N_CELLS=80
ROOT=/nfs/stak/users/thakarr/hpc-share/SMORL/results/tmlr_migration_300m/humanoid_fixedenv

THRESHOLDS=("90,60" "100,70" "110,80")

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
GROUP="tmlr-humanoid-thr${TAG}-300m-fixedenv"
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
echo "Humanoid (fixed env, 300M) threshold: [${THRESHOLD}]  Group: ${GROUP}"
echo "SaveDir: ${SAVE_DIR}  Cell: ${CELL_IDX}"
echo "Budget: ${TOTAL_M}M  Splits: ${SPLITS}  Seeds: ${SEEDS}  num_evals: ${NUM_EVALS}"
nvidia-smi -L || true

"${ENV_DIR}/bin/python" -m scripts.migration_sweep \
    --base "${BASE}" \
    --save-dir "${SAVE_DIR}" \
    --group "${GROUP}" \
    --threshold "${THRESHOLD}" \
    --total-m "${TOTAL_M}" \
    --splits "${SPLITS}" \
    --num-evals "${NUM_EVALS}" \
    --seeds "${SEEDS}" \
    --index "${CELL_IDX}" \
    --skip-existing
