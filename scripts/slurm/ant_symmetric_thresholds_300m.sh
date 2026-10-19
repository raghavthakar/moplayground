#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=ant-sym-300m
#SBATCH --output=ant-sym-300m_%A_%a.out
# Equal vx/vy gates. 6 thresholds x 80 cells = 480 jobs.
#   0-79    100x100
#   80-159  150x150
#   160-239 200x200
#   240-319 250x250
#   320-399 300x300
#   400-479 350x350
#SBATCH --array=0-479

# 300M baseline vs explore->BC->finetune on MOAnt, same threshold on vx and
# vy. Easier equal gates than the 100M symmetric sweep (400/450/500), at
# 3x the sample budget.
#
# Each threshold is baseline-300m + e60-f240 .. e150-f150 (3x the 100M
# e20-f80 .. e50-f50 splits), seeds 0-9. Eval cadence matches the 100M
# runs (every 5M frames on the baseline; num_evals=61).
#
# W&B project SMORL-TMLR. Groups and save dirs are new so these do NOT mix
# with the 100M symmetric or the asymmetric 400x600 / 450x650 runs:
#   tmlr-ant-thr100x100-300m
#   tmlr-ant-thr150x150-300m
#   tmlr-ant-thr200x200-300m
#   tmlr-ant-thr250x250-300m
#   tmlr-ant-thr300x300-300m
#   tmlr-ant-thr350x350-300m
# Group by variant. Filter job_type to {baseline, finetune} for the
# comparison. Explore is a separate run and its archive is ungated.
#
# Submit from the moplayground checkout on the login node:
#   sbatch scripts/slurm/ant_symmetric_thresholds_300m.sh

set -euo pipefail

ENV_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL
CODE_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL/moplayground
BASE=config/morlax/moant_sparse_migration_100m.yaml
TOTAL_M=300
SPLITS="60,240;75,225;90,210;105,195;120,180;135,165;150,150"
SEEDS="0,1,2,3,4,5,6,7,8,9"
NUM_EVALS=61
N_CELLS=80
ROOT=/nfs/stak/users/thakarr/hpc-share/SMORL/results/tmlr_migration_300m/ant

THRESHOLDS=("100,100" "150,150" "200,200" "250,250" "300,300" "350,350")

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
GROUP="tmlr-ant-thr${TAG}-300m"
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
echo "Ant symmetric threshold: [${THRESHOLD}]  Group: ${GROUP}"
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
