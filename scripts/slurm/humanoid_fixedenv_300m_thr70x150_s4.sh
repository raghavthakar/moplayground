#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=humanoid-70x150-300m-s4
#SBATCH --output=humanoid-70x150-300m-s4_%A_%a.out
# 1 threshold x 8 variants x 4 seeds = 32 jobs.
#   variants: baseline-300m + e60-f240 .. e150-f150
#   seeds: 0,1,2,3
#SBATCH --array=0-31

# 300M baseline vs explore->BC->finetune on the fixed MOHumanoid env at
# threshold 70x150 (speed 70, energy 150). Same eval cadence as the other
# 300M humanoid sweeps (num_evals=61, every 5M frames). Four seeds only, so
# this fits beside the remaining reachable-threshold array under the 2-GPU cap.
#
# W&B project SMORL-TMLR. Group:
#   tmlr-humanoid-thr70x150-300m-fixedenv
# Group by variant. Filter job_type to {baseline, finetune} for the
# comparison. Explore is logged as its own run.
#
# Submit from the moplayground checkout on the login node:
#   sbatch scripts/slurm/humanoid_fixedenv_300m_thr70x150_s4.sh

set -euo pipefail

ENV_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL
CODE_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL/moplayground
BASE=config/morlax/mohumanoid_sparse_migration_100m.yaml
TOTAL_M=300
SPLITS="60,240;75,225;90,210;105,195;120,180;135,165;150,150"
SEEDS="0,1,2,3"
NUM_EVALS=61
N_CELLS=32
ROOT=/nfs/stak/users/thakarr/hpc-share/SMORL/results/tmlr_migration_300m/humanoid_fixedenv

THRESHOLD="70,150"

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
CELL_IDX="${INDEX}"
LAST=$((N_CELLS - 1))

if (( CELL_IDX < 0 || CELL_IDX > LAST )); then
    echo "Index ${INDEX} out of range (need 0-${LAST})"
    exit 1
fi

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
