#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=humanoid-120x180-acbc-s4
#SBATCH --output=humanoid-120x180-acbc-s4_%A_%a.out
# Migration splits only (baseline cells 0-3 excluded):
#   7 splits (e60-f240 .. e150-f150) x 4 seeds (0..3) = 28 jobs -> array 4-31.
#SBATCH --array=4-31

# 300M explore->BC->finetune on the fixed MOHumanoid env at threshold 120x180
# with:
#   - 13-D root/CoM RND exploration
#   - Noiseless teacher mean action targets + damped demo rollouts
#   - Joint actor-critic hypernetwork BC distillation (policy + value branches)
#
# Baseline cells (0..3) are intentionally skipped because baseline MORLAX does
# not use exploration or BC and was already evaluated at 120x180.
#
# W&B project SMORL-TMLR. Group:
#   tmlr-humanoid-thr120x180-300m-rootrnd-acbc
#
# Submit from the moplayground checkout on the login node:
#   sbatch scripts/slurm/humanoid_fixedenv_300m_thr120x180_rootrnd_acbc_s4.sh

set -euo pipefail

ENV_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL
CODE_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL/moplayground
BASE=config/morlax/mohumanoid_sparse_migration_100m.yaml
TOTAL_M=300
SPLITS="60,240;75,225;90,210;105,195;120,180;135,165;150,150"
SEEDS="0,1,2,3"
NUM_EVALS=61
N_CELLS=32
ROOT=/nfs/stak/users/thakarr/hpc-share/SMORL/results/tmlr_migration_300m/humanoid_rootrnd_acbc

THRESHOLD="120,180"

INDEX="${SLURM_ARRAY_TASK_ID:-4}"
CELL_IDX="${INDEX}"
LAST=$((N_CELLS - 1))

if (( CELL_IDX < 4 || CELL_IDX > LAST )); then
    echo "Index ${INDEX} out of range for migration-only sweep (need 4-${LAST})"
    exit 1
fi

TAG="${THRESHOLD//,/x}"
GROUP="tmlr-humanoid-thr${TAG}-300m-rootrnd-acbc"
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
echo "Humanoid (root/CoM RND + Actor-Critic BC, 300M) threshold: [${THRESHOLD}]  Group: ${GROUP}"
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
