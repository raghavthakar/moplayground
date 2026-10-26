#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=humanoid-rootrnd-cliff-s4
#SBATCH --output=humanoid-rootrnd-cliff-s4_%A_%a.out
# 3 thresholds x 32 cells (8 variants x 4 seeds) = 96 jobs.
#   0-31    100x150   just above naive MORLAX speed cliff (~87-100); 84% rootrnd dual-unlock
#   32-63   120x180   sweet-spot sweet band: 100% e60 dual-unlock ([3,3,2,2] teachers), well past MORLAX cliff
#   64-95   150x200   upper boundary of rootrnd explore band (100% e60 dual-unlock, 68% overall)
#SBATCH --array=0-95

# 300M baseline vs explore->BC->finetune on the fixed MOHumanoid env with
# 13-D root/CoM RND exploration, at thresholds calibrated from the W&B true-return
# distribution of rootrnd explore runs (true_speed ~ 170-426, true_energy ~ 240-455)
# vs the naive MORLAX collapse cliff (~[87, 48] / 100x80).
#
# Each threshold is 32 cells: baseline-300m + e60-f240 .. e150-f150, seeds 0-3.
# Same eval cadence as the other 300M humanoid sweeps (num_evals=61, every 5M frames).
#
# W&B project SMORL-TMLR. Groups:
#   tmlr-humanoid-thr100x150-300m-rootrnd
#   tmlr-humanoid-thr120x180-300m-rootrnd
#   tmlr-humanoid-thr150x200-300m-rootrnd
#
# Submit all 3 thresholds from the moplayground checkout on the login node:
#   sbatch scripts/slurm/humanoid_fixedenv_300m_rootrnd_cliff_s4.sh
# Or submit a single threshold (32 jobs) with --array:
#   sbatch --array=0-31  scripts/slurm/humanoid_fixedenv_300m_rootrnd_cliff_s4.sh   # 100x150 only
#   sbatch --array=32-63 scripts/slurm/humanoid_fixedenv_300m_rootrnd_cliff_s4.sh   # 120x180 only
#   sbatch --array=64-95 scripts/slurm/humanoid_fixedenv_300m_rootrnd_cliff_s4.sh   # 150x200 only

set -euo pipefail

ENV_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL
CODE_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL/moplayground
BASE=config/morlax/mohumanoid_sparse_migration_100m.yaml
TOTAL_M=300
SPLITS="60,240;75,225;90,210;105,195;120,180;135,165;150,150"
SEEDS="0,1,2,3"
NUM_EVALS=61
N_CELLS=32
ROOT=/nfs/stak/users/thakarr/hpc-share/SMORL/results/tmlr_migration_300m/humanoid_rootrnd

THRESHOLDS=("100,150" "120,180" "150,200")

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
GROUP="tmlr-humanoid-thr${TAG}-300m-rootrnd"
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
echo "Humanoid (root/CoM RND, 300M) threshold: [${THRESHOLD}]  Group: ${GROUP}"
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
