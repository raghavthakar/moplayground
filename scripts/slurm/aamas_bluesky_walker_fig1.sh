#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=aamas-walker-dense-100m
#SBATCH --output=aamas-walker-dense-100m_%A_%a.out
# 10 seeds (0-9) of vanilla MORLAX (baseline-100m) on homogeneous dense MOWalker (T=[0,0]).
#SBATCH --array=0-9

# AAMAS 2027 Blue Sky — Figure 1 Diagnostic Experiment (Left Panel: Homogeneous MORLAX)
#
# Runs 10 seeds (0-9) of standard dense MORLAX (baseline-100m, T=[0,0]) on MOWalker
# for 100M steps with the exact same architecture, hyperparameters, and eval cadence
# (num_evals=21) as the heterogeneous MOWalker runs already in W&B:
#   - Homogeneous (Left Panel):  group = tmlr-walker-thr0x0-100m    (this script, T=[0,0])
#   - Heterogeneous (Right Panel): group = tmlr-walker-thr150x0-100m  (already finished in W&B, T=[150,0])
#
# Note: In scripts.migration_sweep, cells 0..9 correspond to baseline-100m (seeds 0..9),
# and passing --threshold "0,0" sets episodic_threshold.enabled = False (pure per-step rewards).
#
# Submit from the moplayground checkout on the login node:
#   sbatch scripts/slurm/aamas_bluesky_walker_fig1.sh

set -euo pipefail

ENV_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL
CODE_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL/moplayground
BASE=config/morlax/mowalker_sparse_migration_100m.yaml
TOTAL_M=100
SPLITS="20,80"
SEEDS="0,1,2,3,4,5,6,7,8,9"
THRESHOLD="0,0"
TAG="0x0"
GROUP="tmlr-walker-thr${TAG}-100m"
SAVE_DIR="/nfs/stak/users/thakarr/hpc-share/SMORL/results/tmlr_migration_100m/walker/thr${TAG}"

INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if (( INDEX < 0 || INDEX >= 10 )); then
    echo "Index ${INDEX} out of range (need 0-9 for baseline-100m seeds 0-9)"
    exit 1
fi

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
echo "MOWalker homogeneous dense baseline (100M) threshold: [${THRESHOLD}]  Group: ${GROUP}"
echo "SaveDir: ${SAVE_DIR}  Cell (seed): ${INDEX}"
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
