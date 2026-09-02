#!/bin/bash
#SBATCH --time=0-08:00:00
# jax[cuda13] dropped Volta (V100 / sm_70) support, so target H100 (dgxh) and
# A100 (ampere, sm_80) only.
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=RND-Explore-MOWalker
#SBATCH --output=RND-Explore-MOWalker_%A_%a.out
# 5 seeds of return-agnostic RND exploration (scale=1) on MOWalker, 60M frames,
# archive snapshot every ~5M. One array task per seed.
#SBATCH --array=0-4

# Stage-1 master explorer for the walker sparsity study. Training reward is pure
# RND novelty (threshold-independent); every unlocking eval + checkpoint is kept
# in the extrinsic archive. The negligible threshold "1,1" makes retention
# return-agnostic, so this single run is a reusable teacher bank for ANY real
# sparsity threshold (teachers selected offline later).
#
#   sbatch scripts/slurm/intrinsic_ppo_mowalker_explore.sh
# One seed interactively:
#   SLURM_ARRAY_TASK_ID=0 bash scripts/slurm/intrinsic_ppo_mowalker_explore.sh

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
CONFIG=config/intrinsic/mowalker_rnd.yaml
SAVE_DIR=/nfs/hpc/share/thakarr/SMORL/results/walker_explore_agnostic

# Negligible unlock threshold => return-agnostic archive retention.
THRESHOLD="1,1"
SCALES="1"
SEEDS="0,1,2,3,4"
GROUP="walker-explore-agnostic-60m"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"

module load conda
source activate base
conda activate "${ENV_DIR}"

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset WANDB_MODE

echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID:-local} (array ${SLURM_ARRAY_JOB_ID:-NA} task ${INDEX})"
echo "Config: ${CONFIG}  Group: ${GROUP}  Threshold: [${THRESHOLD}] (agnostic)"
nvidia-smi -L || true
"${ENV_DIR}/bin/python" -c "import jax; print('JAX devices:', jax.devices())"

"${ENV_DIR}/bin/python" -m scripts.intrinsic_scale_sweep \
    --base "${CONFIG}" \
    --threshold "${THRESHOLD}" \
    --scales "${SCALES}" \
    --seeds "${SEEDS}" \
    --save-dir "${SAVE_DIR}" \
    --group "${GROUP}" \
    --index "${INDEX}" \
    --skip-existing
