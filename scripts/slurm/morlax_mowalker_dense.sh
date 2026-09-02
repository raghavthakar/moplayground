#!/bin/bash
#SBATCH --time=0-08:00:00
# jax[cuda13] dropped Volta (V100 / sm_70) support, so target H100 (dgxh) and
# A100 (ampere, sm_80) only.
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-MOWalker-dense
#SBATCH --output=MORLAX-MOWalker-dense_%A_%a.out
# Dense (no-sparsity) MORLAX calibration on MOWalker. One array task per seed.
# 3 seeds -> array 0..2. If you edit SEEDS, update this range.
#SBATCH --array=0-2

# Calibration baseline: dense MORLAX MOWalker to read achievable per-objective
# returns (eval/return/<label>/max in W&B). Those set the sparsity ceiling for
# the later threshold sweep. --threshold "0,0" keeps episodic gating disabled.
#
#   sbatch scripts/slurm/morlax_mowalker_dense.sh
# One seed interactively:
#   SLURM_ARRAY_TASK_ID=0 bash scripts/slurm/morlax_mowalker_dense.sh

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
CONFIG=config/morlax/mowalker_dense.yaml

# All-zero threshold => dense (episodic gating stays disabled).
THRESHOLD="0,0"
SEEDS="0,1,2"
GROUP="walker-dense-calib-60m"
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
echo "Config: ${CONFIG}  Group: ${GROUP}  Threshold: [${THRESHOLD}] (dense)"
nvidia-smi -L || true
"${ENV_DIR}/bin/python" -c "import jax; print('JAX devices:', jax.devices())"

"${ENV_DIR}/bin/python" -m scripts.seed_sweep \
    --base "${CONFIG}" \
    --threshold "${THRESHOLD}" \
    --seeds "${SEEDS}" \
    --group "${GROUP}" \
    --index "${INDEX}" \
    --skip-existing
