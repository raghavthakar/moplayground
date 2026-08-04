#!/bin/bash
#SBATCH --time=0-03:00:00
# jax[cuda13] dropped Volta (V100 / sm_70) support, so target H100 (dgxh) and
# A100 (ampere, sm_80) only. See morlax_mocheetah_run0.sh for the rationale.
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-MOHopper-sparse
#SBATCH --output=MORLAX-MOHopper-sparse_%A_%a.out
# 6 sparsity levels -> array indices 0..5.
# If you change --fractions below, update this range to match.
#SBATCH --array=0-5

# Naive episodic-return sparsity sweep for MOHopper (run vs jump), vanilla
# MORLAX, no algorithmic changes. Each array task runs one sparsity level via
# scripts.sparse_threshold_sweep --index. Finds where the basic algo breaks.
# Submit:
#   sbatch scripts/slurm/morlax_mohopper_sparse_sweep.sh
# One level interactively for debugging:
#   SLURM_ARRAY_TASK_ID=0 bash scripts/slurm/morlax_mohopper_sparse_sweep.sh
#
# Results land in: ${save_dir}/${base_name}-sparsity=<fraction>
# (save_dir = morlax_hopper_sparse_sweep, from config/morlax/mohopper_sparse.yaml)

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
CONFIG=config/morlax/mohopper_sparse.yaml

# Sweep grid — must match the --array range above (one task per fraction).
# Thresholds per objective = fraction * OBJ_MAXES (run~2500, jump~2200 @ 50M).
FRACTIONS="0.0,0.25,0.5,0.7,0.85,0.95"
OBJ_MAXES="2500,2200"

# Array index selects the sparsity level. Falls back to 0 for interactive runs.
INDEX="${SLURM_ARRAY_TASK_ID:-0}"

module load conda
source activate base
conda activate "${ENV_DIR}"

cd "${CODE_DIR}"

# Use the git checkout directly (src layout) — no pip install needed.
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# wandb: credentials from `wandb login` on the login node (~/.netrc).
unset WANDB_MODE

echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID:-local} (array ${SLURM_ARRAY_JOB_ID:-NA} task ${SLURM_ARRAY_TASK_ID:-NA})"
echo "Env:  ${ENV_DIR}"
echo "Code: ${CODE_DIR}"
echo "Config: ${CONFIG}"
echo "Fractions: [${FRACTIONS}]  Obj maxes: [${OBJ_MAXES}]"
echo "Sparsity index: ${INDEX}"
echo "PYTHONPATH: ${PYTHONPATH}"
nvidia-smi -L || true
"${ENV_DIR}/bin/python" -c "import jax; print('JAX devices:', jax.devices())"
"${ENV_DIR}/bin/python" -c "import moplayground; print('moplayground:', moplayground.__file__)"

"${ENV_DIR}/bin/python" -m scripts.sparse_threshold_sweep \
    --base "${CONFIG}" \
    --fractions "${FRACTIONS}" \
    --obj-maxes "${OBJ_MAXES}" \
    --index "${INDEX}" \
    --skip-existing
