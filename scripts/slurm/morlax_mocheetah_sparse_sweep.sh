#!/bin/bash
#SBATCH --time=0-02:00:00
# jax[cuda13] dropped Volta (V100 / sm_70) support, so target H100 (dgxh) and
# A100 (ampere, sm_80) only. See morlax_mocheetah_run0.sh for the rationale.
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-MOCheetah-sparse
#SBATCH --output=MORLAX-MOCheetah-sparse_%A_%a.out
# 3 weights x 3 step_sizes = 9 combos -> array indices 0..8.
# If you change the grid below, update this range to match.
#SBATCH --array=0-8

# Sparse-run MOCheetah sanity sweep over (milestone_weight, step_size).
# Each array task runs exactly one grid combo via scripts.sparse_sweep --index.
# Submit:
#   sbatch scripts/slurm/morlax_mocheetah_sparse_sweep.sh
# Run one combo interactively for debugging:
#   SLURM_ARRAY_TASK_ID=0 bash scripts/slurm/morlax_mocheetah_sparse_sweep.sh
#
# Results land in: ${save_dir}/${base_name}-w=<weight>-ss=<step_size>
# (save_dir + base name come from the config; the driver appends the combo tag.)

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
CONFIG=config/morlax/mocheetah_sparse.yaml

# Grid — must match the --array range above (len(weights) * len(step_sizes)).
MILESTONE_WEIGHTS="20,100,500"
STEP_SIZES="1.0,3.0,6.0"

# Array index selects the combo. Falls back to 0 for interactive runs.
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
echo "Grid: weights=[${MILESTONE_WEIGHTS}] step_sizes=[${STEP_SIZES}]"
echo "Combo index: ${INDEX}"
echo "PYTHONPATH: ${PYTHONPATH}"
nvidia-smi -L || true
"${ENV_DIR}/bin/python" -c "import jax; print('JAX devices:', jax.devices())"
"${ENV_DIR}/bin/python" -c "import moplayground; print('moplayground:', moplayground.__file__)"

"${ENV_DIR}/bin/python" -m scripts.sparse_sweep \
    --base "${CONFIG}" \
    --milestone-weights "${MILESTONE_WEIGHTS}" \
    --step-sizes "${STEP_SIZES}" \
    --index "${INDEX}" \
    --skip-existing
