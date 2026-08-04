#!/bin/bash
#SBATCH --time=0-04:00:00
# jax[cuda13] dropped Volta (V100 / sm_70) support, so target H100 (dgxh) and
# A100 (ampere, sm_80) only. See morlax_mocheetah_run0.sh for the rationale.
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-MOHopper
#SBATCH --output=MORLAX-MOHopper_%j.out

# Dense MORLAX multi-objective Hopper (MOHopper: run vs jump) on OSU HPC.
# Pre-sparsification baseline for the heterogeneous-sparsity study.
# Submit from anywhere after git pull:
#   sbatch scripts/slurm/morlax_mohopper_run0.sh
#   RUN_NAME=morlax-hopper-run1 sbatch scripts/slurm/morlax_mohopper_run0.sh
#
# Results land in: ${save_dir}/${RUN_NAME}
# Default RUN_NAME appends the Slurm job id so repeats never overwrite.
#
# Layout assumed (matches local SMORL layout):
#   ENV_DIR  = conda env prefix
#   CODE_DIR = moplayground git checkout

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
CONFIG=config/morlax/mohopper.yaml
# Unique per job unless you explicitly set RUN_NAME=...
RUN_NAME="${RUN_NAME:-morlax-hopper-job${SLURM_JOB_ID}}"

module load conda
source activate base
conda activate "${ENV_DIR}"

cd "${CODE_DIR}"

# Use the git checkout directly (src layout) — no pip install needed.
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# wandb: credentials from `wandb login` on the login node (~/.netrc).
# train.py logs to entity/project set in scripts/train.py.
unset WANDB_MODE

echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID:-local}"
echo "Env:  ${ENV_DIR}"
echo "Code: ${CODE_DIR}"
echo "Config: ${CONFIG}"
echo "Run name: ${RUN_NAME}"
echo "PYTHONPATH: ${PYTHONPATH}"
nvidia-smi -L || true
"${ENV_DIR}/bin/python" -c "import jax; print('JAX devices:', jax.devices())"
"${ENV_DIR}/bin/python" -c "import moplayground; print('moplayground:', moplayground.__file__)"

export SMORL_RUN_NAME="${RUN_NAME}"
"${ENV_DIR}/bin/python" -m scripts.train "${CONFIG}" --name "${RUN_NAME}"
