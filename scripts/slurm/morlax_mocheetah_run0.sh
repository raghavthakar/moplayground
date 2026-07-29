#!/bin/bash
#SBATCH --time=0-02:00:00
#SBATCH --partition=dgx2,dgxh,ampere
#SBATCH --constraint=skylake
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-MOCheetah_run0
#SBATCH --output=MORLAX-MOCheetah_run0_%j.out

# MORLAX multi-objective HalfCheetah (MOCheetah) on OSU HPC.
# Submit from anywhere after git pull:
#   sbatch scripts/slurm/morlax_mocheetah_run0.sh
#
# Layout assumed (matches local SMORL layout):
#   ENV_DIR  = conda env prefix
#   CODE_DIR = moplayground git checkout

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
CONFIG=config/morlax/mocheetah.yaml

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
echo "PYTHONPATH: ${PYTHONPATH}"
"${ENV_DIR}/bin/python" -c "import jax; print('JAX devices:', jax.devices())"
"${ENV_DIR}/bin/python" -c "import moplayground; print('moplayground:', moplayground.__file__)"

"${ENV_DIR}/bin/python" -m scripts.train "${CONFIG}"
