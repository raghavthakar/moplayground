#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-migrate
#SBATCH --output=MORLAX-migrate_%j.out

# Single-run explore -> BC -> finetune migration on sparse MOHopper (50x50).
# One 50M budget: ~10M IntrinsicPPO exploration, BC of the Pareto archive into
# the MORLAX hypernetwork, then ~40M MORLAX finetuning from that init.
#
# Submit:
#   sbatch scripts/slurm/morlax_mohopper_sparse_migration.sh
#   RUN_NAME=migrate-thr50-seed0 sbatch scripts/slurm/morlax_mohopper_sparse_migration.sh

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
CONFIG=config/morlax/mohopper_sparse_migration.yaml
RUN_NAME="${RUN_NAME:-migrate-thr50-seed0}"

module load conda
source activate base
conda activate "${ENV_DIR}"

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export SMORL_RUN_NAME="${RUN_NAME}"
echo "Migration run: ${RUN_NAME}"
"${ENV_DIR}/bin/python" -m scripts.train_migration "${CONFIG}" --name "${RUN_NAME}"
