#!/bin/bash
#SBATCH --time=0-08:00:00
# jax[cuda13] dropped Volta (V100 / sm_70) support, so target H100 (dgxh) and
# A100 (ampere, sm_80) only.
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=cliff-walker
#SBATCH --output=cliff-walker_%j.out

# Sequential sparsity-cliff search for MOWalker (100M probes, early abort).
# 8h GPU slot — not 2 days. If the job times out, resubmit the same script
# (--resume skips finished probes in cliff.json).
# Submit all domains in parallel (one 8h job each):
#   mkdir -p logs
#   sbatch scripts/slurm/cliff_search_walker.sh
#   sbatch scripts/slurm/cliff_search_hopper.sh
#   sbatch scripts/slurm/cliff_search_cheetah.sh
#   sbatch scripts/slurm/cliff_search_ant.sh
#   sbatch scripts/slurm/cliff_search_humanoid.sh
# Result: /nfs/hpc/share/thakarr/SMORL/results/cliff_search/walker/cliff.json

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
DOMAIN=walker

module load conda
source activate base
conda activate "${ENV_DIR}"

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset WANDB_MODE

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Domain: ${DOMAIN}"
nvidia-smi -L || true

"${ENV_DIR}/bin/python" -m scripts.cliff_search --domain "${DOMAIN}" --resume
