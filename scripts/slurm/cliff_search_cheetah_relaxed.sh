#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=cliff-cheetah-relaxed
#SBATCH --output=cliff-cheetah-relaxed_%j.out

# Relaxed sparsity-cliff search for MOCheetah.
#
# The first pass aborted at ~25M and used a 2% dense-HV floor, so Cheetah's
# real ~15x HV drop at p≈0.038 (still ~4% of dense) was never flagged and
# the job reported p=0.5. This run:
#   - waits until 60M before aborting collapsed probes
#   - treats HV < 10% of dense as collapsed (so a 15x drop counts)
#   - writes a NEW save-dir / W&B group so it does not resume the old cliff.json
#
# 8h GPU slot. If the job times out, resubmit the same script (--resume
# skips finished probes in the *relaxed* cliff.json).
#
# Submit:
#   sbatch scripts/slurm/cliff_search_cheetah_relaxed.sh
# W&B group: cliff-search-cheetah-relaxed-100m
# Result: /nfs/hpc/share/thakarr/SMORL/results/cliff_search_relaxed/cheetah/cliff.json

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
DOMAIN=cheetah
SAVE_DIR=/nfs/hpc/share/thakarr/SMORL/results/cliff_search_relaxed/cheetah
GROUP=cliff-search-cheetah-relaxed-100m

module load conda
source activate base
conda activate "${ENV_DIR}"

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset WANDB_MODE

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID:-local}  Domain: ${DOMAIN}"
echo "SaveDir: ${SAVE_DIR}  Group: ${GROUP}"
echo "Relaxed abort: min_steps=60M  consecutive=4  collapse_frac=0.10"
nvidia-smi -L || true

"${ENV_DIR}/bin/python" -m scripts.cliff_search \
    --domain "${DOMAIN}" \
    --save-dir "${SAVE_DIR}" \
    --group "${GROUP}" \
    --abort-min-steps 60000000 \
    --abort-consecutive 4 \
    --collapse-frac 0.10 \
    --resume
