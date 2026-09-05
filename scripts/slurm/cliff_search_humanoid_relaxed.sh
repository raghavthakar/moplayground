#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=cliff-humanoid-relaxed
#SBATCH --output=cliff-humanoid-relaxed_%j.out

# Relaxed sparsity-cliff search for MOHumanoid.
#
# The first pass aborted at ~25M and the 3-seed confirm at p≈0.035 was mixed
# (717k / 257k / 49k). This run waits until 60M before aborting and uses a
# 10% dense-HV floor so a real drop is not missed. NEW save-dir / W&B group
# so it does not resume the old cliff.json.
#
# 8h GPU slot. If the job times out, resubmit the same script (--resume
# skips finished probes in the *relaxed* cliff.json).
#
# Submit:
#   sbatch scripts/slurm/cliff_search_humanoid_relaxed.sh
# W&B group: cliff-search-humanoid-relaxed-100m
# Result: /nfs/hpc/share/thakarr/SMORL/results/cliff_search_relaxed/humanoid/cliff.json

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
DOMAIN=humanoid
SAVE_DIR=/nfs/hpc/share/thakarr/SMORL/results/cliff_search_relaxed/humanoid
GROUP=cliff-search-humanoid-relaxed-100m

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
