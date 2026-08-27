#!/bin/bash
#SBATCH --time=0-04:00:00
# jax[cuda13] dropped Volta (V100 / sm_70) support, so target H100 (dgxh) and
# A100 (ampere, sm_80) only.
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=Archive-RND-MOHopper
#SBATCH --output=Archive-RND-MOHopper_%A_%a.out
# 5 seeds of RND scale=1 at 50x50, with extrinsic archive.
#SBATCH --array=0-4

# Pure-exploration retention: IntrinsicPPO + RND (scale 1) on MOHopper.
# Unlocking evals are copied into an extrinsic archive; each eval plots
# that archive's Pareto front. Compare in W&B against the MORLAX jobs
# from scripts/slurm/morlax_mohopper_archive.sh (same group).
#
#   sbatch scripts/slurm/intrinsic_ppo_mohopper_archive.sh

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
CONFIG=config/intrinsic/mohopper_rnd.yaml
SAVE_DIR=/nfs/hpc/share/thakarr/SMORL/results/retention_archive/rnd

THRESHOLD="50,50"
SCALES="1"
SEEDS="0,1,2,3,4"
GROUP="retention-archive-thr=50x50"
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
echo "Config: ${CONFIG}  Group: ${GROUP}"
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
