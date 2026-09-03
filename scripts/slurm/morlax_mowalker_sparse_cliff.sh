#!/bin/bash
#SBATCH --time=0-16:00:00
# jax[cuda13] dropped Volta (V100 / sm_70) support, so target H100 (dgxh) and
# A100 (ampere, sm_80) only.
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-MOWalker-sparse
#SBATCH --output=MORLAX-MOWalker-sparse_%A_%a.out
# 6 thresholds x 3 seeds = 18 cells. Index = thr_idx * 3 + seed_idx.
#SBATCH --array=0-17

# Naive MORLAX cliff on MOWalker, 100M frames. Thresholds are baked in from
# the dense calibration (group walker-dense-calib-60m), NOT percent-of-max:
#
#   nadir  = [0, -100]      # run min ~0, energy min ~-100
#   ideal  = [2500, 1900]   # run max ~2500, energy max ~1900
#   T      = nadir + p * (ideal - nadir)
#
#   p     run   energy
#   0.01    25     -80
#   0.02    50     -60
#   0.04   100     -20     # hopper-equivalent "4% of range"
#   0.08   200      60
#   0.16   400     220
#   0.32   800     540
#
# p=0 is the dense run already finished — not repeated. Energy at low p is
# easy to unlock (mean dense energy ~1200); run is the sparse objective.
#
#   sbatch scripts/slurm/morlax_mowalker_sparse_cliff.sh

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
CONFIG=config/morlax/mowalker_sparse.yaml

# Order matches p = 0.01, 0.02, 0.04, 0.08, 0.16, 0.32 (run, energy).
THRESHOLDS=(
  "25,-80"
  "50,-60"
  "100,-20"
  "200,60"
  "400,220"
  "800,540"
)
SEEDS="0,1,2"
N_SEEDS=3
GROUP="walker-sparse-cliff-100m"
INDEX="${SLURM_ARRAY_TASK_ID:-0}"

N_THR=${#THRESHOLDS[@]}
N_CELLS=$((N_THR * N_SEEDS))
if (( INDEX < 0 || INDEX >= N_CELLS )); then
  echo "INDEX ${INDEX} out of range for ${N_CELLS} cells" >&2
  exit 1
fi
THR_IDX=$((INDEX / N_SEEDS))
SEED_IDX=$((INDEX % N_SEEDS))
THRESHOLD="${THRESHOLDS[$THR_IDX]}"

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
echo "Cell: thr_idx=${THR_IDX} seed_idx=${SEED_IDX} threshold=[${THRESHOLD}]"
nvidia-smi -L || true

"${ENV_DIR}/bin/python" -m scripts.seed_sweep \
    --base "${CONFIG}" \
    --threshold "${THRESHOLD}" \
    --seeds "${SEEDS}" \
    --group "${GROUP}" \
    --index "${SEED_IDX}" \
    --skip-existing
