#!/bin/bash
#SBATCH --time=0-03:00:00
# jax[cuda13] dropped Volta (V100 / sm_70) support, so target H100 (dgxh) and
# A100 (ampere, sm_80) only. See morlax_mocheetah_run0.sh for the rationale.
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-MOHopper-seed
#SBATCH --output=MORLAX-MOHopper-seed_%A_%a.out
# One array task per seed in SEEDS below.
# 5 seeds -> array indices 0..4. If you edit SEEDS, update this range.
#SBATCH --array=0-4

# Seed sweep for a single sparsity threshold (MOHopper, vanilla MORLAX).
# All seeds land in one W&B group so replicates stay grouped, not scattered.
# Submit:
#   sbatch scripts/slurm/morlax_mohopper_seed_sweep.sh
# One seed interactively for debugging:
#   SLURM_ARRAY_TASK_ID=0 bash scripts/slurm/morlax_mohopper_seed_sweep.sh
#
# Results land in: ${save_dir}/${base_name}-thr=<run>x<jump>-seed=<seed>
# W&B group: sparse-thr=<run>x<jump>  (auto, unless GROUP is set)

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
CONFIG=config/morlax/mohopper_sparse.yaml

# Fixed threshold for this seed sweep (objective order [run, jump]).
THRESHOLD="50,50"
# Seeds to run — keep the --array range above equal to the seed count.
SEEDS="0,1,2,3,4"
# W&B group name. Leave empty to auto-derive as sparse-thr=<run>x<jump>.
GROUP=""

# Array index selects the seed. Falls back to 0 for interactive runs.
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
echo "Threshold: [${THRESHOLD}]  Seeds: [${SEEDS}]  Group: '${GROUP:-auto}'"
echo "Seed index: ${INDEX}"
echo "PYTHONPATH: ${PYTHONPATH}"
nvidia-smi -L || true
"${ENV_DIR}/bin/python" -c "import jax; print('JAX devices:', jax.devices())"
"${ENV_DIR}/bin/python" -c "import moplayground; print('moplayground:', moplayground.__file__)"

GROUP_ARGS=()
if [[ -n "${GROUP}" ]]; then
  GROUP_ARGS=(--group "${GROUP}")
fi

"${ENV_DIR}/bin/python" -m scripts.seed_sweep \
    --base "${CONFIG}" \
    --threshold "${THRESHOLD}" \
    --seeds "${SEEDS}" \
    --index "${INDEX}" \
    "${GROUP_ARGS[@]}" \
    --skip-existing
