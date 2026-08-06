#!/bin/bash
#SBATCH --time=0-03:00:00
# jax[cuda13] dropped Volta (V100 / sm_70) support, so target H100 (dgxh) and
# A100 (ampere, sm_80) only. See morlax_mocheetah_run0.sh for the rationale.
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-MOHopper-sparse
#SBATCH --output=MORLAX-MOHopper-sparse_%A_%a.out
# One array task per threshold set in THRESHOLDS below.
# 6 sets (0..500 in steps of 100) -> array indices 0..5. If you edit THRESHOLDS,
# update this range to match the set count.
#SBATCH --array=0-5

# Episodic-return sparsity sweep for MOHopper (run vs jump), vanilla MORLAX,
# no algorithmic changes. Each array task runs one literal per-objective
# threshold set via scripts.sparse_threshold_sweep --index. Finds where the
# basic algo breaks.
#
# NOTE: the reward structure is now clean (no shared `alive`/`ctrl_cost` added
# to the objective dimensions; `jump` is zero at rest). Objective returns are
# therefore on a different — and asymmetric — scale than the old runs, so the
# previous 125,125 failure point is void. Run the dense baseline (index 0 =
# 0,0) FIRST to read the run/jump return ceilings, then recalibrate THRESHOLDS
# (likely asymmetric, e.g. run threshold >> jump threshold) and the array range.
# Submit:
#   sbatch scripts/slurm/morlax_mohopper_sparse_sweep.sh
# One set interactively for debugging:
#   SLURM_ARRAY_TASK_ID=0 bash scripts/slurm/morlax_mohopper_sparse_sweep.sh
#
# Results land in: ${save_dir}/${base_name}-thr=<run>x<jump>
# (save_dir = morlax_hopper_sparse_sweep, from config/morlax/mohopper_sparse.yaml)

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
CONFIG=config/morlax/mohopper_sparse.yaml

# Literal per-objective episodic-return thresholds, one set per ';'-group,
# each ','-separated in objective order [run, jump]. All-zero => dense baseline.
# Placeholder ladder — recalibrate after the dense (0,0) run (see header note).
# Edit these directly; keep the --array range above equal to the set count.
THRESHOLDS="0,0;25,25;50,50;75,75;100,100;125,125;150,150;175,175;200,200"

# Array index selects the threshold set. Falls back to 0 for interactive runs.
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
echo "Thresholds: [${THRESHOLDS}]"
echo "Threshold-set index: ${INDEX}"
echo "PYTHONPATH: ${PYTHONPATH}"
nvidia-smi -L || true
"${ENV_DIR}/bin/python" -c "import jax; print('JAX devices:', jax.devices())"
"${ENV_DIR}/bin/python" -c "import moplayground; print('moplayground:', moplayground.__file__)"

"${ENV_DIR}/bin/python" -m scripts.sparse_threshold_sweep \
    --base "${CONFIG}" \
    --thresholds "${THRESHOLDS}" \
    --index "${INDEX}" \
    --skip-existing
