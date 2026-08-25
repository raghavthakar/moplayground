#!/bin/bash
#SBATCH --time=0-04:00:00
# jax[cuda13] dropped Volta (V100 / sm_70) support, so target H100 (dgxh) and
# A100 (ampere, sm_80) only.
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=IntrinsicPPO-MOHopper-scale
#SBATCH --output=IntrinsicPPO-MOHopper-scale_%A_%a.out
# One array task per (scale, seed) cell. Scale-major:
#   index = scale_idx * n_seeds + seed_idx
# 5 scales × 5 seeds = 25 cells -> array indices 0..24.
# If you edit SCALES or SEEDS, update this range.
#SBATCH --array=0-24

# IntrinsicPPO + RND: 50x50 unlock threshold, 5 novelty scales × 5 seeds.
# All 25 runs land in one W&B group.
# Submit after git pull on the HPC checkout:
#   sbatch scripts/slurm/intrinsic_ppo_mohopper_scale_sweep.sh
# One cell interactively:
#   SLURM_ARRAY_TASK_ID=0 bash scripts/slurm/intrinsic_ppo_mohopper_scale_sweep.sh
#
# Results: ${save_dir}/${base_name}-thr=50x50-iscale=<s>-seed=<seed>
# W&B group: intrinsic-rnd-thr=50x50  (auto, unless GROUP is set)

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
CONFIG=config/intrinsic/mohopper_rnd.yaml
SAVE_DIR=/nfs/hpc/share/thakarr/SMORL/results/intrinsic_ppo_hopper_scale_sweep

# Unlock bar for eval (objective order [run, jump]). Matches MORLAX collapse.
THRESHOLD="50,50"
# Log-spaced RND novelty coefficients (multiplier on std-normalized prediction error).
SCALES="0.01,0.1,1,10,100"
SEEDS="0,1,2,3,4"
# W&B group. Leave empty to auto-derive as intrinsic-rnd-thr=<run>x<jump>.
GROUP=""

INDEX="${SLURM_ARRAY_TASK_ID:-0}"

module load conda
source activate base
conda activate "${ENV_DIR}"

cd "${CODE_DIR}"

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
echo "Save dir: ${SAVE_DIR}"
echo "Threshold: [${THRESHOLD}]  Scales: [${SCALES}]  Seeds: [${SEEDS}]"
echo "Group: '${GROUP:-auto}'  Cell index: ${INDEX}"
echo "PYTHONPATH: ${PYTHONPATH}"
nvidia-smi -L || true
"${ENV_DIR}/bin/python" -c "import jax; print('JAX devices:', jax.devices())"
"${ENV_DIR}/bin/python" -c "import moplayground; print('moplayground:', moplayground.__file__)"

GROUP_ARGS=()
if [[ -n "${GROUP}" ]]; then
  GROUP_ARGS=(--group "${GROUP}")
fi

"${ENV_DIR}/bin/python" -m scripts.intrinsic_scale_sweep \
    --base "${CONFIG}" \
    --threshold "${THRESHOLD}" \
    --scales "${SCALES}" \
    --seeds "${SEEDS}" \
    --save-dir "${SAVE_DIR}" \
    --index "${INDEX}" \
    "${GROUP_ARGS[@]}" \
    --skip-existing
