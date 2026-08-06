#!/bin/bash
#SBATCH --time=0-04:00:00
# jax[cuda13] dropped Volta (V100 / sm_70) support, so target H100 (dgxh) and
# A100 (ampere, sm_80) only.
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=IntrinsicPPO-MOHopper-RND
#SBATCH --output=IntrinsicPPO-MOHopper-RND_%j.out

# Stage-1 IntrinsicPPO on MOHopper: PPO trained on RND novelty only.
# Extrinsic breakthroughs measured at eval vs unlock thresholds [125, 125]
# (Hard point from the MORLAX sparse sweep).
#
# Submit after git pull on the HPC checkout:
#   sbatch scripts/slurm/intrinsic_ppo_mohopper_rnd_run0.sh
#   RUN_NAME=intrinsic-ppo-hopper-rnd-run1 sbatch scripts/slurm/intrinsic_ppo_mohopper_rnd_run0.sh
#
# Results: ${save_dir}/${RUN_NAME}
#   save_dir = /nfs/hpc/share/thakarr/SMORL/results/intrinsic_ppo_hopper

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
CONFIG=config/intrinsic/mohopper_rnd.yaml
RUN_NAME="${RUN_NAME:-intrinsic-ppo-hopper-rnd-job${SLURM_JOB_ID}}"

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
echo "Job:  ${SLURM_JOB_ID:-local}"
echo "Env:  ${ENV_DIR}"
echo "Code: ${CODE_DIR}"
echo "Config: ${CONFIG}"
echo "Run name: ${RUN_NAME}"
echo "PYTHONPATH: ${PYTHONPATH}"
nvidia-smi -L || true
"${ENV_DIR}/bin/python" -c "import jax; print('JAX devices:', jax.devices())"
"${ENV_DIR}/bin/python" -c "import moplayground; print('moplayground:', moplayground.__file__)"
"${ENV_DIR}/bin/python" -c "from moplayground.learning.training import _ALGO_HANDLERS; print('algos:', list(_ALGO_HANDLERS))"

export SMORL_RUN_NAME="${RUN_NAME}"
"${ENV_DIR}/bin/python" -m scripts.train "${CONFIG}" --name "${RUN_NAME}"
