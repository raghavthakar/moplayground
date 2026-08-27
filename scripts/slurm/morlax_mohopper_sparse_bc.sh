#!/bin/bash
#SBATCH --time=0-04:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-sparse-BC
#SBATCH --output=MORLAX-sparse-BC_%j.out

# Sparse MOHopper MORLAX + offline BC transfer validation (50x50 trap).
# Collects teacher demos once, then trains with morlax_mohopper_sparse_bc.yaml.
#
# Submit:
#   sbatch scripts/slurm/morlax_mohopper_sparse_bc.sh
#   RUN_NAME=morlax-bc-thr50-seed0 sbatch scripts/slurm/morlax_mohopper_sparse_bc.sh
#
# Optional sweep over BC anchor strength (array index -> bc_coef):
#   BC_COEF=2.0 BC_COEF_FINAL=0.2 RUN_NAME=morlax-bc-lam2-seed0 sbatch ...

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
CONFIG=config/morlax/mohopper_sparse_bc.yaml
DEMO_BUFFER=/nfs/hpc/share/thakarr/SMORL/data/mohopper_sparse_bc_demos.npz
RUN_NAME="${RUN_NAME:-morlax-bc-thr50-seed0}"

module load conda
source activate base
conda activate "${ENV_DIR}"

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset WANDB_MODE

mkdir -p "$(dirname "${DEMO_BUFFER}")"

if [[ ! -f "${DEMO_BUFFER}" ]]; then
  echo "Collecting teacher demos -> ${DEMO_BUFFER}"
  "${ENV_DIR}/bin/python" -m scripts.collect_teacher_demos \
    "${CONFIG}" \
    --output "${DEMO_BUFFER}"
else
  echo "Reusing existing demo buffer: ${DEMO_BUFFER}"
fi

export SMORL_RUN_NAME="${RUN_NAME}"
echo "Training run: ${RUN_NAME}"
"${ENV_DIR}/bin/python" -m scripts.train "${CONFIG}" --name "${RUN_NAME}"
