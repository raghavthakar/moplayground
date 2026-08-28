#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --job-name=MORLAX-bc-probe
#SBATCH --output=MORLAX-bc-probe_%j.out

# Sparse MOHopper behavior-cloning transfer probe (50x50 trap).
# Collects teacher demos once, then runs a standalone BC probe: a cold MORLAX
# hypernetwork is trained purely by supervised BC on the demo buffer (no PPO)
# and evaluated on the ungated env at each teacher's labeled preference.
#
# Submit:
#   sbatch scripts/slurm/morlax_mohopper_sparse_bc.sh
#   BC_STEPS=30000 BC_LR=5e-4 sbatch scripts/slurm/morlax_mohopper_sparse_bc.sh

set -euo pipefail

ENV_DIR=/nfs/hpc/share/thakarr/SMORL
CODE_DIR=/nfs/hpc/share/thakarr/SMORL/moplayground
CONFIG=config/morlax/mohopper_sparse_bc.yaml
DEMO_BUFFER=/nfs/hpc/share/thakarr/SMORL/data/mohopper_sparse_bc_demos.npz
BC_STEPS="${BC_STEPS:-20000}"
BC_BATCH="${BC_BATCH:-512}"
BC_LR="${BC_LR:-1e-3}"

module load conda
source activate base
conda activate "${ENV_DIR}"

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "$(dirname "${DEMO_BUFFER}")"

if [[ ! -f "${DEMO_BUFFER}" ]]; then
  echo "Collecting teacher demos -> ${DEMO_BUFFER}"
  "${ENV_DIR}/bin/python" -m scripts.collect_teacher_demos \
    "${CONFIG}" \
    --output "${DEMO_BUFFER}"
else
  echo "Reusing existing demo buffer: ${DEMO_BUFFER}"
fi

echo "Running BC probe (steps=${BC_STEPS} batch=${BC_BATCH} lr=${BC_LR})"
"${ENV_DIR}/bin/python" -m scripts.bc_probe "${CONFIG}" \
  --steps "${BC_STEPS}" --batch "${BC_BATCH}" --lr "${BC_LR}"
