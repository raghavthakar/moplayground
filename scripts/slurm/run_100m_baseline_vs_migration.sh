#!/bin/bash
#SBATCH --time=0-08:00:00
#SBATCH --partition=dgxh,ampere
#SBATCH --mem=32G
#SBATCH -c 12
#SBATCH -G 1
#SBATCH --output=%x_%A_%a.out

# Slurm body for the 100M baseline-vs-migration sweep.
# Do not sbatch this file alone. Submit it through
# scripts/slurm/submit_100m_baseline_vs_migration.sh
# or pass DOMAIN and --array yourself for one domain.
#
# Each threshold is 80 cells: baseline-100m + e20-f80 .. e50-f50, seeds 0-9.
# Save dirs and W&B groups are new, so --skip-existing will not resume the
# September runs.
#
#   DOMAIN=walker    thresholds 300x50, 150x0, 200x0, 250x10     array 0-319
#   DOMAIN=ant       thresholds 450x650, 400x600                 array 0-159
#   DOMAIN=cheetah   threshold  90x240                           array 0-79
#   DOMAIN=hopper    thresholds 50x50, 100x100                   array 0-159
#   DOMAIN=humanoid  thresholds 70x40, 80x50                     array 0-159
#
# W&B project SMORL-TMLR, group tmlr-<domain>-thr<tag>-100m.
# Charts: group by variant, filter job_type in {baseline, finetune}.
# Each eval appends the full return matrix to <run>/eval_fronts.jsonl and
# uploads that file onto the W&B run.

set -euo pipefail

if [[ -z "${DOMAIN:-}" ]]; then
    echo "DOMAIN is unset. Submit with --export=ALL,DOMAIN=<walker|ant|cheetah|hopper|humanoid>."
    exit 1
fi

ENV_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL
CODE_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL/moplayground
TOTAL_M=100
SPLITS="20,80;25,75;30,70;35,65;40,60;45,55;50,50"
SEEDS="0,1,2,3,4,5,6,7,8,9"
N_CELLS=80
ROOT=/nfs/stak/users/thakarr/hpc-share/SMORL/results/tmlr_migration_100m

case "${DOMAIN}" in
    walker)
        BASE=config/morlax/mowalker_sparse_migration_100m.yaml
        THRESHOLDS=("300,50" "150,0" "200,0" "250,10")
        ;;
    ant)
        BASE=config/morlax/moant_sparse_migration_100m.yaml
        THRESHOLDS=("450,650" "400,600")
        ;;
    cheetah)
        BASE=config/morlax/mocheetah_sparse_migration_100m.yaml
        THRESHOLDS=("90,240")
        ;;
    hopper)
        BASE=config/morlax/mohopper_sparse_migration_100m.yaml
        THRESHOLDS=("50,50" "100,100")
        ;;
    humanoid)
        BASE=config/morlax/mohumanoid_sparse_migration_100m.yaml
        THRESHOLDS=("70,40" "80,50")
        ;;
    *)
        echo "Unknown DOMAIN=${DOMAIN}"
        exit 1
        ;;
esac

INDEX="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${INDEX}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is unset. Pass --array=0-$(( ${#THRESHOLDS[@]} * N_CELLS - 1 ))."
    exit 1
fi

THR_IDX=$((INDEX / N_CELLS))
CELL_IDX=$((INDEX % N_CELLS))
LAST=$(( ${#THRESHOLDS[@]} * N_CELLS - 1 ))
if (( THR_IDX < 0 || THR_IDX >= ${#THRESHOLDS[@]} )); then
    echo "Index ${INDEX} out of range for ${DOMAIN} (need 0-${LAST})"
    exit 1
fi

THRESHOLD="${THRESHOLDS[$THR_IDX]}"
TAG="${THRESHOLD//,/x}"
GROUP="tmlr-${DOMAIN}-thr${TAG}-100m"
SAVE_DIR="${ROOT}/${DOMAIN}/thr${TAG}"

module load conda
source activate base
conda activate "${ENV_DIR}"

cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
unset WANDB_MODE

export WANDB_DATA_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL/wandb/data
export WANDB_CACHE_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL/wandb/cache
export WANDB_DIR=/nfs/stak/users/thakarr/hpc-share/SMORL/wandb/runs
mkdir -p "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DIR}" "${SAVE_DIR}"

echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID:-local} (array ${SLURM_ARRAY_JOB_ID:-NA} task ${INDEX})"
echo "Domain: ${DOMAIN}  Threshold: [${THRESHOLD}]  Group: ${GROUP}"
echo "SaveDir: ${SAVE_DIR}  Cell: ${CELL_IDX}"
echo "Budget: ${TOTAL_M}M  Splits: ${SPLITS}  Seeds: ${SEEDS}"
nvidia-smi -L || true

"${ENV_DIR}/bin/python" -m scripts.migration_sweep \
    --base "${BASE}" \
    --save-dir "${SAVE_DIR}" \
    --group "${GROUP}" \
    --threshold "${THRESHOLD}" \
    --total-m "${TOTAL_M}" \
    --splits "${SPLITS}" \
    --seeds "${SEEDS}" \
    --index "${CELL_IDX}" \
    --skip-existing
