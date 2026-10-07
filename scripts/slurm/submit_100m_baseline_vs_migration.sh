#!/bin/bash
# Submit the 100M baseline-vs-migration sweep.
#
# Compares cold MORLAX (baseline-100m) with explore -> BC -> finetune at
# e20-f80, e25-f75, e30-f70, e35-f65, e40-f60, e45-f55, and e50-f50.
# Seeds 0-9. 80 jobs per threshold, 880 jobs total. W&B project SMORL-TMLR.
#
#   walker    300x50, 150x0, 200x0, 250x10    320 jobs
#   ant       450x650, 400x600                160 jobs
#   cheetah   90x240                           80 jobs
#   hopper    50x50, 100x100                  160 jobs
#   humanoid  70x40, 80x50                    160 jobs
#
# On the HPC login node:
#   bash /nfs/stak/users/thakarr/hpc-share/SMORL/moplayground/scripts/slurm/submit_100m_baseline_vs_migration.sh
#
# A second run skips any cell that already wrote run_meta.json.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB="${SCRIPT_DIR}/run_100m_baseline_vs_migration.sh"

submit() {
    local domain=$1
    local array=$2
    echo "sbatch mig100m-${domain} --array=${array}"
    sbatch --array="${array}" --job-name="mig100m-${domain}" \
        --export=ALL,DOMAIN="${domain}" \
        "${JOB}"
}

submit walker   0-319
submit ant      0-159
submit cheetah  0-79
submit hopper   0-159
submit humanoid 0-159
