#!/bin/bash
# Submit the fresh TMLR 100M migration campaign (880 cells).
# Run this on the HPC login node, from anywhere:
#   bash /nfs/stak/users/thakarr/hpc-share/SMORL/moplayground/scripts/slurm/tmlr_submit_all.sh
#
# Resubmitting is safe: finished cells have run_meta.json and are skipped.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB="${SCRIPT_DIR}/tmlr_migration_100m.sh"

submit() {
    local domain=$1
    local array=$2
    echo "sbatch ${domain} --array=${array}"
    sbatch --array="${array}" --job-name="tmlr-${domain}" \
        --export=ALL,DOMAIN="${domain}" \
        "${JOB}"
}

submit walker   0-319
submit ant      0-159
submit cheetah  0-79
submit hopper   0-159
submit humanoid 0-159
