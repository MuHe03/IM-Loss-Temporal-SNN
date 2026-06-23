#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="${1:-}"

if [[ ! "${TASK_ID}" =~ ^[0-9]+$ ]] || ((TASK_ID < 0 || TASK_ID > 4)); then
  cat >&2 <<'EOF'
Usage: scripts/train_ff_output_threshold_im_local.sh TASK_ID

TASK_ID selects one seed:
  0=2020, 1=42, 2=123, 3=309, 4=969
EOF
  exit 2
fi

export SLURM_ARRAY_JOB_ID="local"
export SLURM_ARRAY_TASK_ID="${TASK_ID}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

exec bash "${SCRIPT_DIR}/train_ff_output_threshold_im.sh"
