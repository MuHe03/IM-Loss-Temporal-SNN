#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="${1:-}"

if [[ ! "${TASK_ID}" =~ ^[0-9]+$ ]] || ((TASK_ID < 0 || TASK_ID > 14)); then
  cat >&2 <<'EOF'
Usage: scripts/train_ff_output_3way_local.sh TASK_ID

TASK_ID selects one experiment from the original 15-task Slurm array:
  0-4    non-spiking final membrane
  5-9    spiking output count
  10-14  spiking output count + rate IM

Seeds within each group are 2020/42/123/309/969.
EOF
  exit 2
fi

export SLURM_ARRAY_JOB_ID="local"
export SLURM_ARRAY_TASK_ID="${TASK_ID}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

exec bash "${SCRIPT_DIR}/train_ff_output_3way.sh"
