#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="${1:-}"

if [[ ! "${TASK_ID}" =~ ^[0-9]+$ ]] || ((TASK_ID < 0 || TASK_ID > 15)); then
  cat >&2 <<'EOF'
Usage: scripts/train_temporal_multiseed_local.sh TASK_ID

TASK_ID selects one experiment from the original 16-task Slurm array:
  0-3    FF-LIF baseline, seeds 42/123/309/969
  4-7    FF-LIF + rate IM, seeds 42/123/309/969
  8-11   RSNN baseline, seeds 42/123/309/969
  12-15  RSNN + rate IM, seeds 42/123/309/969
EOF
  exit 2
fi

export SLURM_ARRAY_JOB_ID="local"
export SLURM_ARRAY_TASK_ID="${TASK_ID}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

exec bash "${SCRIPT_DIR}/train_temporal_multiseed.sh"
