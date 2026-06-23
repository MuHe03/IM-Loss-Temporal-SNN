#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${MODE:-checkpoints}" != "summary" && "$#" -eq 0 && -z "${CHECKPOINTS:-}" ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/eval_temporal_local.sh runs/.../best.pth [runs/.../best.pth ...]

Environment examples:
  COMPUTE_ACTIVITY=1 scripts/eval_temporal_local.sh runs/.../best.pth
  MODE=summary SUMMARY_OUTPUT_DIR=runs/summary_local scripts/eval_temporal_local.sh
EOF
  exit 2
fi

export SLURM_JOB_ID="local"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-8}"

exec bash "${SCRIPT_DIR}/eval_temporal_slurm.sh" "$@"
