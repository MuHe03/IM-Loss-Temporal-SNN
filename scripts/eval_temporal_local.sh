#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"

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

if [[ "${MODE:-checkpoints}" == "summary" ]]; then
  cmd=(
    "${PYTHON_BIN}" summarize_shd_results.py
    --run_root "${RUNS_ROOT:-runs}"
    --output_dir "${SUMMARY_OUTPUT_DIR:-runs/summary}"
    --batch_size "${BATCH_SIZE:-256}"
    --num_workers "${NUM_WORKERS:-0}"
    --device "${DEVICE:-cuda:0}"
  )
  echo "Running SHD summary"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exec "${cmd[@]}"
fi

checkpoints=("$@")
if [[ "$#" -eq 0 && -n "${CHECKPOINTS:-}" ]]; then
  read -r -a checkpoints <<< "${CHECKPOINTS}"
fi

cmd=(
  "${PYTHON_BIN}" eval_temporal.py
  "${checkpoints[@]}"
  --batch_size "${BATCH_SIZE:-256}"
  --num_workers "${NUM_WORKERS:-0}"
  --device "${DEVICE:-cuda:0}"
)

if [[ -n "${DATA_ROOT:-}" ]]; then
  cmd+=(--data_root "${DATA_ROOT}")
fi
if [[ -n "${DT:-}" ]]; then
  cmd+=(--dt "${DT}")
fi
if [[ -n "${T_STOP:-}" ]]; then
  cmd+=(--t_stop "${T_STOP}")
fi
if [[ "${COMPUTE_ACTIVITY:-0}" == "1" ]]; then
  cmd+=(--compute_activity)
fi
if [[ -n "${EVAL_OUTPUT:-}" ]]; then
  cmd+=(--output "${EVAL_OUTPUT}")
fi

echo "Evaluating ${#checkpoints[@]} checkpoint(s)"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
