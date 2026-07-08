#!/usr/bin/env bash

set -euo pipefail

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

PYTHON_BIN="${PYTHON:-python}"
DATASET="${DATASET:-Randman}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
DATA_ROOT="${DATA_ROOT:-datasets/SHD}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DT="${DT:-0.005}"
T_STOP="${T_STOP:-1.4}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
NUM_LAYERS="${NUM_LAYERS:-1}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
IM_LOSS_WEIGHT="${IM_LOSS_WEIGHT:-0.001}"
IM_TARGET_RATE="${IM_TARGET_RATE:-0.02}"
READOUT="${READOUT:-max_membrane}"
RANDMAN_TRAIN_SAMPLES="${RANDMAN_TRAIN_SAMPLES:-1024}"
RANDMAN_VAL_SAMPLES="${RANDMAN_VAL_SAMPLES:-256}"
RANDMAN_TEST_SAMPLES="${RANDMAN_TEST_SAMPLES:-256}"
RANDMAN_INPUT_SIZE="${RANDMAN_INPUT_SIZE:-128}"
RANDMAN_NUM_CLASSES="${RANDMAN_NUM_CLASSES:-10}"
RANDMAN_NUM_STEPS="${RANDMAN_NUM_STEPS:-100}"
RANDMAN_MANIFOLD_DIM="${RANDMAN_MANIFOLD_DIM:-2}"
RANDMAN_SPIKE_PROB="${RANDMAN_SPIKE_PROB:-0.35}"
RANDMAN_NOISE_RATE="${RANDMAN_NOISE_RATE:-0.001}"
RANDMAN_JITTER_STD="${RANDMAN_JITTER_STD:-1.0}"

seeds=(42 123 309 969)
group=$((TASK_ID / 4))
seed_idx=$((TASK_ID % 4))
seed="${seeds[$seed_idx]}"

arch="lif_mlp"
run_root="${RUNS_ROOT}/${DATASET,,}"
output_dir="${run_root}/temporal_multiseed/lif_mlp_baseline"
extra_args=()

case "${group}" in
  0)
    arch="lif_mlp"
    output_dir="${run_root}/temporal_multiseed/lif_mlp_baseline"
    ;;
  1)
    arch="lif_mlp"
    output_dir="${run_root}/temporal_multiseed/lif_mlp_imloss"
    extra_args+=(--use_im_loss --im_loss_type rate --im_loss_weight "${IM_LOSS_WEIGHT}" --im_target_rate "${IM_TARGET_RATE}")
    ;;
  2)
    arch="rsnn_lif"
    output_dir="${run_root}/temporal_multiseed/rsnn_lif_baseline"
    ;;
  3)
    arch="rsnn_lif"
    output_dir="${run_root}/temporal_multiseed/rsnn_lif_imloss"
    extra_args+=(--use_im_loss --im_loss_type rate --im_loss_weight "${IM_LOSS_WEIGHT}" --im_target_rate "${IM_TARGET_RATE}")
    ;;
esac

cmd=(
  "${PYTHON_BIN}" train_temporal.py
  --dataset "${DATASET}"
  --data_root "${DATA_ROOT}"
  --output_dir "${output_dir}"
  --arch "${arch}"
  --readout "${READOUT}"
  --seed "${seed}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --dt "${DT}"
  --t_stop "${T_STOP}"
  --hidden_size "${HIDDEN_SIZE}"
  --num_layers "${NUM_LAYERS}"
  --learning_rate "${LEARNING_RATE}"
  --randman_train_samples "${RANDMAN_TRAIN_SAMPLES}"
  --randman_val_samples "${RANDMAN_VAL_SAMPLES}"
  --randman_test_samples "${RANDMAN_TEST_SAMPLES}"
  --randman_input_size "${RANDMAN_INPUT_SIZE}"
  --randman_num_classes "${RANDMAN_NUM_CLASSES}"
  --randman_num_steps "${RANDMAN_NUM_STEPS}"
  --randman_manifold_dim "${RANDMAN_MANIFOLD_DIM}"
  --randman_spike_prob "${RANDMAN_SPIKE_PROB}"
  --randman_noise_rate "${RANDMAN_NOISE_RATE}"
  --randman_jitter_std "${RANDMAN_JITTER_STD}"
)

if [[ "${NO_CUDA:-0}" == "1" ]]; then
  cmd+=(--no_cuda)
fi
if [[ -n "${MAX_TRAIN_BATCHES:-}" ]]; then
  cmd+=(--max_train_batches "${MAX_TRAIN_BATCHES}")
fi
if [[ -n "${MAX_EVAL_BATCHES:-}" ]]; then
  cmd+=(--max_eval_batches "${MAX_EVAL_BATCHES}")
fi

cmd+=("${extra_args[@]}")

echo "Running temporal multiseed task ${TASK_ID}: dataset=${DATASET} arch=${arch} seed=${seed} output_dir=${output_dir}"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
