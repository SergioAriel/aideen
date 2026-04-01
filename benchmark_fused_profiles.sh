#!/usr/bin/env zsh
set -euo pipefail

ROOT="/Users/sergiosolis/Programacion/AIDEEN"
DATASET="$ROOT/aideen-bench/tinyshakespeare.txt"

PROFILE="${1:-stable}"

cd "$ROOT"

common=(
  "cargo" "run" "--release" "--features" "wgpu" "-p" "aideen-training" "--bin" "train" "--"
  "--file" "$DATASET" "--epochs" "1" "--log-every" "1" "--save-every" "0"
)

case "$PROFILE" in
  stable)
    env \
      AIDEEN_BATCH_SIZE=4 \
      AIDEEN_CTX_LEN=512 \
      AIDEEN_LM_FUSED_B19=1 \
      AIDEEN_DEBUG_SAMPLE=0 \
      AIDEEN_LOSS_READBACK_EVERY=0 \
      AIDEEN_TPS_SYNC_EVERY=0 \
      AIDEEN_VAL_EVERY=0 \
      AIDEEN_PROGRESS_EVERY=0 \
      AIDEEN_MAX_CHUNKS=20 \
      "${common[@]}"
    ;;
  roof)
    env \
      AIDEEN_BATCH_SIZE=8 \
      AIDEEN_CTX_LEN=512 \
      AIDEEN_LM_FUSED_B19=1 \
      AIDEEN_DEBUG_SAMPLE=0 \
      AIDEEN_LOSS_READBACK_EVERY=0 \
      AIDEEN_TPS_SYNC_EVERY=0 \
      AIDEEN_VAL_EVERY=0 \
      AIDEEN_PROGRESS_EVERY=0 \
      AIDEEN_MAX_CHUNKS=20 \
      "${common[@]}"
    ;;
  roof-nohist)
    env \
      AIDEEN_BATCH_SIZE=8 \
      AIDEEN_CTX_LEN=512 \
      AIDEEN_LM_FUSED_B19=1 \
      AIDEEN_DEQ_MODE=no_mamba \
      AIDEEN_DEBUG_SAMPLE=0 \
      AIDEEN_LOSS_READBACK_EVERY=0 \
      AIDEEN_TPS_SYNC_EVERY=0 \
      AIDEEN_VAL_EVERY=0 \
      AIDEEN_PROGRESS_EVERY=0 \
      AIDEEN_MAX_CHUNKS=20 \
      "${common[@]}"
    ;;
  validation)
    env \
      AIDEEN_BATCH_SIZE=1 \
      AIDEEN_CTX_LEN=512 \
      AIDEEN_LM_FUSED_B19=1 \
      AIDEEN_DEBUG_SAMPLE=10 \
      AIDEEN_LOSS_READBACK_EVERY=10 \
      AIDEEN_TPS_SYNC_EVERY=10 \
      AIDEEN_VAL_EVERY=20 \
      AIDEEN_PROGRESS_EVERY=10 \
      AIDEEN_MAX_CHUNKS=10 \
      "${common[@]}"
    ;;
  report)
    env \
      AIDEEN_BATCH_SIZE=4 \
      AIDEEN_CTX_LEN=512 \
      AIDEEN_LM_FUSED_B19=1 \
      AIDEEN_DEBUG_SAMPLE=0 \
      AIDEEN_LOSS_READBACK_EVERY=20 \
      AIDEEN_TPS_SYNC_EVERY=20 \
      AIDEEN_VAL_EVERY=200 \
      AIDEEN_PROGRESS_EVERY=20 \
      AIDEEN_MAX_CHUNKS=200 \
      "${common[@]}"
    ;;
  report-nohist)
    env \
      AIDEEN_BATCH_SIZE=4 \
      AIDEEN_CTX_LEN=512 \
      AIDEEN_LM_FUSED_B19=1 \
      AIDEEN_DEQ_MODE=no_mamba \
      AIDEEN_DEBUG_SAMPLE=0 \
      AIDEEN_LOSS_READBACK_EVERY=20 \
      AIDEEN_TPS_SYNC_EVERY=20 \
      AIDEEN_VAL_EVERY=200 \
      AIDEEN_PROGRESS_EVERY=20 \
      AIDEEN_MAX_CHUNKS=200 \
      "${common[@]}"
    ;;
  *)
    echo "usage: $0 {stable|roof|roof-nohist|validation|report|report-nohist}" >&2
    exit 2
    ;;
esac
