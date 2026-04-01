#!/usr/bin/env zsh
set -euo pipefail

ROOT="/Users/sergiosolis/Programacion/AIDEEN"
DATASET="$ROOT/aideen-bench/tinyshakespeare.txt"

PROFILE="${1:-stable}"

cd "$ROOT"

common=(
  "cargo" "run" "--release" "--features" "wgpu" "-p" "aideen-training" "--bin" "train" "--"
  "train" "--file" "$DATASET" "--epochs" "1" "--log-every" "1" "--save-every" "0"
)

case "$PROFILE" in
  stable)
    env \
      AIDEEN_BATCH_SIZE=4 \
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
      AIDEEN_DEQ_HIST_GATED=0 \
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
      AIDEEN_DEBUG_SAMPLE=10 \
      AIDEEN_LOSS_READBACK_EVERY=10 \
      AIDEEN_TPS_SYNC_EVERY=10 \
      AIDEEN_VAL_EVERY=20 \
      AIDEEN_PROGRESS_EVERY=10 \
      AIDEEN_MAX_CHUNKS=10 \
      "${common[@]}"
    ;;
  *)
    echo "usage: $0 {stable|roof|roof-nohist|validation}" >&2
    exit 2
    ;;
esac
