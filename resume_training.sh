#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/sergiosolis/Programacion/AIDEEN"
cd "$ROOT"

EPOCHS="${1:-5}"
RESUME_BASE="${AIDEEN_RESUME_BASE:-model_large}"
OUTPUT_BASE="${AIDEEN_CHECKPOINT_BASE:-$RESUME_BASE}"
DATASET="${AIDEEN_DATASET:-$ROOT/corpus_combined.txt}"

if [[ ! -f "${RESUME_BASE}.aidn" ]]; then
  echo "ERROR: No se encontró ${RESUME_BASE}.aidn"
  echo "Exportá AIDEEN_RESUME_BASE=<base> si querés reanudar otro checkpoint."
  exit 1
fi

echo "Resume base : ${RESUME_BASE}"
echo "Output base : ${OUTPUT_BASE}"
echo "Dataset     : ${DATASET}"
echo "Epochs      : ${EPOCHS}"
echo

env \
  AIDEEN_CHECKPOINT_BASE="${OUTPUT_BASE}" \
  AIDEEN_BATCH_SIZE="${AIDEEN_BATCH_SIZE:-8}" \
  AIDEEN_CTX_LEN="${AIDEEN_CTX_LEN:-512}" \
  AIDEEN_DEQ_MODE="${AIDEEN_DEQ_MODE:-hist_gated}" \
  AIDEEN_LM_FUSED_B19="${AIDEEN_LM_FUSED_B19:-1}" \
  AIDEEN_LOSS_READBACK_EVERY="${AIDEEN_LOSS_READBACK_EVERY:-20}" \
  AIDEEN_TPS_SYNC_EVERY="${AIDEEN_TPS_SYNC_EVERY:-20}" \
  AIDEEN_VAL_EVERY="${AIDEEN_VAL_EVERY:-0}" \
  AIDEEN_PROGRESS_EVERY="${AIDEEN_PROGRESS_EVERY:-20}" \
  cargo run --release --features wgpu -p aideen-training --bin train -- \
    --file "${DATASET}" \
    --resume "${RESUME_BASE}" \
    --epochs "${EPOCHS}" \
    --log-every 1 \
    --save-every 1
