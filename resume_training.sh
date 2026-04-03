#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/sergiosolis/Programacion/AIDEEN"
cd "$ROOT"

EPOCHS="${1:-5}"
RESUME_BASE="${AIDEEN_RESUME_BASE:-$ROOT/artifacts/checkpoints/model_histv2_clean_pretrain_latest}"
OUTPUT_BASE="${AIDEEN_CHECKPOINT_BASE:-$RESUME_BASE}"
DATASET="${AIDEEN_DATASET:-$ROOT/corpus_pretrain_minimal.txt}"

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
  AIDEEN_BATCH_SIZE="${AIDEEN_BATCH_SIZE:-4}" \
  AIDEEN_CTX_LEN="${AIDEEN_CTX_LEN:-512}" \
  AIDEEN_LR="${AIDEEN_LR:-0.00002}" \
  AIDEEN_ADJ_ITERS_OVERRIDE="${AIDEEN_ADJ_ITERS_OVERRIDE:-2}" \
  AIDEEN_DEQ_HIST_GATED="${AIDEEN_DEQ_HIST_GATED:-0}" \
  AIDEEN_DEQ_TOKEN_CARRY="${AIDEEN_DEQ_TOKEN_CARRY:-1}" \
  AIDEEN_LOSS_READBACK_EVERY="${AIDEEN_LOSS_READBACK_EVERY:-0}" \
  AIDEEN_TPS_SYNC_EVERY="${AIDEEN_TPS_SYNC_EVERY:-0}" \
  AIDEEN_VAL_EVERY="${AIDEEN_VAL_EVERY:-0}" \
  AIDEEN_PROGRESS_EVERY="${AIDEEN_PROGRESS_EVERY:-0}" \
  cargo run --release --features wgpu -p aideen-training --bin train -- \
    --file "${DATASET}" \
    --resume "${RESUME_BASE}" \
    --epochs "${EPOCHS}" \
    --log-every 1 \
    --save-every 1
