#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/sergiosolis/Programacion/AIDEEN"
cd "$ROOT"

MODE="${1:-both}"
BASE="${AIDEEN_LADDER_BASE:-model_ladder}"
CORPUS_FILE="${AIDEEN_CORPUS_FILE:-$ROOT/corpus_combined.txt}"

TINY_BASE="${BASE}_tiny"
CORPUS_BASE="${BASE}_corpus"

run_tiny() {
  echo "== Stage 1: tinyshakespeare warmup =="
  env \
    AIDEEN_CHECKPOINT_BASE="${TINY_BASE}" \
    AIDEEN_BATCH_SIZE="${AIDEEN_BATCH_SIZE:-4}" \
    AIDEEN_CTX_LEN="${AIDEEN_CTX_LEN:-512}" \
    AIDEEN_DEQ_MODE="${AIDEEN_DEQ_MODE:-hist_gated}" \
    AIDEEN_LM_FUSED_B19="${AIDEEN_LM_FUSED_B19:-1}" \
    AIDEEN_LOSS_READBACK_EVERY="${AIDEEN_LOSS_READBACK_EVERY:-20}" \
    AIDEEN_TPS_SYNC_EVERY="${AIDEEN_TPS_SYNC_EVERY:-20}" \
    AIDEEN_VAL_EVERY="${AIDEEN_VAL_EVERY:-0}" \
    AIDEEN_PROGRESS_EVERY="${AIDEEN_PROGRESS_EVERY:-10}" \
    AIDEEN_MAX_CHUNKS="${AIDEEN_MAX_CHUNKS:-40}" \
    cargo run --release --features wgpu -p aideen-training --bin train -- \
      --file "$ROOT/aideen-bench/tinyshakespeare.txt" \
      --epochs "${AIDEEN_TINY_EPOCHS:-1}" \
      --log-every 1 \
      --save-every 1

  echo
  echo "== Stage 1 report =="
  ./report_checkpoint.sh "${TINY_BASE}"
}

run_corpus() {
  if [[ ! -f "${TINY_BASE}.aidn" ]]; then
    echo "ERROR: falta ${TINY_BASE}.aidn; corré primero stage tiny o MODE=both"
    exit 1
  fi

  echo
  echo "== Stage 2: corpus_combined resume =="
  env \
    AIDEEN_CHECKPOINT_BASE="${CORPUS_BASE}" \
    AIDEEN_BATCH_SIZE="${AIDEEN_BATCH_SIZE_CORPUS:-8}" \
    AIDEEN_CTX_LEN="${AIDEEN_CTX_LEN_CORPUS:-512}" \
    AIDEEN_DEQ_MODE="${AIDEEN_DEQ_MODE_CORPUS:-hist_gated}" \
    AIDEEN_LM_FUSED_B19="${AIDEEN_LM_FUSED_B19_CORPUS:-1}" \
    AIDEEN_LOSS_READBACK_EVERY="${AIDEEN_LOSS_READBACK_EVERY_CORPUS:-20}" \
    AIDEEN_TPS_SYNC_EVERY="${AIDEEN_TPS_SYNC_EVERY_CORPUS:-20}" \
    AIDEEN_VAL_EVERY="${AIDEEN_VAL_EVERY_CORPUS:-0}" \
    AIDEEN_PROGRESS_EVERY="${AIDEEN_PROGRESS_EVERY_CORPUS:-20}" \
    AIDEEN_MAX_CHUNKS="${AIDEEN_MAX_CHUNKS_CORPUS:-18446744073709551615}" \
    cargo run --release --features wgpu -p aideen-training --bin train -- \
      --file "${CORPUS_FILE}" \
      --resume "${TINY_BASE}" \
      --epochs "${AIDEEN_CORPUS_EPOCHS:-1}" \
      --log-every 1 \
      --save-every 1

  echo
  echo "== Stage 2 report =="
  ./report_checkpoint.sh "${CORPUS_BASE}"
}

case "${MODE}" in
  tiny)
    run_tiny
    ;;
  corpus)
    run_corpus
    ;;
  both)
    run_tiny
    run_corpus
    ;;
  *)
    echo "usage: $0 {tiny|corpus|both}" >&2
    exit 2
    ;;
esac
