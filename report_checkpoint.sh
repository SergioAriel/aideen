#!/usr/bin/env zsh
set -euo pipefail

ROOT="/Users/sergiosolis/Programacion/AIDEEN"
MODEL="${1:-model_large}"

cd "$ROOT"

run_case() {
  local title="$1"
  shift
  echo
  echo "================ ${title} ================"
  "$@"
}

COMMON=(
  cargo run --release --features wgpu -p aideen-training --bin infer --
  --model "$MODEL"
  --max-tokens 48
  --temperature 0.15
  --top-p 0.75
  --top-k 6
  --rep-penalty 1.15
)

run_case "STATS / HISTORY ON" env AIDEEN_DEQ_HIST_GATED=1 "${COMMON[@]}" --stats --prompt "The Rust Programming Language is"
run_case "PROMPT A / HISTORY ON" env AIDEEN_DEQ_HIST_GATED=1 "${COMMON[@]}" --prompt "The Rust Programming Language is"
run_case "PROMPT B / HISTORY ON" env AIDEEN_DEQ_HIST_GATED=1 "${COMMON[@]}" --prompt "Chapter 1. Getting Started"

run_case "STATS / HISTORY OFF" env AIDEEN_DEQ_HIST_GATED=0 "${COMMON[@]}" --stats --prompt "The Rust Programming Language is"
run_case "PROMPT A / HISTORY OFF" env AIDEEN_DEQ_HIST_GATED=0 "${COMMON[@]}" --prompt "The Rust Programming Language is"
run_case "PROMPT B / HISTORY OFF" env AIDEEN_DEQ_HIST_GATED=0 "${COMMON[@]}" --prompt "Chapter 1. Getting Started"
