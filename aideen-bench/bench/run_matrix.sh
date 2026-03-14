#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILES="${1:-$ROOT/aideen-bench/bench/profiles.csv}"
OUT_DIR="${2:-$ROOT/aideen-bench/bench/results/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.csv"

echo "profile,dr,h_slots,max_iters,cg_iters,eps,lr,iters,iter20_loss,iter20_ms,hit_ratio_pct,conv,mode,contractivity,max_delta,rs_cg,shared,total" > "$SUMMARY"

echo "Running stress matrix from: $PROFILES"

while IFS=, read -r name dr h_slots max_iters cg_iters eps lr iters; do
  [[ "$name" == "name" ]] && continue
  LOG="$OUT_DIR/${name}.log"
  echo "\n>>> [$name] d_r=$dr h_slots=$h_slots max_iters=$max_iters cg_iters=$cg_iters eps=$eps lr=$lr iters=$iters"

  (
    cd "$ROOT"
    AIDEEN_STRESS_DR="$dr" \
    AIDEEN_STRESS_HSLOTS="$h_slots" \
    AIDEEN_STRESS_MAX_ITERS="$max_iters" \
    AIDEEN_STRESS_CG_ITERS="$cg_iters" \
    AIDEEN_STRESS_EPS="$eps" \
    AIDEEN_STRESS_LR="$lr" \
    AIDEEN_STRESS_ITERS="$iters" \
    cargo run --release --features wgpu -p aideen-training --bin stress_test
  ) | tee "$LOG"

  iter20_line=$(grep "\[STRESS-TEST\] Iter 20" "$LOG" | tail -n1 || true)
  debug_line=$(grep "\[GPU-DEBUG\] Step 20" "$LOG" | tail -n1 || true)

  iter20_loss=$(echo "$iter20_line" | sed -n 's/.*Loss: \([0-9.]*\).*/\1/p')
  iter20_ms=$(echo "$iter20_line" | sed -n 's/.*Time: *\([0-9]*\)ms.*/\1/p')

  hit_ratio_pct=$(echo "$debug_line" | sed -n 's/.*( *\([0-9.]*\)%) .*/\1/p')
  conv=$(echo "$debug_line" | sed -n 's/.* conv=\([A-Z]*\) .*/\1/p')
  mode=$(echo "$debug_line" | sed -n 's/.* mode=\([A-Z]*\) .*/\1/p')
  contractivity=$(echo "$debug_line" | sed -n 's/.* contr=\([0-9.]*\) .*/\1/p')
  max_delta=$(echo "$debug_line" | sed -n 's/.* maxΔ=\([0-9.e+-]*\) .*/\1/p')
  rs_cg=$(echo "$debug_line" | sed -n 's/.* rs_cg=\([0-9.e+-]*\) .*/\1/p')
  shared=$(echo "$debug_line" | sed -n 's/.* shared=\([A-Z]*\) .*/\1/p')
  total=$(echo "$debug_line" | sed -n 's/.* total=\([0-9.]*\).*/\1/p')

  echo "$name,$dr,$h_slots,$max_iters,$cg_iters,$eps,$lr,$iters,${iter20_loss:-},${iter20_ms:-},${hit_ratio_pct:-},${conv:-},${mode:-},${contractivity:-},${max_delta:-},${rs_cg:-},${shared:-},${total:-}" >> "$SUMMARY"

done < "$PROFILES"

echo "\nDone. Summary: $SUMMARY"
cat "$SUMMARY"
