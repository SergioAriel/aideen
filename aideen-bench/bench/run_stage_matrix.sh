#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PROFILES="${1:-$ROOT/aideen-bench/bench/profiles_stage.csv}"
OUT_DIR="${2:-$ROOT/aideen-bench/bench/results/stage_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.csv"
BEST="$OUT_DIR/best_by_stage.csv"

echo "profile,stage,dr,h_slots,max_iters,cg_iters,eps,lr,iters,last_iter_loss,last_iter_ms,hit_ratio_pct,conv,mode,contractivity,last_delta,max_delta,rs_cg,avg_iters,fb_deq,fb_lm,fb_emb" > "$SUMMARY"

echo "Running stage matrix from: $PROFILES"

while IFS=, read -r name stage dr h_slots max_iters cg_iters eps lr iters; do
  [[ "$name" == "name" ]] && continue
  LOG="$OUT_DIR/${name}.log"
  echo ""
  echo ">>> [$name] stage=$stage d_r=$dr h_slots=$h_slots max_iters=$max_iters cg_iters=$cg_iters eps=$eps lr=$lr iters=$iters"

  (
    cd "$ROOT"
    unset AIDEEN_DEQ_ONLY AIDEEN_DEQ_NO_MAMBA
    case "$stage" in
      core) export AIDEEN_DEQ_ONLY=1 ;;
      attn) export AIDEEN_DEQ_NO_MAMBA=1 ;;
      full) ;;
      *) echo "Unknown stage: $stage"; exit 1 ;;
    esac
    AIDEEN_STRESS_DR="$dr" \
    AIDEEN_STRESS_HSLOTS="$h_slots" \
    AIDEEN_STRESS_MAX_ITERS="$max_iters" \
    AIDEEN_STRESS_CG_ITERS="$cg_iters" \
    AIDEEN_STRESS_EPS="$eps" \
    AIDEEN_STRESS_LR="$lr" \
    AIDEEN_STRESS_ITERS="$iters" \
    cargo run --release --features wgpu -p aideen-training --bin stress_test
  ) | tee "$LOG"

  iter_line=$(grep -E "\\[STRESS-TEST\\] Iter +[0-9]+" "$LOG" | tail -n1 || true)
  debug_line=$(grep -E "\\[GPU-DEBUG\\] Step +[0-9]+:" "$LOG" | tail -n1 || true)

  last_loss=$(echo "$iter_line" | sed -n 's/.*Loss: \([0-9.]*\).*/\1/p')
  last_ms=$(echo "$iter_line" | sed -n 's/.*Time: *\([0-9]*\)ms.*/\1/p')
  hit_ratio_pct=$(echo "$debug_line" | sed -n 's/.*( *\([0-9.]*\)%) .*/\1/p')
  conv=$(echo "$debug_line" | sed -n 's/.* conv=\([A-Z]*\) .*/\1/p')
  mode=$(echo "$debug_line" | sed -n 's/.* mode=\([A-Z]*\) .*/\1/p')
  contractivity=$(echo "$debug_line" | sed -n 's/.* contr=\([0-9.e+-]*\) .*/\1/p')
  max_delta=$(echo "$debug_line" | sed -n 's/.* maxΔ=\([0-9.e+-]*\) .*/\1/p')
  last_delta=$(echo "$debug_line" | sed -n 's/.* lastΔ=\([0-9.e+-]*\) .*/\1/p')
  rs_cg=$(echo "$debug_line" | sed -n 's/.* rs_cg=\([0-9.e+-]*\) .*/\1/p')
  avg_iters=$(echo "$debug_line" | sed -n 's/.* iters=\([0-9.e+-]*\) .*/\1/p')
  fb=$(echo "$debug_line" | sed -n 's/.*fb(deq\/lm\/emb)=\([0-9]*\)\/\([0-9]*\)\/\([0-9]*\).*/\1,\2,\3/p')
  fb_deq=$(echo "$fb" | cut -d, -f1)
  fb_lm=$(echo "$fb" | cut -d, -f2)
  fb_emb=$(echo "$fb" | cut -d, -f3)

  echo "$name,$stage,$dr,$h_slots,$max_iters,$cg_iters,$eps,$lr,$iters,${last_loss:-},${last_ms:-},${hit_ratio_pct:-},${conv:-},${mode:-},${contractivity:-},${last_delta:-},${max_delta:-},${rs_cg:-},${avg_iters:-},${fb_deq:-},${fb_lm:-},${fb_emb:-}" >> "$SUMMARY"
done < "$PROFILES"

# Best per stage: prefer conv=OK, then lower loss, then lower ms
awk -F, '
BEGIN {
  OFS=",";
  print "stage,best_profile,last_iter_loss,last_iter_ms,conv,mode,last_delta,contractivity";
}
NR==1 { next }
{
  s=$2; p=$1; loss=$10+0; ms=$11+0; conv=$13; mode=$14; lastd=$16; contr=$15;
  ok = (conv=="OK") ? 1 : 0;
  key_ok[s]=ok;
  if (!(s in best_loss) || ok>best_ok[s] || (ok==best_ok[s] && (loss<best_loss[s] || (loss==best_loss[s] && ms<best_ms[s])))) {
    best_ok[s]=ok; best_p[s]=p; best_loss[s]=loss; best_ms[s]=ms; best_conv[s]=conv; best_mode[s]=mode; best_lastd[s]=lastd; best_contr[s]=contr;
  }
}
END {
  for (s in best_p) {
    print s,best_p[s],best_loss[s],best_ms[s],best_conv[s],best_mode[s],best_lastd[s],best_contr[s];
  }
}
' "$SUMMARY" | sort > "$BEST"

echo ""
echo "Done. Summary: $SUMMARY"
cat "$SUMMARY"
echo ""
echo "Best by stage: $BEST"
cat "$BEST"

