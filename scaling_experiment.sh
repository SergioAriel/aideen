#!/usr/bin/env bash
# Scaling Law Experiment: d_r = 64 / 128 / 256 / 512
# Misma data, mismo seed, mismo LR. Mide loss final por tamaño.
# Uso: ./scaling_experiment.sh | tee scaling_results.log

set -e
cd "$(dirname "$0")"

TEXT_FILE="aideen-bench/tinyshakespeare.txt"
SEED=42
ITERS=60          # pasos de gradiente por tamaño (1 ventana de ctx_len por iter)
LR=0.0003
CTX_LEN=256
SEQ_LEN=256       # 1 ventana por iter — evita que sea 1 epoch por "iter"

echo "=== SCALING LAW EXPERIMENT ==="
echo "Text: $TEXT_FILE  Seed: $SEED  Iters: $ITERS  LR: $LR  CTX: $CTX_LEN"
echo "Start: $(date)"
echo ""

cargo build --release --bin stress_test --features wgpu 2>&1 | tail -3

# d_r mínimo válido = WG_SIZE = 256 (shaders tienen workgroupSize=256)
# d_r < 256 produce logits degenerados (threads 0..d_r-1 solo, reducción incorrecta)
# Solo potencias de 2 >= WG_SIZE=256. d_r=384 (no power-of-2) produce loss=23 degenerado.
for DR in 512; do
    echo ""
    echo "--- d_r=$DR ---"
    AIDEEN_STRESS_DR=$DR \
    AIDEEN_STRESS_ITERS=$ITERS \
    AIDEEN_STRESS_SEED=$SEED \
    AIDEEN_STRESS_LR=$LR \
    AIDEEN_STRESS_CTX_LEN=$CTX_LEN \
    AIDEEN_STRESS_SEQ_LEN=$SEQ_LEN \
    AIDEEN_STRESS_FILE=$TEXT_FILE \
    cargo run --release --bin stress_test --features wgpu 2>&1 \
        | grep --line-buffered -E "^\[STRESS-TEST\] (Iter|Config|TPS prom)"
done

echo ""
echo "End: $(date)"
echo ""
echo "Para ajustar la power law, ejecutá:"
echo "  python3 fit_scaling_law.py scaling_results.log"
