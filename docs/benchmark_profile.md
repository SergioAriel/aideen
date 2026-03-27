# Benchmark Profile (Mac M1 Pro)

## Perfil estándar (estable)
Objetivo: medir TPS real y estabilidad con un único set de flags, sin overhead de debug/readback.

**Comando**
```
cd /Users/sergiosolis/Programacion/AIDEEN && \
AIDEEN_BATCH_SIZE=4 AIDEEN_DEBUG_SAMPLE=0 AIDEEN_LM_FUSED_B19=0 \
AIDEEN_LOSS_READBACK_EVERY=0 AIDEEN_TPS_SYNC_EVERY=10 AIDEEN_MAX_CHUNKS=40 \
cargo run --release --features wgpu -p aideen-training --bin train -- \
  --file /Users/sergiosolis/Programacion/AIDEEN/aideen-bench/tinyshakespeare.txt \
  --epochs 1 --log-every 1 --save-every 0
```

**Env**
- `AIDEEN_BATCH_SIZE=4`
- `AIDEEN_TPS_SYNC_EVERY=10`
- `AIDEEN_LOSS_READBACK_EVERY=0`
- `AIDEEN_DEBUG_SAMPLE=0`
- `AIDEEN_LM_FUSED_B19=0`
- `AIDEEN_MAX_CHUNKS=40`

**Notas**
- TPS real se toma del log `[progress] ... tps=...` (sync cada 10).
- No activar debug/readback en este perfil; rompe comparabilidad.

## Perfil throughput (comparativo, no default)
Objetivo: medir escalado con batch > 1.

**Env (además del perfil estándar)**
- `AIDEEN_BATCH_SIZE=4`
