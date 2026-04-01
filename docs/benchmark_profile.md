# Benchmark Profile (Mac M1 Pro)

## Perfil estándar (estable)
Objetivo: medir TPS real del fused path sin contaminarlo con validación, readbacks o progreso.

**Comando**
```
cd /Users/sergiosolis/Programacion/AIDEEN && \
AIDEEN_BATCH_SIZE=4 AIDEEN_DEBUG_SAMPLE=0 AIDEEN_LM_FUSED_B19=1 \
AIDEEN_LOSS_READBACK_EVERY=0 AIDEEN_TPS_SYNC_EVERY=0 AIDEEN_VAL_EVERY=0 \
AIDEEN_PROGRESS_EVERY=0 AIDEEN_MAX_CHUNKS=20 \
cargo run --release --features wgpu -p aideen-training --bin train -- \
  --file /Users/sergiosolis/Programacion/AIDEEN/aideen-bench/tinyshakespeare.txt \
  --epochs 1 --log-every 1 --save-every 0
```

**Env**
- `AIDEEN_BATCH_SIZE=4`
- `AIDEEN_TPS_SYNC_EVERY=0`
- `AIDEEN_LOSS_READBACK_EVERY=0`
- `AIDEEN_DEBUG_SAMPLE=0`
- `AIDEEN_LM_FUSED_B19=1`
- `AIDEEN_VAL_EVERY=0`
- `AIDEEN_PROGRESS_EVERY=0`
- `AIDEEN_MAX_CHUNKS=20`

**Notas**
- `ctx_len` default del binario de train: `512`.
- TPS real se toma del log final `tps_epoch` y, cuando aplique, de `tps_gpu`.
- No activar debug/readback/progress/val en este perfil; rompe comparabilidad.

**Resultados recientes**
- `ctx=512`, `batch=4`, `B19=1`, history default: `tps_epoch = 4123.0`
- mismo perfil con `AIDEEN_DEQ_HIST_GATED=0`: medir aparte si se busca techo absoluto

## Perfil techo (comparativo, no default del modelo)
Objetivo: buscar el máximo throughput disponible del sistema, aceptando desactivar history.

**Comando**
```
cd /Users/sergiosolis/Programacion/AIDEEN && \
AIDEEN_BATCH_SIZE=8 AIDEEN_DEBUG_SAMPLE=0 AIDEEN_LM_FUSED_B19=1 \
AIDEEN_DEQ_HIST_GATED=0 AIDEEN_LOSS_READBACK_EVERY=0 AIDEEN_TPS_SYNC_EVERY=0 \
AIDEEN_VAL_EVERY=0 AIDEEN_PROGRESS_EVERY=0 AIDEEN_MAX_CHUNKS=20 \
cargo run --release --features wgpu -p aideen-training --bin train -- \
  --file /Users/sergiosolis/Programacion/AIDEEN/aideen-bench/tinyshakespeare.txt \
  --epochs 1 --log-every 1 --save-every 0
```

**Resultado reciente**
- `tps_epoch = 5129.0`

## Perfil throughput (comparativo, no default)
Objetivo: medir escalado con batch > 1.

**Env (además del perfil estándar)**
- `AIDEEN_BATCH_SIZE=4`
