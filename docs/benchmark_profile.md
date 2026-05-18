# Benchmark Profile (Mac M1 Pro)

## Standard profile (stable)
Goal: measure real TPS of the fused path without contaminating it with validation, readbacks or progress.

**Command**
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

**Notes**
- `ctx_len` default of the train binary: `512`.
- Real TPS is taken from the final log `tps_epoch` and, where applicable, from `tps_gpu`.
- Do not enable debug/readback/progress/val in this profile; it breaks comparability.

**Recent results**
- `ctx=512`, `batch=4`, `B19=1`, history default: `tps_epoch = 4123.0`
- `ctx=512`, `batch=8`, `B19=1`, history default: `tps_epoch = 5197.8`
- same profile with `AIDEEN_DEQ_HIST_GATED=0`: measure separately if the absolute ceiling is sought

## Ceiling profile (comparative, not the model default)
Goal: find the maximum throughput available from the system, accepting that history is disabled.

**Command**
```
cd /Users/sergiosolis/Programacion/AIDEEN && \
AIDEEN_BATCH_SIZE=8 AIDEEN_DEBUG_SAMPLE=0 AIDEEN_LM_FUSED_B19=1 \
AIDEEN_DEQ_HIST_GATED=0 AIDEEN_LOSS_READBACK_EVERY=0 AIDEEN_TPS_SYNC_EVERY=0 \
AIDEEN_VAL_EVERY=0 AIDEEN_PROGRESS_EVERY=0 AIDEEN_MAX_CHUNKS=20 \
cargo run --release --features wgpu -p aideen-training --bin train -- \
  --file /Users/sergiosolis/Programacion/AIDEEN/aideen-bench/tinyshakespeare.txt \
  --epochs 1 --log-every 1 --save-every 0
```

**Recent result**
- `tps_epoch = 5129.0`

## Throughput profile (comparative, not default)
Goal: measure scaling with batch > 1.

**Env (in addition to the standard profile)**
- `AIDEEN_BATCH_SIZE=4`
