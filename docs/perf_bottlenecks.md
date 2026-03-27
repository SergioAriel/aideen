# AIDEEN Performance Bottlenecks & Improvement Roadmap

**Baseline** (2026-03-23): ~11-12 TPS on Apple M1/M2 (batch=1, ctx=256, h_slots=8, d=512)
**nanoGPT baseline**: ~9,500 TPS (same hardware, batch=1, block=256, n_layer=3)
**Gap**: ~860× — decomposed into:
- DEQ structural overhead: ~7× (8 fwd + 6 adj Picard iters vs 1 fwd + 1 bwd pass)
- wgpu/Metal vs PyTorch/MPS kernel overhead: ~120× residual gap

---

### 2026-03-26 — Baseline provisional (TPS real, batch=4, no model changes)
**Contexto**: throughput real en M1 Pro, entrenamiento completo GPU, sin tocar la matemática.
**Objetivo**: establecer baseline fijo para medir mejoras.

**Configuración exacta**
- Cmd:
  `AIDEEN_BATCH_SIZE=4 AIDEEN_DEBUG_SAMPLE=0 AIDEEN_LM_FUSED_B19=0 AIDEEN_LOSS_READBACK_EVERY=10 AIDEEN_TPS_SYNC_EVERY=10 AIDEEN_VAL_EVERY=20 AIDEEN_MAX_CHUNKS=40 cargo run --release --features wgpu -p aideen-training --bin train -- --file /Users/sergiosolis/Programacion/AIDEEN/aideen-bench/tinyshakespeare.txt --epochs 1 --log-every 1 --save-every 0`
- Dataset: `tinyshakespeare.txt` (token cache reutilizado)
- Perfil: training real (LM + DEQ + adjoint)
- Código: `e88dcb5`

**Resultados**
- chunk 10: loss=5.1464, tps=81.3
- chunk 20: loss=5.6460, tps=83.3
- VAL chunk 0: 5.7344

**Criterios de invalidez**
- Si la misma config en otro commit produce TPS < 70 o loss divergente, revalidar.
- Si se modifica `aideen-training-lab/src/trainer.rs` o `aideen-backbone/src/shaders/lm_train.wgsl`, repetir este baseline.

**Alcance**
- Válido para Apple M1 Pro, batch=4, ctx=256, h_slots=8, d=512.

---

## Benchmark Profiles (fixed usage)

No usar una sola configuración para todo. En AIDEEN hay tres preguntas distintas:
- si un cambio rompe o altera la estabilidad,
- si mejora el throughput del hot path,
- o si baja el overhead total del sistema.

Cada una requiere un perfil distinto. No comparar números entre perfiles.

### Perfil A — Validación
**Objetivo**: validar corrección/estabilidad cuando un cambio toca backward, historia, adjoint, DEQ o numerics.

**Mirar**
- `loss`
- `val_loss`
- `DEQ-INVALID`
- `contr`
- `maxΔ`
- `iters`
- NaNs / crash / divergencia

**Configuración**
- Cmd:
  `AIDEEN_BATCH_SIZE=1 AIDEEN_DEBUG_SAMPLE=10 AIDEEN_LM_FUSED_B19=0 AIDEEN_LOSS_READBACK_EVERY=10 AIDEEN_TPS_SYNC_EVERY=10 AIDEEN_VAL_EVERY=20 AIDEEN_MAX_CHUNKS=10 cargo run --release --features wgpu -p aideen-training --bin train -- --file /Users/sergiosolis/Programacion/AIDEEN/aideen-bench/tinyshakespeare.txt --epochs 1 --log-every 1 --save-every 0`

**Uso**
- fusiones de kernels en solver/backward
- cambios en staged adjoint
- cambios en historia
- rutas nuevas matemáticamente equivalentes que todavía no fueron validadas

### Perfil B — Régimen
**Objetivo**: medir throughput real del hot path repetitivo. Esta es la métrica principal para performance estructural.

**Mirar**
- `progress chunk 10`
- `progress chunk 20`
- `tps` de régimen

**Configuración**
- Cmd:
  `AIDEEN_BATCH_SIZE=4 AIDEEN_DEBUG_SAMPLE=0 AIDEEN_LM_FUSED_B19=0 AIDEEN_LOSS_READBACK_EVERY=10 AIDEEN_TPS_SYNC_EVERY=10 AIDEEN_VAL_EVERY=20 AIDEEN_MAX_CHUNKS=40 cargo run --release --features wgpu -p aideen-training --bin train -- --file /Users/sergiosolis/Programacion/AIDEEN/aideen-bench/tinyshakespeare.txt --epochs 1 --log-every 1 --save-every 0`

**Uso**
- solo después de pasar Perfil A si el cambio toca matemática
- optimizaciones de shader/dispatch/lifecycle de buffers que apunten a subir TPS de régimen

**Referencia provisional**
- chunk 10: loss=5.1464, tps=81.3
- chunk 20: loss=5.6460, tps=83.3

### Perfil C — Sistema Total
**Objetivo**: medir overhead total del run corto. Útil para startup, readbacks, bind groups, observabilidad y costos fuera del hot path.

**Mirar**
- `tps` final del run
- tiempo total
- tokens procesados

**Configuración**
- Cmd:
  `AIDEEN_BATCH_SIZE=4 AIDEEN_DEBUG_SAMPLE=0 AIDEEN_LM_FUSED_B19=0 AIDEEN_LOSS_READBACK_EVERY=10 AIDEEN_TPS_SYNC_EVERY=10 AIDEEN_VAL_EVERY=20 AIDEEN_MAX_CHUNKS=10 cargo run --release --features wgpu -p aideen-training --bin train -- --file /Users/sergiosolis/Programacion/AIDEEN/aideen-bench/tinyshakespeare.txt --epochs 1 --log-every 1 --save-every 0`

**Uso**
- costos de driver/setup
- pérdida de tiempo por readbacks
- churn de bind groups
- mejoras que no necesariamente mueven el régimen pero sí el tiempo total del sistema

### Regla de uso
- Si un cambio toca matemática o solver: `Perfil A` -> `Perfil B`
- Si un cambio toca solo overhead: `Perfil C`, y luego `Perfil B` si parece promisor
- No decidir por TPS de un perfil usando números de otro perfil
- No promover defaults por “compila y no crashea”; deben pasar el perfil correspondiente

---

## Status Legend
- ✅ FIXED — already implemented
- 🔥 HIGH — significant gain, actionable now
- 🟡 MEDIUM — moderate gain or moderate effort
- 🔵 LOW — small gain or experimental
- ⛔ STRUCTURAL — inherent to DEQ architecture, not fixable without changing the model

---

## Part 1: Already Fixed

### ✅ B1 — Debug buffer readback every step
**Was**: `read_debug_buffer()` called every step → `device.poll(Maintain::Wait)` every ~90ms.
**Fix**: cache + sample every 4 steps (`AIDEEN_DEBUG_SAMPLE=4`).
**Gain**: ~+25% TPS (observed 7.6 → 11 TPS).
**Files**: `aideen-training-lab/src/trainer.rs:687-695`

### ✅ B2 — Hot-loop heap allocations
**Was**: `carry`, `read_buf`, `tokens` allocated fresh on every chunk in `train_on_file`.
**Fix**: pre-allocated outside loop, reused via `clear() + extend_from_slice()`.
**Gain**: ~+5-10% (less GC pressure, avoids OS allocation).
**Files**: `aideen-training-lab/src/trainer.rs` (train_on_file loop)

### ✅ B3 — `targets_u32.clone()` in LM step
**Was**: extra Vec<u32> clone passed to `train_step_no_readback`.
**Fix**: pass `&targets_u32` directly.
**Gain**: minor (saves 256×4 = 1 KB copy per step).
**Files**: `aideen-training-lab/src/trainer.rs:752`

### ✅ B4 — Batch size bug (AMD training)
**Was**: `AIDEEN_BATCH_SIZE=8` with 256-token chunks → `per_seq_len = 256/8 = 32` tokens.
**Fix**: accumulate batch_size_file chunks before calling `train_sequence`.
**Gain**: correct training (AMD was learning on 32-token windows, not 256).
**Files**: `aideen-training-lab/src/trainer.rs` (train_on_file batch accumulation)

### ✅ B5 — Hist adjoint double-run
**Was**: `apply_fused_deq_update` called `run_staged_adjoint_picard_no_readback` twice —
second call with stale `rhs_slot_buf` injected garbage gradients.
**Fix**: `clear_slot_rhs=true` on main adjoint, removed second run (~45 lines).
**Gain**: ~3.6% TPS + correct gradients.
**Files**: `aideen-backbone/src/gpu_deq.rs`

---

## Part 2: High-Priority Improvements (actionable now)

### 🔥 B6 — LM loss readback blocks CPU before adjoint starts
**Problem**: `train_step_no_readback(..., read_loss=true)` at trainer.rs:756 does:
```
queue.submit(LM kernels)
map_async(loss_staging)
device.poll(Maintain::Wait)   ← CPU STALL ~2-3ms
```
This stall happens BEFORE the adjoint Picard starts. GPU is idle (waiting for CPU).
The loss value is only needed for logging every 10 steps, not for gradient computation.

**Fix**: Switch to `read_loss=false`, start the adjoint immediately, then call
`gpu_lm.read_cached_loss()` AFTER `apply_fused_deq_update` completes. By then the GPU
has already computed the loss (it ran during the adjoint), so the readback is free.

**Estimated gain**: ~+15-20% TPS (removes 1 device.poll per step from critical path).
**Files**: `trainer.rs:756`, `gpu_lm_head.rs:1093-1107`
**Risk**: low — `read_cached_loss()` already exists, just need to call at right time.

### 🔥 B7 — `std::env::var()` called in hot path per training step
**Problem**: 52 `env::var` calls in `gpu_deq.rs` + 26 in `trainer.rs` = ~78 per step.
Most are in `apply_fused_deq_update` and `run_staged_adjoint_picard_no_readback`, called
every single training sequence. `std::env::var()` is a syscall (getenv), not cached by Rust.

At 11 TPS × 78 calls = ~860 syscalls/sec for env var lookups.

**Fix**: Parse all env vars once at `GpuDeq::new()` and store as struct fields (bool/u32/f32).
Example:
```rust
pub struct GpuDeq {
    // ...
    pub cfg_hist_gated: bool,       // was: env::var("AIDEEN_DEQ_HIST_GATED") per step
    pub cfg_fused_profile: bool,    // was: env::var("AIDEEN_FUSED_PROFILE") per step
    pub cfg_hist_train_carrier: bool,
    // etc.
}
```

**Estimated gain**: ~+5-15% (depends on OS; macOS getenv is ~0.3µs × 78 × 11TPS ≈ 0.26ms/s)
**Files**: `aideen-backbone/src/gpu_deq.rs`, `aideen-training-lab/src/trainer.rs`
**Risk**: low — pure refactor, semantics unchanged.

### 🔥 B8 — LM head allocates Vec + clone per step
**Problem**: `train_step_no_readback` at `gpu_lm_head.rs:985-997`:
```rust
let mut sampled_indices = Vec::with_capacity(max_samples + targets.len());
// ... fill ...
sampled_indices.sort_unstable();
sampled_indices.dedup();
self.last_sampled_indices = sampled_indices.clone();  // ← heap alloc + copy
```
Two Vec allocations per step: initial allocation + clone.

**Fix**: Pre-allocate `sampled_indices` as a reusable `Vec<u32>` field in `GpuLmHead`.
Replace `sampled_indices.clone()` with `std::mem::swap` or just track `last_num_samples`
without copying the full vec.

**Estimated gain**: ~+2-5% (saves ~(max_samples+256)×4 bytes × 2 allocs per step)
**Files**: `aideen-backbone/src/gpu_lm_head.rs`

---

## Part 3: Medium-Priority Improvements

### 🟡 B9 — Context length too short (GPU underutilized)
**Problem**: ctx=256 is tiny. With batch=1, the DEQ processes 256 tokens per GPU dispatch.
wgpu kernel launch overhead ~0.5-1ms per dispatch. With 256 tokens:
- Useful work: 256 × forward+backward flops
- Overhead: N_kernels × launch_time
Increasing ctx doubles useful work per launch, keeping overhead constant.

**Fix**: Use `AIDEEN_CTX_LEN=512` or `1024` for training.
Benchmark: measure TPS at ctx=256, 512, 1024.

**Estimated gain**: ~+30-50% TPS at ctx=512 (better GPU utilization per dispatch).
**Constraint**: Metal buffer limit (16 bindings); longer ctx may need more M_state memory.
Verify no OOM before switching.

**Files**: `aideen-training-lab/src/bin/train.rs:261-264`

**Validation (2026-03-24)**:
`AIDEEN_CTX_LEN=512` on tinyshakespeare showed `mode=NORMAL`, `conv=OK`, no DEQ-INVALID.
TPS **12.2** at `chunk 10` (AIDEEN_MAX_CHUNKS=20), contr stayed < 1.

### 🟡 B10 — Separate encoder per adjoint Picard iteration
**Problem**: `run_staged_adjoint_picard_no_readback` submits N_adj=6 separate command encoders,
one per Picard iteration. Each `queue.submit()` has CPU-side overhead for command buffer
creation, encoding, and Metal API call.

Can we batch multiple iterations? No — each iteration reads results written by the previous
(data dependency). BUT: the 4 kernels within each iteration (gcomb, gmix, gscore, accum) are
already batched in one encoder. This is already optimal per-iteration.

**Fix (minor)**: Reduce N_adj (currently default 6). At contr≈0.20, adjoint may converge
in 4 iterations. Profile: does loss change if `adj_iters=4` vs `6`?

**Estimated gain**: ~+15% if adj_iters 6→4 (2 fewer submits per step).
**Files**: `aideen-core/src/state.rs:adj_iters`
**Risk**: medium — may affect gradient quality.

**Validation (2026-03-24, short)**:
`adj_iters=4` vs `6` on stress_test (seed 11, 10 iters).
TPS 7.4 (adj=4) vs 6.7 (adj=6), loss/contr essentially unchanged in short run.
Marked as candidate only; keep default at 6 until longer-run quality is verified.

### 🟡 B11 — Second `read_debug_buffer` call in `train_on_tokens`
**Problem**: `trainer.rs:1047` — inside `train_on_tokens` diagnostic block, `gpu.read_debug_buffer()`
is called every 10 steps WITHOUT using the cache. This is a separate code path from `train_sequence`.

**Fix**: Use `self.cached_debug_buf` or same cache logic as in `train_sequence`.
**Estimated gain**: minor for `train_on_tokens` path (not used in benchmark).
**Files**: `aideen-training-lab/src/trainer.rs:1045-1047`

### 🟡 B12 — queue.write_buffer for params per step
**Problem**: `apply_fused_deq_update` writes `UpdateUniforms` (52 bytes) to GPU buffer every step.
`train_step_no_readback` writes `TrainParams` (48 bytes) + `target_indices_buf` (256×4=1KB) +
`sampled_indices_buf` (~8KB) to GPU every step.

These `queue.write_buffer` calls for params are fine (tiny). But `sampled_indices_buf` write is
~8KB/step = ~88KB/s — negligible on Metal PCIe. Keep as-is.

**Estimated gain**: negligible.

### 🟡 B16 — Fuse `apply_gradient_update` into `apply_fused_deq_update`
**Problem**: After the main fused update, we were still submitting a separate encoder/pass
to apply accumulated gradients when `grad_accum=1`. This adds extra CPU-side submit overhead.

**Fix**: Add `apply_accum` flag to `UpdateUniforms` and run `apply_grad_update_main` as a
final pass in the same encoder when needed.

**Estimated gain**: small (~+3% TPS observed).
**Files**: `aideen-backbone/src/gpu_deq.rs`, `aideen-backbone/src/shaders/fused_deq_update.wgsl`,
`aideen-training-lab/src/trainer.rs`

**Validation (2026-03-24)**:
Baseline TPS 6.6 vs fused TPS 6.8 on stress_test (20 iters, same config). `contr` stayed < 1.

---

## Part 4: Structural (unavoidable DEQ overhead)

### ⛔ S1 — DEQ requires multiple Picard iterations
**Nature**: Fixed-point solving requires iterating until convergence. With 8 fwd iters + 6 adj
iters = 14 total passes vs Transformer's ~2 (1 fwd + 1 bwd). This is the fundamental ~7×
overhead that cannot be removed without changing the model.

**Mitigations**:
- Reduce N_fwd if convergence is faster (check hit_ratio; if >0.8, can try 6 fwd iters)
- Warm-start: h_{t+1} starts from h_t (already implemented via H_curr carry)
- Anderson acceleration: replaces simple Picard with faster-converging fixed-point iteration.
  Would reduce iterations 8→4 at same accuracy. Significant engineering effort.

### ⛔ S2 — wgpu/Metal kernel dispatch overhead vs PyTorch
**Nature**: nanoGPT uses PyTorch's Metal backend with fused kernels (attention, LayerNorm,
Adam all fused). wgpu submits individual compute passes, each with Metal API overhead.

Measured: nanoGPT 9,500 TPS vs AIDEEN 11 TPS. Even a pure-Transformer in wgpu would likely
achieve only ~100-200 TPS (50-100× gap remains from kernel dispatch overhead).

**Mitigations** (long-term):
- Fuse more kernels in WGSL shaders (e.g., fuse LM probs+update into one pass)
- Move to Metal shader directly (bypass wgpu overhead for bottleneck kernels)
- Increase batch_size and ctx_len (amortizes per-dispatch cost)

### ⛔ S3 — Single-token attention (O(h²) slots, but O(n) via SSM)
**Clarification**: AIDEEN's attention is slot-to-slot O(h_slots²)=O(64), NOT token-to-token.
Already O(n) via SSM recurrence. Flash Attention does NOT apply here.
The NGI Zero proposal mentioning Flash Attention was incorrect for AIDEEN's architecture.

---

## Part 5: Longer-Term Improvements

### 🔵 B13 — Gradient accumulation (batch_size > 1)
Already implemented via `AIDEEN_GRAD_ACCUM=N`. Use N=4-8 to amortize dispatch overhead.
With N=4: 4 forward passes before weight update = ~4× more tokens per update.
**Files**: `aideen-training-lab/src/trainer.rs` (AIDEEN_GRAD_ACCUM loop)

### 🔵 B14 — Embedding gather batching
Currently: embedding lookup submits one encoder per batch. When GRAD_ACCUM>1, each
accumulation step resubmits embeddings. Could merge into one larger batch gather.

### 🔵 B15 — Async checkpoint saves
`save_checkpoint` calls `device.poll(Maintain::Wait)` to readback weights. Currently blocking.
Could serialize weights in background thread. Not a TPS bottleneck (save every 30min).

---

## Summary Table

| ID | Description | Status | Est. Gain | Effort |
|----|-------------|--------|-----------|--------|
| B1 | Debug buffer sampling | ✅ Fixed | +25% | — |
| B2 | Hot-loop pre-alloc | ✅ Fixed | +5-10% | — |
| B3 | targets_u32 clone | ✅ Fixed | +1% | — |
| B4 | Batch size bug | ✅ Fixed | critical | — |
| B5 | Hist adjoint double-run | ✅ Fixed | +3.6% | — |
| B6 | LM loss async readback | 🔥 High | +15-20% | 1h |
| B7 | Cache env vars | 🔥 High | +5-15% | 2h |
| B8 | LM sampled_indices alloc | 🔥 High | +2-5% | 0.5h |
| B9 | Larger ctx_len | 🟡 Medium | +30-50% | 0.5h (config) |
| B10 | Reduce adj_iters 6→4 | 🟡 Medium | +15% | 1h (validate) |
| B11 | train_on_tokens debug cache | 🟡 Medium | minor | 0.5h |
| B12 | write_buffer params | 🟡 Medium | negligible | — |
| B16 | fuse apply_grad into fused_update | 🟡 Medium | +3% | 0.5h |
| B13 | Gradient accumulation | 🔵 Low | already impl | — |
| S1 | DEQ Picard iters | ⛔ Structural | ~7× gap | — |
| S2 | wgpu vs PyTorch overhead | ⛔ Structural | ~120× gap | — |
| S3 | Flash Attention (N/A) | ⛔ N/A | — | — |

**Projected TPS after B6+B7+B8+B9**: ~11 × 1.20 × 1.10 × 1.03 × 1.40 ≈ **21-23 TPS**

---

## Next Actions (ordered by ROI)

1. **B9 first** — try `AIDEEN_CTX_LEN=512` in benchmark script, measure TPS. Zero code change.
2. **B10** — test `adj_iters=4` on stress_test, verify loss convergence unaffected.
3. **B6** — async LM loss: change `read_loss=true` → `false`, move `read_cached_loss()` after adjoint.
4. **B7** — cache env vars: add `cfg_hist_gated`, `cfg_hist_train_*` etc. as GpuDeq fields.
5. **B8** — pre-alloc sampled_indices in GpuLmHead.
