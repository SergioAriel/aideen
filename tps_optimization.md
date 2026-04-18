# AIDEEN — Plan de optimización de TPS

**Rama**: `sequential-fpm-v2`  
**Baseline actual (Mac M1, d_r=512, h_slots=4, FPM_STAGE=4, NO_SLOT_ATTN)**:  
~91–137 TPS (AR training, ctx_len=128)

**Referencia parallelizacion-real (RDNA3, mismos params)**: 490–1020 TPS

---

## Bottlenecks identificados

### Forward shader (`deq_slot_attn_unified_clean.wgsl`)

| # | Problema | Impacto estimado |
|---|----------|-----------------|
| F1 | Token loop **serial dentro del shader** — todos los T tokens en un for-loop, sin paralelismo entre tokens | Alto |
| F2 | **12 Picard iters × 4 slots** por token — 48 sub-pasos por token, ~24k unidades seriales para T=512 | Alto |
| F3 | **k/v bottleneck write**: solo 32/256 threads activos (`if tid < RETAIN_RANK`) — 87.5% threads idle en esa fase | Medio |
| F4 | **RMS reductions con workgroupBarrier** — múltiples tree-reductions por token (h_rms, gate, proposal, prev_norm) serializan el pipeline | Medio |

### Backward shader (`fused_fpm_retain_bwd.wgsl`)

| # | Problema | Impacto estimado |
|---|----------|-----------------|
| B1 | **WG_SIZE=64, solo 4 workgroups totales** — un workgroup por slot, 7+ compute units idle durante todo el backward | Alto |
| B2 | **Token loop serial** — T tokens en for-loop dentro de 4 workgroups. No hay paralelismo cross-token | Alto |
| B3 | **Wout backward: O(T × d²/WG_SIZE)** — `for row in 0..d` anidado en `for k in 0..dims_per_lane`, operación más pesada. Para d=512: 4096 ops/thread/token | Alto |
| B4 | **W_k_write grad: O(T × d × r)** — loop sobre d=512 × r=32 por token, completamente serial dentro del workgroup | Medio |
| B5 | **Phase 2 bottleneck usa 32/64 threads** — mismo problema que F3 pero en WG_SIZE=64 | Bajo |

### Arquitectura / Sistema

| # | Problema | Impacto estimado |
|---|----------|-----------------|
| A1 | **Mac M1 Metal vs RDNA3 Vulkan** — overhead por dispatch mayor en Metal, wavefront de 32 vs 64, menor VRAM bandwidth | Estructural |
| A2 | **Dispatch overhead por token** — múltiples `begin_compute_pass` por token. En Metal cada uno cuesta ~50–100µs | Medio |
| A3 | **fpm_m_buf acceso no-coalescente** — `[t × h × d + slot × d + dim]`: para el token loop inverso salta h×d floats entre tokens del mismo slot | Bajo |

---

## Soluciones propuestas (ordenadas por relación costo/impacto)

---

### S1 — Backward paralelo sobre tokens [ALTO impacto, alto costo]

**Qué**: Reemplazar el for-loop de tokens en el backward por un dispatch de T workgroups (uno por token). Cada workgroup procesa un token, todos en paralelo.

**Cómo**: Restructurar `fused_fpm_retain_bwd.wgsl` para que el dispatch sea `(h_slots, T, 1)` en vez de `(h_slots, 1, 1)`. Los gradientes de pesos (`W_k_write`, `W_v_write`, `W_retain_up`, etc.) se acumulan con `atomicAdd` o en un buffer intermedio que se reduce aparte.

**Problema**: Los gradientes de `dm_new` (TBPTT carry) tienen dependencia serial hacia atrás — el token t necesita `dm_new` del token t+1. Solución: dos pasadas — primero computar todos los `g_m_inner` en paralelo, luego hacer la reducción del carry serial (solo T escalares, no T×d).

**Ganancia esperada**: 4–8× en el backward sobre GPU con ≥8 CUs.

---

### S2 — Bottleneck write distribuido entre todos los threads [MEDIO impacto, bajo costo]

**Qué**: En vez de `if (tid < RETAIN_RANK)` con 32 threads haciendo O(d) cada uno, distribuir el trabajo de la reducción d→r entre todos los 256 (forward) o 64 (backward) threads.

**Cómo** (forward, WG_SIZE=256, d=512, r=32):
```wgsl
// Cada thread calcula su contribución parcial para TODOS los r slots
// Reducción en shared memory: particionada en r bloques de WG_SIZE/r threads
for (var r = 0u; r < RETAIN_RANK; r++) {
    var partial = 0.0;
    for (var j = tid; j < d_model; j += WG_SIZE) {
        partial += W_k_write[j * RETAIN_RANK + r] * c[j];
    }
    // reducción dentro del bloque asignado a r
    shared_vals[tid] = partial;
}
workgroupBarrier();
// luego prefix-sum por r
```

**Ganancia esperada**: 2–3× en el bloque k/v del forward/backward.

---

### S3 — Fusionar dispatches: mini-batch de tokens por compute pass [MEDIO impacto, medio costo]

**Qué**: En vez de un compute pass por paso de Picard por token, agrupar N tokens por pass (e.g., N=4 o N=8). Reduce el overhead fijo de Metal por `begin_compute_pass`.

**Cómo**: Agregar dimensión Z al dispatch (`dispatch_workgroups(batch, h_slots, N)`). Dentro del shader, `wid.z` indica el token dentro del mini-batch.

**Restricción**: Los tokens dentro del mini-batch deben ser **independientes** entre sí (no dependen del resultado del token anterior para ese Picard iter). Esto es verdad dentro de la misma iteración de Picard — el token carry (`fpm_m_cache`) entre tokens sigue siendo serial pero solo en el update post-convergencia.

**Ganancia esperada**: 1.5–2× por reducción de overhead de dispatch, sin cambio en lógica.

---

### S4 — Wout backward: convertir a matmul [ALTO impacto en backward, medio costo]

**Qué**: El backward de `W_out` es actualmente un loop serial `O(d²)` por token dentro del workgroup. Con T tokens, es `O(T × d²)` total — la operación más costosa.

Esto es esencialmente una multiplicación de matrices: `∂L/∂W_out = Σ_t (g_m_outer[t] ⊗ m_inner[t])`. Se puede reformular como `G_out = H_g^T × H_m` donde `H_g` y `H_m` son matrices `T × d`.

**Cómo**: Usar un dispatch separado `fpm_wout_bwd` con workgroup `(d/16, d/16, 1)` y `tile_size=16`, igual al `fpm_shared_wout` que ya existe en el pipeline. Aprovechar el pipeline ya existente o extenderlo para el backward.

**Ganancia esperada**: 3–5× en el costo de Wout backward.

---

### S5 — Coalescing del fpm_m_buf [BAJO impacto, bajo costo]

**Qué**: Cambiar el layout de `fpm_m_buf` de `[T, h_slots, d]` a `[h_slots, T, d]`. En el backward (que itera tokens en orden inverso por slot), los accesos quedan coalescentes: `buf[slot * T * d + t * d + dim]`.

**Cómo**: Cambiar `fpm_m_off()` en el shader y el equivalente Rust. Requiere actualizar el forward también para escribir en el nuevo layout.

**Ganancia esperada**: 10–20% en backward por mejor cache line utilization.

---

### S6 — Picard iters adaptativos más agresivos [BAJO impacto, cero costo]

**Qué**: Reducir `min_deq_iters` para tokens donde el DEQ converge rápido. El homeostatic convergence check ya existe — bajar `FPM_HOMEO_MIN_ITERS` de 4 a 2.

**Ganancia esperada**: 0–20% dependiendo de la distribución de convergencia.

---

## Priorización recomendada

| Prioridad | Solución | Esfuerzo | Ganancia |
|-----------|----------|----------|----------|
| 1 | **S2** — Bottleneck distribuido | 1 día | 2–3× en fases FPM |
| 2 | **S3** — Mini-batch tokens | 2 días | 1.5–2× |
| 3 | **S4** — Wout matmul | 2 días | 3–5× en backward |
| 4 | **S1** — Backward paralelo | 3–5 días | 4–8× en backward |
| 5 | **S5** — M-buf coalescing | 0.5 días | ~15% |
| 6 | **S6** — Picard adaptativo | 0 días | ~10% |

**Stack acumulado** (todas implementadas): estimado 8–20× sobre baseline actual → **700–2000 TPS en RDNA3**, **200–500 TPS en M1**.

---

## Nota sobre avg_loss=NaN

El AR benchmark (2026-04-15, sequential-fpm-v2, 600 seq, ctx_len=128) produjo avg_loss=NaN durante todo el training. Esto es un bug separado de TPS — probablemente el sampled softmax con vocab=50 produce logits en rango que causa NaN en cross-entropy antes de la normalización. A investigar: clip de logits, escala de embeddings, o el flujo de gradientes en la primera iteración.
