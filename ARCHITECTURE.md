# AIDEEN Architecture — Forward Pass Step by Step

> Canonical runtime updated as of 2026-04-03.
> Older sections below are preserved as archival notes where explicitly marked.

---

## Canonical Runtime (2026-04-03)

El runtime canónico actual ya no usa `deq_forward.wgsl`, `deq_forward_exact.wgsl`
ni `deq_forward_subgroup.wgsl`. El path real de forward es:

```text
embedding_train -> deq_slot_attn_unified_clean -> deq_forward_pool -> lm_train
```

La ecuación efectiva del bloque DEQ por token/slot es:

```text
signal_t,s   = W_in,s x_t
slot_ctx_t,s = SlotAttn(signal_t,1..S)
pre_t,s      = signal_t,s + H_curr_t-1,s + slot_ctx_t,s + slot_anchor_s
f_t,s        = NormScale ⊙ RMSNorm(pre_t,s)
h_t,s(next)  = damping * f_t,s + (1 - damping) * H_curr_t-1,s
```

con estas decisiones canónicas:

- `slot_ctx` se construye desde `signal`, no desde `H_curr`
- `H_curr` se mantiene como memoria local rápida entre tokens
- `H_curr` entra explícitamente en `pre`
- `H_curr` también ancla el damping
- la historia explícita `HistCtx/MState` queda fuera del runtime canónico por ahora

Consecuencia práctica:

- el único selector estructural del solve canónico que sigue activo es
  `AIDEEN_DEQ_TOKEN_CARRY`
- los shaders de historia (`hist_v2_*`) se conservan como material de referencia,
  pero no forman parte del path baseline actual

---

## Overview

AIDEEN is a **Deep Equilibrium Model (DEQ)** operating on `h_slots` parallel hidden states.
For each token position `t`, the canonical solver now computes:

```
h*_s = f(h*_s ; s_t, H_{t-1})   for s = 0..h_slots-1
```

where `f` involves slot attention built from the current token signal, per-slot input
injection, `H_curr` carry, and RMSNorm normalization. The Picard iteration finds the
fixed point, and `H_curr` is the only active temporal carrier in the baseline.

---

## 1. Token Embedding → `S_in`

**File:** `aideen-backbone/src/shaders/embedding_train.wgsl`

Input: token indices `[batch, seq_len]`
Output: `S_in[batch, seq_len, d_model]`

Each token index is looked up in the embedding table. The result is a dense vector of
`d_model` floats per token, shared as input to all `h_slots`.

---

## 2. DEQ Block — Per-Token Loop

**File:** `aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl`

One GPU workgroup per batch item. Iterates over `t = 0..seq_len-1` sequentially
(temporal dependency through `M_{t-1}`).

### 2a. Scratch Buffer Layout

Per token, `scratch_stride = 2 * h_slots * d_model` floats:

| Region | Offset | Size | Content |
|--------|--------|------|---------|
| `signal_base` | `0` | `h_slots × d_model` | `W_in,s × s_t` per slot |
| `attn_base` | `+h_slots*d` | `h_slots × d_model` | slot attention output `slot_ctx` |

### 2c. Prelude: Per-Slot Input Injection

**Before the Picard loop**, for each slot `s`:

```
signal_s = W_in_s × s_t        (d_model × d_model matrix multiply)
```

`W_in` is **per-slot**: GPU buffer `W_in[h_slots * d_model * d_model]`, slot `s`
starts at offset `s * d_model * d_model`. All slots are currently initialized from
the same CPU matrix (replicated by `w_in_gpu_flat()`).

Stored at `Scratch[signal_base + s * d_model]`.

### 2d. Slot Attention Context

For each slot `s`, Q/K/V are computed from the per-slot `signal` of the current token,
not from `H_curr`. That yields:

```text
slot_ctx_s = SlotAttn(signal_1, ..., signal_S)_s
```

This separation is intentional:

- `signal` handles token content
- `slot_ctx` handles inter-slot coordination
- `H_curr` handles token-to-token memory

---

## 3. Picard Fixed-Point Iteration

Runs for up to `max_iters` iterations, exits early when `max|h_next - h_curr| < epsilon`.

### 3a. Q/K/V Projection (shared matrices across all slots)

```
Q_s = W_q × h_s
K_s = W_k × h_s
V_s = W_v × h_s
```

`W_q`, `W_k`, `W_v` are `d_model × d_model` matrices, **shared** across all `h_slots`.
Diversity between slot representations comes from their different `signal_s` states.

The resulting attention output is stored at `Scratch[attn_base + s*d]`.

### 3b. Cross-Slot Attention

Score matrix (clamped to `[-4, 4]` to prevent over-sharpening):
```
score[qs, ks] = clamp(Q_qs · K_ks / sqrt(d_model), -4, 4)
```

Softmax over key slots per query slot:
```
attn[qs, ks] = softmax_k(score[qs, :])
```

Stored at `Scratch[attn_weight_base + qs * h_slots + ks]`.

Mix vector (attention-weighted sum of V):
```
mix_qs = Σ_ks  attn[qs,ks] * V_ks
```

Temporarily stored at `Scratch[mamba_base + qs*d]` during the iteration
(overwritten by `M_t` post-convergence).

Output projection:
```
attn_out_qs = W_o × mix_qs
```

Stored at `Scratch[attn_base + qs*d]`.

### 3c. Combined Update

```
attn_signal_s = attn_out_s + signal_s            (h-dependent terms)
combined_s    = attn_signal_s + slot_anchor_s + hist_ctx_s
```

**`slot_anchor_s`** — trainable per-slot bias `d_model` floats per slot. Provides
slot identity / specialization. `∂slot_anchor/∂h = 0`.

**RMSNorm** — anchored to `attn_signal_s` only (stop-gradient on `slot_anchor` and
`hist_ctx`), so the Lipschitz constant of `f` depends only on h-dependent terms:

```
rms_s = RMS(attn_signal_s) = sqrt(mean(attn_signal_s²) + rms_floor² + 1e-6)
h_new_s = NormScale ⊙ (combined_s / rms_s)
```

`NormScale` is a learned `d_model` vector (LayerNorm-style per-channel scale).

**Damped Picard step:**
```
h_next_s = damping * h_new_s + (1 - damping) * h_curr_s
```

### 3e. Associative Memory (FPM Integration)

Si `ENABLE_ASSOC_EVENT_GATE` está activo, cada slot procesa una rama de memoria asociativa:
1.  **Read Query**: `q_assoc = W_q_assoc × h_s`.
2.  **Softmax Attention**: Se compara `q_assoc` contra las llaves en los `ASSOC_BANKS`.
    *   **Clipping**: Los logits de atención se claman a `[-25, 25]` para estabilidad.
3.  **Context Retrieval**: `assoc_ctx = Σ_b match_b * bank_value_b`.
4.  **Normalization**: `assoc_ctx = assoc_ctx / RMS(assoc_ctx)`.
5.  **Injection**: `fpm_ctx = fpm_ctx + alpha_assoc * assoc_ctx`.

Esta señal inyectada es lo que permite al modelo realizar **Associative Recall** de largo plazo.

### 3f. Convergence Check and Contractivity

```
max_delta = max_s,d |h_next_s[d] - h_curr_s[d]|
converged = (max_delta < epsilon)
```

Contractivity (ratio of consecutive deltas) only measured when
`d_prev > epsilon * 10` to avoid noise artifacts near convergence.

---

## 4. Post-Convergence: Fixed-Point Memory Temporal Update

After the loop exits (h* found), for each slot `s`:

```
x_proj_s = (I + W_x) × RMSUnit(h*_s)          (identity skip + learned transform)
M_inner_s = a_s * M_{t-1}_s + (1 - a_s) * x_proj_s     (gated accumulation)
M_t_s     = (I + W_out) × M_inner_s             (identity skip + learned projection)
```

where `a_s = sigmoid(A_log_s)` ∈ (0,1) is the per-dimension decay.

In `hist_selective` mode: `a_s` is further modulated by a content-dependent delta:
```
delta_factor = 1 + 0.5 * tanh(W_delta × RMSUnit(h*) + b_delta)
a_s = pow(a_base, delta_factor)     clamped ≥ alpha_min
```

`M_t_s` is written to `Scratch[mamba_base + s*d]` for token `t`, available as
`M_{t-1}` at the next token.

---

## 5. Slot Pooling → `H_pooled`

After each token's fixed point, the `h_slots` states are merged into one vector:

```
w_s = 0.7 * (Σ_q attn[q→s] / h_slots) + 0.3 * (1/h_slots)
H_pooled[t] = Σ_s  w_s * h*_s
```

Weights are computed from the **last iteration's attention** (stop-gradient).
- Active slots (high attention-received) get weight ≈ `0.7/h_active + 0.3/h_slots`
- Dead slots get weight ≈ `0.3/h_slots` (non-zero, enabling recovery)

This is a deliberate design choice: active slots propagate stronger gradients to the
LM head, dead slots maintain a gradient path.

---

## 6. LM Head and Loss

**File:** `aideen-backbone/src/shaders/lm_train.wgsl`

```
h_norm = RMSNorm(H_pooled[t])     = g_lm ⊙ (H_pooled[t] / RMS(H_pooled[t]))
logit_v = W_lm[v] · h_norm + b_lm[v]
p_v     = softmax(logit_v)
L_t     = -log(p_{target_t})
```

Loss is averaged over the sequence. `W_lm` is `[vocab_size, d_model]`.

Uses **sampled softmax** for efficiency (a subset of vocab tokens per step).

---

## 7. Backward Pass — Picard Adjoint

**File:** `aideen-backbone/src/shaders/staged_adjoint_picard.wgsl`

Uses the **implicit function theorem**: if `h* = f(h*)`, then
`dL/dθ = (∂f/∂h*)^{-∞} * ∂f/∂θ` solved via another Picard iteration
on the adjoint equation.

The LM head produces `dl_dh[t, d]` (gradient of loss w.r.t. `H_pooled[t]`).
This becomes the backward RHS `b_in`.

### Stages (dispatched sequentially):

| Kernel | Purpose |
|--------|---------|
| `picard_init` | Initialize adjoint v⁰ = b_in * w_s + rhs_slot |
| `picard_gcomb` | Jacobian-vector product through RMSNorm: g = (N/rms) * v - z*(z·v)/rms³ |
| `picard_gmix` | W_o^T: g_mix = W_o^T * g_comb |
| `picard_gscore` | Softmax backward: g_score[qs,ks] = attn[qs,ks] * (g_mix·V_ks - Σ_j attn[qs,j]*g_mix·V_j) |
| `picard_accum` | Accumulate: V-path, K-path, Q-path → new adjoint |

The adjoint iteration converges when v converges (same contractivity as forward).

---

## 8. Weight Update

**File:** `aideen-backbone/src/shaders/fused_deq_update.wgsl`

Accumulates gradients over all `seq_len * h_slots` entries, then applies SGD with weight decay:

| Weight | Type | Shape | Notes |
|--------|------|-------|-------|
| `W_q` | shared | `d × d` | Cross-slot attention query |
| `W_k` | shared | `d × d` | Cross-slot attention key |
| `W_v` | shared | `d × d` | Cross-slot attention value |
| `W_o` | shared | `d × d` | Attention output projection |
| `W_in[s]` | **per-slot** | `d × d` each | Input injection, h_slots copies |
| `W_hist` | shared | `d × d` | History projection (in HistParams) |
| `W_x` | shared | `d × d` | Fixed-Point Memory input projection |
| `W_out` | shared | `d × d` | Fixed-Point Memory output projection |
| `slot_anchor[s]` | per-slot | `d` each | Per-slot identity bias |
| `NormScale` | shared | `d` | RMSNorm per-channel scale |
| `W_lm` | shared | `vocab × d` | LM head (AdamW) |
| `g_lm` | shared | `d` | LM RMSNorm scale (AdamW) |

Weight decay: `w ← w * (1 - lr * 0.01) - lr * g * grad_scale`
Gradient clipping: per-element clip to `[-0.5, +0.5] * lr * grad_scale`

---

## 9. HistParams Buffer Layout

**File:** `aideen-backbone/src/fixed_point_memory_reasoning.rs`, serialized in `trainer.rs`

Starting offset 0:

| Offset | Size | Content |
|--------|------|---------|
| `0` | `d²` | `W_hist` (shared, d×d matrix) |
| `d²` | `h_slots * d` | `hist_slot_scale` (per-slot scale, d per slot) |
| `d² + h*d` | `h_slots * d` | `hist_slot_bias` (unused/zero) |
| `d² + 2h*d` | `h_slots` | `hist_gate_logit` (per-slot gate scalar) |
| `d² + 2h*d + h` | `h_slots * d` | `slot_anchor` (per-slot identity bias, d per slot) |
| `+ h*d` | `d²` | `W_delta` (selective SSM, d×d) |
| `+ d²` | `d` | `b_delta` (selective SSM bias) |
| `+ d` | `1` | `hist_selective_flag` |
| `+ 1` | `1` | `hist_warmup` (0→1 ramp) |
| `+ 1` | `1` | `rms_floor` |
| `+ 1` | `1` | `contr_floor` |
| `+ 1` | `1` | `hist_inject_flag` |
| `+ 1` | `1` | `hist_minner_zero` |
| `+ 1` | `1` | `hist_force_nomamba` |
| `+ 1` | `1` | `hist_prelude_skip` |
| `+ 1` | `1` | `hist_loop_force_nomamba` |

---

## 10. Debug Log Layout (DebugLog[])

The table below is archival and refers to the old `deq_forward.wgsl` family. It is no
longer the canonical runtime path after 2026-04-03.

Written by `deq_forward.wgsl` (batch 0, thread 0), then **partially overwritten**
by `fused_deq_update.wgsl` at indices 40-44.

| Index | Written by forward | Overwritten by update |
|-------|-------------------|----------------------|
| 0 | 777.7 (sentinel) | — |
| 1 | batch_size | — |
| 2 | d_model | — |
| 10 | seq_len | — |
| 11 | max_h_seen | — |
| 12 | H_curr[0] (first element) | — |
| 13 | avg iters/token | — |
| 14 | 1 if fully converged else 0 | — |
| 15 | non-converged token count | — |
| 16 | max_delta (over seq) | — |
| 17 | last_delta (final token) | — |
| 21 | max_contractivity | — |
| 22 | avg inj_rms | — |
| 23 | avg hist_rms | — |
| 24 | avg hist/inj ratio | — |
| 25 | avg mamba_rms | — |
| 26 | avg Q_rms (slot 0) | — |
| 27 | avg K_rms (slot 0) | — |
| 28 | avg V_rms (slot 0) | — |
| 29 | avg mix_rms | — |
| 30 | avg attn_out_rms | — |
| 31 | avg attn_max | — |
| 32 | avg attn_entropy | — |
| 33 | avg combined_rms | — |
| 40 | HistParams[0] (W_hist[0,0]) | **g_wq** (gradient) |
| 41 | HistParams[1] (W_hist[0,1]) | **g_wo** (gradient) |
| 42 | HistParams[2] (W_hist[0,2]) | **g_wv** (gradient) |
| 43 | slot_anchor[0,0] (first slot anchor value) | **g_win[0]** (slot-0 W_in grad) |
| 44 | slot_anchor[0,1] (second slot anchor value) | **g_wk** (gradient) |
| 45 | rms_floor | — |
| 46 | contr_floor | — |
| 47 | hist_inject_flag | — |
| 48 | hist_minner_zero | — |
| 49 | hist_force_nomamba | — |
| 50 | hist_prelude_skip | — |
| 51 | hist_loop_force_nomamba | — |

**Important:** Indices 40-44 show the FINAL value (after the update shader runs).
The forward-written values are overwritten. To observe slot_anchor or W_hist values,
read indices 45-51 or add dedicated debug slots.

---

## 11. Complete Data Flow Summary

The diagram below is archival and describes the older history-coupled runtime. The
canonical runtime now uses `deq_slot_attn_unified_clean.wgsl` with `H_curr` as the
active temporal carrier.

```
token_ids[batch, seq_len]
    │
    ▼ embedding_train.wgsl
S_in[batch, seq_len, d_model]
    │
    ▼ deq_forward.wgsl  (one workgroup per batch item, seq loop inside)
    │
    │  For each t = 0..seq_len-1:
    │    ┌─────────────────────────────────────────────────────┐
    │    │  signal_s = W_in_s × s_t           (per-slot W_in) │
    │    │  hist_ctx_s = f(W_hist, M_{t-1}_s) (stop-grad)     │
    │    │                                                      │
    │    │  Picard loop (max_iters):                           │
    │    │    Q_s, K_s, V_s = W_q/k/v × h_s  (shared W)      │
    │    │    attn[qs,ks] = softmax(Q·K/√d)                   │
    │    │    mix_qs = Σ_ks attn[qs,ks] * V_ks                │
    │    │    attn_out_s = W_o × mix_s                        │
    │    │    combined_s = attn_out_s + signal_s               │
    │    │               + slot_anchor_s + hist_ctx_s          │
    │    │    rms_s = RMS(attn_out_s + signal_s)   (h-only)   │
    │    │    h_new_s = NormScale ⊙ (combined_s / rms_s)      │
    │    │    h_next_s = β*h_new_s + (1-β)*h_curr_s           │
    │    │    [converged?] → exit loop                         │
    │    │                                                      │
    │    │  Fixed-Point Memory update (post-convergence):                   │
    │    │    x_proj_s = (I + W_x) × RMSUnit(h*)              │
    │    │    M_t_s = (I+W_out)(a*M_{t-1} + (1-a)*x_proj)     │
    │    │                                                      │
    │    │  Slot pooling:                                       │
    │    │    w_s = 0.7*(attn_recv_s/h) + 0.3*(1/h)           │
    │    │    H_pooled[t] = Σ_s w_s * h*_s                    │
    │    └─────────────────────────────────────────────────────┘
    │
    ▼ lm_train.wgsl
H_pooled[batch, seq_len, d_model]
    │  h_norm_t = g_lm ⊙ (H_pooled[t] / RMS(H_pooled[t]))
    │  logit_v  = W_lm[v] · h_norm_t + b_lm[v]
    │  L_t = -log softmax(logit_{target_t})
    │  → dl_dh[t] = dL/dH_pooled[t]
    │
    ▼ staged_adjoint_picard.wgsl  (Picard adjoint backward)
    │  Solves: v* = (I - J_f^T)^{-1} * dl_dh (via Picard on adjoint)
    │  Stages: init → gcomb → gmix → gscore → accum
    │  Produces: rhs_slot_buf[t, s, d] = ∂L/∂W_in contribution
    │
    ▼ fused_deq_update.wgsl
    W_q, W_k, W_v, W_o ← SGD update (shared)
    W_in[s]             ← SGD update (per-slot)
    W_lm, g_lm         ← AdamW update
    HistParams          ← SGD update (W_hist, slot_anchor, etc.)
```

---

## 12. Key Design Invariants

1. **Contractivity**: `f` is contractive if `‖J_f‖ < 1`. RMSNorm with stop-gradient
   on non-h terms keeps the Lipschitz constant bounded by `‖W_q‖‖W_k‖‖W_v‖‖W_o‖`.
   Spectral renorm (`renorm_every_steps=4`) keeps individual σ ≤ 0.10.

2. **History as additive bias**: `hist_ctx` and `slot_anchor` shift the fixed point
   but do not change its convergence properties. This is the key stability guarantee.

3. **Temporal separation**: `M_t` is computed **after** convergence. During the Picard
   loop it is read-only (from the previous token). This means the DEQ fixed-point
   equation for token `t` is fully self-contained.

4. **Pooling and gradients**: The 70/30 attention-received pooling ensures dead slots
   still receive gradients. This is a recent change (2026-03-18). Previously, uniform
   pooling `1/h_slots` was used.
