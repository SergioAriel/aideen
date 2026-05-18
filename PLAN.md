# Final Implementation Plan: Selective, DEQ-Compatible Temporal Memory for AIDEEN

> Update 2026-04-03:
> this document describes a historical plan centered on `hist_gated` / `HistCtx`.
> The current canonical baseline no longer uses that route as the main runtime.
> The active base is:
>
> - `slot_ctx = Attn(signal)`
> - `pre = signal + H_curr + slot_ctx + slot_anchor`
> - `H_curr` as fast local memory between tokens
> - explicit history out of the baseline for now
>
> From this point on, the next phases should be reinterpreted as reference
> material for reconstructing history on top of `H_curr`, not as a currently active operational plan.

## Summary

A strong temporal memory will be implemented, inspired by Fixed-Point Memory, but integrated in a way compatible with the DEQ fixed point.

The productive strategy will be:

1. Keep the temporal memory **outside the Picard loop**.
2. Make history enter the DEQ as **per-token fixed context**, per-slot, projected, with controlled magnitude and an explicit gate.
3. Train the **historical interface** first.
4. Then train the **external temporal dynamics**:
   - simple first,
   - then input-dependent selective.
5. Keep `fixed_mamba` and `init_mamba` only as ablations.
6. Do not increase the scratch stride to `7K+1`; keep `6K+1` with lifetime reuse.

This plan is the productive route. The internal integration of memory inside the DEQ is explicitly deferred to a later experimental phase.

## Naming note (avoid ambiguity)

In this plan and in the current flags, "**Fixed-Point Memory**" means **external temporal memory / historical channel** (the context `c_{t,k}` or the historical initialization), **not** an "internal mamba" inside the DEQ.

Historical flag reference:
- `AIDEEN_DEQ_ONLY=1` → DEQ with no attention or history.
- `AIDEEN_DEQ_NO_MAMBA=1` → attention ON, **history OFF**.
- `AIDEEN_DEQ_HIST_GATED=1` → attention ON, **history ON** (gated).

The current baseline should no longer depend on those selectors.

## Current state (2026-04-03)

Validated canonical baseline:

- single runtime via `deq_slot_attn_unified_clean.wgsl`
- `slot_ctx` built from `signal`
- `H_curr` active in `pre`
- `H_curr` active in damping
- `AIDEEN_DEQ_TOKEN_CARRY` as the single central selector of the solve
- explicit history `HistCtx/MState` out of the baseline path

Controlled comparisons against the previous baseline:

- `seed=42`, `80 chunks`
  - previous: `loss=8.6026`, `tps=3103.3`
  - current canonical: `loss=8.4108`, `tps=3359.6`
- `seed=11`, `20 chunks`
  - previous: `loss=10.0097`, `tps=2921.5`
  - current canonical: `loss=9.7964`, `tps=3346.7`

Current structural conclusion:

- `slot_ctx` is indeed essential
- `H_curr` is indeed essential
- the previous explicit history did not justify its cost on a short horizon
- the next memory should be designed starting from `H_curr` or very close to `H_curr`

---

## Historical state (2026-03-14)

**Phase 1 closed** with a stable baseline:
- `hist_cap_floor_mult = 0.08`
- `α_min = 0.070`, `α_max = 0.20`
- warmup of `α_min` via `AIDEEN_HIST_ALPHA_WARMUP_STEPS=20`
- epsilon warmup via `AIDEEN_DEQ_EPS_WARMUP_STEPS=10`, `AIDEEN_DEQ_EPS_WARMUP_VALUE=3e-4`
- stability confirmed on seeds `7/11/13/42` (no BOOST/FAIL)

**Phase 2 closed**: input-dependent selectivity with trainable `W_Δ`/`b_Δ`.
Validation: seeds `7/11/13/42`, `AIDEEN_HIST_SELECTIVE=1`, `AIDEEN_HIST_TRAIN_DELTA=1`, `AIDEEN_LM_FROZEN=1`,
no BOOST/FAIL, `conv=OK` at steps 10/20/30/40, `hist/inj ≈ 0.08` stable.
Structural requirement (solver): when `AIDEEN_DEQ_HIST_GATED=1`, `adaptive_max_iters >= 20`
(convergence criterion, not a patch).

**Phase 3 closed**: LM active with stable selective history.
Validation: seeds `11/13/42`, `AIDEEN_HIST_SELECTIVE=1`, `AIDEEN_HIST_TRAIN_DELTA=1`, LM **not** frozen,
no BOOST/FAIL, `conv=OK` at steps 10/20/30/40, `hist/inj ≈ 0.08` stable.
Structural fix: LM backward on GPU now includes RMSNorm; `dl_dh` correct (post-RMSNorm).

**LM policy (updated)**:
- LM enabled by default from Phase 3.
- `AIDEEN_LM_FROZEN=1` remains as an isolation flag if needed in later phases.

---

## 1. Target failure mode

The goal is to fix this structural failure:

- `no_mamba` converges cleanly.
- any current attempt to reintroduce temporal memory (`fixed_mamba`, `init_mamba`) degrades:
  - contractivity,
  - convergence,
  - or total cost.

### Structural cause
The temporal history enters the DEQ with an incorrect interface:
- too raw,
- without adequate projection,
- without correct magnitude control,
- without sufficient per-slot specialization.

The problem is not "having memory", but **how the memory modulates the DEQ operator**.

---

## 2. Target architecture

## 2.1 Variables

For token `t` and slot `k`:

- `s_t ∈ R^D`: current input
- `M_{t-1,k} ∈ R^D`: previous external temporal memory
- `h_{t,k}^{(\ell)} ∈ R^D`: Picard iterate
- `h_{t,k}^* ∈ R^D`: converged DEQ state
- `c_{t,k} ∈ R^D`: fixed historical context of the token
- `x_{t,k} ∈ R^D`: temporal projection of `h_t^*`

---

## 2.2 Per-slot historical interface

### Historical projection

\[
u_{t,k} = W_{hist}^{shared} M_{t-1,k} + d_k \odot M_{t-1,k}
\]

where:

- `W_hist^{shared} ∈ R^{D×D}`
- `d_k ∈ R^D`, one per slot

### Initialization

- `W_hist^{shared} = I + 0.01\xi`
- `d_k = 0`

### Justification
- avoids an identity bypass via `d_k`
- keeps the historical channel alive from the start
- does not force the memory to enter "raw"

---

## 2.3 Magnitude control of the historical channel

Define:

\[
r_u = RMS(u_{t,k}), \qquad \tau_t = RMS(inj_t)
\]

\[
\tilde u_{t,k} = \frac{u_{t,k}}{\max\left(1,\frac{r_u}{\tau_t + \epsilon}\right)}
\]

### Important decision
`\tau_t` will be treated as **detach** in the backward of the cap.

### Justification
- `\tau_t` is a reference scale of the input, not a training path toward `W_in`
- avoid spurious gradients `\partial \tau_t / \partial W_in`
- keep the historical channel decoupled from the amplitude learning of the main input

This must be implemented as an explicit invariant.

---

## 2.4 Gate with a positive floor

\[
\alpha_k = \alpha_{min} + (\alpha_{max} - \alpha_{min}) \sigma(g_k)
\]

Operational values:
- `α_min = 0.070`
- `α_max = 0.20`
- `α_k(0) ≈ 0.10` from the initial logits.

### Justification
- a non-zero floor avoids a dead historical channel; 0.070 keeps contr < 0.85 and hist/inj in 0.04–0.08 under stress.
- a 0.20 ceiling limits the gate so it does not erode contractivity over long steps.

### Final context

\[
c_{t,k} = \alpha_k \tilde u_{t,k}
\]

---

## 2.5 DEQ operator

For each Picard iteration:

\[
q_k = W_q h_k^{(\ell)},\quad
k_j = W_k h_j^{(\ell)},\quad
v_j = W_v h_j^{(\ell)}
\]

\[
a_{k,j} = softmax_j\left(\frac{q_k^\top k_j}{\sqrt{D}}\right)
\]

\[
attn_k(h^{(\ell)}) = W_o \sum_j a_{k,j} v_j
\]

\[
inj_t = W_{in}s_t
\]

\[
z_k^{(\ell)} = attn_k(h^{(\ell)}) + inj_t + c_{t,k}
\]

\[
f_k(h^{(\ell)}; s_t, M_{t-1}) = RMSNorm(z_k^{(\ell)})
\]

\[
h_k^{(\ell+1)} = \beta f_k(h^{(\ell)}; s_t, M_{t-1}) + (1-\beta)h_k^{(\ell)}
\]

### Key property
- `c_{t,k}` is fixed during all iterations of the token
- `M_{t-1,k}` does not change inside the DEQ
- the DEQ keeps solving a fixed operator in `h`

---

## 3. External temporal memory

## 3.1 Simple temporal phase

\[
x_{t,k} = W_x h_{t,k}^*
\]

\[
a = \sigma(-A_{log})
\]

\[
\tilde m_{t,k} = a \odot M_{t-1,k} + (1-a)\odot x_{t,k}
\]

\[
M_{t,k} = W_{out}\tilde m_{t,k}
\]

This phase serves to stabilize the historical interface and the basic temporal backward.

## 3.2 Selective temporal phase

It is then replaced by:

\[
\Delta_{t,k} = softplus(W_\Delta h_{t,k}^* + b_\Delta)
\]

\[
a_{t,k} = \exp(-\Delta_{t,k} \odot A)
\]

\[
x_{t,k} = W_x h_{t,k}^*
\]

\[
\tilde m_{t,k} = a_{t,k} \odot M_{t-1,k} + (1-a_{t,k})\odot x_{t,k}
\]

\[
M_{t,k} = W_{out}\tilde m_{t,k}
\]

### Justification
This restores the input-dependent selectivity that the current memory does not have, without introducing temporal dynamics inside Picard.

---

## 4. Backward of the historical interface

## 4.1 Gradient toward the context

Since `c_{t,k}` enters as a sum in `z_k`:

\[
g^c_{t,k} = \frac{\partial L}{\partial c_{t,k}} = g^{comb}_{t,k}
\]

## 4.2 Backward of the gate

\[
\frac{\partial L}{\partial g_k}
=
(\alpha_{max}-\alpha_{min})\sigma(g_k)(1-\sigma(g_k))
\sum_t \left\langle g^c_{t,k}, \tilde u_{t,k} \right\rangle
\]

## 4.3 Exact backward of the cap

Let:

\[
\tilde u = s u,\qquad
s = \min\left(1,\frac{\tau}{r}\right),\qquad
r = \sqrt{\frac{1}{D}\sum_i u_i^2 + \epsilon}
\]

and \(g = \partial L / \partial \tilde u\).

### Unclipped branch
\[
\frac{\partial L}{\partial u} = g
\]

### Clipped branch
\[
\frac{\partial L}{\partial u}
=
s g
-
\frac{s}{D r^2} u (u^\top g)
\]

### Requirement
This formula will be implemented exactly.  
The incomplete approximation `s g` is not accepted.

## 4.4 Backward of the projection

\[
\frac{\partial L}{\partial W_{hist}^{shared}}
+=
\sum_{t,k} g^u_{t,k}\otimes M_{t-1,k}
\]

\[
\frac{\partial L}{\partial d_k}
+=
\sum_t g^u_{t,k}\odot M_{t-1,k}
\]

\[
g^{(deq)}_{M_{t-1,k}}
=
(W_{hist}^{shared})^\top g^u_{t,k}
+
d_k \odot g^u_{t,k}
\]

---

## 5. External temporal backward

## 5.1 Simple phase

### Temporal forward
\[
M_t = g(M_{t-1}, h_t^*)
\]

### Temporal backward
It will be done by `TBPTT` with chunks:

- first `L = 16`
- then `L = 32`

### Order
1. LM backward produces `dl_dh_t`
2. temporal backward produces:
   - `g_h^{temporal}(t)`
   - gradients of `W_x`, `W_out`, `A_log`
3. it forms:
\[
b_t = dl_dh_t + g_h^{temporal}(t)
\]
4. staged Picard solves the DEQ for that `b_t`

## 5.2 Selective phase

Add:

\[
\frac{\partial L}{\partial h_t^*}
\leftarrow
\frac{\partial L}{\partial h_t^*}
+
W_\Delta^\top \frac{\partial L}{\partial \Delta_t}
\]

\[
\frac{\partial L}{\partial \Delta_t}
=
\frac{\partial L}{\partial a_t}\odot (-a_t \odot A)
\]

This will be implemented explicitly in the temporal shader.

---

## 6. Scratch and memory

## 6.1 Constraint

Kept:

\[
scratch\_stride = D(6K + 1) + K^2
\]

It is not increased to `7K+1`.

## 6.2 Lifetime reuse

- `q_base`: Q
- `k_base`: K
- `v_base`: V
- `attn_base`: attention output / temporary
- `mamba_base`:
  - during DEQ: `c_{t,k}`
  - afterward: `M_t`
- `signal_base`: `inj_t`
- `m_inner_base`:
  - during DEQ: temporary
  - afterward: `\tilde m_t`
- `attn_weight_base`: attention weights

### Justification
This avoids the bandwidth and cache tax of a larger stride.

---

## 7. Changes per file

## 7.1 `fixed_point_memory_reasoning.rs`

Add:
- `w_hist_shared: DMatrix<f32>`
- `hist_slot_scale: DMatrix<f32>` (`K×D`)
- `hist_gate_logit: DVector<f32>` (`K`)
- `w_delta: DMatrix<f32>` (`D×D`) for the selective phase
- `b_delta: DVector<f32>` (`D`) for the selective phase

### Initialization
- `w_hist_shared = I + 0.01\xi`
- `hist_slot_scale = 0`
- `hist_gate_logit` for an effective gate of `0.10`
- `w_delta`, `b_delta` initialized but frozen until the selective phase

### Renorm
- `w_hist_shared` does **not** use the `0.10` threshold
- reason:
  - `W_hist` does not enter the Jacobian with respect to `h`
  - it does not affect DEQ contractivity
  - the effective scale is already controlled by the cap relative to `\tau_t`

If numerical control is applied, it will only be an optional soft clip (`σ ≤ 1.5`), not an aggressive amputation.

## 7.2 `gpu_deq.rs`

Add buffers:
- `hist_w_buf`
- `hist_slot_scale_buf`
- `hist_gate_buf`
- `w_delta_buf`
- `b_delta_buf`

Update the bind groups of:
- forward DEQ
- fused DEQ update
- temporal backward

## 7.3 `deq_forward.wgsl`

New mode:
- `AIDEEN_DEQ_HIST_GATED=1`

Implement:
1. read of `M_{t-1,k}`
2. computation of `u`
3. computation of `\tau_t`
4. relative cap
5. gate
6. write of `c_{t,k}` to `mamba_base`
7. `combined = attn + inj + c`

Keep the post-convergence temporal update.

## 7.4 `fused_deq_update.wgsl`

Add the update of:
- `W_hist_shared`
- `hist_slot_scale`
- `hist_gate_logit`

It must include the exact backward of the cap.

## 7.5 New shader: `temporal_mamba_backward.wgsl`

Simple phase:
- gradients of `W_x`, `W_out`, `A_log`

Selective phase:
- gradients of `W_\Delta`, `b_\Delta`

## 7.6 `trainer.rs`

Add the stage:
- `AttnHistFixed-Point Memory`

Execution order:
1. LM backward
2. if phase 1:
   - staged Picard with `dl_dh`
3. if temporal phase:
   - external temporal backward
   - form `b_total`
   - staged Picard with `b_total`
4. fused DEQ update
5. temporal update

---

## 8. Test cases and acceptance

## 8.1 Numerical tests

1. `hist_gated` small reference:
- `D=16`, `K=2`
- cosine `> 0.99`

2. Grad-check of the cap:
- clipped branch
- cosine `> 0.99`

3. Grad-check of:
- `W_hist_shared`
- `hist_slot_scale`
- `hist_gate_logit`

4. Temporal grad-check:
- simple phase: `W_x/W_out/A_log`
- selective phase: `W_\Delta/b_\Delta`

## 8.2 Stress tests

Baselines:
- `AIDEEN_DEQ_NO_MAMBA=1`
- `AIDEEN_DEQ_FIXED_MAMBA=1`
- `AIDEEN_DEQ_INIT_MAMBA=1`

Candidate:
- `AIDEEN_DEQ_HIST_GATED=1`

Runs:
- `20` iter
- `100` iter

## 8.3 Acceptance criteria

### Phase 1 (hist_gated + selective with α_min=0.070, α_max=0.20)
- `mode = NORMAL` (occasional BOOST acceptable if contr < 1 and it self-corrects)
- `conv = OK`
- `hit ≈ 0`
- `contr < 0.85` at Steps 10/20/30/40
- `loss20 <= loss20(no_mamba) + 1%`
- `loss40` not explosively growing; `max_h < 3.5`

### Historical channel
- `hist_ctx_rms / inj_rms` median in `[0.04, 0.08]`
- maximum `< 0.12`
- mean effective gate per slot in `[0.06, 0.12]`

### Dead-channel criterion
If after `100` steps:
\[
median(hist\_ctx\_rms / inj\_rms) < 0.06
\]
the phase fails.

### Temporal phase
- non-zero and finite gradients
- no `BOOST/FAIL`
- cost per iteration `≤ 1.5×` the `no_mamba` baseline

---

## 9. Assumptions and defaults

1. `\tau_t` is treated as `detach` in the backward of the cap.
2. `d_k` starts at `0`.
3. `W_hist_shared` starts near identity.
4. `W_hist_shared` is not renormalized to `0.10`.
5. Gate with a floor of 0.070 and a ceiling of 0.20 (clamp active).
6. The historical channel enters per-slot.
7. The selective temporal memory is the final goal.
8. The temporal backward is done by `TBPTT`.
9. `fixed_mamba` and `init_mamba` remain as ablations.
10. The scratch stride is not increased.

---

## 10. Expected result

When the selective phase is finished, AIDEEN should have:

- a historical channel alive from the start,
- a temporal memory that is truly input-selective,
- a historical interface that preserves both direction and useful magnitude,
- clean DEQ convergence,
- and a cost controlled via truncated external temporal backward.

That will be the serious baseline before considering a memory truly internal to the DEQ.
