# Why Mamba Lives Outside the DEQ Loop

*Technical note — March 2026*

## The Problem

Deep Equilibrium Models (DEQ) find a fixed point h* by iterating a function f until convergence:

```
h^(k+1) = f(h^(k); x)    until    |h^(k+1) - h^(k)| < epsilon
```

For this to converge, f must be a **contraction**: its Lipschitz constant L must satisfy L < 1. In AIDEEN, we enforce this via spectral normalization (sigma <= 0.10 every 4 gradient steps).

Mamba-style selective state space models (SSM) maintain a temporal memory M that evolves across tokens:

```
M_t = a_t * M_{t-1} + (1 - a_t) * x_proj(h_t)
```

where `a_t = sigmoid(A_log)` is an input-dependent decay rate.

The natural instinct is to put M inside the DEQ loop — let the fixed-point solver jointly optimize h* and M. But this breaks the convergence guarantee.

## Why It Breaks

If M participates in the fixed-point iteration, the function becomes:

```
h^(k+1) = f(h^(k), M^(k); x)
M^(k+1) = g(h^(k); M^(k-1))
```

Now M depends on h, which depends on M. The Jacobian of the combined system (h, M) has cross-terms that make the Lipschitz bound much harder to satisfy. In practice, we observed that:

1. The spectral norm of the combined Jacobian exceeded 1.0 during training
2. Picard iteration failed to converge within the iteration cap
3. The model oscillated instead of finding a stable fixed point

## The Solution: Frozen Context

AIDEEN places the Mamba state **outside** the Picard iteration loop. During fixed-point solving, M_{t-1} enters as a **frozen context** — it does not update until h* is found:

```
# Step 1: Picard iteration with frozen M
for k in 0..max_iters:
    h^(k+1) = f(h^(k); x_t, M_{t-1}_frozen)   # M does not change
    if converged: break

# Step 2: Post-convergence Mamba update
M_t = a_t * M_{t-1} + (1 - a_t) * x_proj(h*)   # Now update M using converged h*
```

This preserves the contraction property because:
- f depends on h (the variable being iterated) and fixed quantities (x_t, M_{t-1})
- The Jacobian df/dh has no cross-terms with M
- Spectral norm enforcement on df/dh alone is sufficient for convergence

## The Trade-off

The temporal memory M does not benefit from the iterative refinement of the DEQ — it only sees the final h*. This means M captures a "one-shot" summary of the converged state rather than participating in the reasoning process.

In practice, we find this is acceptable:
- The DEQ's iterative refinement handles intra-token reasoning (cross-slot attention, input injection)
- The Mamba state handles inter-token temporal context (what happened in previous tokens)
- These are naturally separable concerns

## Implementation

In the GPU shader (`deq_forward.wgsl`), the separation is explicit:

1. **Lines 140-993:** Picard iteration loop. Reads `M_{t-1}` from scratch memory but never writes to it.
2. **Lines 996-1098:** Post-convergence Mamba update. Reads converged `H_curr`, computes new `M_t`, writes to scratch memory for the next token.

The history context enters the DEQ as a stop-gradient additive bias:

```
hist_ctx_s = alpha_s * gate(M_{t-1,s})
combined = RMSNorm(attn_signal + slot_bias + hist_ctx_s)
```

where `alpha_s` and `gate` are learnable but their gradients do not flow through M during the Picard iteration.

## Results

With this architecture:
- **100% Picard convergence** across all tokens in training (0 unconverged tokens out of thousands)
- **Stable training** for 12+ hours on consumer GPU (AMD Radeon 780M, 2GB VRAM)
- **Contractivity** maintained below 0.85 throughout training
- **Average 5-6 Picard iterations** per token (cap at 20, rarely needed)

The history mechanism contributes a stable context signal (hist_rms ~ 9.5e-4, hist/inj ratio ~ 0.25), confirming that temporal information flows into the DEQ without destabilizing convergence.

## References

- Bai, Kolter & Koltun (2019). "Deep Equilibrium Models." NeurIPS.
- Bai et al. (2021). "Stabilizing Equilibrium Models by Jacobian Regularization." ICML.
- Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."
- Winston & Kolter (2020). "Monotone Operator Equilibrium Networks." NeurIPS.
