# Why DEQ Models Need Spectral Warm-Up: Lessons from Training on Consumer Hardware

**Author:** Juan Marchetto ([@JuanMarchetto](https://github.com/JuanMarchetto))
**Date:** March 30, 2026
**Project:** [AIDEEN](https://github.com/SergioAriel/aideen) — Open-Source DEQ+Mamba AI Engine in Rust

---

## TL;DR

Deep Equilibrium Models (DEQ) require their weight matrices to satisfy a contractivity condition (spectral norm < 1.0) before the fixed-point iteration can converge. When training from random initialization, this condition is violated — spectral norms are typically 2-3x above the threshold. We found that our benchmark showed 100% unconverged Picard iterations (contractivity > 2.0) while our training pipeline showed 0% unconverged (contractivity < 0.85). The difference: the training pipeline loads a pre-conditioned checkpoint where spectral renormalization has already been applied. The benchmark initializes from scratch.

This is not a bug — it's an inherent property of DEQ architectures that has practical implications for benchmarking and evaluation.

## Background

AIDEEN is an open-source AI engine that combines Deep Equilibrium Models with Mamba-style selective state memory. The core idea: instead of stacking 24-96 transformer layers, use a single parameter block and iterate it to a mathematical fixed point via Picard iteration.

For convergence, the Banach fixed-point theorem requires the mapping to be a contraction — meaning the spectral norm of the Jacobian must be strictly less than 1.0. In practice, we enforce this through spectral normalization of the weight matrices (Q, K, V, O, W_in) at regular intervals during training.

## The Problem

We built a benchmark harness (`aideen-bench`) that trains both AIDEEN (DEQ+Mamba) and a baseline transformer (via Candle) on the same data, same order, same seeds, with paired t-tests for statistical rigor.

The results were alarming:

```
AIDEEN DEQ:   val_loss 4.17 ± 0.00  (100% unconverged, contractivity 2.0-50.0)
Transformer:  val_loss 2.98 ± 0.02  (normal training)
```

Every single Picard iteration in the benchmark failed to converge. The DEQ was in permanent emergency mode. Meanwhile, our training pipeline on the same architecture showed perfect convergence:

```
Training pipeline: 0% unconverged, contractivity 0.4-0.85, 5.9 avg iterations
```

## The Investigation

We compared the two environments:

| | Benchmark (fails) | Training (works) |
|---|---|---|
| Initialization | Random weights | Pre-trained checkpoint |
| Spectral norms at init | ~2.0-3.0 (above threshold) | ~0.08-0.10 (below threshold) |
| Contractivity | 2.0-50.0 | 0.4-0.85 |
| Renorm frequency | Every 4 steps | Every 4 steps |
| Damping | 0.60 (emergency) | 0.85 (normal) |

The root cause became clear: **random initialization produces weight matrices with spectral norms well above the contractivity threshold.** The spectral renormalization (applied every 4 training steps) gradually brings these norms down, but it takes hundreds of steps before the contraction mapping condition is satisfied.

Our training pipeline worked because it loaded a checkpoint where this conditioning had already happened. The benchmark started from scratch and never had the chance to get the spectral norms under control.

## The Math

For a DEQ with mapping `f(x) = g(Wx + b)`, convergence requires:

```
||J_f|| < 1  (spectral norm of Jacobian < 1)
```

With multi-head attention and per-slot projections (Q, K, V, O, W_in), the effective Jacobian depends on the spectral norms of all weight matrices. Our spectral normalization enforces `σ(W) ≤ 0.10` for attention weights, but this enforcement is **iterative** — it clips the spectral norm via power iteration every N training steps.

At random initialization (e.g., Xavier uniform in [-0.2, 0.2] for a 512x512 matrix), the spectral norm is approximately:

```
σ(W_init) ≈ 0.2 × sqrt(512) ≈ 4.5
```

This is 45x above our target threshold of 0.10. Even with renormalization every 4 steps, it takes many passes to bring all weight matrices below threshold simultaneously.

## The Solution

DEQ benchmarks need a **spectral warm-up phase** before the comparison is fair:

1. Initialize both models from random weights (same seed)
2. Run the DEQ for N warm-up steps with aggressive spectral renormalization (every step, not every 4)
3. Do NOT count warm-up steps in the comparison
4. Start the iso-data comparison only after the DEQ's spectral norms are below threshold

This mirrors how DEQ models work in practice: you train once to establish the spectral conditioning, then subsequent training (fine-tuning, continued training) benefits from the pre-conditioned state.

We are implementing this protocol and will publish corrected benchmark results.

## Implications

1. **DEQ vs. Transformer benchmarks must account for warm-up.** A naive iso-data comparison disadvantages the DEQ because it includes the spectral conditioning phase where the model literally cannot converge.

2. **Checkpointing is more important for DEQ than for transformers.** A transformer checkpoint saves learned knowledge. A DEQ checkpoint saves both knowledge AND the spectral conditioning state. Losing the checkpoint means re-doing the warm-up.

3. **The spectral norm at initialization is a design parameter.** Smaller initialization ranges (e.g., [-0.05, 0.05] instead of [-0.2, 0.2]) would reduce the warm-up period but might slow learning. This is a trade-off worth investigating.

4. **This property is inherent to DEQ architectures**, not specific to AIDEEN. Any DEQ implementation using spectral normalization for convergence guarantees will face the same warm-up requirement.

## Hardware Context

All experiments were run on an AMD Radeon 780M integrated GPU (2 GB VRAM, Vulkan) via wgpu. Training throughput: 11 tokens/second (v0.1) and 32 tokens/second (v0.2 with Anderson acceleration and batched GPU encoders). The entire engine is written in Rust with WGSL compute shaders — no CUDA, no Python, no cloud dependency.

## Code

The full codebase is open source: [github.com/SergioAriel/aideen](https://github.com/SergioAriel/aideen)

- Benchmark harness: `aideen-bench/src/main.rs`
- Spectral normalization: `aideen-backbone/src/spectral_norm.rs`
- DEQ forward shader: `aideen-block/src/shaders/deq_forward.wgsl`
- Training pipeline: `aideen-training-lab/src/trainer.rs`

## What's Next

- Corrected benchmark with warm-up protocol
- Ablation study: DEQ with/without SSM, effect of spectral norm threshold
- Investigation of the W_hist learning rate (history gate receives ~1e-11 effective gradient steps — needs dedicated higher learning rate)
- Browser inference demo via WebGPU

---

*AIDEEN is applying for NGI Zero Commons Fund (EU) to fund rigorous benchmarks, a browser demo, and community foundations for hardware-sovereign AI. All code is MIT licensed.*
