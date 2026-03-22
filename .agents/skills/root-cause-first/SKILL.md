---
name: root-cause-first
description: Use when debugging model quality, convergence, stability, or performance regressions. Prioritizes structural and mathematical root causes (architecture, equations, Jacobians, invariants) over masking symptoms with any type of intervention.
---

# Root Cause First

Use this skill for AI/DEQ/SSM/attention instability, divergence, parity failures, or sudden regressions.

## Core Rule

Do not hide failures with symptom-masking before identifying the structural cause.
Every proposed code change must be analyzed first to verify it is not only a patch and that it is a real fix for the target failure mode.

Patch vs real fix is decided by causal correctness, not by the type of intervention applied.
- A change is **not a patch** if it fixes the failure where it originates, restores a broken invariant/property, does not rely on fragile compensations, and remains valid across regimes.
- A change **is a patch** if it mostly suppresses symptoms, depends on narrow tuning/guards, or leaves the broken mechanism intact.
- The anti-patch argument must be causal. It is not enough to justify a change by the category of intervention made or avoided; the report must explain the restored invariant or corrected causal path.

Note: any type of intervention can be either structural or a patch — equation changes, hyperparameter changes, architecture changes, guard additions. The category is not the evidence; the causal argument is.

Allowed temporary guards only if:
- They are clearly labeled as temporary.
- A root-cause investigation runs in parallel.
- Exit criteria to remove the guard are defined.

## Change Gate (Mandatory Before Editing)

For each planned change, explicitly state:
- Failure mode targeted.
- Why this change addresses the cause (equation/path/invariant level).
- Why it is not just symptom masking under the definition above.
- How success will be validated (specific metric/test/log).

If these four points cannot be justified, do not implement the change.

## Workflow

1. Reproduce and freeze the failing case:
   - Record exact command, seed, profile, stage flags, and commit.
   - Capture minimal logs needed to compare before/after (`loss`, `conv`, `mode`, `hit`, `lastΔ`, `maxΔ`, `iters`, fallback counters).

2. Isolate by ablation:
   - Run `deq-only` first.
   - Add one component at a time (`+attn`, then `+mamba`).
   - Keep dataset and seed fixed.
   - Stop when the first failing component is identified.

3. Verify mathematical consistency:
   - Confirm forward map `f(h,s)` matches design equations.
   - Confirm backward/Jacobian path matches forward dependencies.
   - Reject any gradient path that drops terms from active forward branches.

4. Check invariants:
   - Contractivity/convergence invariants.
   - CPU/GPU parity at step level (same inputs, same state).
   - State semantics (stateless vs stateful blocks inside DEQ iteration).
   - **Buffer/accumulator lifecycle**: For every iterative solver call (Picard, CG, power iteration, adjoint),
     verify that each accumulator and RHS buffer is explicitly cleared before the solve begins.
     A call site that reuses a buffer without clearing injects stale state into the solver equation
     (`v_{k+1} = Jᵀvₖ + b + rhs_stale` instead of `v_{k+1} = Jᵀvₖ + b`), corrupting the result.
     Check every call site of the solver, not just the first one.

5. Apply structural fix:
   - Prefer the intervention that corrects the failure at its origin.
   - If backend math is incomplete, route to mathematically consistent path until backend is corrected.

6. Revalidate:
   - Short stress (sanity), long stress (stability drift), and parity tests.
   - Report what changed, why it is mathematically correct, and what remains open.

## Anti-Patterns

- Using any single intervention as sole fix without establishing causal connection to the failure origin.
- Treating the category of change made or avoided as proof that a fix is structural.
- Declaring "stable" from 10-20 iterations only.
- Comparing heterogeneous paths without controlling seed/state.
- **Analyzing forward symptoms while backward state is broken**: If the failure manifests in the forward
  pass (divergence, contractivity > 1, loss spike), do not assume the cause is in the forward path.
  Backward buffer corruption (stale accumulators, wrong gradient accumulation) produces corrupted
  parameter updates that then cause forward-pass failures at the next step. Always audit the backward
  path — and specifically its buffer lifecycle — before building forward-only theories.
- **Theoretical analysis without empirical falsification**: Gain/contractivity estimates derived from
  initialization statistics can be orders of magnitude wrong (e.g., theorized inj_rms ≈ 0.004 vs
  observed 0.055). Run one concrete measurement to anchor theory before constructing multi-step causal
  chains from unverified numbers.

## Required Output in Debug Reports

- Failing component (first ablation stage that breaks).
- Structural cause (equation/path mismatch).
- Structural fix applied.
- Temporary guards still active (if any).
- Evidence: before/after metrics and commands.
- Open items: what was not closed and what the next correct step is.

## GPU-Only Default (AIDEEN Policy)

Default is GPU-only for AIDEEN. CPU execution is allowed only as an explicit, temporary fallback.
When CPU paths exist or CPU↔GPU sync appears:

1. **Identify the CPU owner**
   - Is CPU the source-of-truth for weights/state?
   - Is CPU path still used for training, or only as fallback?

2. **Evaluate urgency**
   - **Urgent** if CPU ownership breaks invariants (e.g., per-slot weights lost on sync, mismatched layouts, divergent math, or CPU-only updates that re-upload stale tensors).
   - **Can wait** only if CPU path is fully disabled, cannot be selected by default, and does not receive or overwrite state.

3. **Decision rule**
   - If CPU causes loss of information or changes semantics: **prioritize GPU-only conversion now**.
   - If CPU is unused and cannot be selected without explicit flags: document it, but defer cleanup until stability is closed.

4. **Actions**
   - Remove or hard-gate CPU fallbacks that can silently reintroduce stale state.
   - Make GPU buffers the sole source-of-truth for training.
   - If CPU must remain, require explicit opt-in and ensure exact parity with no lossy conversions.

Report: whether GPU-only is achieved, what CPU code remains, and whether any CPU sync can alter training behavior.
