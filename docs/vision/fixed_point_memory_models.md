# Fixed-Point Memory Models for AIDEEN

## Purpose of this document

This document compares the two memory architectures that are relevant for AIDEEN:

1. **H-only DEQ with post-solve memory update**
2. **Joint fixed point over (H, M)**

The goal is not to declare both equally ready.
The goal is to make the distinction precise, so we can:

- start from the simpler and safer model first;
- build a clean implementation and training path;
- later design a true fixed-point memory model `(H, M)` on top of that base.

This document should be read together with:
- [memory_reference.md](/Users/sergiosolis/Programacion/AIDEEN/docs/vision/memory_reference.md)

---

## Executive summary

There are two conceptually valid ways to build memory in AIDEEN.

### Model A. H-only DEQ + post-write memory

The token solve is only over `H`.
Memory is read from the past as fixed context and updated only after `H*` has been found.

This is the simpler model.
It is the correct place to start.

### Model B. Joint fixed point over `(H, M)`

The token solve is over the combined state `(H, M)`.
`H` and `M` interact with each other during the same fixed-point search.

This is closer to the original intuition of “fixed-point memory”.
It is also much harder to stabilize and specify correctly.

### Recommended order

The correct sequence is:

1. implement and cleanly validate **Model A**;
2. use it as the architectural baseline;
3. only then design **Model B** as a new system, not as a patch on top of a messy implementation.

---

## 1. Why we need two models explicitly

The phrase “fixed-point memory” can mean two very different things.

### Interpretation 1
Memory influences the fixed point of `H`, but memory itself is not part of the fixed-point state.

### Interpretation 2
Memory and token state co-evolve together until both reach a joint equilibrium.

If we do not separate these interpretations, the implementation becomes ambiguous:

- some equations assume memory is frozen;
- other equations assume memory is dynamic inside the token;
- control and convergence logic become incoherent;
- it becomes impossible to tell whether a failure comes from read, write, or the coupled operator itself.

This document exists to prevent that ambiguity.

---

## 2. Model A: H-only DEQ with post-solve memory update

This is the simpler architecture.

### 2.1 Core equation

For token `t`:

```text
read_t = Read(M_{t-1})
H_t    = solve_H(H ; signal_t, slot_ctx_t, read_t)
M_t    = Update(M_{t-1}, H_t)
```

The fixed point is over `H` only.

### 2.2 What is fixed during the token solve

During the Picard iterations of token `t`:

- backbone weights are fixed;
- `M_{t-1}` is fixed;
- `read_t` is fixed or recomputed from a frozen snapshot only;
- write is not applied yet.

### 2.3 What changes after the token solve

After `H_t` converges:

- the model computes a memory write proposal;
- retain / forget / update logic is applied;
- the new memory state `M_t` becomes available for future tokens.

### 2.4 Conceptual meaning

This model says:

- memory is a causal context for the token solve;
- memory is updated from the solved token result;
- memory does not participate as a live variable of the same fixed-point system.

### 2.5 Why it is a good starting point

This model is better as the first implementation because:

- it preserves a clean DEQ over `H`;
- it keeps the Jacobian analysis simpler;
- causality is easier to enforce;
- inference with fixed weights and runtime memory is straightforward;
- debugging read and write separately is possible.

### 2.6 What must be true for Model A to be valid

1. The token reads only past memory.
2. The read enters as residual context, not dominant signal.
3. The write happens only after `H*` is available.
4. The write result is what future tokens may observe.
5. No implicit write-to-read leak occurs inside the same token solve.

### 2.7 What this model is *not*

Model A is **not** a joint memory fixed point.

It is memory around a fixed point, not memory inside the same equilibrium state.

That is acceptable and still meaningful.
It just has to be named correctly.

---

## 3. Model B: joint fixed point over (H, M)

This is the more ambitious architecture.

### 3.1 Core equation

For token `t`, the system seeks:

```text
(H_t*, M_t*) = solve_{H,M}(H, M ; signal_t, slot_ctx_t)
```

Equivalently, a joint operator:

```text
H_{k+1} = F_H(H_k, M_k ; signal_t)
M_{k+1} = F_M(H_k, M_k ; signal_t)
```

with the target fixed point:

```text
H* = F_H(H*, M* ; signal_t)
M* = F_M(H*, M* ; signal_t)
```

### 3.2 Conceptual meaning

This model says:

- token state and memory state co-adapt inside the same token;
- memory is not merely past context;
- memory is one of the variables being solved jointly with `H`.

### 3.3 Why this is closer to the original intuition

This is the architecture that most literally deserves the name:

> Fixed-Point Memory

because memory is not external to the fixed point; it is part of it.

### 3.4 Why this is much harder

The stability problem is no longer only about:

```text
∂H_next / ∂H
```

It becomes the stability of the full joint Jacobian:

```text
[ ∂H_next/∂H   ∂H_next/∂M ]
[ ∂M_next/∂H   ∂M_next/∂M ]
```

That means:

- memory feedback can destroy contractivity;
- read and write are no longer cleanly separated;
- controller logic must reason about a larger state space;
- it becomes much easier to hide instability in gating or damping tricks.

### 3.5 Additional design decisions required

Model B cannot be implemented seriously without deciding all of these explicitly:

1. What part of `M` is allowed to move inside one token solve?
2. Is the read recomputed every iteration?
3. Is the write proposal recomputed every iteration?
4. Is `M` updated in full every iteration, or partially?
5. Is the solve Picard-only, alternating, or split-operator?
6. What is the convergence criterion over the joint state?
7. What does inference persistence mean if `M` is solved jointly per token?

### 3.6 Why it is still compatible with fixed inference weights

This is important.

A joint `(H, M)` fixed point does **not** require changing the backbone weights during inference.

It still fits the intended inference model as long as:

- weights remain frozen;
- only runtime state `(H, M)` evolves.

So Model B is still compatible with the AIDEEN design goal of:

- fixed parameters,
- dynamic runtime memory.

### 3.7 Why we should not jump to it first

Even though Model B is conceptually attractive, it should not be the first implementation target because:

- failures are much harder to localize;
- every bug looks like “memory instability”;
- causality is easier to accidentally violate;
- debugging read/write/storage ownership becomes far more difficult;
- the implementation tends to accumulate hidden guards if the conceptual model is not already clean.

---

## 4. Side-by-side comparison

| Aspect | Model A: H-only DEQ + post-write | Model B: joint fixed point `(H, M)` |
|---|---|---|
| Fixed-point variable | `H` only | `(H, M)` jointly |
| Memory during token solve | frozen past memory | live variable inside solve |
| Read semantics | causal snapshot | part of joint operator |
| Write semantics | post-solve update | inside same equilibrium system |
| Stability difficulty | moderate | high |
| Causality reasoning | simpler | harder |
| Debugging | much easier | much harder |
| Inference with fixed weights | natural | still possible |
| Closeness to original “FPM” idea | medium | high |
| Recommended implementation order | first | second |

---

## 5. Why Model A should come first

Model A is not a concession in the bad sense.
It is the correct foundational step.

### 5.1 It lets us validate the right things separately

Model A allows us to validate, one by one:

- storage semantics;
- causal snapshot rules;
- read path;
- inject path;
- write path;
- persistence.

If those are not clean in Model A, then Model B will only amplify confusion.

### 5.2 It gives a deployable memory even if Model B takes longer

Even if we never ship joint `(H, M)` equilibrium immediately, Model A still gives:

- dynamic inference memory;
- fixed weights during inference;
- per-slot memory state;
- causal adaptation over time;
- compatibility with KV-cache/TurboQuant-like runtime state.

So Model A is already valuable on its own.

### 5.3 It is the right baseline for evaluating Model B later

If Model B is ever built, it should be judged against a clean Model A baseline.

Otherwise we will never know whether:

- the improvement comes from genuine joint equilibrium,
- or just from mixing more moving parts together.

---

## 6. What Model A should look like in practice

This section makes the starting target explicit.

### 6.1 Storage semantics

Model A needs three clearly distinct roles:

1. **persistent memory owner**
   - authoritative memory carried forward across chunk boundaries

2. **token-local history**
   - memory snapshots produced token by token inside the chunk

3. **staging / scratch**
   - temporary storage for read/write computation only

These roles must not be silently conflated.

### 6.2 Token flow

For token `t`:

```text
M_prev   = persistent_owner or previous token-local snapshot
read_t   = Read(M_prev)
H_t      = solve_H(signal_t, slot_ctx_t, read_t)
M_t      = Update(M_prev, H_t)
store M_t as token-local history
if needed, promote last token to persistent owner
```

### 6.3 Inference behavior

At inference time:

- weights remain fixed;
- only the memory owner/history evolves;
- memory state can be kept in runtime cache form.

### 6.4 What must be prohibited

For Model A, these should be explicitly forbidden:

- write influencing the same token solve before `H*` is finalized;
- ambiguous ownership of memory state;
- using scratch buffers as if they were persistent memory;
- hidden parameter mutation during inference.

---

## 7. What Model B would require when the time comes

Model B should only be attempted after Model A is clean.

### 7.1 Preconditions

Before starting Model B, we should already have:

1. clean storage semantics;
2. stable read path;
3. stable post-solve write path;
4. clear inference runtime memory representation;
5. validated baseline quality/cost with Model A.

### 7.2 New design tasks for Model B

Then, and only then, we should specify:

1. the joint state variable;
2. the exact `(H, M)` update equations;
3. whether to use simultaneous or alternating updates;
4. the joint convergence metric;
5. the backward path for the coupled operator;
6. the inference semantics of a jointly solved memory state.

### 7.3 What must not happen

We must not arrive at Model B by gradually stuffing write logic into Model A until the distinction disappears.

That path almost always creates:

- hidden coupling,
- poor observability,
- and a system that is neither a clean Model A nor a real Model B.

Model B must be treated as a new architecture.

---

## 8. Recommended roadmap

### Phase 1. Clean Model A

Goal:

- memory as causal runtime state around an `H`-only DEQ.

Checklist:

1. define persistent owner;
2. define token-local history;
3. define staging buffers;
4. validate read semantics;
5. validate inject semantics;
6. validate write semantics;
7. validate persistence;
8. validate training/inference alignment.

### Phase 2. Baseline and hardening

Goal:

- make Model A stable, auditable, and deployable.

Checklist:

1. long-run stability;
2. cross-seed stability;
3. cost profile;
4. inference cache semantics;
5. optional quantization path.

### Phase 3. Design Model B from scratch

Goal:

- true joint fixed-point memory `(H, M)`.

Checklist:

1. formal coupled equations;
2. joint invariants;
3. joint convergence controller;
4. exact comparison against Model A.

---

## 9. Final recommendation

The correct technical position is:

- **Yes**, a true fixed-point memory over `(H, M)` is possible.
- **Yes**, it is closer to the original intuition.
- **No**, it should not be the first implementation target.

The right starting point is:

> Build a clean `H`-only DEQ with causal post-solve memory first, then use that as the baseline and foundation for a later joint `(H, M)` fixed-point design.

That is the path that best protects:

- conceptual clarity,
- causal correctness,
- stability,
- and deployment realism.
