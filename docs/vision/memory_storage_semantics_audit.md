# Memory Storage Semantics Audit

## Purpose

This document audits the current storage-side implementation of memory against **Model A** from:
- [memory_reference.md](/Users/sergiosolis/Programacion/AIDEEN/docs/vision/memory_reference.md)
- [fixed_point_memory_models.md](/Users/sergiosolis/Programacion/AIDEEN/docs/vision/fixed_point_memory_models.md)

The goal is not to defend the current code. The goal is to classify what each buffer currently means, where it is used, and whether that meaning is compatible with the target architecture.

---

## Model A storage contract

Model A requires exactly three roles:

1. **Persistent memory owner**
   - authoritative memory carried across chunk boundaries

2. **Token-local history**
   - per-token memory snapshots inside the active chunk

3. **Staging / scratch**
   - temporary computation buffers that are not themselves the semantic memory

If a single buffer plays multiple semantic roles, that must be treated as a design smell until justified.

---

## Current buffers involved in memory

### 1. `MState`

Code references:
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/deq_bridge.rs:619](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/deq_bridge.rs:619)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:437](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:437)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:557](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:557)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:762](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:762)

### What it does today

`MState` is a persistent per-batch, per-slot buffer without token dimension.

In practice today it is used as:
- first-token fallback for memory reads;
- inter-chunk persistent state;
- seed for `fpm_m_cache` before the token loop.

### Classification against Model A

This is the closest thing to the intended **persistent memory owner**.

### Verdict

`MState` should remain the leading candidate for:
- **persistent owner of memory**

This part of the design is directionally correct.

---

### 2. `HistCtx`

Code references:
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:458](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:458)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:533](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:533)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:1304](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:1304)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/deq_bridge.rs:1487](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/deq_bridge.rs:1487)

### What it does today

`HistCtx` has token dimension and per-slot content.

In practice today it is used as:
- previous-token memory source inside the chunk;
- storage for per-token persisted memory snapshots;
- the source from which the last token is copied back into `MState` at the end of the chunk.

### Classification against Model A

This is the natural fit for:
- **token-local history**

### Verdict

`HistCtx` is semantically defensible as:
- token-local memory history inside the chunk

It should not be the global long-term owner.

This is also directionally correct.

---

### 3. `H_hist`

Code references:
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/deq_bridge.rs:634](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/deq_bridge.rs:634)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/deq_bridge.rs:1407](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/deq_bridge.rs:1407)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:509](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:509)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:1308](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:1308)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-backbone/src/gpu_deq.rs:2324](/Users/sergiosolis/Programacion/AIDEEN/aideen-backbone/src/gpu_deq.rs:2324)

### What it does today

`H_hist` originated as a persistent hidden-state history mechanism.

In FPM today it is also used as:
- first-chunk seed for `MState`;
- blended persistence target with `fpm_persist_beta`;
- a persistent slow-memory companion living in the same binding slot as the legacy history path.

### Classification against Model A

This buffer does **not** have a clean role under Model A.

It is currently doing at least two things:
- legacy hidden-state carry
- FPM persistence/blend support

### Verdict

`H_hist` is the main semantic ambiguity in the current design.

Under a clean Model A, `H_hist` should be one of only two things:

1. either a legacy mechanism that stays outside FPM memory semantics;
2. or a removed/replaced buffer if FPM memory already has `MState` + `HistCtx`.

It should **not** remain an implicit second persistent memory owner.

---

### 4. `fpm_m_cache`

Code references:
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:439](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:439)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:1292](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:1292)

### What it does today

`fpm_m_cache` is a workgroup-local runtime carry that:
- is initialized from `MState`;
- evolves token to token inside the active dispatch;
- becomes the source for `HistCtx` persistence when enabled.

### Classification against Model A

This is not the semantic memory owner.
It is:
- **staging/runtime working state**

### Verdict

`fpm_m_cache` is valid as staging/runtime scratch for the current chunk execution.
It should never be treated as the authoritative long-term semantic memory by itself.

---

## Current semantic picture

If we reinterpret the current code using Model A semantics, the cleanest reading is:

- `MState` = persistent memory owner
- `HistCtx` = token-local history
- `fpm_m_cache` = runtime staging / working state
- `H_hist` = ambiguous legacy overlap that should not remain an unexamined second memory owner

---

## What is already directionally correct

These ideas in the current code are conceptually sound for Model A:

1. token `t` reads token `t-1` via `HistCtx` inside the chunk;
2. only the first token falls back to `MState`;
3. `MState` has no token dimension and naturally fits cross-chunk persistence;
4. `fpm_m_cache` acts like a runtime working copy, not like the formal persistent owner.

These are all compatible with the intended causal design.

---

## What is not clean yet

### 1. `H_hist` overlaps with FPM semantics

This is the biggest storage-side ambiguity.

If FPM already has:
- persistent owner (`MState`)
- token-local history (`HistCtx`)
- runtime working cache (`fpm_m_cache`)

then `H_hist` should not remain a hidden second persistence path unless we can formally justify it.

### 2. Persistence logic is still harder to reason about than necessary

Today, the persistence path involves:
- `MState` seeding
- `fpm_m_cache` runtime evolution
- `HistCtx` per-token writeback
- optional `H_hist` blending
- last-token copy `HistCtx -> MState`

That is too many semantic layers for a storage model that should be simple.

### 3. The implementation still reflects historical layering

The current storage design looks like it evolved by accretion:
- legacy hidden-state memory
- token history
- later FPM additions

Rather than from a single storage contract.

---

## Recommended storage contract for Model A

If we rewrite storage semantics cleanly for Model A, the target should be:

### Persistent owner
- `MState`

### Token-local history
- `HistCtx`

### Runtime working state
- `fpm_m_cache`

### Outside FPM memory semantics
- `H_hist`

Under this contract:

1. `MState` is the only long-term owner;
2. `HistCtx` stores token-local snapshots for causal intra-chunk reads;
3. `fpm_m_cache` is only the execution-time working copy;
4. `H_hist` either belongs to a different mechanism or should be removed from FPM flow.

---

## Concrete next audit

The next correct audit is no longer “what does each buffer mean?”
That part is now much clearer.

The next audit should be:

1. verify the exact causal snapshot rules under this storage contract;
2. verify whether the current read path really respects:
   - `HistCtx[t-1]` within chunk
   - `MState` only as first-token fallback
3. check whether `H_hist` is actually needed for Model A memory at all.

That is the next place where code and concept need to be aligned.
