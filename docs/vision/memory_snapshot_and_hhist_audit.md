# Memory Snapshot and H_hist Audit

## Purpose

This document audits two very specific questions for **Model A** memory:

1. Does the current code respect the intended causal snapshot rule?
2. Is `H_hist` actually necessary for Model A memory semantics?

This document extends:
- [memory_reference.md](/Users/sergiosolis/Programacion/AIDEEN/docs/vision/memory_reference.md)
- [fixed_point_memory_models.md](/Users/sergiosolis/Programacion/AIDEEN/docs/vision/fixed_point_memory_models.md)
- [memory_storage_semantics_audit.md](/Users/sergiosolis/Programacion/AIDEEN/docs/vision/memory_storage_semantics_audit.md)

---

## 1. Model A causal snapshot rule

Under Model A, the token solve must follow this principle:

- token `t` may read only memory finalized before token `t` begins;
- inside the chunk, token `t` should read the previous token snapshot `t-1`;
- only the first token of the active range may fall back to the persistent owner;
- memory write must not feed back into the same token solve.

---

## 2. What the current shader does

Relevant code:
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:457](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:457)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:458](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:458)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:557](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:557)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:762](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:762)

The current logic is:

- `use_prev_token_mem = global_t > 0`
- if true, read from `HistCtx[(t-1)]`
- if false, read from `MState`

This means:

- token 0 in the active range reads `MState`
- token `t>0` reads `HistCtx[t-1]`

### Audit result

This is conceptually correct for Model A.

It matches the intended causal rule:

- intra-chunk causal read comes from previous token history;
- inter-chunk / first-token fallback comes from the persistent owner.

### Important nuance

This conclusion is about **read-source semantics only**.
It does not yet validate:

- read scaling,
- query/key equations,
- write correctness,
- or stability.

It only says the snapshot source rule itself is sound.

---

## 3. Does write leak into the same token solve?

For Model A, the token must not observe its own memory write before the solve is finished.

Relevant code:
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:1299](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:1299)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:1304](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:1304)

Current logic:

- `fpm_m_cache` is updated after the solve section;
- `HistCtx[h_base_t + ...] = working` only in the persist block;
- future tokens may read that token-local write, but the current token does not re-read it during its own solve.

### Audit result

This is also directionally correct for Model A.

The current code does **not** look like a same-token write-to-read leak by default.

---

## 4. What `H_hist` is doing today in FPM

Relevant code:
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:530](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:530)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:1306](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl:1306)
- [/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/deq_bridge.rs:1407](/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/deq_bridge.rs:1407)

In FPM mode, `H_hist` is not used as the read source for memory lookup.

Instead, it is used as:

1. a seed source for `MState` on the first chunk;
2. a persistence/blend target after the token write;
3. a slow carry associated with the old hidden-state memory path.

### Key observation

`H_hist` is **not** part of the causal read rule itself.

The read rule is already satisfied by:
- `MState`
- `HistCtx`

Therefore `H_hist` is not needed to explain causal memory access under Model A.

---

## 5. Is `H_hist` necessary for Model A?

### Short answer

Not obviously.

### Longer answer

For Model A, the minimal memory semantics only require:

1. a persistent owner (`MState`)
2. token-local history (`HistCtx`)
3. working state (`fpm_m_cache`)

`H_hist` is not required by that minimal contract.

Its current use is secondary:

- seed convenience,
- blended persistence,
- compatibility with an older hidden-state mechanism.

### Interpretation

That means `H_hist` currently behaves more like:

- a transitional persistence helper,
- or a legacy overlap,

than like a conceptually necessary piece of Model A memory.

---

## 6. Practical implication

This is the most important conclusion of this audit:

> The current Model A memory path already has enough buffers to be well-defined without `H_hist`.

That means the right next question is not:

- “what does `H_hist` mean?”

The right next question is:

- “should `H_hist` stay outside FPM Model A entirely?”

That is a much better question because it treats `H_hist` as an optional overlap to justify, not as a presumed requirement.

---

## 7. Audit conclusion

### Confirmed

1. the current snapshot read rule is causally correct for Model A;
2. the first-token fallback / previous-token history split is conceptually sound;
3. same-token write-to-read leakage does not appear to be the primary issue;
4. `H_hist` is not required to explain the memory read semantics.

### Open

1. whether `H_hist` should remain in FPM Model A at all;
2. whether the read/inject equations are correctly scaled;
3. whether write/persist should depend only on `MState + HistCtx` instead of also involving `H_hist`.

---

## 8. Recommended next step

The next correct step is:

1. treat `H_hist` as non-authoritative for Model A memory;
2. audit whether FPM Model A can be specified cleanly with only:
   - `MState`
   - `HistCtx`
   - `fpm_m_cache`
3. only after that, revisit whether any slow blended persistence is still needed.

That is the next point where concept and implementation need to be aligned.
