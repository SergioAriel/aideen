# Memory Reference for AIDEEN

## Purpose of this document

This document is the reference specification for the memory system we want in AIDEEN.

It is intentionally written as a design and audit parameter, not as a description of whatever the current code happens to do.

The current implementation may contain useful ideas, partial structures, legacy carryovers, or incorrect placements of logic. None of those are authoritative by themselves. The authority for future design and refactoring should be the conceptual model described here.

This document exists to answer four questions clearly:

1. What problem is memory supposed to solve?
2. What kind of memory do we want conceptually?
3. What must remain fixed during inference, and what is allowed to evolve?
4. Which invariants must hold so memory does not break DEQ stability, causal correctness, or deployment practicality?

---

## Executive summary

The intended memory for AIDEEN is:

- a **dynamic execution-state memory**, not online weight training during inference;
- organized **per slot**;
- read **causally** from past state only;
- injected into the token solve as **residual context**, not as a second primary signal source;
- updated **after** the token solve, not during it;
- persistent across tokens and optionally across longer horizons;
- compatible with **fixed backbone weights** during inference;
- compatible with a **KV-cache/TurboQuant-style runtime state** that evolves while parameters remain frozen.

In one sentence:

> AIDEEN memory should behave like learned, selective, slot-wise adaptive state, while keeping the model weights fixed during inference.

---

## 1. What problem memory is supposed to solve

AIDEEN already has a token-local solve (`H`) and slot coordination (`slot_coord`). Memory is not meant to replace those components.

Memory is meant to solve a different problem:

- the model should be able to accumulate and reuse information across tokens;
- that accumulated information should influence the solve of the next tokens;
- the influence should be selective and structured, not a blind carry;
- this should happen **without mutating the backbone weights during inference**.

### What memory is *not* supposed to be

Memory is not supposed to be:

- a disguised form of online finetuning;
- a hidden way of mutating `W_q`, `W_k`, `W_v`, `W_o`, `W_in`, etc. during inference;
- an uncontrolled recurrent feedback loop that contaminates the DEQ fixed-point search;
- a second model stuffed into buffers without a clear causal role.

### Why memory exists even if KV cache exists

Standard KV cache stores short-horizon inference traces for attention reuse. That is useful, but conceptually narrow.

The intended AIDEEN memory is broader:

- it can preserve slot-specific latent traces;
- it can carry structured information that is not identical to standard transformer KV cache;
- it can be quantized or runtime-managed similarly to KV/TurboQuant systems;
- but its semantics are learned as a dedicated memory mechanism, not only as inference acceleration.

---

## 2. Core conceptual distinction: weights vs memory state

This distinction must stay explicit at all times.

### 2.1 Fixed weights

The backbone parameters represent learned knowledge of the model:

- language knowledge;
- structural priors;
- read/write policies;
- slot interaction rules;
- memory usage rules.

These weights:

- are trained offline;
- may be updated during normal training by gradient descent;
- should remain **fixed during inference**.

### 2.2 Dynamic memory state

The memory state represents runtime experience:

- what has happened recently;
- what each slot retained;
- what should be reused by the next tokens;
- what is worth carrying across chunk boundaries or sessions.

This state:

- evolves during training;
- also evolves during inference;
- is the correct place for online adaptation;
- is allowed to be cached, compressed, quantized, truncated, reset, or persisted.

### 2.3 Why this distinction matters

If the memory mechanism requires modifying the backbone weights during inference, then we are no longer building runtime memory state; we are building online plastic parameter updates.

That can be interesting research, but it is not the current target.

The intended target is:

- **plastic behavior** in runtime,
- **fixed parameters** in inference,
- **dynamic state** as the adaptation carrier.

---

## 3. Neuroplasticity interpretation

The memory idea is inspired by neuroplasticity, but it should not be interpreted naively.

### 3.1 What we want from the neuroplasticity analogy

We want:

- adaptation based on experience;
- gradual accumulation of useful traces;
- selective strengthening or preservation;
- state-dependent future processing.

### 3.2 What we do *not* want literally

We do **not** currently want:

- literal online mutation of all backbone parameters during inference;
- unconstrained self-modification of the model function;
- inference behavior that depends on rewriting the network weights themselves.

### 3.3 Working interpretation

For AIDEEN, “neuroplasticity-like memory” should mean:

- the model learns **how to read and write memory**;
- the runtime system carries **memory state** between tokens;
- that state changes the effective behavior of inference;
- but the backbone parameters remain fixed.

This is a pragmatic and deployable version of neuroplasticity:

- plastic in behavior,
- not plastic in frozen weights.

---

## 4. Minimal semantic model of memory

The minimal concept should be this.

### 4.1 There is a persistent memory state per slot

For each slot `k`, there exists a persistent memory state:

`m_k[t]`

This state should mean:

- what slot `k` currently retains from the past,
- before token `t` is solved.

### 4.2 The token reads only past memory

For token `t`, the model must read only a frozen snapshot from the past:

`read_t = Read(M_{t-1})`

The token must not read a memory that is simultaneously being modified by its own solve.

### 4.3 The solve uses memory as context

The token-local DEQ solve should be conceptually:

`H_t = solve(H ; signal_t, slot_ctx_t, read_t)`

Memory is an input/context term to the solve.

It is **not** a free-changing variable inside the same Picard fixed point unless the entire joint system is explicitly and correctly formulated as such.

For the intended current design, the safer and more interpretable principle is:

- memory read is frozen during the token solve;
- memory write happens after the solve.

### 4.4 After the solve, memory is updated

Once `H_t` is available, memory may be updated:

`M_t = Update(M_{t-1}, H_t)`

The update should be:

- selective,
- bounded,
- slot-aware,
- and causal.

### 4.5 Output may depend on both solved state and updated memory

Conceptually, output may be formed from:

- `H_t`,
- optionally `M_t`,
- or some pooled representation derived from them.

But this is secondary to the core invariant:

- memory read must be causally prior to the solve,
- memory write must be causally posterior to the solve.

---

## 5. Timescales

The design only makes sense if the timescales are explicit.

### 5.1 Within one Picard solve

Inside the solve of token `t`:

- backbone weights are constant;
- memory read is constant;
- the operator being iterated must be stable and well-defined.

This means:

- no hidden mutation of the read snapshot while iterating;
- no write path feeding back into the same fixed-point search unless explicitly modeled and re-validated mathematically.

### 5.2 Between tokens in the same chunk

Between token `t-1` and token `t`:

- memory state may evolve;
- token `t` may read the memory produced by `t-1`;
- this is the main intended causal memory path.

### 5.3 Between chunks

A compact persistent state may be carried across chunk boundaries.

That state should represent:

- the best current memory summary at the end of the previous chunk,
- not arbitrary scratch buffers.

### 5.4 During inference sessions

A runtime memory cache may persist across long sequences or sessions if desired.

This is where a TurboQuant-like runtime store becomes relevant.

### 5.5 During training

Weights are updated by gradient descent between steps.

Memory state may still evolve token-to-token inside the forward process.

The crucial rule is that training updates to model weights and runtime updates to memory state must remain conceptually distinct.

---

## 6. What the memory should store semantically

This is one of the most important unresolved design questions, so the acceptable possibilities should be explicit.

The memory state should store one of the following, or a clearly defined combination:

### Option A. Latent state trace

Memory stores a filtered trace of solved slot state:

- some function of `H_t`,
- preserving useful latent information for later reads.

### Option B. Context summary

Memory stores a dedicated summary representation:

- not the raw `H_t`,
- but a transformed state optimized for future retrieval.

### Option C. Write-proposal state

Memory stores something already shaped by a learned write map:

- a representation explicitly built to be persistent and queryable.

### Constraint on all options

Whatever the stored object is, it must satisfy:

- it can be read causally by future tokens;
- it can be updated incrementally;
- it is stable enough to persist;
- it is representable in inference runtime storage;
- it does not require mutating backbone weights to exist.

---

## 7. Read path: what it should mean

The read path should answer:

> Given the current token/slot state, which past slot memories matter now, and what information should be injected into the current solve?

### 7.1 Content-based read is conceptually correct

A content-based read over slot memories is reasonable:

- current slot state produces a query;
- memory slots produce keys;
- attention-like matching selects relevant memories;
- values produce a memory context vector.

This is conceptually aligned with the goal.

### 7.2 The read should be residual, not dominant

The memory read should enter the solve as a contextual branch.

Therefore:

- it should not automatically be forced to full signal scale;
- it should not dominate the token signal unless the learned mechanism justifies it;
- it should behave like an auxiliary informative source, not a second main input stream by default.

### 7.3 The read must be based on a frozen snapshot

Within token `t`:

- query can depend on current token state;
- keys/values must come from memory frozen before the solve;
- the read result should be stable across Picard iterations unless the design explicitly justifies otherwise.

---

## 8. Write path: what it should mean

The write path should answer:

> What should become part of future memory after the current token has been solved?

### 8.1 Write must happen after solve

The current token should not change its own past-memory snapshot while its solve is still being computed.

The canonical order is:

1. read frozen memory
2. solve token state
3. compute write proposal
4. update memory
5. expose updated memory to future tokens

### 8.2 Write must be selective

A useful write path should include:

- proposal content,
- write magnitude / permission,
- retention / forgetting,
- bounded update budget.

### 8.3 Write should not start nearly frozen

If the retain prior or gate initialization makes writing almost impossible from step 0, then the memory mechanism is present in name only.

A healthy prior should:

- favor preservation,
- but still leave real dynamic headroom.

---

## 9. Retain / forget semantics

Retain is conceptually valid.

It should mean:

- how much of previous memory is preserved,
- before new information is mixed in.

### Good retain behavior

Good retain behavior is:

- slot-dependent,
- input-dependent,
- not globally frozen,
- not saturating by default,
- compatible with write budget.

### Bad retain behavior

Bad retain behavior is:

- identical across all slots by construction,
- near-constant regardless of input,
- so strong that writes have no practical effect,
- or so weak that memory becomes unstable overwrite.

---

## 10. Inference constraints

This section is critical.

### 10.1 Inference must keep weights fixed

During inference:

- no online parameter mutation of the backbone;
- no hidden update of model matrices as a substitute for memory.

### 10.2 Inference memory is allowed to evolve

During inference, the following is allowed and intended:

- a memory cache changes over time;
- slot memory state persists across tokens;
- that state may be truncated, reset, serialized, quantized, or evicted.

### 10.3 Compatibility with TurboQuant/KV-like runtime caches

The runtime memory should be representable in a storage form analogous to a KV cache:

- external to the frozen weights,
- efficient to store and update,
- possibly quantized,
- portable across inference steps.

This is not a compromise. It is exactly the right place for online adaptation state.

---

## 11. Training constraints

Training is where the model learns how to use memory.

### 11.1 What training should learn

Training should learn:

- how to form read queries;
- how to represent memory keys/values;
- how to decide write amount;
- how to retain or forget;
- how to use memory context inside the solve.

### 11.2 What training should not rely on

Training should not rely on:

- online mutation of frozen inference weights as the core memory carrier;
- heuristics that only work because training code can intervene from outside the architecture;
- hidden coupling between runtime memory and parameter updates.

### 11.3 Training/inference alignment

A memory system is only well-designed if the mechanism used in training is the same conceptual mechanism that will exist in inference.

That means:

- if inference will use runtime memory state and frozen weights,
- then training should optimize a mechanism compatible with exactly that regime.

---

## 12. What absolutely must remain true (invariants)

These are the non-negotiable design invariants.

### 12.1 Causality

Token `t` can read only memory finalized before token `t` begins.

### 12.2 Separation of solve and write

The read snapshot must be fixed during the token solve.

### 12.3 Fixed inference parameters

Memory must not require backbone weight updates during inference.

### 12.4 Slot semantics

Memory is per-slot. If slots are used as carriers, then read/write must preserve slot structure and not silently collapse into a global undifferentiated state.

### 12.5 Stable entry into solve

Memory context must enter as a bounded contextual term, not as an uncontrolled dominant source.

### 12.6 Auditability

At any point, it must be possible to identify:

- what is persistent memory,
- what is token-local history,
- what is just staging/scratch.

If those roles are blurred, the design is not clean enough.

---

## 13. How to classify storage roles

To avoid confusion in implementation, every buffer used by memory should belong to exactly one semantic class.

### Class A. Persistent memory owner

A buffer that represents the authoritative memory state to be carried forward.

Properties:

- persists beyond a single token;
- may persist across chunk boundaries;
- is the canonical source for future reads when no fresher token-local snapshot exists.

### Class B. Token-local history

A buffer that stores per-token memory results inside the active chunk.

Properties:

- causal within the chunk;
- future tokens in the same chunk may read from it;
- not necessarily the global long-term owner.

### Class C. Staging / scratch / cache

A buffer used only to compute reads, writes, or transitions.

Properties:

- not semantically part of memory by itself;
- should never silently become the authoritative meaning of memory.

This classification should be enforced in code review.

---

## 14. Evaluation questions for the current code

Any current or future implementation of memory should be audited with these questions.

### Storage questions

1. Which buffer is the persistent owner?
2. Which buffer is only token-local history?
3. Which buffer is staging only?
4. Is any buffer playing more than one semantic role? If yes, why?

### Read questions

1. What exact object is being read?
2. Is that object causally prior to the token solve?
3. Is the read content-based for a good reason, or just inherited implementation?
4. Does the read enter as residual context or as dominant signal?

### Solve questions

1. Is memory frozen during Picard for the current token?
2. Does memory improve solve quality without destroying contractivity?
3. Are invalids caused by the memory signal itself, or by controller logic around it?

### Write questions

1. Is write computed only from post-solve information?
2. Is the write budget meaningful at initialization?
3. Is retain selective or effectively constant?
4. Does write improve future tokens causally, or only create noise?

### Inference questions

1. Can the full mechanism run with frozen weights?
2. Can the memory state be represented as a cache?
3. Can the state be quantized or compressed without changing semantics too much?

---

## 15. Current implementation should be treated as hypothesis, not truth

This is the most important procedural rule.

The current implementation may contain:

- correct concepts,
- useful partial structures,
- wrong semantic ownership,
- legacy overlaps,
- equations that need relocation or rewriting,
- buffers that do not correspond cleanly to the intended design.

Therefore the correct workflow is:

1. start from the conceptual memory model;
2. inspect each code path against that model;
3. keep only what can be justified;
4. rewrite or remove what cannot.

---

## 16. Working design statement

The current best design statement for AIDEEN memory is:

> AIDEEN should have a slot-wise dynamic memory state that is read causally from the past, enters the token solve as residual context, is updated only after the solve, persists across tokens through explicit runtime state, and remains fully compatible with frozen model weights during inference.

Everything in the implementation should be judged against that statement.

---

## 17. What this document does not decide yet

This document intentionally does **not** fully decide:

- the exact stored representation of memory;
- the exact read equation;
- the final write equation;
- the final choice of persistent buffer layout;
- whether future AIDEEN versions should add true effective-weight plasticity.

Those remain open design questions.

What this document does decide is the conceptual frame within which those choices must be made.

---

## 18. Practical consequence for ongoing implementation work

For the ongoing memory implementation, the correct sequence is:

1. clarify storage semantics first;
2. define the persistent owner vs token-local history vs staging;
3. validate causal snapshot rules;
4. validate read path;
5. validate solve injection;
6. validate write;
7. validate persistence;
8. only then optimize or expand functionality.

That sequence should take precedence over convenience or inheritance from current code structure.
