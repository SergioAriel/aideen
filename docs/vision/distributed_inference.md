# AIDEEN Distributed Inference

Status: design draft, updated after the current DEQ/Assoc/FPM token-circuit audit.

This document defines the intended distributed-inference architecture for AIDEEN. It is intentionally separate from federated training; training across user machines is covered in [distributed_training_users.md](/Users/sergiosolis/Programacion/AIDEEN/docs/distributed_training_users.md).

## Core Position

Distributed inference should not make remote experts replace the local model.

The preferred contract is:

```text
remote experts provide evidence, bindings, context, or memory patches
local AIDEEN integrates them into its own state/memory
local LMHead decodes from the refined local state
```

Therefore, expert consultation should happen before the final LMHead decode, not after the LMHead has already committed to logits.

## Current Local Circuit

The current training/inference-relevant model path is:

```text
tokens -> embeddings/S_in
       -> DEQ/FPM/Assoc forward over the known sequence
       -> h_t / h_pooled_t / local memory state
       -> LMHead over h_pooled for logits/loss
```

Important clarification:

```text
known prompt tokens do not require LMHead to advance to the next prompt token
```

The serial dependency is inside the DEQ/FPM/Assoc state update, because token `t` can depend on memory/history from token `t-1`.

For a known prompt/document, the model can process all tokens through DEQ first and run the LMHead only when logits are needed.

For autoregressive generation, LMHead is required for every generated token because the next token is unknown until decoded:

```text
state_t -> LMHead -> sample/select token_{t+1} -> feed token_{t+1} back into DEQ
```

## Distributed Inference Loop

The intended high-level loop is:

```text
1. Local prefill
   prompt/document tokens -> local DEQ/FPM/Assoc

2. Routing decision
   inspect uncertainty, memory misses, task/domain, confidence, and local state

3. P2P expert query
   send compact query package to selected peers

4. Expert response
   receive memory/context/evidence patches, not final text as the main contract

5. Integration
   validate, score, and gate expert patches

6. Refinement
   run a local DEQ refinement pass using local state + accepted expert patches

7. Decode
   h_refined -> LMHead -> logits/token/answer
```

In compact form:

```text
prompt -> local state/memory -> router -> P2P experts -> memory patches
       -> local DEQ refinement -> LMHead -> output
```

## What Is Refinement?

Refinement means running an additional local forward/inference step after expert information has been integrated.

It is not full training and it is not necessarily a full document replay.

Possible refinement levels:

### 1. Final-state refinement

Cheapest option:

```text
h_final + local FPM/Assoc + expert patch -> DEQ refinement -> h_refined
h_refined -> LMHead
```

Use when expert information can be treated as additional memory/context for the current final state.

### 2. Last-K-token replay

More faithful, more expensive:

```text
last K tokens + local memory + expert patch -> DEQ forward -> h_refined
h_refined -> LMHead
```

Use when the expert patch changes interpretation of recent context.

Candidate K values for experiments:

```text
K = 16, 32, 64
```

### 3. Memory-only injection, no DEQ rerun

Simplest but weaker:

```text
expert patch -> memory buffers
old h_final -> LMHead
```

This is only useful if the LMHead or final read path can actually observe the injected memory. In the current architecture, a mini-DEQ refinement is usually the cleaner path because it lets the accepted expert evidence affect `h_refined` before logits are computed.

## What To Send To Experts

Do not send the full prompt by default.

Prefer compact packages:

```text
QueryState {
  task_id,
  domain_hint,
  h_query or selected h slots,
  uncertainty metrics,
  local assoc misses / low-confidence reads,
  compact FPM context summary,
  optional text excerpt if policy allows,
  segment/document fingerprint,
  time_budget_ms,
  requested_response_type
}
```

Possible payloads:

- `h_query`: final or selected slot state.
- `assoc_miss_summary`: keys/queries where local Assoc had low confidence.
- `fpm_context_summary`: compact context vector or low-rank summary.
- `domain_hint`: code, math, science, memory lookup, consistency check, etc.
- `segment_fingerprint`: hash/fingerprint for cache or peer-side retrieval.
- `privacy_policy`: whether raw text snippets are allowed.

## What Experts Should Return

Preferred expert output is structured evidence, not final prose:

```text
ExpertPatch {
  task_id,
  peer_id,
  patch_type,
  bindings,
  context_delta,
  evidence_refs,
  confidence,
  compatibility_score,
  patch_norm,
  ttl,
  signature
}
```

Patch types:

- `AssocBindings`: explicit key/value bindings relevant to the current query.
- `FpmContext`: contextual vector/delta for graded memory.
- `Evidence`: text/document references or compact excerpts when allowed.
- `Critique`: contradiction, uncertainty, or correction signal.
- `DomainDelta`: expert-domain state delta.

The local node remains responsible for deciding whether and how to use the patch.

## Integration Policy

Remote patches must be gated before affecting local state.

Minimum gates:

```text
accepted_weight = trust(peer)
                * confidence
                * compatibility(local_state, patch)
                * freshness
                * budget_gate
```

A patch should be rejected or downweighted when:

- peer signature/trust is invalid,
- patch norm is an outlier,
- confidence is low,
- compatibility with local state is low,
- it conflicts with stronger local evidence,
- it exceeds the inference time budget,
- it violates privacy or policy constraints.

Do not let experts directly dominate `h` or durable memory.

Use bounded integration:

```text
h_candidate = h_local + beta * accepted_delta
beta <= configured cap
```

For memory patches, prefer staging first:

```text
expert patch -> candidate memory
candidate memory -> gate/promotion -> local Assoc/FPM
```

## Local vs Durable Memory

The distributed design should preserve the distinction we need for local training as well:

```text
local candidates: many possible transient bindings/context traces
durable memory: small set of high-evidence promoted bindings/context
```

This matters because the current real-text audit showed that ordinary text can fill all Assoc banks quickly if every moderate transition becomes durable memory.

Distributed inference should therefore avoid treating every expert response as durable truth.

Recommended memory flow:

```text
expert patch
  -> transient candidate buffer
  -> score against local state and uncertainty
  -> optional DEQ refinement
  -> promote only if useful/confident
```

## Router Triggers

The router should consult experts only when useful.

Candidate triggers:

- LMHead margin would be low if decoded now.
- Assoc read confidence is low.
- Assoc query misses or retrieves conflicting bindings.
- FPM confidence is low or entropy is high.
- Slot disagreement is high.
- Domain classifier detects a specialized domain.
- User explicitly requests external/distributed help.
- Local context is insufficient for required factual recall.

Do not query experts on every token by default.

Recommended first policy:

```text
prefill query: allowed
per generated token query: only on uncertainty or explicit need
max expert hops per answer: small bounded number
```

## Timing Modes

### Prefill-time distributed enrichment

Best default for long prompts/documents:

```text
process prompt locally -> detect need -> query experts -> refine -> decode
```

Benefits:

- LMHead can run only after local/expert context is integrated.
- Network latency is paid before generation starts.
- Expert work can happen in parallel.

### Decode-time expert check

Use sparingly during generation:

```text
h_t -> uncertainty high -> query experts -> refine h_t -> LMHead -> token
```

This is more expensive because it can stall token generation.

### Background expert enrichment

For long sessions:

```text
local generation continues
background peers enrich memory/cache
future turns benefit
```

This is useful when latency is high and immediate correctness is not critical.

## Relation To Existing `aideen-node` Code

Current code in [aideen-node/src/inference.rs](/Users/sergiosolis/Programacion/AIDEEN/aideen-node/src/inference.rs) implements an older concept:

```text
inside DEQ iteration k:
  query ExpertPipeline using slot 0
  receive delta
  inject delta into all slots
  continue DEQ loop
```

That path is useful as an MVP proof of expert deltas, but it is not the final desired architecture.

Problems with the old approach:

- It queries during DEQ iterations rather than after local prefill/state assessment.
- It assumes expert output is a direct `delta` to inject into every slot.
- It does not distinguish evidence, bindings, context, and final text.
- It does not stage or promote remote memory patches.
- It risks letting remote deltas perturb convergence instead of refining a stable local state.

Recommended evolution:

```text
ExpertResult(delta)       -> ExpertPatch(structured)
iteration injection       -> post-prefill / uncertainty-triggered refinement
slot-0-only query         -> router over h_final, Assoc/FPM metrics, domain hints
blind delta injection     -> gated integration + bounded DEQ refinement
```

## Relation To Protocol v1

[protocol_v1.md](/Users/sergiosolis/Programacion/AIDEEN/docs/protocol_v1.md) currently defines:

```text
ExpertTask { s_r, target_id, time_budget_ms, ... }
ExpertResult { delta, q_total, iters, stop, ... }
```

This can support the MVP, but the distributed-inference design needs a richer future message family:

```text
ExpertQuery / ExpertPatch
```

Backward-compatible path:

1. Keep `ExpertTask` / `ExpertResult` for MVP delta experts.
2. Add optional structured patch fields or new enum variants in protocol v2.
3. Ensure every patch has confidence, provenance, norm, and signature metadata.

## Safety And Trust

Distributed inference is not just a performance feature. It is a trust boundary.

Required invariants:

- Remote nodes cannot directly write local weights.
- Remote nodes cannot directly force final logits.
- Remote patches must be signed/provenanced.
- Local node can reject, downweight, or ignore any patch.
- Patches should be bounded by norm and time budget.
- Privacy policy decides whether raw text may leave the node.

## Interaction With Federated Training

Federated training and distributed inference are related but not identical.

Federated training:

```text
nodes train on documents -> upload weight/gradient deltas -> coordinator aggregates
```

Distributed inference:

```text
nodes answer expert queries -> return evidence/memory/context patches -> local node refines state
```

The shared lesson is selectivity.

If every node writes or returns noisy local transitions, aggregation/integration becomes unstable. Both systems need bounded contribution:

```text
training: bounded local rounds + accepted deltas
inference: bounded expert patches + local gating/refinement
```

## Minimal Implementation Plan

### Phase 0: Documentation and alignment

- Treat this document as the target design.
- Mark the current in-iteration expert delta path as MVP/legacy experimental.
- Keep protocol v1 stable until protocol v2 is explicitly designed.

### Phase 1: State-only prefill/decode split

Add explicit APIs:

```text
forward_state_only(prompt_tokens) -> LocalInferenceState
lmhead_decode(state) -> logits
```

This makes it possible to process known context without calling LMHead until needed.

### Phase 2: Expert router inputs

Expose router metrics from local state:

- h_final / selected slots,
- Assoc read confidence,
- Assoc miss/conflict summary,
- FPM uncertainty/confidence,
- LMHead margin when available,
- domain hints.

### Phase 3: Structured expert patches

Define internal structs before changing wire protocol:

```text
ExpertQuery
ExpertPatch
PatchScore
PatchIntegrationResult
```

Initially they can be local/in-process and mapped to protocol v1 deltas only where needed.

### Phase 4: Mini-refinement path

Implement:

```text
apply_expert_patch_to_candidate_memory(...)
run_final_state_refinement(...)
decode_from_refined_state(...)
```

Start with final-state refinement before last-K replay.

### Phase 5: P2P protocol v2

Extend protocol after local semantics are validated.

New or extended messages should include:

- query state metadata,
- structured bindings/context/evidence,
- confidence,
- provenance/signature,
- time budget,
- patch norm and compatibility metadata.

## Open Questions

- What is the exact representation of an expert `AssocBinding` across nodes?
- Should remote context be injected into FPM, Assoc, or a separate candidate buffer first?
- Which uncertainty metric best predicts when expert help improves answer quality?
- How many expert patches can be integrated before refinement becomes noisy?
- Should last-K replay become default for complex prompts, or only for high-stakes answers?
- How do we cache expert patches across a session without turning them into untrusted durable memory?

## Current Recommendation

Do not build distributed inference as remote text generation first.

Build it as:

```text
remote memory/context/evidence -> local gated integration -> DEQ refinement -> local LMHead
```

This preserves AIDEEN's model philosophy:

- local fixed-point reasoning remains sovereign,
- experts provide bounded evidence,
- memory stays explicit and inspectable,
- final logits remain a local decision.
