# Aideen AI — Architecture Decisions

## ADR-001: GenerationStrategy — How to generate text from H*

**Status:** UNDER EVALUATION — no strategy definitively selected.
**Context:** H* is the DEQ fixed point (8 slots × D_R). We need to decide how to convert it into a token sequence. There are 3 viable alternatives.

---

### Strategy A: SlotDirect

```
H* (8 slots × D_R)
   ↓
Each slot → LmHead → 1 token
   ↓
Sequence of up to K=8 tokens
```

- Simplest, no extra layers
- K slots already hold DEQ state — no redundancy
- Minimal latency (one matrix multiply)
- Maximum K tokens per response
- Slots are "parallel reasoners", not sequential positions

**Wins when:** The DEQ learns to specialize slots by token position.

---

### Strategy B: Decoder (current)

```
H* ──FiLM──► scale/bias per layer
<bos> → Mamba layer 0..N → LmHead → tokens
```

- Arbitrary-length sequences
- H* conditions each layer via FiLM (semantic guidance)
- Decoder has its own memory of generated text
- Extra layers = more parameters, more latency
- Possible redundancy with the DEQ

**Wins when:** The query requires long, coherent responses.

---

### Strategy C: DeqAutoReg

```
query_0 → DEQ → H*_0 → LmHead → token_0
query_1 = query_0 + token_0 → DEQ → H*_1 → LmHead → token_1
query_2 = query_1 + token_1 → DEQ → H*_2 → LmHead → token_2
...
```

- Autoregressive at the DEQ level — each token is the result of full reasoning
- No extra decoder — reuses the existing DEQ
- Maximum coherence: each token "understands" all previous ones
- Cost = max_iters × tokens_to_generate (expensive without GPU)
- DEQ must converge for each token

**Wins when:** The DEQ is fast on GPU and responses require deep coherence.

---

### Evaluation plan

All strategies are implemented in `aideen-backbone/src/generation_strategy.rs`.
The benchmark `cargo test -p aideen-backbone benchmark_all_three_strategies -- --nocapture` compares:

1. Latency per token
2. Token diversity (no repetitive degeneration)
3. Input sensitivity (different tokens for different queries)

**The winning strategy will be decided with trained weights, not random weights.**

---

### Results with random weights, D_R=1024 (historical)

```
SlotDirect   diversity=1.00  elapsed=24ms
Decoder      diversity=0.38  elapsed=1905ms
DeqAutoReg   diversity=0.12  elapsed=31903ms
```

### Results with random weights, D_R=256 + Picard β=0.9

```
SlotDirect   diversity=1.00  elapsed=6ms       ← 4x faster
Decoder      diversity=0.12  elapsed=133ms     ← 14x faster
DeqAutoReg   diversity=0.12  elapsed=3662ms    ← 8.7x faster
Winner by diversity: SlotDirect
```

**Note:** Decoder and DeqAutoReg now collapse equally (diversity=0.12).
This is due to Picard β smoothing: β=0.9 smooths H* and reduces trivial
diversity with random weights. With trained weights, the DEQ will generate
genuinely distinct H* per query.

**Hypotheses to validate with training:**
- SlotDirect: Do slots learn distinct semantic positions?
- Decoder: Does FiLM conditioning via H* reduce degeneration?
- DeqAutoReg: Does each token generate a genuinely different H* with trained weights?

---

## ADR-002: D_R dimension

**Status:** PENDING

- Mobile: D_R = 256
- Desktop: D_R = 512
- Cloud/training: D_R = 1024

Change in `aideen-core/src/state.rs`.

---

## ADR-003: Spectral Normalization in MambaSlotReasoning

**Status:** IMPLEMENTED

Spectral norm enforcement (σ ≤ 0.10 every 4 steps) guarantees DEQ convergence with trained weights. Without spectral norm, the DEQ may diverge or oscillate with non-trivial weights. Implementation: `spectral_renorm.wgsl` shader with power iteration method.
