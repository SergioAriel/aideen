# LOXI-AI Architecture & Skills Reference

**Last Updated:** 2026-02-25  
**Scope:** Unified AI system with delta-based expert routing  
**Critical:** The mathematical foundation MUST be preserved

---

## Core Philosophy

```
Input Tokens
    ↓
Embedding (vocab=64K)
    ↓
[Mamba + Attention2D Hybrid Backbone] ← Efficient, tokens stay fresh
    ↓
Pooling → S_l (local state, 4096D)
    ↓
Project → S_g (global state, 2048D) ← Travels on network
    ↓
[DELTA REFINEMENT LOOP] ← THE HEART
├─ Semantic Router: which experts?
├─ Await expert deltas: Δᵢ from each
├─ Integrate: S_g += tanh(α·Σ wᵢ·Δᵢ) ← CRITICAL MATH
├─ Cap: wᵢ.min(0.15) "socialist constraint"
├─ Check convergence: ||ΔS|| > ε → continue iteration
└─ Reintegrate: S_l += E(ΔS_g)
    ↓
Output logits (vocab=64K)
```

**Why this works:**
- Mamba: O(seq_len) memory, not O(seq²)
- Attention2D: Sparse (Dim16 + Dim64), hybrid cognition
- Deltas: External experts inject knowledge without retraining backbone
- Refinement: Iterative convergence with math guarantees (tanh stability)

---

## Architecture Tiers

### Tier 1: Server (Desktop/M1 Pro+)
```
d_model: 4096
n_layers: 12 total
├─ Mamba blocks: 0-5 (6 blocks)
├─ Attention2D: blocks 6-7
└─ Mamba blocks: 8-11 (4 blocks)

d_state (Mamba): 16
Memory footprint: ~8GB activations
Batch size: 1-8
Speed: ~100ms/token on M1 Pro
Best for: Server inference, training
```

### Tier 2: Hybrid (Laptop/Tablet)
```
d_model: 2048
n_layers: 12 (same structure as T1)
├─ Mamba: 0-5, 8-11
├─ Attention2D: 6-7
d_state: 8
Memory: ~2GB activations
Batch: 1-2
Speed: ~50ms/token
Best for: Laptop deployment, edge inference
```

### Tier 3: Mobile (iPhone/Android)
```
d_model: 512
n_layers: 8
├─ Mamba: 0-3, 5-7
├─ No Attention2D (only Mamba)
d_state: 4
Memory: ~256MB
Batch: 1
Speed: ~20ms/token
Best for: On-device, battery efficient
```

---

## Key Components

### 1. **state.rs** (Refinement Loop)
**Status:** ✅ COMPLETE & PROVEN

**Math preserved:**
- `integrate_deltas()`: tanh(α·Σ wᵢ·Δᵢ) with cap=0.15
- `has_converged()`: ||S_g_new - S_g_old|| < 1e-3
- `reintegrate_local()`: S_l += E(ΔS_g)

**Must NOT change:**
- Tanh stabilization (prevents NaN explosion)
- Cap at 0.15 (prevents expert dominance)
- Epsilon convergence check

---

### 2. **MoE Router (New)**
**Source:** loxi-compute/moe.rs → adapt to deltas

**Function:**
- Input: S_g (2048D) + routing_scores
- Output: Expert allocation tensor [batch, n_experts]
- Key: Routes to delta-producing experts, not attention gates

**Optimization:** Tensor-based routing (not probabilistic)
- Router: S_g → [router_hidden] → n_experts logits
- Softmax + Top-K selection
- Send only to K best experts (not all)

---

### 3. **Mamba + Attention2D Hybrid**
**Status:** ⚠️ NEEDS IMPLEMENTATION

**Mamba blocks:**
- SSM scan: h_{t+1} = A·h_t + B·x_t (parametric)
- O(seq_len) complexity
- Preserves token context naturally

**Attention2D streams (kernel_v8.wgsl):**
- Dim16 Pool: lightweight reasoning (16D per token)
- Dim64 Pool: heavy cognition (64D per token)
- Sparse: not O(seq_len²) full attention
- Fusion: merge both pools after attention

**Placement:** After Mamba blocks 5 & 7 to capture global context

---

### 4. **WGSL Kernels** (GPU acceleration)
**Source:** loxi-compute/src/shaders/

**Kernels to use:**
- `mamba.wgsl` → parallel scan (Hillis-Steele)
- `kernel_v8.wgsl` → two-stream attention
- `embedding.wgsl` → token lookup
- `lm_head.wgsl` → output logits
- `layernorm.wgsl` → RMSNorm

**Note:** Used for inference only (not training)

---

### 5. **Sharding / Distributed**
**Source:** loxi/protocol/crates/ai/loxi-ai/shard_manager

**For:** Multi-node inference (if scaling across machines)

**Shard types:**
- `Shared`: embedding, norm
- `LayerAttention{layer}`: per-layer weights
- `LayerExpert{layer, expert}`: MoE weights
- `FinalNorm`: output norm

---

## Trade-offs & Decisions

| What | Choice | Why | Cost |
|------|--------|-----|------|
| **Base Arch** | Mamba+Attention2D | O(n) + sparse = best memory/speed | Hybrid complexity |
| **d_model** | 4096 (T1), 2048 (T2), 512 (T3) | More dims = better token maturation | 8GB/2GB/256MB |
| **ffn_ratio** | 2 | Pools to vector, FFN expansion wasted | Smaller hidden |
| **cap** | 0.15 | "Socialist" prevents expert dominance | Slower expert influence |
| **integration_alpha** | 0.2 | Dampens delta magnitude | More stable, less aggressive |
| **routing** | Tensor-based MoE | Sparse expert selection | K experts vs all |
| **Deltas** | Per-expert tensors | External knowledge injection | Network latency |

---

## Critical Math to Preserve

### Stability (tanh)
```rust
delta_val = (alpha * weighted_sum).tanh()
// Guarantees: [-1, 1] bounded output
// Prevents: NaN, explosion, weight collapse
```

### Cap Constraint
```rust
let cap = 0.15f32;
let weights = w_raw.iter().map(|w| w.min(cap)).collect()
// Guarantees: no expert > 15% influence
// Prevents: single expert takeover
```

### Convergence Check
```rust
let delta_norm = (delta_sum_squared).sqrt()
return delta_norm > epsilon  // Keep iterating if true
// Guarantees: graceful termination
// Prevents: infinite loops, hung inference
```

---

## Performance Targets

| Metric | T1 (Server) | T2 (Hybrid) | T3 (Mobile) |
|--------|-----------|-----------|-----------|
| **Latency** | <100ms | <50ms | <20ms |
| **Memory Peak** | <8GB | <2GB | <256MB |
| **Tokens/sec** | 10+ | 20+ | 50+ |
| **Precision** | FP32 | FP32 | FP16 (optional) |

---

## Files Affected

### Core (Must implement)
- `loxi-backbone/src/backbone_mamba_attention2d.rs` **← NEW**
- `loxi-backbone/src/state.rs` **← PRESERVE MATH**
- `loxi-backbone/src/routing/moe.rs` **← FROM loxi-compute**
- `loxi-runtime/src/kernels/` **← COPY WGSL**

### Support (Refactor as needed)
- `loxi-backbone/src/types.rs` (consolidate GlobalState, CognitiveDelta)
- `loxi-backbone/src/distributed/` (optional, from shard_manager)
- `loxi-training/src/config.rs` (add new tier configs)

### No changes
- `loxi-backbone/src/cognitive_architect.rs` (semantic routing)
- `loxi-backbone/src/delta_dynamics.rs` (training dynamics)

---

## Session Continuity Checklist

When resuming work, verify:

- [ ] Is state.rs refinement loop still intact?
- [ ] Are convergence epsilon values unchanged (1e-3)?
- [ ] Is cap still 0.15 everywhere?
- [ ] Tanh used for stability, not other functions?
- [ ] MoE routing produces deltas (not weights)?
- [ ] WGSL shaders point to latest kernel_v8?
- [ ] Three tiers (4096D, 2048D, 512D) documented?

---

## Next Major Milestones

1. **[BLOCKER] Create backbone_mamba_attention2d.rs**
   - Implement MambaBlock struct
   - Implement Attention2DBlock struct
   - Wire together 12 total layers
   - Test forward pass

2. **Integrate MoE Router**
   - Load from loxi-compute/moe.rs
   - Adapt to tensor routing (not attention gates)
   - Send deltas to refinement loop

3. **WGSL Kernel Integration**
   - Copy shaders to loxi-runtime/kernels/
   - Create GPU executor for inference
   - Verify numerical correctness vs Rust

4. **Three-Tier Configuration**
   - BackboneConfig::server() → 4096D
   - BackboneConfig::hybrid() → 2048D
   - BackboneConfig::mobile() → 512D

5. **Training Pipeline**
   - Adapt loxi-training to new backbone
   - Preserve C+D dynamics (delta ecosystem)
   - Test convergence on small corpus

---

## Questions to Ask Next Session

1. MoE deltas: tensor shape = [n_experts, D_GLOBAL]?
2. Attention2D placement: after layer 5 & 7? (positions 6-7 in 0-11 indexing)
3. Should mobile (512D) skip Attention2D entirely?
4. WGSL compilation: statically linked or runtime JIT?
5. Training: preserve full Tier1 for training, quantize at inference?

