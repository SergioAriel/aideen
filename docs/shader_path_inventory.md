# Shader Path Inventory

## Purpose
This document maps the shader/runtime paths that are actually relevant in the current
training stack, so cleanup and performance work can distinguish:

- core paths we must preserve
- optional experimental paths
- legacy or likely-dead paths that should be reviewed before removal

It is intentionally scoped to the DEQ/history training path in this branch.

## Core Training Paths

| Path | File | Selected by | Role |
|---|---|---|---|
| Unified slot-attn DEQ | `/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_slot_attn_unified_clean.wgsl` | canonical runtime path | Main DEQ forward path with `slot_ctx = Attn(signal)` and `H_curr` carry in the DEQ solve |
| DEQ forward portable | `/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_forward.wgsl` | legacy fallback / review | Legacy non-canonical DEQ forward path |
| DEQ forward subgroup | `/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_forward_subgroup.wgsl` | legacy fallback / review | Legacy fast-path variant of the old forward family |
| DEQ pool | `/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_forward_pool.wgsl` | always after forward | Pools per-slot output for LM head |
| Hist v2 project | `/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/hist_v2_project.wgsl` | legacy review only | Builds explicit frozen `hist_ctx` from temporal memory |
| Hist v2 temporal | `/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/hist_v2_temporal.wgsl` | legacy review only | Updates explicit temporal carrier `m_t` from `H*` |
| Staged adjoint Picard | `/Users/sergiosolis/Programacion/AIDEEN/aideen-backbone/src/shaders/staged_adjoint_picard.wgsl` | training GPU path | Main DEQ adjoint/backward path |
| Fused DEQ update | `/Users/sergiosolis/Programacion/AIDEEN/aideen-backbone/src/shaders/fused_deq_update.wgsl` | training GPU path | Weight update path, including attn/history updates |
| Embedding train | `/Users/sergiosolis/Programacion/AIDEEN/aideen-backbone/src/shaders/embedding_train.wgsl` | training GPU path | Embedding gather/build/update |
| LM train | `/Users/sergiosolis/Programacion/AIDEEN/aideen-backbone/src/shaders/lm_train.wgsl` | training GPU path | LM head forward/backward/update |

## Optional / Experimental Paths

| Path | File | Selector | Notes |
|---|---|---|---|
| Exact DEQ forward | `/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_forward_exact.wgsl` | `AIDEEN_DEQ_FORWARD_EXACT=1` | Alternate forward path; not the default training path |
| Clean staged adjoint Picard | `/Users/sergiosolis/Programacion/AIDEEN/aideen-backbone/src/shaders/staged_adjoint_picard_clean.wgsl` | clean-path selection inside GPU backend | Secondary adjoint path, not the primary one we benchmarked |

## Likely Legacy / Review Candidates

| File | Current evidence | Action |
|---|---|---|
| `/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/kernel_v8.wgsl` | No references found in `aideen-block`, `aideen-backbone`, or `aideen-training-lab` | Audit before removal |
| `/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/shaders/deq_sgd_update.wgsl` | Still wired through `deq_bridge.rs`, but not part of the main fused training path | Review if still needed |

## Runtime Selection Hotspots

- `/Users/sergiosolis/Programacion/AIDEEN/aideen-block/src/deq_bridge.rs`
  - selects `deq_forward` vs `deq_forward_exact`
  - enables subgroup fast path
  - enables staged slot-attention paths
  - canonical runtime currently bypasses the explicit history loop and runs:
    `deq_slot_attn_unified_clean -> deq_forward_pool`

- `/Users/sergiosolis/Programacion/AIDEEN/aideen-backbone/src/gpu_deq.rs`
  - selects history behavior and training/update probes
  - owns staged adjoint Picard and fused update pipeline orchestration

## Current Cleanup Policy

1. Do not remove optional paths until their selector and benchmark purpose are documented.
2. Prefer consolidating runtime selection before deleting shader files.
3. Treat `fused_deq_update.wgsl` as the primary locus for history-cost investigations.
4. Keep forward/reference equation alignment ahead of large shader deletions.
