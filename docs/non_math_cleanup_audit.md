# Non-Math Cleanup Audit

Date: 2026-03-27

## Scope

This audit covers the remaining non-mathematical technical debt in the GPU training path after
the dead-code sweep. The goal is to separate:

- code that can be removed now,
- code that is legacy but still reachable,
- synchronization points that are intentional,
- conservative buffer clears that still protect a live invariant.

## 1. Removed Dead Code

Removed because it was not referenced by the compiled training path or tests:

- `GpuDeqBackend`
  - `fused_update_hist_bias_pipeline`
  - `fused_update_hist_xprep_pipeline`
  - `fused_update_hist_hrhs_pipeline`
  - `picard_damping_from_env()`
  - `env_flag()`
- `GpuLmHead`
  - `dw_accum_pipeline`
  - `apply_adamw_pipeline`
  - dead local `d_parts` declarations
- `Trainer`
  - `env_u32()`
- `fixed_point_memory_reasoning`
  - `identity_like_mat_with_rng()`
- `train` CLI
  - `print_banner()`
  - `run_small_dataset()`

Validation:

- `cargo check --release --features wgpu -p aideen-training` completed without warnings.

## 2. Legacy Fused Adjoint

### Finding

The old fused Picard adjoint path was not part of the real training call chain. It only survived
through stale integration tests that no longer compiled against the current backend API.

### Decision

Delete the path now.

Reason:

- training no longer uses it,
- tests were already broken against the current API,
- keeping it only preserved dead code and dead shader surface.

Action taken:

- removed `run_fused_adjoint_picard_no_readback()`
- removed its dedicated shader and pipeline/bind-group/buffer wiring
- removed the stale fused/CG test coverage and kept only staged/forward coverage aligned with the
  current adjoint API

## 3. Profiling

Profiling here means instrumentation for timing and internal inspection, not normal training.

Examples:

- `cfg_fused_profile`
- `cfg_picard_profile`
- `cfg_picard_stage_profile`
- `cfg_hist_internal_probe`
- `cfg_picard_internal_probe`

These modes intentionally:

- break batched execution into smaller submits,
- insert `poll(Wait)` barriers,
- zero more scratch buffers than the normal hot path,
- perform readbacks for timing or internal signal inspection.

They should not be judged by regime TPS.

## 4. Remaining `poll()` / `poll(Wait)` Classification

### Real-time training hot path

- normal staged adjoint path:
  - no hard `Wait` barrier between adjoint and fused update
- normal fused update path:
  - uses `poll(Poll)`, not `poll(Wait)`

Interpretation:

- this path is intentionally asynchronous,
- current waits are not the main regime bottleneck.

### Intentional synchronization points

- trainer progress/TPS sync
- epoch boundary metrics
- final validation / end-of-run metrics
- checkpoint moment readback
- immediate CPU readback helpers in LM/deq code
- profiling/stage timing paths

### Decision

Do not remove waits blindly.

Reason:

- most remaining waits sit at API boundaries where CPU consumes data immediately,
- or inside profiling/debug modes where completion timing is the point of the path.

Action taken:

- documented the purpose of the surviving waits in code.

## 5. Remaining `clear_buffer()` Classification

### Clears that are still structural

#### Staged adjoint

- `adj_bufs.b_v_out`
  - required because each Picard solve must start from `v_state = 0`
- `fused_hist_ctx_buf` when `clear_slot_rhs`
  - required when slot RHS state must not leak across calls
- `fused_qgrad_buf`
- `fused_gscore_buf`
  - earlier experiments showed removing these changes solver trajectory

#### Fused update normal path

- `fused_hist_ctx_buf`
- `fused_hist_delta_buf`

Reason:

- their producers do not yet have proof of full overwrite coverage across all selective/gated
  regimes.

### Clears already removed in prior work

- staged path:
  - `fused_hist_delta_buf`
- normal fused update path:
  - `fused_weighted_h_buf`
  - `fused_gmix_buf`
  - `fused_gscore_buf`
  - `fused_qgrad_buf`

Those removals were retained only where overwrite coverage had already been validated. In addition,
the old `fused_v_next_buf` no longer exists because the fused adjoint path was removed entirely.

### Decision

Do not remove the remaining conservative clears yet.

Reason:

- there is still no causal proof that all consuming regimes are fully covered without them.

## 6. Net Result

### Closed now

- dead code sweep
- legacy-path reachability audit
- wait/poll classification
- clear classification

### Still open by design

- removal of remaining conservative clears
- deeper synchronization reduction in debug/profile/helper APIs

These remain open because they require either:

- test migration,
- stronger path-by-path overwrite proof,
- or an explicit product decision about retaining debug/reference paths.
