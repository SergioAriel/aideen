struct SharedUniforms {
    d_model: u32,
    h_slots: u32,
    lr: f32,
    grad_scale: f32,
    ternary_flag: u32,
    weight_decay: f32,
    seq_len: u32,
    damping: f32,
    residual_alpha: f32,
    grad_accum_mode: u32,
    n_accum: u32,
    n_total_weights: u32,
    batch_size: u32,
    apply_accum: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform>             params:          SharedUniforms;
@group(0) @binding(1) var<storage, read>       fpm_dm_buf:      array<f32>;
@group(0) @binding(2) var<storage, read>       fpm_minner_buf:  array<f32>;
@group(0) @binding(3) var<storage, read>       fpm_hunit_buf:   array<f32>;
@group(0) @binding(4) var<storage, read_write> AllGradients:    array<f32>;
@group(0) @binding(5) var<storage, read>       fpm_gx_buf:      array<f32>;
@group(0) @binding(6) var<storage, read_write> fpm_wout_partial: array<f32>;
@group(0) @binding(7) var<storage, read_write> fpm_wx_partial:   array<f32>;

@group(1) @binding(0) var<storage, read_write> AllWeights: array<f32>;

fn aw_wq_base(d: u32, h: u32) -> u32 { return 0u; }
fn aw_wk_base(d: u32, h: u32) -> u32 { return h * d * d + h * d; }
fn aw_wv_base(d: u32, h: u32) -> u32 { return aw_wk_base(d, h) + h * d * d + h * d; }
fn aw_wo_base(d: u32, h: u32) -> u32 { return aw_wv_base(d, h) + h * d * d; }
fn aw_win_base(d: u32, h: u32) -> u32 { return aw_wo_base(d, h) + h * d * d; }
fn aw_wx_base(d: u32, h: u32) -> u32 { return aw_win_base(d, h) + h * d * d; }
fn aw_wout_base(d: u32, h: u32) -> u32 { return aw_wx_base(d, h) + d * d; }
fn aw_alog_base(d: u32, h: u32) -> u32 { return aw_wout_base(d, h) + d * d; }

fn fpm_off(t_abs: u32, slot: u32, d: u32, h: u32) -> u32 {
    return t_abs * h * d + slot * d;
}

fn wx_scale() -> f32 {
    return 0.5;
}

// ── FPM_BLOCK: number of (token,slot) entries per partial-reduction block ─────
const FPM_BLOCK: u32 = 64u;

// ══════════════════════════════════════════════════════════════════════════════
// W_out gradient — factorized path (replaces the naive triple-loop kernel)
//
// Original naive kernel: for each (row,col) workgroup, loops over all T×h
// (token,slot) pairs reading fpm_dm_buf and fpm_minner_buf → 135× read
// amplification when T=512, h=4, d=512.
//
// Factorized path (valid when fpm_dm_buf is token-invariant, i.e. TBPTT carry
// writes the same dm value into every token slot):
//
//   dW_out[row,col] = (1/n) * Σ_slot dm[slot,row] * Σ_t minner[t,slot,col]
//
// Step 1 — fpm_minner_colsum_main
//   Sums fpm_minner_buf over t_abs for each (slot, col).
//   Dispatch: (h_slots, ceil(d/64), 1)  @workgroup_size(64,1,1)
//   Writes result into fpm_wout_partial[slot * d + col] (first h*d elements).
//   Access is stride-1 coalesced for consecutive col lanes.
//
// Step 2 — fpm_wout_factored_main
//   Outer product dm_const × minner_sum → dW_out.
//   Dispatch: (ceil(d/16), ceil(d/16), 1)  @workgroup_size(16,16,1)
//   Reads only 2*h*d ≈ 16KB (fully L1-cached); writes d²=1MB once.
// ══════════════════════════════════════════════════════════════════════════════

@compute
@workgroup_size(64, 1, 1)
fn fpm_minner_colsum_main(
    @builtin(workgroup_id)        wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let slot = wid.x;
    let col  = wid.y * 64u + lid.x;
    let d = params.d_model;
    let h = params.h_slots;
    let total_tokens = params.batch_size * params.seq_len;
    if (slot >= h || col >= d) { return; }

    var sum = 0.0;
    for (var t_abs = 0u; t_abs < total_tokens; t_abs = t_abs + 1u) {
        sum = sum + fpm_minner_buf[fpm_off(t_abs, slot, d, h) + col];
    }
    // Reuse fpm_wout_partial[slot * d + col] as temp storage (first h*d entries).
    fpm_wout_partial[slot * d + col] = sum;
}

@compute
@workgroup_size(16, 16, 1)
fn fpm_wout_factored_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    let d = params.d_model;
    let h = params.h_slots;
    if (row >= d || col >= d) { return; }

    let n_entries = max(1.0, f32(params.batch_size * params.seq_len * h));
    var grad = 0.0;
    for (var slot = 0u; slot < h; slot = slot + 1u) {
        // dm is token-invariant: read from t_abs=0 slice (fpm_off(0,slot,d,h) = slot*d).
        let dm   = fpm_dm_buf[slot * d + row];
        // minner_sum written by fpm_minner_colsum_main into fpm_wout_partial[slot*d + col].
        let msum = fpm_wout_partial[slot * d + col];
        grad = grad + dm * msum;
    }
    grad = grad / n_entries;

    let idx = aw_wout_base(d, h) + row * d + col;
    let clip = 0.5;
    let wd_factor = 1.0 - params.lr * params.weight_decay;
    let raw_step = params.lr * grad;
    if (params.grad_accum_mode == 1u) {
        AllGradients[idx] = AllGradients[idx] + raw_step;
    } else {
        AllWeights[idx] = AllWeights[idx] * wd_factor - clamp(raw_step, -clip, clip);
    }
}

// ── Fallback naive W_out kernel (kept for non-TBPTT paths / debugging) ───────
@compute
@workgroup_size(16, 16, 1)
fn fused_fpm_stage_wout_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    let d = params.d_model;
    let h = params.h_slots;
    if (row >= d || col >= d) { return; }

    let n_entries = max(1.0, f32(params.batch_size * params.seq_len * h));
    var grad = 0.0;
    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        for (var t = 0u; t < params.seq_len; t = t + 1u) {
            let t_abs = b * params.seq_len + t;
            for (var slot = 0u; slot < h; slot = slot + 1u) {
                let off = fpm_off(t_abs, slot, d, h);
                grad = grad + fpm_dm_buf[off + row] * fpm_minner_buf[off + col];
            }
        }
    }
    grad = grad / n_entries;

    let idx = aw_wout_base(d, h) + row * d + col;
    let clip = 0.5;
    let wd_factor = 1.0 - params.lr * params.weight_decay;
    let raw_step = params.lr * grad;
    if (params.grad_accum_mode == 1u) {
        AllGradients[idx] += raw_step;
    } else {
        AllWeights[idx] = AllWeights[idx] * wd_factor - clamp(raw_step, -clip, clip);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// W_x gradient — partial + reduce path
//
// Original: single kernel loops over all T×h entries per output dim → serial.
// Partial path: splits the T×h loop into blocks of FPM_BLOCK, each block
// dispatched as a separate workgroup → parallel across blocks.
// ══════════════════════════════════════════════════════════════════════════════

@compute
@workgroup_size(64, 1, 1)
fn fused_fpm_stage_wx_partial_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim   = gid.x;
    let block = gid.y;
    let d = params.d_model;
    let h = params.h_slots;
    if (dim >= d) { return; }

    let n_entries = params.batch_size * params.seq_len * h;
    let start = block * FPM_BLOCK;
    if (start >= n_entries) { return; }
    let end = min(start + FPM_BLOCK, n_entries);

    var grad = 0.0;
    for (var entry = start; entry < end; entry = entry + 1u) {
        let t_abs = entry / h;
        let slot  = entry % h;
        let off   = fpm_off(t_abs, slot, d, h);
        grad = grad + fpm_gx_buf[off + dim] * fpm_hunit_buf[off + dim];
    }
    fpm_wx_partial[block * d + dim] = grad;
}

@compute
@workgroup_size(64, 1, 1)
fn fused_fpm_stage_wx_reduce_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let d = params.d_model;
    let h = params.h_slots;
    if (dim >= d) { return; }

    let n_entries = max(1.0, f32(params.batch_size * params.seq_len * h));
    let n_blocks = (params.batch_size * params.seq_len * h + FPM_BLOCK - 1u) / FPM_BLOCK;
    var grad = 0.0;
    for (var block = 0u; block < n_blocks; block = block + 1u) {
        grad = grad + fpm_wx_partial[block * d + dim];
    }
    grad = grad / n_entries;

    let idx = aw_wx_base(d, h) + dim * d + dim;
    let clip = 0.5;
    let wd_factor = 1.0 - params.lr * params.weight_decay;
    let wx_raw = AllWeights[idx];
    let wx_tanh = tanh(wx_raw);
    let wx_grad = grad * wx_scale() * (1.0 - wx_tanh * wx_tanh);
    let raw_step = params.lr * wx_grad;
    if (params.grad_accum_mode == 1u) {
        AllGradients[idx] += raw_step;
    } else {
        AllWeights[idx] = wx_raw * wd_factor - clamp(raw_step, -clip, clip);
    }
}

// ── Fallback naive W_x kernel ─────────────────────────────────────────────────
@compute
@workgroup_size(64, 1, 1)
fn fused_fpm_stage_wx_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let d = params.d_model;
    let h = params.h_slots;
    if (dim >= d) { return; }

    let n_entries = max(1.0, f32(params.batch_size * params.seq_len * h));
    var grad = 0.0;
    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        for (var t = 0u; t < params.seq_len; t = t + 1u) {
            let t_abs = b * params.seq_len + t;
            for (var slot = 0u; slot < h; slot = slot + 1u) {
                let off = fpm_off(t_abs, slot, d, h);
                grad = grad + fpm_gx_buf[off + dim] * fpm_hunit_buf[off + dim];
            }
        }
    }
    grad = grad / n_entries;

    let idx = aw_wx_base(d, h) + dim * d + dim;
    let clip = 0.5;
    let wd_factor = 1.0 - params.lr * params.weight_decay;
    let wx_raw = AllWeights[idx];
    let wx_tanh = tanh(wx_raw);
    let wx_grad = grad * wx_scale() * (1.0 - wx_tanh * wx_tanh);
    let raw_step = params.lr * wx_grad;
    if (params.grad_accum_mode == 1u) {
        AllGradients[idx] += raw_step;
    } else {
        AllWeights[idx] = wx_raw * wd_factor - clamp(raw_step, -clip, clip);
    }
}
