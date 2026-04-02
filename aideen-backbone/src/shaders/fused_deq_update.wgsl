struct UpdateUniforms {
    d_model: u32,
    h_slots: u32,
    lr: f32,
    grad_scale: f32,
    ternary_flag: u32,
    weight_decay: f32,
    seq_len: u32,
    damping: f32,
    residual_alpha: f32,
    grad_accum_mode: u32,  // 0=direct update, 1=accumulate into AllGradients
    n_accum: u32,          // accumulation count (for apply_grad_update_main)
    n_total_weights: u32,  // total AllWeights elements (for apply_grad_update_main bounds)
    batch_size: u32,       // number of sequences processed in parallel
    apply_accum: u32,      // 1=apply accumulated gradients (apply_grad_update_main)
    grid_stride_x: u32,    // X dimension of the dispatch grid (for 2D dispatch linearisation)
};

@group(0) @binding(0) var<uniform> params: UpdateUniforms;
@group(0) @binding(1) var<storage, read> v_adjoint: array<f32>;
@group(0) @binding(2) var<storage, read> q_input: array<f32>;
@group(0) @binding(3) var<storage, read> h_star: array<f32>;
@group(0) @binding(4) var<storage, read_write> debug_log: array<f32>;
@group(0) @binding(5) var<storage, read> dl_dh_temp_buf: array<f32>;
@group(0) @binding(6) var<storage, read> Scratch: array<f32>;
@group(0) @binding(7) var<storage, read_write> mix_buf: array<f32>;
@group(0) @binding(8) var<storage, read_write> weighted_h_buf: array<f32>;
@group(0) @binding(9) var<storage, read_write> gmix_buf: array<f32>;
@group(0) @binding(10) var<storage, read_write> gscore_buf: array<f32>;
@group(0) @binding(11) var<storage, read_write> qgrad_buf: array<f32>;
@group(0) @binding(12) var<storage, read_write> hist_ctx_buf: array<f32>;
@group(0) @binding(13) var<storage, read_write> hist_delta_buf: array<f32>;
@group(0) @binding(14) var<storage, read_write> tbptt_carry_buf: array<f32>;

@group(0) @binding(15) var<storage, read_write> AllGradients: array<f32>;

@group(1) @binding(0) var<storage, read_write> AllWeights: array<f32>;

// AllWeights layout offset functions (must match deq_forward.wgsl and deq_bridge.rs).
fn aw_wq_base(d: u32, h: u32) -> u32 { return 0u; }
fn aw_wk_base(d: u32, h: u32) -> u32 { return h*d*d + h*d; }
fn aw_wv_base(d: u32, h: u32) -> u32 { return aw_wk_base(d,h) + h*d*d + h*d; }
fn aw_wo_base(d: u32, h: u32) -> u32 { return aw_wv_base(d,h) + h*d*d; }
fn aw_win_base(d: u32, h: u32) -> u32 { return aw_wo_base(d,h) + h*d*d; }
fn aw_wx_base(d: u32, h: u32) -> u32 { return aw_win_base(d,h) + h*d*d; }
fn aw_wout_base(d: u32, h: u32) -> u32 { return aw_wx_base(d,h) + d*d; }
fn aw_alog_base(d: u32, h: u32) -> u32 { return aw_wout_base(d,h) + d*d; }
fn aw_nscale_base(d: u32, h: u32) -> u32 { return aw_alog_base(d,h) + h*d; }
fn aw_hist_base(d: u32, h: u32) -> u32 { return aw_nscale_base(d,h) + d; }

fn entry_base(entry: u32, d: u32) -> u32 {
    return entry * d;
}

fn scratch_stride(d: u32, h_slots: u32) -> u32 {
    return d * (h_slots * 8u) + h_slots * h_slots + h_slots;
}

fn hist_mat_len(d: u32) -> u32 {
    return d * d;
}

fn hist_scale_base(d: u32, h_slots: u32) -> u32 {
    return hist_mat_len(d);
}

fn hist_bias_base(d: u32, h_slots: u32) -> u32 {
    return hist_scale_base(d, h_slots) + h_slots * d;
}

fn hist_gate_base(d: u32, h_slots: u32) -> u32 {
    return hist_bias_base(d, h_slots) + h_slots * d;
}

fn slot_anchor_base(d: u32, h_slots: u32) -> u32 {
    return hist_gate_base(d, h_slots) + h_slots;
}

fn hist_delta_base(d: u32, h_slots: u32) -> u32 {
    return slot_anchor_base(d, h_slots) + h_slots * d;
}

fn hist_delta_bias_base(d: u32, h_slots: u32) -> u32 {
    return hist_delta_base(d, h_slots) + h_slots * d * d;
}

fn hist_selective_flag_base(d: u32, h_slots: u32) -> u32 {
    return hist_delta_bias_base(d, h_slots) + d;
}

fn hist_warmup_base(d: u32, h_slots: u32) -> u32 {
    return hist_selective_flag_base(d, h_slots) + 1u;
}

fn hist_rms_floor_base(d: u32, h_slots: u32) -> u32 {
    return hist_warmup_base(d, h_slots) + 1u;
}

fn hist_contr_floor_base(d: u32, h_slots: u32) -> u32 {
    return hist_rms_floor_base(d, h_slots) + 1u;
}

fn hist_inject_flag_base(d: u32, h_slots: u32) -> u32 {
    return hist_contr_floor_base(d, h_slots) + 1u;
}

fn hist_minner_zero_base(d: u32, h_slots: u32) -> u32 {
    return hist_inject_flag_base(d, h_slots) + 1u;
}

fn hist_force_nomamba_base(d: u32, h_slots: u32) -> u32 {
    return hist_minner_zero_base(d, h_slots) + 1u;
}

fn hist_prelude_skip_base(d: u32, h_slots: u32) -> u32 {
    return hist_force_nomamba_base(d, h_slots) + 1u;
}

fn hist_loop_force_nomamba_base(d: u32, h_slots: u32) -> u32 {
    return hist_prelude_skip_base(d, h_slots) + 1u;
}

fn signal_zero_base(d: u32, h_slots: u32) -> u32 {
    return hist_loop_force_nomamba_base(d, h_slots) + 1u;
}

fn attn_out_mode_base(d: u32, h_slots: u32) -> u32 {
    return signal_zero_base(d, h_slots) + 1u;
}

fn attn_uniform_base(d: u32, h_slots: u32) -> u32 {
    return attn_out_mode_base(d, h_slots) + 1u;
}

fn attn_freeze_base(d: u32, h_slots: u32) -> u32 {
    return attn_uniform_base(d, h_slots) + 1u;
}

// W_gate_hist follows the 21 scalars (hist_selective_flag .. v_norm_scale)
fn hist_gate_query_base(d: u32, h_slots: u32) -> u32 {
    return hist_selective_flag_base(d, h_slots) + 21u;
}

// W_forget: h_slots × d_model forget gate query (follows W_gate_hist)
fn w_forget_base(d: u32, h: u32) -> u32 {
    return hist_gate_query_base(d, h) + h * d;
}

// b_forget: h_slots bias for forget gate (follows W_forget)
fn b_forget_base(d: u32, h: u32) -> u32 {
    return w_forget_base(d, h) + h * d;
}

// f_gate scratch offset: 1 value per (t_abs, slot) stored after attn_weight region
fn token_forget_base(t: u32, d: u32, h_slots: u32) -> u32 {
    return token_scratch_base(t, d, h_slots) + 8u * d * h_slots + h_slots * h_slots;
}

fn hist_alpha_min_target() -> f32 {
    return 0.070;
}

fn hist_alpha_min_start() -> f32 {
    return 0.030;
}

fn hist_alpha_max() -> f32 {
    return 0.20;
}

fn hist_alpha_min(d: u32, h_slots: u32) -> f32 {
    let warmup = clamp(AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_warmup_base(d, h_slots)], 0.0, 1.0);
    return hist_alpha_min_start() + (hist_alpha_min_target() - hist_alpha_min_start()) * warmup;
}

fn hist_wx_max() -> f32 {
    return 0.5;
}

fn hist_gate_alpha(d: u32, h_slots: u32, logit: f32) -> f32 {
    let amin = hist_alpha_min(d, h_slots);
    let amax = hist_alpha_max();
    return amin + (amax - amin) * (1.0 / (1.0 + exp(-logit)));
}

fn hist_gate_sigma(logit: f32) -> f32 {
    return 1.0 / (1.0 + exp(-logit));
}

fn hist_cap_mult() -> f32 {
    return 1.0;
}

fn hist_cap_floor_mult() -> f32 {
    return 0.0;
}

fn hist_selective_enabled(d: u32, h_slots: u32) -> bool {
    return AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_selective_flag_base(d, h_slots)] > 0.5;
}

fn hist_selective_a_floor() -> f32 {
    return 0.070;
}

fn token_scratch_base(t: u32, d: u32, h_slots: u32) -> u32 {
    return t * scratch_stride(d, h_slots);
}

fn token_mamba_base(t: u32, d: u32, h_slots: u32) -> u32 {
    return token_scratch_base(t, d, h_slots) + h_slots * 4u * d;
}

fn token_signal_base(t: u32, d: u32, h_slots: u32) -> u32 {
    return token_mamba_base(t, d, h_slots) + h_slots * d;
}

fn token_minner_base(t: u32, d: u32, h_slots: u32) -> u32 {
    return token_signal_base(t, d, h_slots) + h_slots * d;
}

fn token_hist_ctx_base(t: u32, d: u32, h_slots: u32) -> u32 {
    return token_minner_base(t, d, h_slots) + h_slots * d;
}

var<workgroup> hist_reduce_u: array<f32, 64>;
var<workgroup> hist_reduce_aux: array<f32, 64>;
var<workgroup> hist_carry_vec: array<f32, 1024>;
var<workgroup> hist_total_vec: array<f32, 1024>;
var<workgroup> hist_ginner_vec: array<f32, 1024>;
var<workgroup> hist_gx_vec: array<f32, 1024>;
var<workgroup> stage4_mat_a_tile: array<f32, 256>;
var<workgroup> stage4_mat_b_tile: array<f32, 256>;

@compute
@workgroup_size(16, 16, 1)
fn fused_attn_stage1a_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let d = params.d_model;
    let h_slots = params.h_slots;
    let row = gid.x;
    let token = gid.y;
    let slot = gid.z;
    let local_col = lid.x;
    let local_row = lid.y;
    let n_tokens = params.batch_size * params.seq_len;
    if (row >= d || token >= n_tokens || slot >= h_slots) { return; }

    var gmix = 0.0;
    for (var tile = 0u; tile < d; tile = tile + 16u) {
        let k_a = tile + local_col;
        if (k_a < d) {
            stage4_mat_a_tile[local_row * 16u + local_col] =
                v_adjoint[token * h_slots * d + slot * d + k_a];
        } else {
            stage4_mat_a_tile[local_row * 16u + local_col] = 0.0;
        }

        let k_b = tile + local_row;
        if (k_b < d) {
            stage4_mat_b_tile[local_row * 16u + local_col] =
                AllWeights[aw_wo_base(params.d_model, params.h_slots) + slot * d * d + row * d + k_b];
        } else {
            stage4_mat_b_tile[local_row * 16u + local_col] = 0.0;
        }
        workgroupBarrier();

        let tile_limit = min(16u, d - tile);
        for (var k = 0u; k < tile_limit; k = k + 1u) {
            gmix = gmix
                + stage4_mat_a_tile[local_row * 16u + k]
                * stage4_mat_b_tile[k * 16u + local_col];
        }
        workgroupBarrier();
    }
    let entry = token * h_slots + slot;
    gmix_buf[entry_base(entry, d) + row] = gmix;
}

@compute
@workgroup_size(64, 1, 1)
fn fused_attn_stage1b_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let lane = lid.x;
    // 2D dispatch reconstruction: entry = wid.y * grid_stride_x + wid.x.
    // When n <= 65535, gid.y == 0 and grid_stride_x == n, so entry == wid.x.
    let entry = wid.y * params.grid_stride_x + wid.x;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (entry >= n_entries) { return; }

    let t_abs = entry / h_slots;
    let t = t_abs % params.seq_len;
    let qs = entry % h_slots;
    let sstride = scratch_stride(d, h_slots);
    let base = t_abs * sstride;
    let v_base = base + h_slots * d * 2u;
    let signal_base = base + d * (h_slots * 5u);
    let attn_weight_base = signal_base + 3u * h_slots * d;
    let slot_anchor = slot_anchor_base(d, h_slots);

    for (var dim = lane; dim < d; dim = dim + 64u) {
        var mix = 0.0;
        var weighted_h = 0.0;
        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
            let a = Scratch[attn_weight_base + qs * h_slots + ks];
            let k_off = ks * d;
            mix = mix + a * Scratch[v_base + k_off + dim];
            // External slot-context path: W_v is applied to (signal + slot_anchor + hist_ctx),
            // so g_wv must use the same source vector to keep forward/backward consistent.
            // the same source vector to keep forward/backward consistent.
            let src = Scratch[signal_base + k_off + dim]
                + AllWeights[aw_hist_base(params.d_model, params.h_slots) +slot_anchor + k_off + dim]
                + Scratch[token_hist_ctx_base(t_abs, d, h_slots) + k_off + dim];
            weighted_h = weighted_h + a * src;
        }
        let out_idx = entry_base(entry, d) + dim;
        mix_buf[out_idx] = mix;
        weighted_h_buf[out_idx] = weighted_h;
    }
}

@compute
@workgroup_size(16, 16, 1)
fn fused_attn_stage2_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ks = gid.x;
    let entry = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (ks >= h_slots || entry >= n_entries) { return; }

    let t_abs = entry / h_slots;
    let t = t_abs % params.seq_len;
    let qs = entry % h_slots;
    let sstride = scratch_stride(d, h_slots);
    let base = t_abs * sstride;
    let v_base = base + h_slots * d * 2u;
    let signal_base = base + d * (h_slots * 5u);
    let attn_weight_base = signal_base + 3u * h_slots * d;
    let off = entry_base(entry, d);

    var g_alpha = 0.0;
    let k_off = ks * d;
    for (var dim = 0u; dim < d; dim = dim + 1u) {
        g_alpha = g_alpha + gmix_buf[off + dim] * Scratch[v_base + k_off + dim];
    }

    var alpha_dot_g = 0.0;
    for (var js = 0u; js < h_slots; js = js + 1u) {
        let j_off = js * d;
        var g_alpha_j = 0.0;
        for (var dim = 0u; dim < d; dim = dim + 1u) {
            g_alpha_j = g_alpha_j + gmix_buf[off + dim] * Scratch[v_base + j_off + dim];
        }
        let a_j = Scratch[attn_weight_base + qs * h_slots + js];
        alpha_dot_g = alpha_dot_g + a_j * g_alpha_j;
    }

    let a = Scratch[attn_weight_base + qs * h_slots + ks];
    gscore_buf[entry * h_slots + ks] = a * (g_alpha - alpha_dot_g);
}

@compute
@workgroup_size(16, 16, 1)
fn fused_attn_stage3_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let entry = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (dim >= d || entry >= n_entries) { return; }

    let t_abs = entry / h_slots;
    let t = t_abs % params.seq_len;
    let scale = inverseSqrt(max(1.0, f32(d)));
    let sstride = scratch_stride(d, h_slots);
    let base = t_abs * sstride;
    let k_base = base + h_slots * d;
    var gq = 0.0;
    for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
        gq = gq + scale * gscore_buf[entry * h_slots + ks] * Scratch[k_base + ks * d + dim];
    }
    qgrad_buf[entry_base(entry, d) + dim] = gq;
}

@compute
@workgroup_size(16, 16, 1)
fn fused_attn_stage4_wo_win_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    let d = params.d_model;
    if (row >= d || col >= d) { return; }

    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    let idx = row * d + col;
    let lr = params.lr;
    let wd_factor = 1.0 - lr * params.weight_decay;
    let grad_scale = params.grad_scale;

    var g_wo: array<f32, 16>;
    var g_win: array<f32, 16>;

    for (var entry = 0u; entry < n_entries; entry = entry + 1u) {
        let t_abs = entry / h_slots;
        let qs = entry % h_slots;
        let q_off = qs * d;
        let off = entry_base(entry, d);
        let g_attn_col = v_adjoint[t_abs * h_slots * d + q_off + col];
        g_wo[qs] = g_wo[qs] + g_attn_col * mix_buf[off + row];
        g_win[qs] = g_win[qs] + q_input[t_abs * d + row] * g_attn_col;
    }

    let seq_scale = 1.0 / max(1.0, f32(n_entries));
    let clip = 0.5;
    for (var s = 0u; s < h_slots; s = s + 1u) {
        let raw_wo = lr * g_wo[s] * seq_scale * grad_scale;
        let wo_idx = aw_wo_base(params.d_model, params.h_slots) + s * d * d + idx;
        if (params.grad_accum_mode == 1u) {
            AllGradients[wo_idx] += raw_wo;
        } else {
            AllWeights[wo_idx] = AllWeights[wo_idx] * wd_factor - clamp(raw_wo, -clip, clip);
        }
    }

    for (var s = 0u; s < h_slots; s = s + 1u) {
        let g_win_s = g_win[s] * seq_scale;
        let raw_win = lr * g_win_s * grad_scale;
        if (params.grad_accum_mode == 1u) {
            AllGradients[aw_win_base(params.d_model, params.h_slots) +s * d * d + idx] += raw_win;
        } else {
            AllWeights[aw_win_base(params.d_model, params.h_slots) +s * d * d + idx] = AllWeights[aw_win_base(params.d_model, params.h_slots) +s * d * d + idx] * wd_factor - clamp(raw_win, -clip, clip);
        }
    }

    if (idx == 0u) {
        debug_log[8] = 241.0;
        debug_log[41] = g_wo[0];
        debug_log[43] = g_win[0];
    }
}

@compute
@workgroup_size(16, 16, 1)
fn fused_attn_stage4_wq_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let col = gid.x;
    let row = gid.y;
    let slot = gid.z;
    let local_col = lid.x;
    let local_row = lid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (row >= d || col >= d || slot >= h_slots) { return; }

    let n_entries = params.batch_size * params.seq_len * h_slots;
    let n_tokens = params.batch_size * params.seq_len;
    let idx = row * d + col;
    let lr = params.lr;
    let wd_factor = 1.0 - lr * params.weight_decay;
    let grad_scale = params.grad_scale;

    var g_wq = 0.0;
    for (var tile = 0u; tile < n_tokens; tile = tile + 16u) {
        let token_a = tile + local_col;
        if (token_a < n_tokens) {
            let entry = token_a * h_slots + slot;
            stage4_mat_a_tile[local_row * 16u + local_col] =
                qgrad_buf[entry_base(entry, d) + row];
        } else {
            stage4_mat_a_tile[local_row * 16u + local_col] = 0.0;
        }

        let token_b = tile + local_row;
        if (token_b < n_tokens) {
            stage4_mat_b_tile[local_row * 16u + local_col] =
                Scratch[token_signal_base(token_b, d, h_slots) + slot * d + col]
                + AllWeights[aw_hist_base(d, h_slots) + slot_anchor_base(d, h_slots) + slot * d + col]
                + Scratch[token_hist_ctx_base(token_b, d, h_slots) + slot * d + col];
        } else {
            stage4_mat_b_tile[local_row * 16u + local_col] = 0.0;
        }
        workgroupBarrier();

        let tile_limit = min(16u, n_tokens - tile);
        for (var k = 0u; k < tile_limit; k = k + 1u) {
            g_wq = g_wq
                + stage4_mat_a_tile[local_row * 16u + k]
                * stage4_mat_b_tile[k * 16u + local_col];
        }
        workgroupBarrier();
    }

    let seq_scale = 1.0 / max(1.0, f32(n_entries));
    let clip = 0.5;
    let per_idx = slot * d * d + idx;
    let raw_wq = lr * g_wq * seq_scale * grad_scale;
    if (params.grad_accum_mode == 1u) {
        AllGradients[aw_wq_base(params.d_model, params.h_slots) + per_idx] += raw_wq;
    } else {
        AllWeights[aw_wq_base(params.d_model, params.h_slots) + per_idx] =
            AllWeights[aw_wq_base(params.d_model, params.h_slots) + per_idx] * wd_factor - clamp(raw_wq, -clip, clip);
    }

    if (idx == 0u && slot == 0u) {
        debug_log[8] = 246.0;
        debug_log[40] = g_wq;
    }
}

@compute
@workgroup_size(16, 16, 1)
fn fused_attn_stage4_prep_wk_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let tk = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_tokens = params.batch_size * params.seq_len;
    let n_token_slots = n_tokens * h_slots;
    if (dim >= d || tk >= n_token_slots) { return; }

    let t_abs = tk / h_slots;
    let ks = tk % h_slots;
    let scale = inverseSqrt(max(1.0, f32(d)));
    let sstride = scratch_stride(d, h_slots);
    let q_base = t_abs * sstride;

    var coeff = 0.0;
    for (var qs = 0u; qs < h_slots; qs = qs + 1u) {
        let entry = t_abs * h_slots + qs;
        coeff = coeff + scale * gscore_buf[entry * h_slots + ks]
            * Scratch[q_base + qs * d + dim];
    }
    hist_delta_buf[entry_base(tk, d) + dim] = coeff;
}

@compute
@workgroup_size(16, 16, 1)
fn fused_attn_stage4_prep_wv_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let tk = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_tokens = params.batch_size * params.seq_len;
    let n_token_slots = n_tokens * h_slots;
    if (dim >= d || tk >= n_token_slots) { return; }

    let t_abs = tk / h_slots;
    let ks = tk % h_slots;
    let signal_base_t = token_signal_base(t_abs, d, h_slots);
    let attn_weight_base_t = signal_base_t + 3u * h_slots * d;

    var coeff = 0.0;
    for (var qs = 0u; qs < h_slots; qs = qs + 1u) {
        let entry = t_abs * h_slots + qs;
        coeff = coeff
            + Scratch[attn_weight_base_t + qs * h_slots + ks]
            * gmix_buf[entry_base(entry, d) + dim];
    }
    hist_ctx_buf[entry_base(tk, d) + dim] = coeff;
}

@compute
@workgroup_size(16, 16, 1)
fn fused_attn_stage4_wk_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let col = gid.x;
    let row = gid.y;
    let slot = gid.z;
    let local_col = lid.x;
    let local_row = lid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (row >= d || col >= d || slot >= h_slots) { return; }

    let n_entries = params.batch_size * params.seq_len * h_slots;
    let n_tokens = params.batch_size * params.seq_len;
    let idx = row * d + col;
    let lr = params.lr;
    let wd_factor = 1.0 - lr * params.weight_decay;
    let grad_scale = params.grad_scale;

    var g_wk = 0.0;
    for (var tile = 0u; tile < n_tokens; tile = tile + 16u) {
        let token_a = tile + local_col;
        if (token_a < n_tokens) {
            stage4_mat_a_tile[local_row * 16u + local_col] =
                hist_delta_buf[entry_base(token_a * h_slots + slot, d) + row];
        } else {
            stage4_mat_a_tile[local_row * 16u + local_col] = 0.0;
        }

        let token_b = tile + local_row;
        if (token_b < n_tokens) {
            stage4_mat_b_tile[local_row * 16u + local_col] =
                Scratch[token_signal_base(token_b, d, h_slots) + slot * d + col]
                + AllWeights[aw_hist_base(d, h_slots) + slot_anchor_base(d, h_slots) + slot * d + col]
                + Scratch[token_hist_ctx_base(token_b, d, h_slots) + slot * d + col];
        } else {
            stage4_mat_b_tile[local_row * 16u + local_col] = 0.0;
        }
        workgroupBarrier();

        let tile_limit = min(16u, n_tokens - tile);
        for (var k = 0u; k < tile_limit; k = k + 1u) {
            g_wk = g_wk
                + stage4_mat_a_tile[local_row * 16u + k]
                * stage4_mat_b_tile[k * 16u + local_col];
        }
        workgroupBarrier();
    }

    let seq_scale = 1.0 / max(1.0, f32(n_entries));
    let clip = 0.5;
    let per_idx = slot * d * d + idx;
    let raw_wk = lr * g_wk * seq_scale * grad_scale;
    if (params.grad_accum_mode == 1u) {
        AllGradients[aw_wk_base(params.d_model, params.h_slots) + per_idx] += raw_wk;
    } else {
        AllWeights[aw_wk_base(params.d_model, params.h_slots) + per_idx] =
            AllWeights[aw_wk_base(params.d_model, params.h_slots) + per_idx] * wd_factor - clamp(raw_wk, -clip, clip);
    }

    if (idx == 0u && slot == 0u) {
        debug_log[8] = 247.0;
        debug_log[44] = g_wk;
    }
}

@compute
@workgroup_size(16, 16, 1)
fn fused_attn_stage4_wv_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let col = gid.x;
    let row = gid.y;
    let slot = gid.z;
    let local_col = lid.x;
    let local_row = lid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (row >= d || col >= d || slot >= h_slots) { return; }

    let n_entries = params.batch_size * params.seq_len * h_slots;
    let n_tokens = params.batch_size * params.seq_len;
    let idx = row * d + col;
    let lr = params.lr;
    let wd_factor = 1.0 - lr * params.weight_decay;
    let grad_scale = params.grad_scale;

    var g_wv = 0.0;
    for (var tile = 0u; tile < n_tokens; tile = tile + 16u) {
        let token_a = tile + local_col;
        if (token_a < n_tokens) {
            let signal_base_t = token_signal_base(token_a, d, h_slots);
            stage4_mat_a_tile[local_row * 16u + local_col] =
                Scratch[signal_base_t + slot * d + row]
                + AllWeights[aw_hist_base(d, h_slots) + slot_anchor_base(d, h_slots) + slot * d + row]
                + Scratch[token_hist_ctx_base(token_a, d, h_slots) + slot * d + row];
        } else {
            stage4_mat_a_tile[local_row * 16u + local_col] = 0.0;
        }

        let token_b = tile + local_row;
        if (token_b < n_tokens) {
            stage4_mat_b_tile[local_row * 16u + local_col] =
                hist_ctx_buf[entry_base(token_b * h_slots + slot, d) + col];
        } else {
            stage4_mat_b_tile[local_row * 16u + local_col] = 0.0;
        }
        workgroupBarrier();

        let tile_limit = min(16u, n_tokens - tile);
        for (var k = 0u; k < tile_limit; k = k + 1u) {
            g_wv = g_wv
                + stage4_mat_a_tile[local_row * 16u + k]
                * stage4_mat_b_tile[k * 16u + local_col];
        }
        workgroupBarrier();
    }

    let seq_scale = 1.0 / max(1.0, f32(n_entries));
    let clip = 0.5;
    let per_idx = slot * d * d + idx;
    let raw_wv = lr * g_wv * seq_scale * grad_scale;
    if (params.grad_accum_mode == 1u) {
        AllGradients[aw_wv_base(params.d_model, params.h_slots) + per_idx] += raw_wv;
    } else {
        AllWeights[aw_wv_base(params.d_model, params.h_slots) + per_idx] =
            AllWeights[aw_wv_base(params.d_model, params.h_slots) + per_idx] * wd_factor - clamp(raw_wv, -clip, clip);
    }

    if (idx == 0u && slot == 0u) {
        debug_log[8] = 248.0;
        debug_log[42] = g_wv;
    }
}

@compute
@workgroup_size(64, 1, 1)
fn fused_attn_stage4_bias_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (col >= d) { return; }

    let n_entries = params.batch_size * params.seq_len * h_slots;
    let lr = params.lr;
    let wd_factor = 1.0 - lr * params.weight_decay;
    let grad_scale = params.grad_scale;
    let seq_scale = 1.0 / max(1.0, f32(n_entries));
    let clip = 0.5;
    let scale = inverseSqrt(max(1.0, f32(d)));

    var g_qbias: array<f32, 16>;
    var g_kbias: array<f32, 16>;

        for (var entry2 = 0u; entry2 < n_entries; entry2 = entry2 + 1u) {
            let t2_abs  = entry2 / h_slots;
            let qs2    = entry2 % h_slots;
            let off2   = entry_base(entry2, d);
            let sstr2  = scratch_stride(d, h_slots);
            let qbase2 = t2_abs * sstr2;
            g_qbias[qs2] = g_qbias[qs2] + qgrad_buf[off2 + col];
            for (var ks2 = 0u; ks2 < h_slots; ks2 = ks2 + 1u) {
                g_kbias[ks2] = g_kbias[ks2]
                    + scale * gscore_buf[entry2 * h_slots + ks2]
                    * Scratch[qbase2 + qs2 * d + col];
            }
        }
    for (var sb = 0u; sb < h_slots; sb = sb + 1u) {
        let raw_qb = lr * g_qbias[sb] * seq_scale * grad_scale;
        let raw_kb = lr * g_kbias[sb] * seq_scale * grad_scale;
        if (params.grad_accum_mode == 1u) {
            AllGradients[aw_wq_base(params.d_model, params.h_slots) + h_slots * d * d + sb * d + col] += raw_qb;
            AllGradients[aw_wk_base(params.d_model, params.h_slots) + h_slots * d * d + sb * d + col] += raw_kb;
        } else {
            AllWeights[aw_wq_base(params.d_model, params.h_slots) + h_slots * d * d + sb * d + col] =
                AllWeights[aw_wq_base(params.d_model, params.h_slots) + h_slots * d * d + sb * d + col] * wd_factor - clamp(raw_qb, -clip, clip);
            AllWeights[aw_wk_base(params.d_model, params.h_slots) + h_slots * d * d + sb * d + col] =
                AllWeights[aw_wk_base(params.d_model, params.h_slots) + h_slots * d * d + sb * d + col] * wd_factor - clamp(raw_kb, -clip, clip);
        }
    }

    if (col == 0u) {
        debug_log[8] = 243.0;
    }
}


@compute
@workgroup_size(64, 1, 1)
fn fused_hist_stage_prep_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let lane = lid.x;
    // 2D dispatch reconstruction: entry = wid.y * grid_stride_x + wid.x.
    let entry = wid.y * params.grid_stride_x + wid.x;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (entry >= n_entries) { return; }

    let t_abs = entry / h_slots;
    let t = t_abs % params.seq_len;
    let slot = entry % h_slots;
    let off = slot * d;
    let hist_mat = hist_mat_len(d);
    let hist_scale = hist_scale_base(d, h_slots);
    let signal_base = token_signal_base(t_abs, d, h_slots);
    let hist_out = entry_base(entry, d);
    let sample_t = select(0u, max(1u, params.seq_len / 2u), params.seq_len > 1u);
    let sample_entry = sample_t * h_slots;
    let selective = hist_selective_enabled(d, h_slots);

    var local_prev_sumsq = 0.0;
    var local_prev_h_sumsq = 0.0;
    for (var dim = lane; dim < d; dim = dim + 64u) {
        var prev_m = 0.0;
        if (t > 0u) {
            prev_m = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + dim];
        }
        local_prev_sumsq = local_prev_sumsq + prev_m * prev_m;
        var prev_h = 0.0;
        if (t > 0u) {
            prev_h = h_star[(t_abs - 1u) * h_slots * d + off + dim];
        }
        local_prev_h_sumsq = local_prev_h_sumsq + prev_h * prev_h;
    }
    hist_reduce_aux[lane] = local_prev_sumsq;
    hist_reduce_u[lane] = local_prev_h_sumsq;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            hist_reduce_aux[lane] = hist_reduce_aux[lane] + hist_reduce_aux[lane + stride];
            hist_reduce_u[lane] = hist_reduce_u[lane] + hist_reduce_u[lane + stride];
        }
        workgroupBarrier();
    }
    let prev_rms = sqrt(hist_reduce_aux[0] / max(1.0, f32(d)) + 1e-6);
    let prev_h_rms = sqrt(hist_reduce_u[0] / max(1.0, f32(d)) + 1e-6);
    let inv_prev_rms = 1.0 / max(prev_rms, 1e-6);
    if (lane == 0u) {
        gscore_buf[entry] = inv_prev_rms;
    }

    var local_u_sumsq = 0.0;
    var local_inj_sumsq = 0.0;
    for (var dim = lane; dim < d; dim = dim + 64u) {
        var prev_m = 0.0;
        if (t > 0u) {
            prev_m = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + dim] / prev_rms;
        }
        // The production historical interface is carrier-only: no additive per-slot bias.
        // Otherwise the DEQ sees "history" even when M_{t-1}=0, which breaks the intended
        // semantics and makes the branch look artificially strong.
        var u = AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_scale + off + dim] * prev_m;
        for (var j = 0u; j < d; j = j + 1u) {
            var prev_j = 0.0;
            if (t > 0u) {
                prev_j = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + j];
            }
            u = u + AllWeights[aw_hist_base(params.d_model, params.h_slots) +dim * d + j] * prev_j;
        }
        gmix_buf[hist_out + dim] = u;
        local_u_sumsq = local_u_sumsq + u * u;
        let inj = Scratch[signal_base + off + dim];
        local_inj_sumsq = local_inj_sumsq + inj * inj;
        var prev_h = 0.0;
        if (t > 0u) {
            prev_h = h_star[(t_abs - 1u) * h_slots * d + off + dim];
        }
        qgrad_buf[hist_out + dim] = prev_h / prev_h_rms;
    }
    hist_reduce_u[lane] = local_u_sumsq;
    hist_reduce_aux[lane] = local_inj_sumsq;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            hist_reduce_u[lane] = hist_reduce_u[lane] + hist_reduce_u[lane + stride];
            hist_reduce_aux[lane] = hist_reduce_aux[lane] + hist_reduce_aux[lane + stride];
        }
        workgroupBarrier();
    }

    let r_u = sqrt(hist_reduce_u[0] / max(1.0, f32(d)) + 1e-6);
    let inj_rms = sqrt(hist_reduce_aux[0] / max(1.0, f32(d)) + 1e-6);
    let tau = max(
        hist_cap_mult() * inj_rms,
        hist_cap_floor_mult() * r_u,
    );
    let scale = min(1.0, tau / max(r_u, 1e-6));
    let gate_logit = AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_gate_base(d, h_slots) + slot];
    let alpha = hist_gate_alpha(d, h_slots, gate_logit);

    var local_dot = 0.0;
    for (var dim = lane; dim < d; dim = dim + 64u) {
        let g_c = mix_buf[hist_out + dim];
        let u = gmix_buf[hist_out + dim];
        let u_tilde = scale * u;
        gmix_buf[hist_out + dim] = u_tilde;
        local_dot = local_dot + g_c * u_tilde;
    }
    hist_reduce_u[lane] = local_dot;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            hist_reduce_u[lane] = hist_reduce_u[lane] + hist_reduce_u[lane + stride];
        }
        workgroupBarrier();
    }

    let denom = max(1e-6, f32(d) * r_u * r_u);
    let clip_active = r_u > tau + 1e-6;
    for (var dim = lane; dim < d; dim = dim + 64u) {
        let g_c = mix_buf[hist_out + dim];
        let u = gmix_buf[hist_out + dim] / max(scale, 1.0e-6);
        let g_utilde = alpha * g_c;
        var g_u = g_utilde;
        if (clip_active) {
            g_u = scale * g_utilde - (scale / denom) * u * hist_reduce_u[0];
        }
        weighted_h_buf[hist_out + dim] = g_u;

        let a_base = 1.0 / (1.0 + exp(AllWeights[aw_alog_base(params.d_model, params.h_slots) +slot * d + dim]));
        var a_t = a_base;
        var delta_coeff = 0.0;
        if (selective) {
            var delta_pre = AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_delta_bias_base(d, h_slots) + dim];
            for (var j = 0u; j < d; j = j + 1u) {
                delta_pre = delta_pre
                    + AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_delta_base(d, h_slots) + slot * d * d + j * d + dim]
                    * qgrad_buf[hist_out + j];
            }
            let delta_factor = 1.0 + 0.5 * tanh(delta_pre);
            let a_core = pow(max(a_base, 1.0e-6), delta_factor);
            let a_floor = hist_alpha_min(d, h_slots);
            if (a_core < a_floor) {
                a_t = a_floor;
                delta_coeff = 0.0;
            } else {
                a_t = a_core;
                let log_a = log(max(a_base, 1.0e-6));
                let x_unit = qgrad_buf[hist_out + dim];
                let wx = hist_wx_max() * tanh(AllWeights[aw_wx_base(params.d_model, params.h_slots) +dim * d + dim]);
                let x_proj = x_unit + wx * x_unit;
                let prev_m_raw = select(
                    0.0,
                    Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + dim],
                    t > 0u
                );
                let gpre_factor = a_core * log_a * 0.5 * (1.0 - tanh(delta_pre) * tanh(delta_pre));
                delta_coeff = (prev_m_raw - x_proj) * gpre_factor;
            }
        }
        hist_ctx_buf[hist_out + dim] = a_t;
        hist_delta_buf[hist_out + dim] = delta_coeff;
    }
    workgroupBarrier();

    if (entry == params.seq_len * h_slots - 1u && lane == 0u) {
        debug_log[53] = 0.0;
        debug_log[54] = 0.0;
    }
    var local_prev_dot = 0.0;
    for (var dim = lane; dim < d; dim = dim + 64u) {
        var g_m_unit = AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_scale + off + dim]
            * weighted_h_buf[hist_out + dim];
        for (var row = 0u; row < d; row = row + 1u) {
            g_m_unit = g_m_unit
                + AllWeights[aw_hist_base(params.d_model, params.h_slots) +row * d + dim]
                * weighted_h_buf[hist_out + row];
        }
        gmix_buf[hist_out + dim] = g_m_unit;
    }
    workgroupBarrier();
    for (var dim = lane; dim < d; dim = dim + 64u) {
        var prev_m = 0.0;
        if (t > 0u) {
            prev_m = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + dim];
        }
        local_prev_dot = local_prev_dot + prev_m * gmix_buf[hist_out + dim];
    }
    hist_reduce_aux[lane] = local_prev_dot;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            hist_reduce_aux[lane] = hist_reduce_aux[lane] + hist_reduce_aux[lane + stride];
        }
        workgroupBarrier();
    }
    let prev_denom = max(1e-6, f32(d) * prev_rms * prev_rms * prev_rms);
    for (var dim = lane; dim < d; dim = dim + 64u) {
        var prev_m = 0.0;
        if (t > 0u) {
            prev_m = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + dim];
        }
        let g_m_unit = gmix_buf[hist_out + dim];
        gmix_buf[hist_out + dim] = (g_m_unit / prev_rms)
            - (prev_m * hist_reduce_aux[0] / prev_denom);
    }
    workgroupBarrier();
    if (entry == sample_entry && lane == 0u) {
        var sum_gc = 0.0;
        var sum_gu = 0.0;
        var sum_gm = 0.0;
        for (var dim = 0u; dim < d; dim = dim + 1u) {
            sum_gc = sum_gc + abs(mix_buf[hist_out + dim]);
            sum_gu = sum_gu + abs(weighted_h_buf[hist_out + dim]);
            sum_gm = sum_gm + abs(gmix_buf[hist_out + dim]);
        }
        let inv_d = 1.0 / max(1.0, f32(d));
        debug_log[55] = sum_gc * inv_d;
        debug_log[56] = sum_gu * inv_d;
        debug_log[57] = sum_gm * inv_d;
        if (t == sample_t) {
            debug_log[60] = sum_gc * inv_d;
            debug_log[61] = sum_gu * inv_d;
            debug_log[62] = sum_gm * inv_d;
        }
    }
    if (entry == params.seq_len * h_slots - 1u && lane == 0u) {
        var sum_gu = 0.0;
        var sum_gm = 0.0;
        for (var dim = 0u; dim < d; dim = dim + 1u) {
            sum_gu = sum_gu + abs(weighted_h_buf[hist_out + dim]);
            sum_gm = sum_gm + abs(gmix_buf[hist_out + dim]);
        }
        let inv_d = 1.0 / max(1.0, f32(d));
        debug_log[53] = sum_gu * inv_d;
        debug_log[54] = sum_gm * inv_d;
    }
}

@compute
@workgroup_size(16, 16, 1)
fn fused_hist_stage_mat_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (row >= d || col >= d) { return; }

    let n_entries = params.batch_size * params.seq_len * h_slots;
    let n_tokens = params.batch_size * max(1u, params.seq_len - 1u);
    var grad = 0.0;
    for (var entry = 0u; entry < n_entries; entry = entry + 1u) {
        let t_abs = entry / h_slots;
        let t = t_abs % params.seq_len;
        if (t == 0u) { continue; }
        let slot = entry % h_slots;
        let off = slot * d;
        let inv_prev_rms = gscore_buf[entry];
        let prev_val = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + col] * inv_prev_rms;
        grad = grad + weighted_h_buf[entry_base(entry, d) + row] * prev_val;
    }

    // Normalize by the effective token count to keep updates invariant to seq_len.
    // Without this, gradients scale with sequence length and can destabilize long runs.
    let idx = row * d + col;
    let clip = 0.5;
    let token_norm = 1.0 / max(1.0, f32(n_tokens));
    let raw_step = params.lr * (grad * token_norm) * params.grad_scale;
    if (params.grad_accum_mode == 1u) {
        AllGradients[aw_hist_base(params.d_model, params.h_slots) +idx] += raw_step;
    } else {
        let step = clamp(raw_step, -clip, clip);
        let before = AllWeights[aw_hist_base(params.d_model, params.h_slots) +idx];
        let after = before - step;
        AllWeights[aw_hist_base(params.d_model, params.h_slots) +idx] = after;
        if (row == 0u && col == 0u) {
            debug_log[64] = grad;
            debug_log[65] = step;
            debug_log[66] = before;
            debug_log[67] = after;
            debug_log[80] = params.lr;
            debug_log[81] = params.grad_scale;
            debug_log[82] = params.weight_decay;
        }
    }
}

@compute
@workgroup_size(16, 16, 1)
fn fused_hist_stage_scale_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let slot = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (dim >= d || slot >= h_slots) { return; }

    var grad = 0.0;
    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        for (var t = 1u; t < params.seq_len; t = t + 1u) {
            let t_abs = b * params.seq_len + t;
            let entry = t_abs * h_slots + slot;
            let off = slot * d;
            let inv_prev_rms = gscore_buf[entry];
            let prev_val = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + dim] * inv_prev_rms;
            grad = grad + weighted_h_buf[entry_base(entry, d) + dim] * prev_val;
        }
    }

    let idx = hist_scale_base(d, h_slots) + slot * d + dim;
    let clip = 0.5;
    let n_tokens = params.batch_size * (max(1u, params.seq_len) - 1u);
    let token_norm = 1.0 / max(1.0, f32(n_tokens));
    let raw_step = params.lr * (grad * token_norm) * params.grad_scale;
    if (params.grad_accum_mode == 1u) {
        AllGradients[aw_hist_base(params.d_model, params.h_slots) +idx] += raw_step;
    } else {
        AllWeights[aw_hist_base(params.d_model, params.h_slots) +idx] = AllWeights[aw_hist_base(params.d_model, params.h_slots) +idx] - clamp(raw_step, -clip, clip);
    }
}

@compute
@workgroup_size(16, 16, 1)
fn fused_hist_stage_bias_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let slot = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (dim >= d || slot >= h_slots) { return; }

    var grad = 0.0;
    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        for (var t = 0u; t < params.seq_len; t = t + 1u) {
            let t_abs = b * params.seq_len + t;
            let entry = t_abs * h_slots + slot;
            grad = grad + weighted_h_buf[entry_base(entry, d) + dim];
        }
    }

    grad = grad / max(1.0, f32(params.batch_size * params.seq_len));
    let idx = hist_bias_base(d, h_slots) + slot * d + dim;
    let clip = 0.5;
    let raw_step = params.lr * grad * params.grad_scale;
    if (params.grad_accum_mode == 1u) {
        AllGradients[aw_hist_base(params.d_model, params.h_slots) +idx] += raw_step;
    } else {
        AllWeights[aw_hist_base(params.d_model, params.h_slots) +idx] = AllWeights[aw_hist_base(params.d_model, params.h_slots) +idx] - clamp(raw_step, -clip, clip);
    }
}

@compute
@workgroup_size(64, 1, 1)
fn fused_hist_stage_gate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (slot >= h_slots) { return; }

    var grad = 0.0;
    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        for (var t = 0u; t < params.seq_len; t = t + 1u) {
            let t_abs = b * params.seq_len + t;
            let entry = t_abs * h_slots + slot;
            let off = entry_base(entry, d);
            for (var dim = 0u; dim < d; dim = dim + 1u) {
                grad = grad + mix_buf[off + dim] * gmix_buf[off + dim];
            }
        }
    }

    let idx = hist_gate_base(d, h_slots) + slot;
    let sigma = hist_gate_sigma(AllWeights[aw_hist_base(params.d_model, params.h_slots) +idx]);
    let token_norm = 1.0 / max(1.0, f32(params.batch_size * params.seq_len) * f32(d));
    let gate_grad = 0.20 * sigma * (1.0 - sigma) * grad * token_norm;
    let clip = 0.5;
    let raw_step = params.lr * gate_grad * params.grad_scale;
    if (params.grad_accum_mode == 1u) {
        AllGradients[aw_hist_base(params.d_model, params.h_slots) +idx] += raw_step;
    } else {
        let step = clamp(raw_step, -clip, clip);
        let before = AllWeights[aw_hist_base(params.d_model, params.h_slots) +idx];
        let after = before - step;
        AllWeights[aw_hist_base(params.d_model, params.h_slots) +idx] = after;
        if (slot == 0u) {
            debug_log[68] = gate_grad;
            debug_log[69] = step;
            debug_log[70] = before;
            debug_log[71] = after;
        }
    }
}

@compute
@workgroup_size(64, 1, 1)
fn fused_hist_stage_mprev_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let lane = lid.x;
    let entry = wid.x;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (entry >= n_entries) { return; }

    let t_abs = entry / h_slots;
    let t = t_abs % params.seq_len;
    let slot = entry % h_slots;
    let off = slot * d;
    let hist_out = entry_base(entry, d);
    let hist_scale = hist_scale_base(d, h_slots);
    let sample_t = select(0u, max(1u, params.seq_len / 2u), params.seq_len > 1u);
    let sample_entry = sample_t * h_slots;
    if (entry == params.seq_len * h_slots - 1u && lane == 0u) {
        debug_log[53] = 0.0;
        debug_log[54] = 0.0;
    }

    var local_prev_sumsq = 0.0;
    for (var dim = lane; dim < d; dim = dim + 64u) {
        var prev_m = 0.0;
        if (t > 0u) {
            prev_m = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + dim];
        }
        local_prev_sumsq = local_prev_sumsq + prev_m * prev_m;
    }
    hist_reduce_aux[lane] = local_prev_sumsq;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            hist_reduce_aux[lane] = hist_reduce_aux[lane] + hist_reduce_aux[lane + stride];
        }
        workgroupBarrier();
    }
    let prev_rms = sqrt(hist_reduce_aux[0] / max(1.0, f32(d)) + 1e-6);

    for (var dim = lane; dim < d; dim = dim + 64u) {
        var g_m_unit = AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_scale + off + dim] * weighted_h_buf[hist_out + dim];
        for (var row = 0u; row < d; row = row + 1u) {
            g_m_unit = g_m_unit + AllWeights[aw_hist_base(params.d_model, params.h_slots) +row * d + dim] * weighted_h_buf[hist_out + row];
        }
        gmix_buf[hist_out + dim] = g_m_unit;
    }
    workgroupBarrier();
    var local_dot = 0.0;
    for (var dim = lane; dim < d; dim = dim + 64u) {
        var prev_m = 0.0;
        if (t > 0u) {
            prev_m = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + dim];
        }
        local_dot = local_dot + prev_m * gmix_buf[hist_out + dim];
    }
    hist_reduce_aux[lane] = local_dot;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            hist_reduce_aux[lane] = hist_reduce_aux[lane] + hist_reduce_aux[lane + stride];
        }
        workgroupBarrier();
    }
    let denom = max(1e-6, f32(d) * prev_rms * prev_rms * prev_rms);
    for (var dim = lane; dim < d; dim = dim + 64u) {
        var prev_m = 0.0;
        if (t > 0u) {
            prev_m = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + dim];
        }
        let g_m_unit = gmix_buf[hist_out + dim];
        gmix_buf[hist_out + dim] = (g_m_unit / prev_rms)
            - (prev_m * hist_reduce_aux[0] / denom);
    }
    workgroupBarrier();
    if (entry == params.seq_len * h_slots - 1u && lane == 0u) {
        var sum_gu = 0.0;
        var sum_gm = 0.0;
        for (var dim = 0u; dim < d; dim = dim + 1u) {
            sum_gu = sum_gu + abs(weighted_h_buf[hist_out + dim]);
            sum_gm = sum_gm + abs(gmix_buf[hist_out + dim]);
        }
        let inv_d = 1.0 / max(1.0, f32(d));
        debug_log[53] = sum_gu * inv_d;
        debug_log[54] = sum_gm * inv_d;
    }
        if (entry == sample_entry && lane == 0u) {
            var sum_gm_mid = 0.0;
            for (var dim = 0u; dim < d; dim = dim + 1u) {
                sum_gm_mid = sum_gm_mid + abs(gmix_buf[hist_out + dim]);
            }
            debug_log[57] = sum_gm_mid / max(1.0, f32(d));
            if (t == sample_t) {
                debug_log[62] = sum_gm_mid / max(1.0, f32(d));
            }
        }
}

@compute
@workgroup_size(64, 1, 1)
fn fused_hist_stage_tbptt_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let lane = lid.x;
    // hist_tbptt dispatches batch_size * h_slots workgroups.
    // Use 2D grid reconstruction for safety when this product exceeds 65535.
    let wg_idx = wid.y * params.grid_stride_x + wid.x;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let batch_idx = wg_idx / h_slots;
    let slot = wg_idx % h_slots;
    if (slot >= h_slots || batch_idx >= params.batch_size || params.seq_len <= 1u) { return; }
    let selective = hist_selective_enabled(d, h_slots);
    let sample_t = max(1u, params.seq_len / 2u);
    if (slot == 0u && batch_idx == 0u && lane == 0u) {
        debug_log[45] = select(0.0, 1.0, selective);
        debug_log[46] = 0.0;
        debug_log[47] = 0.0;
        debug_log[48] = 0.0;
        debug_log[49] = 0.0;
        debug_log[50] = 0.0;
            debug_log[51] = 0.0;
            debug_log[52] = 0.0;
            debug_log[58] = 0.0;
            debug_log[59] = 0.0;
    }

    let off = slot * d;
    let carry_off = (batch_idx * h_slots + slot) * d;
    for (var dim = lane; dim < d; dim = dim + 64u) {
        hist_carry_vec[dim] = tbptt_carry_buf[carry_off + dim];
    }
    workgroupBarrier();

    var t = params.seq_len;
    loop {
        if (t <= 1u) { break; }
        t = t - 1u;
        let t_abs = batch_idx * params.seq_len + t;

        let entry = t_abs * h_slots + slot;
        let hist_out = entry_base(entry, d);

        for (var dim = lane; dim < d; dim = dim + 64u) {
            let total_gm = gmix_buf[hist_out + dim] + hist_carry_vec[dim];
            hist_total_vec[dim] = total_gm;
            gmix_buf[hist_out + dim] = total_gm;
        }
        workgroupBarrier();

        for (var dim = lane; dim < d; dim = dim + 64u) {
            var g_m_inner = hist_total_vec[dim];
            for (var out = 0u; out < d; out = out + 1u) {
                g_m_inner = g_m_inner + AllWeights[aw_wout_base(params.d_model, params.h_slots) +dim * d + out] * hist_total_vec[out];
            }
            hist_ginner_vec[dim] = g_m_inner;
        }
        workgroupBarrier();

        let prev_h_base = (t_abs - 1u) * h_slots * d + off;
        var local_h_sumsq = 0.0;
        for (var dim = lane; dim < d; dim = dim + 64u) {
            let h_val = h_star[prev_h_base + dim];
            local_h_sumsq = local_h_sumsq + h_val * h_val;
        }
        hist_reduce_u[lane] = local_h_sumsq;
        workgroupBarrier();
        for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
            if (lane < stride) {
                hist_reduce_u[lane] = hist_reduce_u[lane] + hist_reduce_u[lane + stride];
            }
            workgroupBarrier();
        }
        let h_rms = sqrt(hist_reduce_u[0] / max(1.0, f32(d)) + 1e-6);

        for (var dim = lane; dim < d; dim = dim + 64u) {
            let a_t = hist_ctx_buf[hist_out + dim];
            let g_pre = hist_ginner_vec[dim] * hist_delta_buf[hist_out + dim];
            hist_delta_buf[hist_out + dim] = g_pre;
            hist_gx_vec[dim] = (1.0 - a_t) * hist_ginner_vec[dim];
            weighted_h_buf[hist_out + dim] = hist_gx_vec[dim];
            var g_h_unit = hist_gx_vec[dim];
            let wx = hist_wx_max() * tanh(AllWeights[aw_wx_base(params.d_model, params.h_slots) +dim * d + dim]);
            g_h_unit = g_h_unit + wx * hist_gx_vec[dim];
            if (selective) {
                for (var out = 0u; out < d; out = out + 1u) {
                    g_h_unit = g_h_unit
                        + AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_delta_base(d, h_slots) + slot * d * d + dim * d + out]
                        * hist_delta_buf[hist_out + out];
                }
            }
            hist_total_vec[dim] = g_h_unit;
        }

        let dst_entry = (t_abs - 1u) * h_slots + slot;
        let dst_off = entry_base(dst_entry, d);
        var local_dot = 0.0;
        for (var dim = lane; dim < d; dim = dim + 64u) {
            local_dot = local_dot + hist_total_vec[dim] * h_star[prev_h_base + dim];
        }
        hist_reduce_u[lane] = local_dot;
        workgroupBarrier();
        for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
            if (lane < stride) {
                hist_reduce_u[lane] = hist_reduce_u[lane] + hist_reduce_u[lane + stride];
            }
            workgroupBarrier();
        }
        let denom = max(1e-6, f32(d) * h_rms * h_rms * h_rms);
        let dot_gh_h = hist_reduce_u[0];
        for (var dim = lane; dim < d; dim = dim + 64u) {
            let g_h_unit = hist_total_vec[dim];
            let a_t = hist_ctx_buf[hist_out + dim];
            hist_carry_vec[dim] = a_t * hist_ginner_vec[dim];
        }
        workgroupBarrier();

        if (selective && slot == 0u && batch_idx == 0u && lane == 0u && t == params.seq_len - 1u) {
            var sum_gap = 0.0;
            var sum_ginner = 0.0;
            var sum_dpre = 0.0;
            var sum_ga = 0.0;
            var sum_gpre = 0.0;
            var sum_at = 0.0;
            var sum_loga = 0.0;
            for (var dim = 0u; dim < d; dim = dim + 1u) {
                let a_base = 1.0 / (1.0 + exp(AllWeights[aw_alog_base(params.d_model, params.h_slots) +slot * d + dim]));
                let log_a = log(max(a_base, 1.0e-6));
                var x_proj = qgrad_buf[hist_out + dim];
                let wx = hist_wx_max() * tanh(AllWeights[aw_wx_base(params.d_model, params.h_slots) +dim * d + dim]);
                x_proj = x_proj + wx * qgrad_buf[hist_out + dim];
                let prev_m = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + dim];
                var delta_pre = AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_delta_bias_base(d, h_slots) + dim];
                for (var j = 0u; j < d; j = j + 1u) {
                    delta_pre = delta_pre
                        + AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_delta_base(d, h_slots) + slot * d * d + j * d + dim]
                        * qgrad_buf[hist_out + j];
                }
                let delta = log(1.0 + exp(delta_pre));
                let a_core = exp(delta * log_a);
                let a_floor = hist_alpha_min(d, h_slots);
                let a_t = max(a_floor, a_core);
                let g_a = (prev_m - x_proj) * hist_ginner_vec[dim];
                let g_delta = select(0.0, g_a * a_core * log_a, a_core >= a_floor);
                let g_pre = g_delta * (1.0 / (1.0 + exp(-delta_pre)));
                sum_gap = sum_gap + abs(prev_m - x_proj);
                sum_ginner = sum_ginner + abs(hist_ginner_vec[dim]);
                sum_dpre = sum_dpre + abs(delta_pre);
                sum_ga = sum_ga + abs(g_a);
                sum_gpre = sum_gpre + abs(g_pre);
                sum_at = sum_at + a_t;
                sum_loga = sum_loga + abs(log_a);
            }
            let inv_d = 1.0 / max(1.0, f32(d));
            debug_log[46] = sum_gap * inv_d;
            debug_log[47] = sum_ginner * inv_d;
            debug_log[48] = sum_dpre * inv_d;
            debug_log[49] = sum_ga * inv_d;
            debug_log[50] = sum_gpre * inv_d;
            debug_log[51] = sum_at * inv_d;
            debug_log[52] = sum_loga * inv_d;
        }
        if (selective && slot == 0u && batch_idx == 0u && lane == 0u && t == sample_t) {
            var sum_ginner_mid = 0.0;
            var sum_gpre_mid = 0.0;
            for (var dim = 0u; dim < d; dim = dim + 1u) {
                sum_ginner_mid = sum_ginner_mid + abs(hist_ginner_vec[dim]);
                sum_gpre_mid = sum_gpre_mid + abs(hist_delta_buf[hist_out + dim]);
            }
            let inv_d = 1.0 / max(1.0, f32(d));
            debug_log[58] = sum_ginner_mid * inv_d;
            debug_log[59] = sum_gpre_mid * inv_d;
            debug_log[63] = 0.0;
        }
    }
    // Write cross-sequence TBPTT carry: hist_carry_vec = ∂L/∂M_0 for this sequence.
    for (var dim = lane; dim < d; dim = dim + 64u) {
        tbptt_carry_buf[carry_off + dim] = hist_carry_vec[dim];
    }
    workgroupBarrier();
}

@compute
@workgroup_size(64, 1, 1)
fn fused_hist_stage_xprep_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let lane = lid.x;
    let entry = wid.x;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (entry >= n_entries) { return; }

    let t_abs = entry / h_slots;
    let t = t_abs % params.seq_len;
    let hist_out = entry_base(entry, d);
    if (t == 0u) {
        for (var dim = lane; dim < d; dim = dim + 64u) {
            weighted_h_buf[hist_out + dim] = 0.0;
            qgrad_buf[hist_out + dim] = 0.0;
        }
        return;
    }

    let slot = entry % h_slots;
    let off = slot * d;
    let prev_h_base = (t_abs - 1u) * h_slots * d + off;
    var local_h_sumsq = 0.0;
    for (var dim = lane; dim < d; dim = dim + 64u) {
        let h_val = h_star[prev_h_base + dim];
        local_h_sumsq = local_h_sumsq + h_val * h_val;
    }
    hist_reduce_u[lane] = local_h_sumsq;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            hist_reduce_u[lane] = hist_reduce_u[lane] + hist_reduce_u[lane + stride];
        }
        workgroupBarrier();
    }
    let h_rms = sqrt(hist_reduce_u[0] / max(1.0, f32(d)) + 1e-6);

        for (var dim = lane; dim < d; dim = dim + 64u) {
            var g_x = gmix_buf[hist_out + dim];
            for (var out = 0u; out < d; out = out + 1u) {
                g_x = g_x + AllWeights[aw_wout_base(params.d_model, params.h_slots) +dim * d + out] * gmix_buf[hist_out + out];
            }
        let a = 1.0 / (1.0 + exp(AllWeights[aw_alog_base(params.d_model, params.h_slots) +slot * d + dim]));
        weighted_h_buf[hist_out + dim] = (1.0 - a) * g_x;
        qgrad_buf[hist_out + dim] = h_star[prev_h_base + dim] / h_rms;
    }
}

@compute
@workgroup_size(64, 1, 1)
fn fused_hist_stage_hrhs_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let lane = lid.x;
    let entry = wid.x;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (entry >= n_entries) { return; }

    let t_abs = entry / h_slots;
    let t = t_abs % params.seq_len;
    if (t == 0u) { return; }
    let slot = entry % h_slots;
    let dst_entry = (t_abs - 1u) * h_slots + slot;
    let src_off = entry_base(entry, d);
    let dst_off = entry_base(dst_entry, d);
    let prev_h_base = (t_abs - 1u) * h_slots * d + slot * d;

    var local_h_sumsq = 0.0;
    for (var dim = lane; dim < d; dim = dim + 64u) {
        let h_val = h_star[prev_h_base + dim];
        local_h_sumsq = local_h_sumsq + h_val * h_val;
    }
    hist_reduce_u[lane] = local_h_sumsq;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            hist_reduce_u[lane] = hist_reduce_u[lane] + hist_reduce_u[lane + stride];
        }
        workgroupBarrier();
    }
    let h_rms = sqrt(hist_reduce_u[0] / max(1.0, f32(d)) + 1e-6);

    var local_dot = 0.0;
    for (var dim = lane; dim < d; dim = dim + 64u) {
            var g_h_unit = weighted_h_buf[src_off + dim];
            let wx = hist_wx_max() * tanh(AllWeights[aw_wx_base(params.d_model, params.h_slots) +dim * d + dim]);
            g_h_unit = g_h_unit + wx * weighted_h_buf[src_off + dim];
        gmix_buf[dst_off + dim] = g_h_unit;
        local_dot = local_dot + g_h_unit * h_star[prev_h_base + dim];
    }
    hist_reduce_u[lane] = local_dot;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            hist_reduce_u[lane] = hist_reduce_u[lane] + hist_reduce_u[lane + stride];
        }
        workgroupBarrier();
    }

    let denom = max(1e-6, f32(d) * h_rms * h_rms * h_rms);
    let dot_gh_h = hist_reduce_u[0];
    for (var dim = lane; dim < d; dim = dim + 64u) {
        let h_val = h_star[prev_h_base + dim];
        let g_h_unit = gmix_buf[dst_off + dim];
        hist_ctx_buf[dst_off + dim] = (g_h_unit / h_rms)
            - (h_val * dot_gh_h / denom);
    }
}

@compute
@workgroup_size(16, 16, 1)
fn fused_hist_stage_wout_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x; // output dim in M-space
    let row = gid.y; // input dim from temporal inner state
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (row >= d || col >= d) { return; }

    let n_temporal_tokens = params.batch_size * max(1u, params.seq_len - 1u);
    let n_temporal_entries = max(1.0, f32(n_temporal_tokens * h_slots));
    var grad = 0.0;
    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        for (var t = 1u; t < params.seq_len; t = t + 1u) {
            let t_abs = b * params.seq_len + t;
            for (var slot = 0u; slot < h_slots; slot = slot + 1u) {
                let entry = t_abs * h_slots + slot;
                let off = slot * d;
                let prev_inner = Scratch[token_minner_base(t_abs - 1u, d, h_slots) + off + row];
                grad = grad + prev_inner * gmix_buf[entry_base(entry, d) + col];
            }
        }
    }

    // W_out is shared across slots, so the unbiased scale is over all
    // token-slot contributions, not only tokens.
    grad = grad / n_temporal_entries;
    let idx = row * d + col;
    let clip = 0.5;
    let wd_factor = 1.0 - params.lr * params.weight_decay;
    let raw_step = params.lr * grad;
    if (params.grad_accum_mode == 1u) {
        AllGradients[aw_wout_base(params.d_model, params.h_slots) +idx] += raw_step;
    } else {
        let step = clamp(raw_step, -clip, clip);
        let before = AllWeights[aw_wout_base(params.d_model, params.h_slots) +idx];
        let after = before * wd_factor - step;
        AllWeights[aw_wout_base(params.d_model, params.h_slots) +idx] = after;
        if (row == 0u && col == 0u) {
            debug_log[83] = grad;
            debug_log[84] = step;
            debug_log[85] = before;
            debug_log[86] = after;
            debug_log[87] = params.lr;
            debug_log[88] = params.grad_scale;
            debug_log[89] = params.weight_decay;
        }
    }
}

@compute
@workgroup_size(16, 16, 1)
fn fused_hist_stage_wx_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x; // output dim in x_proj space
    let row = gid.y; // input dim from h_star
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (row >= d || col >= d) { return; }

    let n_temporal_tokens = params.batch_size * max(1u, params.seq_len - 1u);
    let n_temporal_entries = max(1.0, f32(n_temporal_tokens * h_slots));
    var grad = 0.0;
    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        for (var t = 1u; t < params.seq_len; t = t + 1u) {
            let t_abs = b * params.seq_len + t;
            for (var slot = 0u; slot < h_slots; slot = slot + 1u) {
                let entry = t_abs * h_slots + slot;
                let prev_h_unit = qgrad_buf[entry_base(entry, d) + row];
                grad = grad + prev_h_unit * weighted_h_buf[entry_base(entry, d) + col];
            }
        }
    }

    // W_x is shared across slots, so the unbiased scale is over all
    // token-slot contributions, not only tokens.
    grad = grad / n_temporal_entries;
    let idx = row * d + col;
    let clip = 0.5;
    let wd_factor = 1.0 - params.lr * params.weight_decay;
    if (row == col) {
        let wx_raw = AllWeights[aw_wx_base(params.d_model, params.h_slots) +idx];
        let wx_scale = hist_wx_max();
        let wx_tanh = tanh(wx_raw);
        let wx_grad = grad * wx_scale * (1.0 - wx_tanh * wx_tanh);
        let raw_step = params.lr * wx_grad;
        if (params.grad_accum_mode == 1u) {
            AllGradients[aw_wx_base(params.d_model, params.h_slots) +idx] += raw_step;
        } else {
            AllWeights[aw_wx_base(params.d_model, params.h_slots) +idx] = wx_raw * wd_factor - clamp(raw_step, -clip, clip);
        }
    } else {
        if (params.grad_accum_mode != 1u) {
            AllWeights[aw_wx_base(params.d_model, params.h_slots) +idx] = 0.0;
        }
    }
}

@compute
@workgroup_size(16, 16, 1)
fn fused_hist_stage_wdelta_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x; // output dim in delta_pre space
    let row = gid.y; // input dim from h_unit
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (col >= d || row >= d || !hist_selective_enabled(d, h_slots)) { return; }

    let n_temporal_tokens = params.batch_size * max(1u, params.seq_len - 1u);
    let n_norm = max(1.0, f32(n_temporal_tokens));

    // Per-slot W_delta: each slot accumulates its own gradient independently.
    for (var slot = 0u; slot < h_slots; slot = slot + 1u) {
        var grad = 0.0;
        for (var b = 0u; b < params.batch_size; b = b + 1u) {
            for (var t = 1u; t < params.seq_len; t = t + 1u) {
                let t_abs = b * params.seq_len + t;
                let entry = t_abs * h_slots + slot;
                grad = grad
                    + qgrad_buf[entry_base(entry, d) + row]
                    * hist_delta_buf[entry_base(entry, d) + col];
            }
        }

        grad = grad / n_norm;
        let idx = hist_delta_base(d, h_slots) + slot * d * d + row * d + col;
        let clip = 0.5;
        let raw_step = params.lr * grad;
        if (params.grad_accum_mode == 1u) {
            AllGradients[aw_hist_base(params.d_model, params.h_slots) +idx] += raw_step;
        } else {
            let step = clamp(raw_step, -clip, clip);
            let before = AllWeights[aw_hist_base(params.d_model, params.h_slots) +idx];
            let after = before - step;
            AllWeights[aw_hist_base(params.d_model, params.h_slots) +idx] = after;
            if (slot == 0u && row == 0u && col == 0u) {
                debug_log[72] = grad;
                debug_log[73] = step;
                debug_log[74] = before;
                debug_log[75] = after;
            }
        }
    }
}

@compute
@workgroup_size(64, 1, 1)
fn fused_hist_stage_bdelta_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (dim >= d || !hist_selective_enabled(d, h_slots)) { return; }

    let n_temporal_tokens = params.batch_size * max(1u, params.seq_len - 1u);
    let n_temporal_entries = max(1.0, f32(n_temporal_tokens * h_slots));
    var grad = 0.0;
    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        for (var t = 1u; t < params.seq_len; t = t + 1u) {
            let t_abs = b * params.seq_len + t;
            for (var slot = 0u; slot < h_slots; slot = slot + 1u) {
                let entry = t_abs * h_slots + slot;
                grad = grad + hist_delta_buf[entry_base(entry, d) + dim];
            }
        }
    }

    // Same rationale as W_Δ: keep the accumulated selective signal intact.
    let idx = hist_delta_bias_base(d, h_slots) + dim;
    let clip = 0.5;
    let raw_step = params.lr * grad;
    if (params.grad_accum_mode == 1u) {
        AllGradients[aw_hist_base(params.d_model, params.h_slots) +idx] += raw_step;
    } else {
        let step = clamp(raw_step, -clip, clip);
        let before = AllWeights[aw_hist_base(params.d_model, params.h_slots) +idx];
        let after = before - step;
        AllWeights[aw_hist_base(params.d_model, params.h_slots) +idx] = after;
        if (dim == 0u) {
            debug_log[76] = grad;
            debug_log[77] = step;
            debug_log[78] = before;
            debug_log[79] = after;
        }
    }
}

@compute
@workgroup_size(64, 1, 1)
fn fused_hist_stage_alog_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (dim >= d) { return; }

    let n_temporal_tokens = params.batch_size * max(1u, params.seq_len - 1u);
    let n_norm = max(1.0, f32(n_temporal_tokens));
    let selective = hist_selective_enabled(d, h_slots);

    // Per-slot A_log: each slot accumulates its own gradient independently.
    for (var slot = 0u; slot < h_slots; slot = slot + 1u) {
        var grad = 0.0;
        let off = slot * d;
        for (var b = 0u; b < params.batch_size; b = b + 1u) {
            for (var t = 1u; t < params.seq_len; t = t + 1u) {
                let t_abs = b * params.seq_len + t;
                let entry = t_abs * h_slots + slot;
                let weighted_h = weighted_h_buf[entry_base(entry, d) + dim];
                var x_proj = qgrad_buf[entry_base(entry, d) + dim];
                let wx = hist_wx_max() * tanh(AllWeights[aw_wx_base(params.d_model, params.h_slots) +dim * d + dim]);
                x_proj = x_proj + wx * qgrad_buf[entry_base(entry, d) + dim];

                var m_prev = 0.0;
                if (t > 1u) {
                    m_prev = Scratch[token_mamba_base(t_abs - 2u, d, h_slots) + off + dim];
                }

                let a_base = 1.0 / (1.0 + exp(AllWeights[aw_alog_base(params.d_model, params.h_slots) +slot * d + dim]));
                if (selective) {
                    let a_t = hist_ctx_buf[entry_base(entry, d) + dim];
                    let a_floor = hist_alpha_min(d, h_slots);
                    if (a_t > a_floor + 1.0e-6) {
                        let log_a = log(max(a_base, 1.0e-6));
                        let delta_factor = log(max(a_t, 1.0e-6)) / min(log_a, -1.0e-6);
                        let g_m_inner = weighted_h / max(1.0 - a_t, 1.0e-6);
                        grad = grad - a_t
                            * (delta_factor * log_a)
                            * (1.0 - a_base) * (m_prev - x_proj) * g_m_inner;
                    }
                } else {
                    grad = grad - a_base * (m_prev - x_proj) * weighted_h;
                }
            }
        }

        grad = grad / n_norm;
        let clip = 0.5;
        // A_log gets 100× higher LR so timescales can adapt quickly.
        let raw_step = params.lr * 100.0 * grad;
        let idx_alog = aw_alog_base(params.d_model, params.h_slots) + slot * d + dim;
        if (params.grad_accum_mode == 1u) {
            AllGradients[idx_alog] += raw_step;
        } else {
            let updated = AllWeights[idx_alog] - clamp(raw_step, -clip, clip);
            // Clamp A_log to [-1.0, 9.0]: keeps a_actual = sigmoid(-A_log) in [0.0001, 0.73].
            AllWeights[idx_alog] = clamp(updated, -1.0, 9.0);
        }
    }
}

// ---- Gradient Apply ----
// Applies accumulated gradients from AllGradients to AllWeights.
// Called once after n_accum accumulation passes (mode=1).
// Weight decay is applied to main weights (indices < aw_hist_base); not to hist params.
@compute
@workgroup_size(256, 1, 1)
fn apply_grad_update_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Reconstruct linear workgroup index from 2D grid (supports n_workgroups > 65535).
    // When n_workgroups <= 65535: grid_stride_x == n_workgroups, gid.y == 0, reduces to gid.x.
    let wg_idx = gid.y * params.grid_stride_x + gid.x;
    let i = wg_idx;
    if (i >= params.n_total_weights) { return; }
    let d = params.d_model;
    let h = params.h_slots;
    let clip = 0.5;
    let hist_start = aw_hist_base(d, h);
    // Main weights (W_q..NormScale) get weight decay; hist params do not.
    let wd_mult = select(1.0, 1.0 - params.lr * params.weight_decay, i < hist_start);
    let avg_step = AllGradients[i] / f32(params.n_accum);
    AllWeights[i] = AllWeights[i] * wd_mult - clamp(avg_step, -clip, clip);
    AllGradients[i] = 0.0;
}

// W_gate_hist gradient: ∂L/∂W_gate_hist[s,j] = Σ_t [ inner_dot(t,s) * h_star[t,s,j] ] / sqrt(d)
// where inner_dot(t,s) = dot(weighted_h[t,s,:], m_inner[t,s,:])  (sech²≈1 approximation at init)
@compute
@workgroup_size(16, 16, 1)
fn fused_hist_stage_wgate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;   // W_gate_hist dim j
    let slot = gid.y;  // slot s
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (col >= d || slot >= h_slots) { return; }

    let off = slot * d;
    var grad = 0.0;

    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        for (var t = 0u; t < params.seq_len; t = t + 1u) {
            let t_abs = b * params.seq_len + t;
            let entry = t_abs * h_slots + slot;

            // inner_dot = dot(weighted_h[t,s,:], m_inner[t,s,:])
            var inner_dot = 0.0;
            for (var dim = 0u; dim < d; dim = dim + 1u) {
                let wh = weighted_h_buf[entry_base(entry, d) + dim];
                let mi = Scratch[token_minner_base(t_abs, d, h_slots) + off + dim];
                inner_dot = inner_dot + wh * mi;
            }

            // h_star[t,s,j] at fixed point
            let h_star_val = h_star[t_abs * h_slots * d + off + col];
            grad = grad + inner_dot * h_star_val;
        }
    }

    let n_tokens = max(1u, params.batch_size * params.seq_len);
    let token_norm = 1.0 / max(1.0, f32(n_tokens));
    let idx = hist_gate_query_base(d, h_slots) + off + col;
    let clip = 0.5;
    let raw_step = params.lr * grad * inverseSqrt(f32(d)) * token_norm * params.grad_scale;
    if (params.grad_accum_mode == 1u) {
        AllGradients[aw_hist_base(params.d_model, params.h_slots) + idx] += raw_step;
    } else {
        AllWeights[aw_hist_base(params.d_model, params.h_slots) + idx] =
            AllWeights[aw_hist_base(params.d_model, params.h_slots) + idx] - clamp(raw_step, -clip, clip);
    }
}

// Forget gate gradient: ∂L/∂W_f[s,j] and ∂L/∂b_f[s].
// g_scalar[t,s] = Σ_d weighted_h_buf[t,s,d] * exp(-A_log[s,d]) * m_prev[t-1,s,d] * f*(1-f)
// Note: a/(1-a) = exp(-A_log) is exact for non-selective case.
// ∂L/∂W_f[s,j] = Σ_{b,t} g_scalar[b,t,s] * h_star[b,t-1,s,j] / sqrt(d)
// ∂L/∂b_f[s]   = Σ_{b,t} g_scalar[b,t,s]
@compute
@workgroup_size(16, 16, 1)
fn fused_hist_stage_forget_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;   // W_forget dim j
    let slot = gid.y;  // slot s
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (col >= d || slot >= h_slots) { return; }

    let off = slot * d;
    var wf_grad = 0.0;
    var bf_grad = 0.0;

    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        for (var t = 1u; t < params.seq_len; t = t + 1u) {
            let t_abs = b * params.seq_len + t;
            let entry = t_abs * h_slots + slot;

            // g_scalar = Σ_d weighted_h * (a/(1-a)) * m_prev * f*(1-f)
            // weighted_h_buf = (1-a_t) * hist_ginner, so weighted_h * a_t/(1-a_t) = a_t * hist_ginner.
            var g_scalar = 0.0;
            for (var dim = 0u; dim < d; dim = dim + 1u) {
                let a_t = hist_ctx_buf[entry_base(entry, d) + dim];
                let a_ratio = a_t / max(1.0 - a_t, 1.0e-6);
                let m_prev = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + dim];
                g_scalar += weighted_h_buf[entry_base(entry, d) + dim] * a_ratio * m_prev;
            }

            // f_gate stored in Scratch at token_forget_base
            let f_val = Scratch[token_forget_base(t_abs, d, h_slots) + slot];
            g_scalar = g_scalar * f_val * (1.0 - f_val);

            // Accumulate W_f gradient using h_star at previous token (h*_{t-1})
            let t_prev_abs = b * params.seq_len + t - 1u;
            let h_prev_val = h_star[t_prev_abs * h_slots * d + off + col];
            wf_grad += g_scalar * h_prev_val;
            // b_f gradient (only one thread per slot updates it)
            if (col == 0u) {
                bf_grad += g_scalar;
            }
        }
    }

    let n_tokens = max(1u, params.batch_size * params.seq_len);
    let token_norm = 1.0 / max(1.0, f32(n_tokens) * f32(d));
    let clip = 0.5;

    let wf_idx = w_forget_base(d, h_slots) + off + col;
    let wf_step = params.lr * wf_grad * inverseSqrt(f32(d)) * token_norm * params.grad_scale;
    if (params.grad_accum_mode == 1u) {
        AllGradients[aw_hist_base(d, h_slots) + wf_idx] += wf_step;
    } else {
        AllWeights[aw_hist_base(d, h_slots) + wf_idx] -= clamp(wf_step, -clip, clip);
    }

    if (col == 0u) {
        let bf_idx = b_forget_base(d, h_slots) + slot;
        let bf_step = params.lr * bf_grad * token_norm * params.grad_scale;
        if (params.grad_accum_mode == 1u) {
            AllGradients[aw_hist_base(d, h_slots) + bf_idx] += bf_step;
        } else {
            AllWeights[aw_hist_base(d, h_slots) + bf_idx] -= clamp(bf_step, -clip, clip);
        }
    }
}
