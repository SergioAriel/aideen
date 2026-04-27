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
    attn_grad_bypass: f32, // Softmax backward bypass: 0.0=exact, 1.0=STE
};

@group(0) @binding(0) var<uniform> params: UpdateUniforms;
@group(0) @binding(1) var<storage, read> v_adjoint: array<f32>;
@group(0) @binding(2) var<storage, read> q_input: array<f32>;
@group(0) @binding(3) var<storage, read_write> h_star: array<f32>;
@group(0) @binding(4) var<storage, read_write> h_next: array<f32>;
@group(0) @binding(5) var<storage, read_write> Scratch: array<f32>;
@group(0) @binding(6) var<storage, read> dl_dh_temp_buf: array<f32>;
@group(0) @binding(7) var<storage, read_write> debug_log: array<f32>;
@group(0) @binding(8) var<storage, read_write> mix_buf: array<f32>;
@group(0) @binding(9) var<storage, read_write> weighted_h_buf: array<f32>;
@group(0) @binding(10) var<storage, read_write> gmix_buf: array<f32>;
@group(0) @binding(11) var<storage, read_write> gscore_buf: array<f32>;
@group(0) @binding(12) var<storage, read_write> qgrad_buf: array<f32>;
@group(0) @binding(13) var<storage, read_write> hist_ctx_buf: array<f32>;
@group(0) @binding(14) var<storage, read_write> hist_delta_buf: array<f32>;

@group(0) @binding(15) var<storage, read_write> tbptt_carry_buf: array<f32>;

// Group 1: Solve Pool (Unified across all solve shaders)
@group(1) @binding(0) var<storage, read_write> AssocBuf: array<f32>;
@group(1) @binding(1) var<storage, read_write> AssocPersistentBuf: array<f32>;
@group(1) @binding(2) var<storage, read_write> AssocHist: array<f32>;
@group(1) @binding(3) var<storage, read_write> AssocReadBuf: array<f32>;
@group(1) @binding(4) var<storage, read_write> AllWeights: array<f32>;
@group(1) @binding(5) var<storage, read_write> AllGradients: array<f32>;
// AllWeights layout offset functions (must match deq_forward.wgsl and deq_bridge.rs).
fn aw_wq_base(d: u32, h: u32) -> u32 { return 0u; }
fn aw_wk_base(d: u32, h: u32) -> u32 { return h * d * d + h * d; }
fn aw_wv_base(d: u32, h: u32) -> u32 { return aw_wk_base(d, h) + h * d * d + h * d; }
fn aw_wo_base(d: u32, h: u32) -> u32 { return aw_wv_base(d, h) + h * d * d; }
fn aw_win_base(d: u32, h: u32) -> u32 { return aw_wo_base(d, h) + h * d * d; }
fn aw_wx_base(d: u32, h: u32) -> u32 { return aw_win_base(d, h) + h * d * d; }
fn aw_wout_base(d: u32, h: u32) -> u32 { return aw_wx_base(d, h) + d * d; }
fn aw_alog_base(d: u32, h: u32) -> u32 { return aw_wout_base(d, h) + d * d; }
fn aw_nscale_base(d: u32, h: u32) -> u32 { return aw_alog_base(d, h) + h * d; }
fn aw_hist_base(d: u32, h: u32) -> u32 { return aw_nscale_base(d, h) + d; }

const RETAIN_RANK: u32 = 32u;
const ASSOC_RANK: u32 = 32u;

fn entry_base(entry: u32, d: u32) -> u32 {
    return entry * d;
}

fn v_adjoint_base(token: u32, slot: u32, d: u32) -> u32 {
    return (slot * params.batch_size * params.seq_len + token) * d;
}

fn hist_entry_base(entry: u32, d: u32, h_slots: u32) -> u32 {
    return entry * d;
}

fn hist_slot_base(token: u32, slot: u32, d: u32) -> u32 {
    return token * params.h_slots * d + slot * d;
}

fn scratch_stride(d: u32, h_slots: u32) -> u32 {
    return d * (h_slots * 8u) + h_slots * h_slots + h_slots;
}

fn hist_mat_len(d: u32) -> u32 { return d * d; }
fn hist_scale_base(d: u32, h_slots: u32) -> u32 { return hist_mat_len(d); }
fn hist_bias_base(d: u32, h_slots: u32) -> u32 { return hist_scale_base(d, h_slots) + h_slots * d; }
fn hist_gate_base(d: u32, h_slots: u32) -> u32 { return hist_bias_base(d, h_slots) + h_slots * d; }
fn slot_anchor_base(d: u32, h_slots: u32) -> u32 { return hist_gate_base(d, h_slots) + h_slots; }
fn w_k_write_base(d: u32, h: u32) -> u32 { return slot_anchor_base(d, h) + h * d; }
fn w_v_write_base(d: u32, h: u32) -> u32 { return w_k_write_base(d, h) + h * d * RETAIN_RANK; }
fn hist_delta_bias_base(d: u32, h_slots: u32) -> u32 { return w_v_write_base(d, h_slots) + h_slots * RETAIN_RANK * d; }
fn hist_selective_flag_base(d: u32, h_slots: u32) -> u32 { return hist_delta_bias_base(d, h_slots) + h_slots * d; }
fn hist_alpha_warmup_factor_base(d: u32, h_slots: u32) -> u32 { return hist_selective_flag_base(d, h_slots) + 1u; }
fn hist_rms_floor_base(d: u32, h_slots: u32) -> u32 { return hist_alpha_warmup_factor_base(d, h_slots) + 1u; }
fn hist_contr_floor_base(d: u32, h_slots: u32) -> u32 { return hist_rms_floor_base(d, h_slots) + 1u; }
fn hist_inject_flag_base(d: u32, h_slots: u32) -> u32 { return hist_contr_floor_base(d, h_slots) + 1u; }
fn hist_minner_zero_base(d: u32, h_slots: u32) -> u32 { return hist_inject_flag_base(d, h_slots) + 1u; }
fn hist_force_nofpm_base(d: u32, h_slots: u32) -> u32 { return hist_minner_zero_base(d, h_slots) + 1u; }
fn hist_prelude_skip_base(d: u32, h_slots: u32) -> u32 { return hist_force_nofpm_base(d, h_slots) + 1u; }
fn hist_loop_force_nofpm_base(d: u32, h_slots: u32) -> u32 { return hist_prelude_skip_base(d, h_slots) + 1u; }
fn signal_zero_base(d: u32, h_slots: u32) -> u32 { return hist_loop_force_nofpm_base(d, h_slots) + 1u; }
fn attn_out_mode_base(d: u32, h_slots: u32) -> u32 { return signal_zero_base(d, h_slots) + 1u; }
fn attn_uniform_base(d: u32, h_slots: u32) -> u32 { return attn_out_mode_base(d, h_slots) + 1u; }
fn attn_freeze_base(d: u32, h_slots: u32) -> u32 { return attn_uniform_base(d, h_slots) + 1u; }
fn signal_scale_base(d: u32, h_slots: u32) -> u32 { return signal_zero_base(d, h_slots) + 7u; }
fn hist_gate_query_base(d: u32, h_slots: u32) -> u32 { return hist_delta_bias_base(d, h_slots) + h_slots * d + 21u; }
fn w_write_gate_base(d: u32, h_slots: u32) -> u32 { return hist_gate_query_base(d, h_slots) + h_slots * d; }
fn b_write_mem_base(d: u32, h_slots: u32) -> u32 { return w_write_gate_base(d, h_slots) + h_slots * d; }
fn hhist_gamma_base(d: u32, h: u32) -> u32 { return b_write_mem_base(d, h) + h; }
fn w_retain_up_base(d: u32, h: u32) -> u32 { return hhist_gamma_base(d, h) + h; }
fn w_retain_down_base(d: u32, h: u32) -> u32 { return w_retain_up_base(d, h) + h * d * RETAIN_RANK; }
fn b_retain_base(d: u32, h: u32) -> u32 { return w_retain_down_base(d, h) + h * RETAIN_RANK * d; }
fn w_q_mem_base(d: u32, h: u32) -> u32 { return b_retain_base(d, h) + h * d; }
fn w_k_mem_base(d: u32, h: u32) -> u32 { return w_q_mem_base(d, h) + h * d * RETAIN_RANK; }
fn b_read_mem_base(d: u32, h: u32) -> u32 { return w_k_mem_base(d, h) + h * d * RETAIN_RANK; }
fn w_k_assoc_base(d: u32, h: u32) -> u32 { return b_read_mem_base(d, h) + h; }
fn w_v_assoc_base(d: u32, h: u32) -> u32 { return w_k_assoc_base(d, h) + h * d * ASSOC_RANK; }
fn w_q_assoc_base(d: u32, h: u32) -> u32 { return w_v_assoc_base(d, h) + h * d * ASSOC_RANK; }
fn alpha_assoc_base(d: u32, h: u32) -> u32 { return w_q_assoc_base(d, h) + h * d * ASSOC_RANK; }
fn w_event_assoc_base(d: u32, h: u32) -> u32 { return alpha_assoc_base(d, h) + h; }
fn b_event_assoc_base(d: u32, h: u32) -> u32 { return w_event_assoc_base(d, h) + h * d; }


// W_gate_hist follows the 21 scalars (hist_selective_flag .. v_norm_scale)

const H_HIST_GAMMA_SCALE: f32 = 0.1;

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
    let warmup = clamp(AllWeights[aw_hist_base(params.d_model, params.h_slots) + hist_alpha_warmup_factor_base(d, h_slots)], 0.0, 1.0);
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

fn token_fpm_base(t: u32, d: u32, h_slots: u32) -> u32 {
    return token_scratch_base(t, d, h_slots) + h_slots * 4u * d;
}

fn token_signal_base(t: u32, d: u32, h_slots: u32) -> u32 {
    return token_fpm_base(t, d, h_slots) + h_slots * d;
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
                v_adjoint[v_adjoint_base(token, slot, d) + k_a];
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
    let entry = wid.x;
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
    // Exact softmax backward: a * (g_alpha - alpha_dot_g)
    // When attention is uniform and slot values are similar, alpha_dot_g ≈ g_alpha → gradient ≈ 0.
    // bypass=1.0 removes the mean-subtraction (STE), giving non-zero gradient at initialization.
    // bypass=0.0 is the unmodified exact backward (current default).
    let retain_mean = 1.0 - params.attn_grad_bypass;
    let gscore_val = a * (g_alpha - retain_mean * alpha_dot_g);
    gscore_buf[entry * h_slots + ks] = gscore_val;

    // Gradient chain diagnostics: log at entry=0, qs=0, ks=0 only.
    // debug_buf[90]=gmix_sample, [91]=g_alpha, [92]=alpha_dot_g, [93]=gscore, [94]=attn_weight
    if (entry == 0u && qs == 0u && ks == 0u) {
        debug_log[90] = gmix_buf[0];          // sample of upstream adjoint signal
        debug_log[91] = g_alpha;              // dot(gmix, V_ks) before subtraction
        debug_log[92] = alpha_dot_g;          // weighted mean of all g_alpha_j
        debug_log[93] = gscore_val;           // final gradient score (should be non-zero)
        debug_log[94] = a;                    // attention weight α[qs=0, ks=0]
    }
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
    let slot = gid.z;
    let d = params.d_model;
    let h_slots = params.h_slots;
    if (row >= d || col >= d || slot >= h_slots) { return; }

    let n_tokens = params.batch_size * params.seq_len;
    let n_entries = n_tokens * h_slots;
    let idx = row * d + col;
    let lr = params.lr;
    let wd_factor = 1.0 - lr * params.weight_decay;
    let grad_scale = params.grad_scale;

    var g_wo = 0.0;
    var g_win = 0.0;
    for (var token = 0u; token < n_tokens; token = token + 1u) {
        let g_attn_col = v_adjoint[v_adjoint_base(token, slot, d) + col];
        let entry = token * h_slots + slot;
        let off = entry_base(entry, d);
        g_wo = g_wo + g_attn_col * mix_buf[off + row];
        g_win = g_win + q_input[token * d + row] * g_attn_col;
    }

    let seq_scale = 1.0 / max(1.0, f32(n_entries));
    let clip = 0.5;
    let raw_wo = lr * g_wo * seq_scale * grad_scale;
    let wo_idx = aw_wo_base(params.d_model, params.h_slots) + slot * d * d + idx;
    if (params.grad_accum_mode == 1u) {
        AllGradients[wo_idx] += raw_wo;
    } else {
        AllWeights[wo_idx] = AllWeights[wo_idx] * wd_factor - clamp(raw_wo, -clip, clip);
    }

    let g_win_s = g_win * seq_scale;
    let raw_win = lr * g_win_s * grad_scale;
    let win_idx = aw_win_base(params.d_model, params.h_slots) + slot * d * d + idx;
    if (params.grad_accum_mode == 1u) {
        AllGradients[win_idx] += raw_win;
    } else {
        AllWeights[win_idx] = AllWeights[win_idx] * wd_factor - clamp(raw_win, -clip, clip);
    }

    if (idx == 0u && slot == 0u) {
        // Keep 8..18 reserved for the forward DEQ snapshot header.
        debug_log[1000] = 241.0;
        debug_log[41] = g_wo;
        debug_log[43] = g_win;
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
        debug_log[1000] = 246.0;
        // Log g_wq at [row=0,col=0,slot=0] for compatibility, plus its square for magnitude check.
        debug_log[40] = g_wq;
        // [45]: squared update magnitude (no sign cancellation), accumulated atomically across threads
        // is not atomic here but single thread writes it — use abs() as magnitude proxy.
        debug_log[45] = abs(raw_wq);
    }
    // [46]: sum of |raw_wq| across ALL elements to measure total Frobenius update magnitude.
    // Use a relaxed atomic add with non-atomic fallback (races acceptable for diagnostics).
    debug_log[46] += raw_wq * raw_wq;
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
    hist_delta_buf[hist_entry_base(tk, d, h_slots) + dim] = coeff;
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
    hist_ctx_buf[hist_entry_base(tk, d, h_slots) + dim] = coeff;
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
                hist_delta_buf[hist_slot_base(token_a, slot, d) + row];
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
        // Keep 8..18 reserved for the forward DEQ snapshot header.
        debug_log[1000] = 247.0;
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
                hist_ctx_buf[hist_slot_base(token_b, slot, d) + col];
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
        // Keep 8..18 reserved for the forward DEQ snapshot header.
        debug_log[1000] = 248.0;
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
        // Keep 8..18 reserved for the forward DEQ snapshot header.
        debug_log[1000] = 243.0;
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




// ---- Gradient Apply ----
// Applies accumulated gradients from AllGradients to AllWeights.
// Called once after n_accum accumulation passes (mode=1).
// Weight decay is applied to main weights (indices < aw_hist_base); not to hist params.
@compute
@workgroup_size(256, 1, 1)
fn apply_grad_update_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
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
