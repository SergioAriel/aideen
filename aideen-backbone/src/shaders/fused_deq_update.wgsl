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
fn aw_wo_base(d: u32, h: u32) -> u32 { return aw_wv_base(d,h) + d*d; }
fn aw_win_base(d: u32, h: u32) -> u32 { return aw_wo_base(d,h) + d*d; }
fn aw_wx_base(d: u32, h: u32) -> u32 { return aw_win_base(d,h) + h*d*d; }
fn aw_wout_base(d: u32, h: u32) -> u32 { return aw_wx_base(d,h) + d*d; }
fn aw_alog_base(d: u32, h: u32) -> u32 { return aw_wout_base(d,h) + d*d; }
fn aw_nscale_base(d: u32, h: u32) -> u32 { return aw_alog_base(d,h) + h*d; }
fn aw_hist_base(d: u32, h: u32) -> u32 { return aw_nscale_base(d,h) + d; }

fn entry_base(entry: u32, d: u32) -> u32 {
    return entry * d;
}

fn scratch_stride(d: u32, h_slots: u32) -> u32 {
    return d * (h_slots * 7u) + h_slots * h_slots;
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

var<workgroup> hist_reduce_u: array<f32, 64>;
var<workgroup> hist_reduce_aux: array<f32, 64>;
var<workgroup> hist_carry_vec: array<f32, 1024>;
var<workgroup> hist_total_vec: array<f32, 1024>;
var<workgroup> hist_ginner_vec: array<f32, 1024>;
var<workgroup> hist_gx_vec: array<f32, 1024>;

@compute
@workgroup_size(64, 1, 1)
fn fused_attn_stage1a_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    let n_tokens = max(1u, params.seq_len - 1u);
    let lane = lid.x;
    let entry = wid.x;
    if (entry >= n_entries) { return; }
    let t_abs = entry / h_slots;
    let t = t_abs % params.seq_len;
    let qs = entry % h_slots;
    let q_off = qs * d;
    let t_off = t_abs * h_slots * d;
    for (var dim = lane; dim < d; dim = dim + 64u) {
        var gmix = 0.0;
        for (var dout = 0u; dout < d; dout = dout + 1u) {
            gmix = gmix + AllWeights[aw_wo_base(params.d_model, params.h_slots) +dim * d + dout] * v_adjoint[t_off + q_off + dout];
        }
        gmix_buf[entry_base(entry, d) + dim] = gmix;
    }
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
    let scratch_stride = d * (h_slots * 7u) + h_slots * h_slots;
    let base = t_abs * scratch_stride;
    let v_base = base + h_slots * d * 2u;
    let signal_base = base + d * (h_slots * 5u);
    let attn_weight_base = signal_base + 2u * h_slots * d;
    let slot_anchor = slot_anchor_base(d, h_slots);

    for (var dim = lane; dim < d; dim = dim + 64u) {
        var mix = 0.0;
        var weighted_h = 0.0;
        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
            let a = Scratch[attn_weight_base + qs * h_slots + ks];
            let k_off = ks * d;
            mix = mix + a * Scratch[v_base + k_off + dim];
            // V_fixed path: W_v is applied to (signal + slot_anchor), so g_wv must use
            // the same source vector to keep forward/backward consistent.
            let src = Scratch[signal_base + k_off + dim] + AllWeights[aw_hist_base(params.d_model, params.h_slots) +slot_anchor + k_off + dim];
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
    let scratch_stride = d * (h_slots * 7u) + h_slots * h_slots;
    let base = t_abs * scratch_stride;
    let v_base = base + h_slots * d * 2u;
    let signal_base = base + d * (h_slots * 5u);
    let attn_weight_base = signal_base + 2u * h_slots * d;
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
    let scratch_stride = d * (h_slots * 7u) + h_slots * h_slots;
    let base = t_abs * scratch_stride;
    let k_base = base + h_slots * d;
    var gq = 0.0;
    for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
        gq = gq + scale * gscore_buf[entry * h_slots + ks] * Scratch[k_base + ks * d + dim];
    }
    qgrad_buf[entry_base(entry, d) + dim] = gq;
}

@compute
@workgroup_size(16, 16, 1)
fn fused_attn_stage4_main(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    var g_wo = 0.0;
    var g_wv = 0.0;
    var g_win: array<f32, 16>;  // per-slot W_in gradient (max 16 slots), zero-initialized
    var g_wq: array<f32, 16>;  // per-slot W_q gradient (max 16 slots), zero-initialized
    var g_wk: array<f32, 16>;  // per-slot W_k gradient
    let scale = inverseSqrt(max(1.0, f32(d)));

    for (var entry = 0u; entry < n_entries; entry = entry + 1u) {
        let t_abs = entry / h_slots;
        let qs = entry % h_slots;
        let q_off = qs * d;
        let off = entry_base(entry, d);
        let scratch_stride = d * (h_slots * 7u) + h_slots * h_slots;
        let base = t_abs * scratch_stride;
        let q_base = base;
        let g_attn_col = v_adjoint[t_abs * h_slots * d + q_off + col];
        g_wo = g_wo + g_attn_col * mix_buf[off + row];
        g_wv = g_wv + weighted_h_buf[off + row] * gmix_buf[off + col];
        // Per-slot W_in gradient: dL/dW_in_s[row,col] += x[t,row] * adjoint[t,s,col]
        g_win[qs] = g_win[qs] + q_input[t_abs * d + row] * g_attn_col;
        let q_src = h_star[t_abs * h_slots * d + q_off + col];
        g_wq[qs] = g_wq[qs] + qgrad_buf[off + row] * q_src;
        let q_row = Scratch[q_base + q_off + row];
        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
            let hk_col = h_star[t_abs * h_slots * d + ks * d + col];
            g_wk[ks] = g_wk[ks] + scale * gscore_buf[entry * h_slots + ks] * q_row * hk_col;
        }
    }

    let seq_scale = 1.0 / max(1.0, f32(n_entries));
    g_wv = g_wv * seq_scale;
    g_wo = g_wo * seq_scale;

    let clip = 0.5;
    let raw_wv = lr * g_wv * grad_scale;
    let raw_wo = lr * g_wo * grad_scale;

    // Per-slot W_q / W_k update
    for (var s = 0u; s < h_slots; s = s + 1u) {
        let raw_wq = lr * g_wq[s] * seq_scale * grad_scale;
        let raw_wk = lr * g_wk[s] * seq_scale * grad_scale;
        let per_idx = s * d * d + idx;
        if (params.grad_accum_mode == 1u) {
            AllGradients[aw_wq_base(params.d_model, params.h_slots) +per_idx] += raw_wq;
            AllGradients[aw_wk_base(params.d_model, params.h_slots) +per_idx] += raw_wk;
        } else {
            AllWeights[aw_wq_base(params.d_model, params.h_slots) +per_idx] = AllWeights[aw_wq_base(params.d_model, params.h_slots) +per_idx] * wd_factor - clamp(raw_wq, -clip, clip);
            AllWeights[aw_wk_base(params.d_model, params.h_slots) +per_idx] = AllWeights[aw_wk_base(params.d_model, params.h_slots) +per_idx] * wd_factor - clamp(raw_wk, -clip, clip);
        }
    }
    if (params.grad_accum_mode == 1u) {
        AllGradients[aw_wv_base(params.d_model, params.h_slots) +idx] += raw_wv;
        AllGradients[aw_wo_base(params.d_model, params.h_slots) +idx] += raw_wo;
    } else {
        AllWeights[aw_wv_base(params.d_model, params.h_slots) +idx] = AllWeights[aw_wv_base(params.d_model, params.h_slots) +idx] * wd_factor - clamp(raw_wv, -clip, clip);
        AllWeights[aw_wo_base(params.d_model, params.h_slots) +idx] = AllWeights[aw_wo_base(params.d_model, params.h_slots) +idx] * wd_factor - clamp(raw_wo, -clip, clip);
    }
    // Per-slot W_in update: each slot's matrix at offset s * d * d
    for (var s = 0u; s < h_slots; s = s + 1u) {
        let g_win_s = g_win[s] * seq_scale;
        let raw_win = lr * g_win_s * grad_scale;
        if (params.grad_accum_mode == 1u) {
            AllGradients[aw_win_base(params.d_model, params.h_slots) +s * d * d + idx] += raw_win;
        } else {
            AllWeights[aw_win_base(params.d_model, params.h_slots) +s * d * d + idx] = AllWeights[aw_win_base(params.d_model, params.h_slots) +s * d * d + idx] * wd_factor - clamp(raw_win, -clip, clip);
        }
    }

    // Per-slot Q/K bias gradient (one thread per col, guarded by row==0).
    // q_bias[s,col] lives at AllWeights[aw_wq_base(params.d_model, params.h_slots) +d*d + s*d + col]; same layout for W_k.
    if (row == 0u) {
        var g_qbias: array<f32, 16>;  // zero-initialized per WGSL spec
        var g_kbias: array<f32, 16>;
        for (var entry2 = 0u; entry2 < n_entries; entry2 = entry2 + 1u) {
            let t2_abs  = entry2 / h_slots;
            let qs2    = entry2 % h_slots;
            let off2   = entry_base(entry2, d);
            let sstr2  = d * (h_slots * 7u) + h_slots * h_slots;
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
                AllGradients[aw_wq_base(params.d_model, params.h_slots) +h_slots * d * d + sb * d + col] += raw_qb;
                AllGradients[aw_wk_base(params.d_model, params.h_slots) +h_slots * d * d + sb * d + col] += raw_kb;
            } else {
                AllWeights[aw_wq_base(params.d_model, params.h_slots) +h_slots * d * d + sb * d + col] = AllWeights[aw_wq_base(params.d_model, params.h_slots) +h_slots * d * d + sb * d + col] * wd_factor - clamp(raw_qb, -clip, clip);
                AllWeights[aw_wk_base(params.d_model, params.h_slots) +h_slots * d * d + sb * d + col] = AllWeights[aw_wk_base(params.d_model, params.h_slots) +h_slots * d * d + sb * d + col] * wd_factor - clamp(raw_kb, -clip, clip);
            }
        }
    }

    if (idx == 0u) {
        debug_log[8] = 204.0;
        debug_log[40] = g_wq[0];
        debug_log[41] = g_wo;
        debug_log[42] = g_wv;
        debug_log[43] = g_win[0]; // log slot-0 gradient for diagnostics
        debug_log[44] = g_wk[0];
    }
}

@compute
@workgroup_size(64, 1, 1)
fn fused_hist_stage_prep_main(
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
    let hist_mat = hist_mat_len(d);
    let hist_scale = hist_scale_base(d, h_slots);
    let signal_base = token_signal_base(t_abs, d, h_slots);
    let hist_out = entry_base(entry, d);
    let sample_t = select(0u, max(1u, params.seq_len / 2u), params.seq_len > 1u);
    let sample_entry = sample_t * h_slots;

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
        qgrad_buf[hist_out + dim] = u;
        local_u_sumsq = local_u_sumsq + u * u;
        let inj = Scratch[signal_base + off + dim];
        local_inj_sumsq = local_inj_sumsq + inj * inj;
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
        let u = qgrad_buf[hist_out + dim];
        let u_tilde = scale * u;
        gmix_buf[hist_out + dim] = u_tilde;
        hist_ctx_buf[hist_out + dim] = alpha * u_tilde;
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
        let u = qgrad_buf[hist_out + dim];
        let g_utilde = alpha * g_c;
        var g_u = g_utilde;
        if (clip_active) {
            g_u = scale * g_utilde - (scale / denom) * u * hist_reduce_u[0];
        }
        weighted_h_buf[hist_out + dim] = g_u;
    }
    workgroupBarrier();
    if (entry == sample_entry && lane == 0u) {
        var sum_gc = 0.0;
        var sum_gu = 0.0;
        for (var dim = 0u; dim < d; dim = dim + 1u) {
            sum_gc = sum_gc + abs(mix_buf[hist_out + dim]);
            sum_gu = sum_gu + abs(weighted_h_buf[hist_out + dim]);
        }
        let inv_d = 1.0 / max(1.0, f32(d));
        debug_log[55] = sum_gc * inv_d;
        debug_log[56] = sum_gu * inv_d;
        if (t == sample_t) {
            debug_log[60] = sum_gc * inv_d;
            debug_log[61] = sum_gu * inv_d;
        }
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
        let prev_val = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + col];
        var local_prev_sumsq = 0.0;
        for (var dim = 0u; dim < d; dim = dim + 1u) {
            let prev = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + dim];
            local_prev_sumsq = local_prev_sumsq + prev * prev;
        }
        let prev_rms = sqrt(local_prev_sumsq / max(1.0, f32(d)) + 1e-6);
        grad = grad + weighted_h_buf[entry_base(entry, d) + row] * (prev_val / prev_rms);
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
            var local_prev_sumsq = 0.0;
            for (var j = 0u; j < d; j = j + 1u) {
                let prev = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + j];
                local_prev_sumsq = local_prev_sumsq + prev * prev;
            }
            let prev_rms = sqrt(local_prev_sumsq / max(1.0, f32(d)) + 1e-6);
            let prev_val = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + dim] / prev_rms;
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
    let wg_idx = wid.x;
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
            qgrad_buf[hist_out + dim] = h_star[prev_h_base + dim] / h_rms;
        }
        workgroupBarrier();

        for (var dim = lane; dim < d; dim = dim + 64u) {
            let a_base = 1.0 / (1.0 + exp(AllWeights[aw_alog_base(params.d_model, params.h_slots) +slot * d + dim]));
            var a_t = a_base;
            var x_proj = qgrad_buf[hist_out + dim];
            let wx = hist_wx_max() * tanh(AllWeights[aw_wx_base(params.d_model, params.h_slots) +dim * d + dim]);
            x_proj = x_proj + wx * qgrad_buf[hist_out + dim];
            var g_pre = 0.0;
            if (selective) {
                var delta_pre = AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_delta_bias_base(d, h_slots) + dim];
                for (var j = 0u; j < d; j = j + 1u) {
                    delta_pre = delta_pre
                        + AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_delta_base(d, h_slots) + slot * d * d + j * d + dim]
                        * qgrad_buf[hist_out + j];
                }
                // delta_factor in [0.5, 1.5], a_core = a_base^{delta_factor}, default a_core=a_base.
                let delta_factor = 1.0 + 0.5 * tanh(delta_pre);
                let a_core = pow(max(a_base, 1.0e-6), delta_factor);
                let prev_m = Scratch[token_mamba_base(t_abs - 1u, d, h_slots) + off + dim];
                let g_a = (prev_m - x_proj) * hist_ginner_vec[dim];
                let log_a = log(max(a_base, 1.0e-6));
                let a_floor = hist_alpha_min(d, h_slots);
                if (a_core < a_floor) {
                    a_t = a_floor;
                    g_pre = 0.0;
                } else {
                    a_t = a_core;
                    let g_delta_factor = g_a * a_core * log_a;
                    // d delta_factor / d delta_pre = 0.5 * (1 - tanh^2(delta_pre))
                    g_pre =
                        g_delta_factor * 0.5 * (1.0 - tanh(delta_pre) * tanh(delta_pre));
                }
            }
            hist_delta_buf[hist_out + dim] = g_pre;
            hist_gx_vec[dim] = (1.0 - a_t) * hist_ginner_vec[dim];
            weighted_h_buf[hist_out + dim] = hist_gx_vec[dim];
            var g_h_unit = hist_gx_vec[dim];
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
            let h_val = h_star[prev_h_base + dim];
            let g_h_unit = hist_total_vec[dim];
            hist_ctx_buf[dst_off + dim] = (g_h_unit / h_rms)
                - (h_val * dot_gh_h / denom);
            let a_base = 1.0 / (1.0 + exp(AllWeights[aw_alog_base(params.d_model, params.h_slots) +slot * d + dim]));
            var a_t = a_base;
            if (selective) {
                var delta_pre = AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_delta_bias_base(d, h_slots) + dim];
                for (var j = 0u; j < d; j = j + 1u) {
                    delta_pre = delta_pre + AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_delta_base(d, h_slots) + slot * d * d + j * d + dim]
                        * qgrad_buf[hist_out + j];
                }
                let delta = log(1.0 + exp(delta_pre));
                let log_a = log(max(a_base, 1.0e-6));
                let a_core = exp(delta * log_a);
                let a_floor = hist_alpha_min(d, h_slots);
                if (a_core < a_floor) {
                    a_t = a_floor;
                } else {
                    a_t = a_core;
                }
            }
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
                var g_m_inner = gmix_buf[entry_base(entry, d) + dim];
                for (var out = 0u; out < d; out = out + 1u) {
                    g_m_inner = g_m_inner + AllWeights[aw_wout_base(params.d_model, params.h_slots) +dim * d + out] * gmix_buf[entry_base(entry, d) + out];
                }

                var x_proj = qgrad_buf[entry_base(entry, d) + dim];
                let wx = hist_wx_max() * tanh(AllWeights[aw_wx_base(params.d_model, params.h_slots) +dim * d + dim]);
                x_proj = x_proj + wx * qgrad_buf[entry_base(entry, d) + dim];

                var m_prev = 0.0;
                if (t > 1u) {
                    m_prev = Scratch[token_mamba_base(t_abs - 2u, d, h_slots) + off + dim];
                }

                let a_base = 1.0 / (1.0 + exp(AllWeights[aw_alog_base(params.d_model, params.h_slots) +slot * d + dim]));
                if (selective) {
                    var delta_pre = AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_delta_bias_base(d, h_slots) + dim];
                    for (var j = 0u; j < d; j = j + 1u) {
                        delta_pre = delta_pre
                            + AllWeights[aw_hist_base(params.d_model, params.h_slots) +hist_delta_base(d, h_slots) + slot * d * d + j * d + dim]
                            * qgrad_buf[entry_base(entry, d) + j];
                    }
                    let delta_factor = 1.0 + 0.5 * tanh(delta_pre);
                    let a_core = pow(max(a_base, 1.0e-6), delta_factor);
                    let a_floor = hist_alpha_min(d, h_slots);
                    let a_t = a_floor + (1.0 - a_floor) * a_core;
                    grad = grad - (a_t - a_floor)
                        * (delta_factor * log(max(a_base, 1.0e-6)))
                        * (1.0 - a_base) * (m_prev - x_proj) * g_m_inner;
                } else {
                    grad = grad - a_base * (1.0 - a_base) * (m_prev - x_proj) * g_m_inner;
                }
            }
        }

        grad = grad / n_norm;
        let clip = 0.5;
        let raw_step = params.lr * grad;
        if (params.grad_accum_mode == 1u) {
            AllGradients[aw_alog_base(params.d_model, params.h_slots) +slot * d + dim] += raw_step;
        } else {
            AllWeights[aw_alog_base(params.d_model, params.h_slots) +slot * d + dim] = AllWeights[aw_alog_base(params.d_model, params.h_slots) +slot * d + dim] - clamp(raw_step, -clip, clip);
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
