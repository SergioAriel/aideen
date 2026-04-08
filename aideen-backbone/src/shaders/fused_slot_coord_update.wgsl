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
    grad_accum_mode: u32,
    n_accum: u32,
    n_total_weights: u32,
    batch_size: u32,
    apply_accum: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<uniform> params: UpdateUniforms;
@group(0) @binding(1) var<storage, read> v_adjoint: array<f32>;
@group(0) @binding(2) var<storage, read> q_input: array<f32>;
@group(0) @binding(3) var<storage, read> h_star: array<f32>;
@group(0) @binding(4) var<storage, read_write> debug_log: array<f32>;
@group(0) @binding(5) var<storage, read> dl_dh_temp_buf: array<f32>;
@group(0) @binding(6) var<storage, read> Scratch: array<f32>;
@group(0) @binding(7) var<storage, read_write> mix_buf: array<f32>;
@group(0) @binding(8) var<storage, read_write> slot_q_buf: array<f32>;
@group(0) @binding(9) var<storage, read_write> gmix_buf: array<f32>;
@group(0) @binding(10) var<storage, read_write> gscore_buf: array<f32>;
@group(0) @binding(11) var<storage, read_write> qgrad_buf: array<f32>;
@group(0) @binding(12) var<storage, read_write> slot_v_buf: array<f32>;
@group(0) @binding(13) var<storage, read_write> slot_k_work_buf: array<f32>;
@group(0) @binding(14) var<storage, read_write> tbptt_carry_buf: array<f32>;
@group(0) @binding(15) var<storage, read_write> AllGradients: array<f32>;

@group(1) @binding(0) var<storage, read_write> AllWeights: array<f32>;

override SLOT_COORD_HEAD_DIM: u32 = 32u;
override SLOT_COORD_TRAIN_BIAS: bool = true;
override SLOT_COORD_LOGIT_GAIN: f32 = 1.0;
const MAX_SLOT_CAP: u32 = 16u;
const SLOT_COORD_QK_SCALE: f32 = 1.0;
const SLOT_COORD_VALUE_SCALE: f32 = 0.1;
const SLOT_COORD_ANCHOR_SCALE: f32 = 0.1;

fn aw_wq_base(d: u32, h: u32) -> u32 { return 0u; }
fn aw_wk_base(d: u32, h: u32) -> u32 { return h * d * d + h * d; }
fn aw_wv_base(d: u32, h: u32) -> u32 { return aw_wk_base(d, h) + h * d * d + h * d; }
fn aw_wo_base(d: u32, h: u32) -> u32 { return aw_wv_base(d, h) + h * d * d; }
fn aw_win_base(d: u32, h: u32) -> u32 { return aw_wo_base(d, h) + h * d * d; }
fn aw_nscale_base(d: u32, h: u32) -> u32 {
    let aw_wx = aw_win_base(d, h) + h * d * d;
    let aw_wout = aw_wx + d * d;
    let aw_alog = aw_wout + d * d;
    return aw_alog + h * d;
}
fn aw_hist_base(d: u32, h: u32) -> u32 { return aw_nscale_base(d, h) + d; }
fn hist_mat_len(d: u32) -> u32 { return d * d; }
fn hist_scale_base(d: u32, h_slots: u32) -> u32 { return hist_mat_len(d); }
fn hist_bias_base(d: u32, h_slots: u32) -> u32 { return hist_scale_base(d, h_slots) + h_slots * d; }
fn hist_gate_base(d: u32, h_slots: u32) -> u32 { return hist_bias_base(d, h_slots) + h_slots * d; }
fn slot_anchor_base(d: u32, h_slots: u32) -> u32 { return hist_gate_base(d, h_slots) + h_slots; }

fn head_dim(d: u32) -> u32 {
    return min(d, SLOT_COORD_HEAD_DIM);
}

fn signal_span(d: u32, h_slots: u32) -> u32 {
    return d * h_slots;
}

fn scratch_stride(d: u32, h_slots: u32) -> u32 {
    let span = signal_span(d, h_slots);
    return span * 3u + h_slots * h_slots;
}

fn token_scratch_base(t: u32, d: u32, h_slots: u32) -> u32 {
    return t * scratch_stride(d, h_slots);
}

fn token_signal_base(t: u32, d: u32, h_slots: u32) -> u32 {
    return token_scratch_base(t, d, h_slots);
}

fn token_alpha_base(t: u32, d: u32, h_slots: u32) -> u32 {
    return token_scratch_base(t, d, h_slots) + signal_span(d, h_slots) * 2u;
}

fn token_slot_base(t: u32, slot: u32, d: u32, h_slots: u32) -> u32 {
    return t * h_slots * d + slot * d;
}

fn entry_base(entry: u32, d: u32) -> u32 {
    return entry * d;
}

fn slot_scale_base(entry: u32) -> u32 {
    return entry * 2u;
}

fn slot_src_rms_cache(entry: u32) -> f32 {
    return tbptt_carry_buf[slot_scale_base(entry)];
}

fn slot_attn_scale_cache(entry: u32) -> f32 {
    return tbptt_carry_buf[slot_scale_base(entry) + 1u];
}

fn signal_at(t: u32, slot: u32, dim: u32, d: u32, h_slots: u32) -> f32 {
    return Scratch[token_signal_base(t, d, h_slots) + slot * d + dim];
}

fn slot_anchor_at(slot: u32, dim: u32, d: u32, h_slots: u32) -> f32 {
    return AllWeights[slot_anchor_idx(slot, dim, d)];
}

fn src_at(t: u32, slot: u32, in_dim: u32, d: u32, h_slots: u32) -> f32 {
    return signal_at(t, slot, in_dim, d, h_slots) + slot_anchor_at(slot, in_dim, d, h_slots);
}

fn src_rms_at(t: u32, slot: u32, d: u32, h_slots: u32) -> f32 {
    var sumsq = 0.0;
    for (var j = 0u; j < d; j = j + 1u) {
        let src = src_at(t, slot, j, d, h_slots);
        sumsq = sumsq + src * src;
    }
    return sqrt(sumsq / max(1.0, f32(d)) + 1.0e-6);
}

fn src_norm_at(t: u32, slot: u32, dim: u32, d: u32, h_slots: u32) -> f32 {
    return src_at(t, slot, dim, d, h_slots) / max(src_rms_at(t, slot, d, h_slots), 1.0e-6);
}

fn src_norm_with_rms(t: u32, slot: u32, dim: u32, d: u32, h_slots: u32, src_rms: f32) -> f32 {
    return src_at(t, slot, dim, d, h_slots) / max(src_rms, 1.0e-6);
}

fn alpha_at(t: u32, qs: u32, ks: u32, d: u32, h_slots: u32) -> f32 {
    return Scratch[token_alpha_base(t, d, h_slots) + qs * h_slots + ks];
}

fn token_attn_base(t: u32, slot: u32, d: u32, h_slots: u32) -> u32 {
    return token_scratch_base(t, d, h_slots) + signal_span(d, h_slots) + slot * d;
}

fn attn_at(t: u32, slot: u32, dim: u32, d: u32, h_slots: u32) -> f32 {
    return Scratch[token_attn_base(t, slot, d, h_slots) + dim];
}

fn attn_scale_at(t: u32, slot: u32, d: u32, h_slots: u32) -> f32 {
    var attn_sumsq = 0.0;
    for (var dim = 0u; dim < d; dim = dim + 1u) {
        let attn = attn_at(t, slot, dim, d, h_slots);
        attn_sumsq = attn_sumsq + attn * attn;
    }
    let attn_rms = sqrt(attn_sumsq / max(1.0, f32(d)) + 1.0e-6);
    return src_rms_at(t, slot, d, h_slots) / max(attn_rms, 1.0e-6);
}

@compute
@workgroup_size(64, 1, 1)
fn slot_coord_stage0_scale_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let entry = gid.x;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (entry >= n_entries) { return; }

    let token = entry / h_slots;
    let slot = entry % h_slots;
    let src_rms = max(src_rms_at(token, slot, d, h_slots), 1.0e-6);
    var attn_sumsq = 0.0;
    for (var dim = 0u; dim < d; dim = dim + 1u) {
        let attn = attn_at(token, slot, dim, d, h_slots);
        attn_sumsq = attn_sumsq + attn * attn;
    }
    let attn_rms = sqrt(attn_sumsq / max(1.0, f32(d)) + 1.0e-6);
    let scale_base = slot_scale_base(entry);
    tbptt_carry_buf[scale_base] = src_rms;
    tbptt_carry_buf[scale_base + 1u] = src_rms / max(attn_rms, 1.0e-6);
}

fn q_weight_idx(slot: u32, in_dim: u32, out_head: u32, d: u32) -> u32 {
    return aw_wq_base(d, params.h_slots) + slot * d * d + in_dim * d + out_head;
}

fn k_weight_idx(slot: u32, in_dim: u32, out_head: u32, d: u32) -> u32 {
    return aw_wk_base(d, params.h_slots) + slot * d * d + in_dim * d + out_head;
}

fn v_weight_idx(slot: u32, in_dim: u32, out_head: u32, d: u32) -> u32 {
    return aw_wv_base(d, params.h_slots) + slot * d * d + in_dim * d + out_head;
}

fn o_weight_idx(slot: u32, in_head: u32, out_dim: u32, d: u32) -> u32 {
    return aw_wo_base(d, params.h_slots) + slot * d * d + in_head * d + out_dim;
}

fn win_weight_idx(slot: u32, in_dim: u32, out_dim: u32, d: u32) -> u32 {
    return aw_win_base(d, params.h_slots) + slot * d * d + in_dim * d + out_dim;
}

fn q_bias_idx(slot: u32, head: u32, d: u32) -> u32 {
    return aw_wq_base(d, params.h_slots) + params.h_slots * d * d + slot * d + head;
}

fn k_bias_idx(slot: u32, head: u32, d: u32) -> u32 {
    return aw_wk_base(d, params.h_slots) + params.h_slots * d * d + slot * d + head;
}

fn slot_anchor_idx(slot: u32, dim: u32, d: u32) -> u32 {
    return aw_hist_base(d, params.h_slots) + slot_anchor_base(d, params.h_slots) + slot * d + dim;
}

@compute
@workgroup_size(16, 16, 1)
fn slot_coord_stage1_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head = gid.x;
    let token = gid.y;
    let slot = gid.z;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let hd = head_dim(d);
    let n_tokens = params.batch_size * params.seq_len;
    if (head >= hd || token >= n_tokens || slot >= h_slots) { return; }

    let entry = token * h_slots + slot;
    let src_rms = max(slot_src_rms_cache(entry), 1.0e-6);
    var q = AllWeights[q_bias_idx(slot, head, d)];
    var k = AllWeights[k_bias_idx(slot, head, d)];
    var v = 0.0;
    for (var j = 0u; j < d; j = j + 1u) {
        let src = src_norm_with_rms(token, slot, j, d, h_slots, src_rms);
        q = q + AllWeights[q_weight_idx(slot, j, head, d)] * src;
        k = k + AllWeights[k_weight_idx(slot, j, head, d)] * src;
        v = v + AllWeights[v_weight_idx(slot, j, head, d)] * src;
    }

    let tk = token * h_slots + slot;
    slot_q_buf[entry_base(entry, d) + head] = q;
    slot_k_work_buf[entry_base(tk, d) + head] = k;
    slot_v_buf[entry_base(tk, d) + head] = v;

    var gmix = 0.0;
    let attn_scale = slot_attn_scale_cache(entry);
    for (var out_dim = 0u; out_dim < d; out_dim = out_dim + 1u) {
        gmix = gmix
            + v_adjoint[entry_base(entry, d) + out_dim]
            * attn_scale
            * AllWeights[o_weight_idx(slot, head, out_dim, d)];
    }
    gmix_buf[entry_base(entry, d) + head] = gmix;
}

@compute
@workgroup_size(16, 16, 1)
fn slot_coord_stage1b_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head = gid.x;
    let entry = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let hd = head_dim(d);
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (head >= hd || entry >= n_entries) { return; }

    let token = entry / h_slots;
    let qs = entry % h_slots;
    var mix = 0.0;
    var mean_v = 0.0;
    for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
        let tk = token * h_slots + ks;
        let v = slot_v_buf[entry_base(tk, d) + head];
        mix = mix + alpha_at(token, qs, ks, d, h_slots) * v;
        mean_v = mean_v + v;
    }
    mix_buf[entry_base(entry, d) + head] = mix - mean_v / max(1.0, f32(h_slots));
}

@compute
@workgroup_size(16, 16, 1)
fn slot_coord_stage2_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ks = gid.x;
    let entry = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let hd = head_dim(d);
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (ks >= h_slots || entry >= n_entries) { return; }

    let token = entry / h_slots;
    let qs = entry % h_slots;
    let tk = token * h_slots + ks;

    var g_alpha = 0.0;
    for (var head = 0u; head < hd; head = head + 1u) {
        g_alpha = g_alpha + gmix_buf[entry_base(entry, d) + head] * slot_v_buf[entry_base(tk, d) + head];
    }

    var alpha_dot_g = 0.0;
    for (var js = 0u; js < h_slots; js = js + 1u) {
        let tj = token * h_slots + js;
        var g_alpha_j = 0.0;
        for (var head = 0u; head < hd; head = head + 1u) {
            g_alpha_j = g_alpha_j + gmix_buf[entry_base(entry, d) + head] * slot_v_buf[entry_base(tj, d) + head];
        }
        alpha_dot_g = alpha_dot_g + alpha_at(token, qs, js, d, h_slots) * g_alpha_j;
    }

    let a = alpha_at(token, qs, ks, d, h_slots);
    gscore_buf[entry * h_slots + ks] = a * (g_alpha - alpha_dot_g);
}

@compute
@workgroup_size(16, 16, 1)
fn slot_coord_stage3_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head = gid.x;
    let entry = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let hd = head_dim(d);
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (head >= hd || entry >= n_entries) { return; }

    let token = entry / h_slots;
    let scale = inverseSqrt(max(1.0, f32(hd))) * SLOT_COORD_LOGIT_GAIN;
    var gq = 0.0;
    for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
        let tk = token * h_slots + ks;
        gq = gq + scale * gscore_buf[entry * h_slots + ks] * slot_k_work_buf[entry_base(tk, d) + head];
    }
    qgrad_buf[entry_base(entry, d) + head] = gq;
}

@compute
@workgroup_size(16, 16, 1)
fn slot_coord_stage4_signal_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let entry = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let hd = head_dim(d);
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (dim >= d || entry >= n_entries) { return; }

    let token = entry / h_slots;
    let slot = entry % h_slots;
    let src_rms = max(slot_src_rms_cache(entry), 1.0e-6);
    var grad = 0.0;
    for (var head = 0u; head < hd; head = head + 1u) {
        grad = grad
            + qgrad_buf[entry_base(entry, d) + head] * AllWeights[q_weight_idx(slot, dim, head, d)]
            + slot_k_work_buf[entry_base(entry, d) + head] * AllWeights[k_weight_idx(slot, dim, head, d)]
            + slot_v_buf[entry_base(entry, d) + head] * AllWeights[v_weight_idx(slot, dim, head, d)];
    }

    var dot_gx = 0.0;
    for (var j = 0u; j < d; j = j + 1u) {
        var grad_j = 0.0;
        for (var head = 0u; head < hd; head = head + 1u) {
            grad_j = grad_j
                + qgrad_buf[entry_base(entry, d) + head] * AllWeights[q_weight_idx(slot, j, head, d)]
                + slot_k_work_buf[entry_base(entry, d) + head] * AllWeights[k_weight_idx(slot, j, head, d)]
                + slot_v_buf[entry_base(entry, d) + head] * AllWeights[v_weight_idx(slot, j, head, d)];
        }
        dot_gx = dot_gx + grad_j * src_norm_with_rms(token, slot, j, d, h_slots, src_rms);
    }
    let src_norm = src_norm_with_rms(token, slot, dim, d, h_slots, src_rms);
    let grad_src = grad / src_rms - src_norm * dot_gx / max(f32(d) * src_rms, 1.0e-6);
    // Reuse gmix_buf after the prep passes: from here on the dedicated slot path no longer
    // needs per-head gmix, but W_in/slot_anchor still need the indirect source gradient carried
    // through the normalized Q/K/V projections.
    gmix_buf[entry_base(entry, d) + dim] = grad_src;
}

@compute
@workgroup_size(16, 16, 1)
fn slot_coord_stage4_anchor_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let slot = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_tokens = params.batch_size * params.seq_len;
    if (dim >= d || slot >= h_slots) { return; }

    var grad = 0.0;
    for (var token = 0u; token < n_tokens; token = token + 1u) {
        let entry = token * h_slots + slot;
        grad = grad + v_adjoint[entry_base(entry, d) + dim] + gmix_buf[entry_base(entry, d) + dim];
    }

    let lr = params.lr;
    let wd_factor = 1.0 - lr * params.weight_decay;
    let seq_scale = 1.0 / max(1.0, f32(n_tokens));
    let raw = lr * grad * seq_scale * params.grad_scale * SLOT_COORD_ANCHOR_SCALE;
    let idx = slot_anchor_idx(slot, dim, d);
    let clip = 0.5;
    if (params.grad_accum_mode == 1u) {
        AllGradients[idx] = AllGradients[idx] + raw;
    } else {
        AllWeights[idx] = AllWeights[idx] * wd_factor - clamp(raw, -clip, clip);
    }
}

@compute
@workgroup_size(16, 16, 1)
fn slot_coord_stage4_wo_win_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dim = gid.x;
    let in_dim = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let hd = head_dim(d);
    if (out_dim >= d || in_dim >= d) { return; }

    let n_entries = params.batch_size * params.seq_len * h_slots;
    let n_tokens = params.batch_size * params.seq_len;
    let lr = params.lr;
    let wd_factor = 1.0 - lr * params.weight_decay;
    let grad_scale = params.grad_scale;
    let seq_scale = 1.0 / max(1.0, f32(n_tokens));
    let clip = 0.5;

    var g_wo: array<f32, MAX_SLOT_CAP>;
    var g_win: array<f32, MAX_SLOT_CAP>;

    for (var entry = 0u; entry < n_entries; entry = entry + 1u) {
        let token = entry / h_slots;
        let slot = entry % h_slots;
        let g_attn = v_adjoint[entry_base(entry, d) + out_dim];
        let attn_scale = slot_attn_scale_cache(entry);
        if (in_dim < hd) {
            g_wo[slot] = g_wo[slot] + g_attn * attn_scale * mix_buf[entry_base(entry, d) + in_dim];
        }
        let g_signal = g_attn + gmix_buf[entry_base(entry, d) + out_dim];
        g_win[slot] = g_win[slot] + q_input[token * d + in_dim] * g_signal;
    }

    for (var slot = 0u; slot < h_slots; slot = slot + 1u) {
        if (in_dim < hd) {
            let wo_idx = o_weight_idx(slot, in_dim, out_dim, d);
            let raw_wo = lr * g_wo[slot] * seq_scale * grad_scale * SLOT_COORD_VALUE_SCALE;
            if (params.grad_accum_mode == 1u) {
                AllGradients[wo_idx] = AllGradients[wo_idx] + raw_wo;
            } else {
                AllWeights[wo_idx] = AllWeights[wo_idx] * wd_factor - clamp(raw_wo, -clip, clip);
            }
        }
        let win_idx = win_weight_idx(slot, in_dim, out_dim, d);
        let raw_win = lr * g_win[slot] * seq_scale * grad_scale * SLOT_COORD_VALUE_SCALE;
        if (params.grad_accum_mode == 1u) {
            AllGradients[win_idx] = AllGradients[win_idx] + raw_win;
        } else {
            AllWeights[win_idx] = AllWeights[win_idx] * wd_factor - clamp(raw_win, -clip, clip);
        }
    }
}

@compute
@workgroup_size(16, 16, 1)
fn slot_coord_stage4_wq_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head = gid.x;
    let in_dim = gid.y;
    let slot = gid.z;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let hd = head_dim(d);
    let n_tokens = params.batch_size * params.seq_len;
    if (head >= hd || in_dim >= d || slot >= h_slots) { return; }

    var grad = 0.0;
    for (var token = 0u; token < n_tokens; token = token + 1u) {
        let entry = token * h_slots + slot;
        let src_rms = max(slot_src_rms_cache(entry), 1.0e-6);
        grad = grad + qgrad_buf[entry_base(entry, d) + head] * src_norm_with_rms(token, slot, in_dim, d, h_slots, src_rms);
    }

    let lr = params.lr;
    let wd_factor = 1.0 - lr * params.weight_decay;
    let seq_scale = 1.0 / max(1.0, f32(n_tokens));
    let raw = lr * grad * seq_scale * params.grad_scale * SLOT_COORD_QK_SCALE;
    let idx = q_weight_idx(slot, in_dim, head, d);
    let clip = 0.5;
    if (params.grad_accum_mode == 1u) {
        AllGradients[idx] = AllGradients[idx] + raw;
    } else {
        AllWeights[idx] = AllWeights[idx] * wd_factor - clamp(raw, -clip, clip);
    }
}

@compute
@workgroup_size(16, 16, 1)
fn slot_coord_stage4_prep_wk_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head = gid.x;
    let tk = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let hd = head_dim(d);
    let n_tokens = params.batch_size * params.seq_len;
    let n_token_slots = n_tokens * h_slots;
    if (head >= hd || tk >= n_token_slots) { return; }

    let token = tk / h_slots;
    let ks = tk % h_slots;
    let scale = inverseSqrt(max(1.0, f32(hd))) * SLOT_COORD_LOGIT_GAIN;
    var coeff = 0.0;
    for (var qs = 0u; qs < h_slots; qs = qs + 1u) {
        let entry = token * h_slots + qs;
        coeff = coeff + scale * gscore_buf[entry * h_slots + ks] * slot_q_buf[entry_base(entry, d) + head];
    }
    slot_k_work_buf[entry_base(tk, d) + head] = coeff;
}

@compute
@workgroup_size(16, 16, 1)
fn slot_coord_stage4_prep_wv_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head = gid.x;
    let tk = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let hd = head_dim(d);
    let n_tokens = params.batch_size * params.seq_len;
    let n_token_slots = n_tokens * h_slots;
    if (head >= hd || tk >= n_token_slots) { return; }

    let token = tk / h_slots;
    let ks = tk % h_slots;
    var coeff = 0.0;
    let uniform = 1.0 / max(1.0, f32(h_slots));
    for (var qs = 0u; qs < h_slots; qs = qs + 1u) {
        let entry = token * h_slots + qs;
        coeff = coeff + (alpha_at(token, qs, ks, d, h_slots) - uniform) * gmix_buf[entry_base(entry, d) + head];
    }
    slot_v_buf[entry_base(tk, d) + head] = coeff;
}

@compute
@workgroup_size(16, 16, 1)
fn slot_coord_stage4_wk_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head = gid.x;
    let in_dim = gid.y;
    let slot = gid.z;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let hd = head_dim(d);
    let n_tokens = params.batch_size * params.seq_len;
    if (head >= hd || in_dim >= d || slot >= h_slots) { return; }

    var grad = 0.0;
    for (var token = 0u; token < n_tokens; token = token + 1u) {
        let tk = token * h_slots + slot;
        let src_rms = max(slot_src_rms_cache(tk), 1.0e-6);
        grad = grad + slot_k_work_buf[entry_base(tk, d) + head] * src_norm_with_rms(token, slot, in_dim, d, h_slots, src_rms);
    }

    let lr = params.lr;
    let wd_factor = 1.0 - lr * params.weight_decay;
    let seq_scale = 1.0 / max(1.0, f32(n_tokens));
    let raw = lr * grad * seq_scale * params.grad_scale * SLOT_COORD_QK_SCALE;
    let idx = k_weight_idx(slot, in_dim, head, d);
    let clip = 0.5;
    if (params.grad_accum_mode == 1u) {
        AllGradients[idx] = AllGradients[idx] + raw;
    } else {
        AllWeights[idx] = AllWeights[idx] * wd_factor - clamp(raw, -clip, clip);
    }
}

@compute
@workgroup_size(16, 16, 1)
fn slot_coord_stage4_wv_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head = gid.x;
    let in_dim = gid.y;
    let slot = gid.z;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let hd = head_dim(d);
    let n_tokens = params.batch_size * params.seq_len;
    if (head >= hd || in_dim >= d || slot >= h_slots) { return; }

    var grad = 0.0;
    for (var token = 0u; token < n_tokens; token = token + 1u) {
        let tk = token * h_slots + slot;
        let src_rms = max(slot_src_rms_cache(tk), 1.0e-6);
        grad = grad + slot_v_buf[entry_base(tk, d) + head] * src_norm_with_rms(token, slot, in_dim, d, h_slots, src_rms);
    }

    let lr = params.lr;
    let wd_factor = 1.0 - lr * params.weight_decay;
    let seq_scale = 1.0 / max(1.0, f32(n_tokens));
    let raw = lr * grad * seq_scale * params.grad_scale * SLOT_COORD_VALUE_SCALE;
    let idx = v_weight_idx(slot, in_dim, head, d);
    let clip = 0.5;
    if (params.grad_accum_mode == 1u) {
        AllGradients[idx] = AllGradients[idx] + raw;
    } else {
        AllWeights[idx] = AllWeights[idx] * wd_factor - clamp(raw, -clip, clip);
    }
}

@compute
@workgroup_size(64, 1, 1)
fn slot_coord_stage4_bias_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!SLOT_COORD_TRAIN_BIAS) { return; }
    let head = gid.x;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let hd = head_dim(d);
    let n_tokens = params.batch_size * params.seq_len;
    if (head >= hd) { return; }

    let lr = params.lr;
    let wd_factor = 1.0 - lr * params.weight_decay;
    let seq_scale = 1.0 / max(1.0, f32(n_tokens));
    let raw_scale = lr * seq_scale * params.grad_scale * SLOT_COORD_QK_SCALE;
    let clip = 0.5;

    for (var slot = 0u; slot < h_slots; slot = slot + 1u) {
        var g_qb = 0.0;
        var g_kb = 0.0;
        for (var token = 0u; token < n_tokens; token = token + 1u) {
            let entry = token * h_slots + slot;
            let tk = entry;
            g_qb = g_qb + qgrad_buf[entry_base(entry, d) + head];
            g_kb = g_kb + slot_k_work_buf[entry_base(tk, d) + head];
        }
        let q_idx = q_bias_idx(slot, head, d);
        let k_idx = k_bias_idx(slot, head, d);
        let raw_q = raw_scale * g_qb;
        let raw_k = raw_scale * g_kb;
        if (params.grad_accum_mode == 1u) {
            AllGradients[q_idx] = AllGradients[q_idx] + raw_q;
            AllGradients[k_idx] = AllGradients[k_idx] + raw_k;
        } else {
            AllWeights[q_idx] = AllWeights[q_idx] * wd_factor - clamp(raw_q, -clip, clip);
            AllWeights[k_idx] = AllWeights[k_idx] * wd_factor - clamp(raw_k, -clip, clip);
        }
    }
}
