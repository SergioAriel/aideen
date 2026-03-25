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
};

@group(0) @binding(0) var<uniform> params: UpdateUniforms;
@group(0) @binding(1) var<storage, read> v_state: array<f32>;
@group(0) @binding(2) var<storage, read> q_input_unused: array<f32>;
@group(0) @binding(3) var<storage, read> H_star: array<f32>;
@group(0) @binding(4) var<storage, read_write> debug_log: array<f32>;
@group(0) @binding(5) var<storage, read> b_in: array<f32>;
@group(0) @binding(6) var<storage, read> Scratch: array<f32>;
@group(0) @binding(7) var<storage, read_write> gcomb_buf: array<f32>;
@group(0) @binding(8) var<storage, read_write> v_next: array<f32>;
@group(0) @binding(9) var<storage, read_write> gmix_buf: array<f32>;
@group(0) @binding(10) var<storage, read_write> gscore_buf: array<f32>;
@group(0) @binding(11) var<storage, read_write> qgrad_buf: array<f32>;
@group(0) @binding(12) var<storage, read_write> rhs_slot_buf: array<f32>;

@group(1) @binding(0) var<storage, read_write> W_q: array<f32>;
@group(1) @binding(1) var<storage, read_write> W_k: array<f32>;
@group(1) @binding(2) var<storage, read_write> W_v: array<f32>;
@group(1) @binding(3) var<storage, read_write> W_o: array<f32>;
@group(1) @binding(8) var<storage, read_write> NormScale: array<f32>;
@group(1) @binding(9) var<storage, read_write> HistParams: array<f32>;

fn entry_base(entry: u32, d: u32) -> u32 {
    return entry * d;
}

fn scratch_stride(d: u32, h_slots: u32) -> u32 {
    return d * (h_slots * 7u) + h_slots * h_slots + h_slots;
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
    return hist_delta_base(d, h_slots) + d * d;
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
    let warmup = clamp(HistParams[hist_warmup_base(d, h_slots)], 0.0, 1.0);
    return hist_alpha_min_start() + (hist_alpha_min_target() - hist_alpha_min_start()) * warmup;
}

fn token_mamba_base(t: u32, d: u32, h_slots: u32) -> u32 {
    return t * scratch_stride(d, h_slots) + h_slots * 4u * d;
}

@compute
@workgroup_size(16, 16, 1)
fn picard_init_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let entry = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (dim >= d || entry >= n_entries) { return; }

    let t = entry / h_slots;
    let slot = entry % h_slots;

    // Compute attention-received weight for this slot from Scratch (stop-gradient).
    // Matches the 70/30 mix used in deq_forward.wgsl pooling — consistent forward/backward.
    let base = t * scratch_stride(d, h_slots);
    let attn_w_base = base + d * (h_slots * 7u);
    var recv = 0.0;
    for (var q = 0u; q < h_slots; q = q + 1u) {
        recv = recv + Scratch[attn_w_base + q * h_slots + slot];
    }
    let w_uniform = 1.0 / f32(h_slots);
    let w_attn = recv / f32(h_slots); // normalized: total received = h_slots
    let w_s = 0.7 * w_attn + 0.3 * w_uniform;

    let rhs = b_in[t * d + dim] * w_s
        + rhs_slot_buf[entry_base(entry, d) + dim];
    v_next[entry_base(entry, d) + dim] = rhs;
}

@compute
@workgroup_size(64, 1, 1)
fn picard_gcomb_main(@builtin(local_invocation_id) lid: vec3<u32>,
                     @builtin(workgroup_id) wid: vec3<u32>) {
    let lane = lid.x;
    let entry = wid.x;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (entry >= n_entries) { return; }

    let t = entry / h_slots;
    let t_within = t % params.seq_len;
    let qs = entry % h_slots;
    let q_off = qs * d;
    let t_off = t * h_slots * d;
    let base = t * scratch_stride(d, h_slots);
    let attn_base = base + h_slots * d * 3u;
    let signal_base = attn_base + 2u * h_slots * d;
    let slot_anchor = slot_anchor_base(d, h_slots);
    let hist_gated_mode = params.residual_alpha > -0.75 && params.residual_alpha <= -0.5;

    var inj_sumsq = 0.0;
    var hist_sumsq = 0.0;
    var hist_alpha = 0.0;
    var hist_scale = 0.0;
    var prev_rms = 1.0;
    if (hist_gated_mode) {
        // Match forward: normalize previous M by its RMS before applying W_hist.
        // t_within guards against cross-sequence contamination when batch_size > 1.
        if (t_within > 0u) {
            let prev_mamba = token_mamba_base(t - 1u, d, h_slots);
            var prev_sumsq = 0.0;
            for (var dim = 0u; dim < d; dim = dim + 1u) {
                let prev_v = Scratch[prev_mamba + q_off + dim];
                prev_sumsq = prev_sumsq + prev_v * prev_v;
            }
            prev_rms = sqrt(prev_sumsq / max(1.0, f32(d)) + 1e-6);
        }
        for (var dim = 0u; dim < d; dim = dim + 1u) {
            let inj = Scratch[signal_base + q_off + dim];
            inj_sumsq = inj_sumsq + inj * inj;
            if (t_within > 0u) {
                let prev_mamba = token_mamba_base(t - 1u, d, h_slots);
                var u = HistParams[hist_scale_base(d, h_slots) + q_off + dim]
                    * (Scratch[prev_mamba + q_off + dim] / prev_rms);
                for (var j = 0u; j < d; j = j + 1u) {
                    let prev_j = Scratch[prev_mamba + q_off + j] / prev_rms;
                    u = u + HistParams[dim * d + j] * prev_j;
                }
                hist_sumsq = hist_sumsq + u * u;
            }
        }
        let inj_rms = sqrt(inj_sumsq / max(1.0, f32(d)) + 1e-6);
        let hist_rms = sqrt(hist_sumsq / max(1.0, f32(d)) + 1e-6);
        hist_scale = min(1.0, inj_rms / max(hist_rms, 1e-6));
        let gate_logit = HistParams[hist_gate_base(d, h_slots) + qs];
        let amin = hist_alpha_min(d, h_slots);
        let amax = hist_alpha_max();
        hist_alpha = amin + (amax - amin) * (1.0 / (1.0 + exp(-gate_logit)));
    }

    var sumsq = 0.0;
    var coeff = 0.0;
    for (var dim = 0u; dim < d; dim = dim + 1u) {
        var hist_ctx = 0.0;
        if (hist_gated_mode && t_within > 0u) {
            let prev_mamba = token_mamba_base(t - 1u, d, h_slots);
            var u = HistParams[hist_scale_base(d, h_slots) + q_off + dim]
                * (Scratch[prev_mamba + q_off + dim] / prev_rms);
            for (var j = 0u; j < d; j = j + 1u) {
                let prev_j = Scratch[prev_mamba + q_off + j] / prev_rms;
                u = u + HistParams[dim * d + j] * prev_j;
            }
            hist_ctx = hist_alpha * hist_scale * u;
        }
        let attn_signal = Scratch[attn_base + q_off + dim]
            + Scratch[signal_base + q_off + dim];
        let z_full = attn_signal + hist_ctx + HistParams[slot_anchor + q_off + dim];
        let up = params.damping * v_state[t_off + q_off + dim];
        sumsq = sumsq + attn_signal * attn_signal;
        coeff = coeff + up * NormScale[dim] * z_full;
    }
    let rms_floor = HistParams[hist_rms_floor_base(d, h_slots)];
    let rms = sqrt(sumsq / max(1.0, f32(d)) + rms_floor * rms_floor + 1e-6);

    for (var dim = lane; dim < d; dim = dim + 64u) {
        var hist_ctx = 0.0;
        if (hist_gated_mode && t_within > 0u) {
            let prev_mamba = token_mamba_base(t - 1u, d, h_slots);
            var u = HistParams[hist_scale_base(d, h_slots) + q_off + dim]
                * (Scratch[prev_mamba + q_off + dim] / prev_rms);
            for (var j = 0u; j < d; j = j + 1u) {
                let prev_j = Scratch[prev_mamba + q_off + j] / prev_rms;
                u = u + HistParams[dim * d + j] * prev_j;
            }
            hist_ctx = hist_alpha * hist_scale * u;
        }
        let attn_signal = Scratch[attn_base + q_off + dim]
            + Scratch[signal_base + q_off + dim];
        let z_full = attn_signal + hist_ctx + HistParams[slot_anchor + q_off + dim];
        let up = params.damping * v_state[t_off + q_off + dim];
        let g = (NormScale[dim] / max(rms, 1e-6)) * up
            - attn_signal * coeff / (max(1.0, f32(d)) * max(rms * rms * rms, 1e-6));
        gcomb_buf[entry_base(entry, d) + dim] = g;
    }
}

@compute
@workgroup_size(64, 1, 1)
fn picard_gmix_main(@builtin(local_invocation_id) lid: vec3<u32>,
                    @builtin(workgroup_id) wid: vec3<u32>) {
    let lane = lid.x;
    let entry = wid.x;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (entry >= n_entries) { return; }
    let slot = entry % h_slots;
    let wo_base = slot * d * d;

    for (var dim = lane; dim < d; dim = dim + 64u) {
        var gmix = 0.0;
        for (var dout = 0u; dout < d; dout = dout + 1u) {
            gmix = gmix + W_o[wo_base + dout * d + dim] * gcomb_buf[entry_base(entry, d) + dout];
        }
        gmix_buf[entry_base(entry, d) + dim] = gmix;
    }
}

// Fused GMix + GScore: computes gmix_buf and gscore_buf (and qgrad) in one dispatch.
// Reduces one dispatch per Picard iteration.
@compute
@workgroup_size(64, 1, 1)
fn picard_gmix_gscore_main(@builtin(local_invocation_id) lid: vec3<u32>,
                           @builtin(workgroup_id) wid: vec3<u32>) {
    let lane = lid.x;
    let entry = wid.x;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (entry >= n_entries) { return; }
    let slot = entry % h_slots;
    let wo_base = slot * d * d;

    // Compute gmix for this entry.
    for (var dim = lane; dim < d; dim = dim + 64u) {
        var gmix = 0.0;
        for (var dout = 0u; dout < d; dout = dout + 1u) {
            gmix = gmix + W_o[wo_base + dout * d + dim] * gcomb_buf[entry_base(entry, d) + dout];
        }
        gmix_buf[entry_base(entry, d) + dim] = gmix;
    }
    workgroupBarrier();

    // Compute gscore for all ks for this entry.
    let t = entry / h_slots;
    let qs = entry % h_slots;
    let base = t * scratch_stride(d, h_slots);
    let v_base = base + h_slots * d * 2u;
    let k_base = base + h_slots * d;
    let q_base = base;
    let attn_weight_base = base + d * (h_slots * 7u);
    let off = entry_base(entry, d);
    for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
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
    workgroupBarrier();

    // qgrad for this entry (per-slot).
    let scale = inverseSqrt(max(1.0, f32(d)));
    for (var dim = lane; dim < d; dim = dim + 64u) {
        var gq = 0.0;
        for (var j = 0u; j < h_slots; j = j + 1u) {
            gq = gq + scale * gscore_buf[entry * h_slots + j]
                * Scratch[k_base + j * d + dim];
        }
        qgrad_buf[off + dim] = gq;
    }
}

@compute
@workgroup_size(16, 16, 1)
fn picard_gscore_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ks = gid.x;
    let entry = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (ks >= h_slots || entry >= n_entries) { return; }

    let t = entry / h_slots;
    let qs = entry % h_slots;
    let base = t * scratch_stride(d, h_slots);
    let v_base = base + h_slots * d * 2u;
    let k_base = base + h_slots * d;
    let q_base = base;
    let attn_weight_base = base + d * (h_slots * 7u);
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
    storageBarrier();
    workgroupBarrier();

    if (ks == 0u) {
        let scale = inverseSqrt(max(1.0, f32(d)));
        for (var dim = 0u; dim < d; dim = dim + 1u) {
            var gq = 0.0;
            for (var j = 0u; j < h_slots; j = j + 1u) {
                gq = gq + scale * gscore_buf[entry * h_slots + j] * Scratch[k_base + j * d + dim];
            }
            qgrad_buf[off + dim] = gq;
        }
    }
}

@compute
@workgroup_size(16, 16, 1)
fn picard_accum_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let entry = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (dim >= d || entry >= n_entries) { return; }

    let t = entry / h_slots;
    let target_slot = entry % h_slots;
    let t_off = t * h_slots * d;
    let target_off = target_slot * d;
    let base = t * scratch_stride(d, h_slots);
    let q_base = base;
    let attn_weight_base = base + d * (h_slots * 7u);
    let scale = inverseSqrt(max(1.0, f32(d)));
    let deq_only_mode = params.residual_alpha <= -1.5;

    var jt_v = (1.0 - params.damping) * v_state[t_off + target_off + dim];

    if (!deq_only_mode) {
        var v_path_acc = 0.0;
        var k_path_acc = 0.0;
        var q_path_acc = 0.0;

        for (var qs = 0u; qs < h_slots; qs = qs + 1u) {
            let qs_off = qs * d;
            let src_entry = t * h_slots + qs;
            let src_off = entry_base(src_entry, d);
            let alpha_qt = Scratch[attn_weight_base + qs * h_slots + target_slot];

            var dot_v = 0.0;
            for (var vd = 0u; vd < d; vd = vd + 1u) {
                dot_v = dot_v + W_v[target_slot * d * d + dim * d + vd] * gmix_buf[src_off + vd];
            }
            v_path_acc = v_path_acc + alpha_qt * dot_v;

            let g_score_t = gscore_buf[src_entry * h_slots + target_slot];
            for (var qd = 0u; qd < d; qd = qd + 1u) {
                k_path_acc = k_path_acc
                    + W_k[dim * d + qd] * (scale * g_score_t * Scratch[q_base + qs_off + qd]);
            }

            if (target_slot == qs) {
                for (var qd = 0u; qd < d; qd = qd + 1u) {
                    q_path_acc = q_path_acc + W_q[dim * d + qd] * qgrad_buf[src_off + qd];
                }
            }
        }
        jt_v = jt_v + v_path_acc + k_path_acc + q_path_acc;
    }

    let b = b_in[t * d + dim] / max(1.0, f32(h_slots))
        + rhs_slot_buf[entry_base(entry, d) + dim];
    v_next[t_off + target_off + dim] = jt_v + b;
}

@compute
@workgroup_size(16, 16, 1)
fn picard_accum_v_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let entry = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (dim >= d || entry >= n_entries) { return; }

    let t = entry / h_slots;
    let target_slot = entry % h_slots;
    let base = t * scratch_stride(d, h_slots);
    let attn_weight_base = base + d * (h_slots * 7u);

    var v_path_acc = 0.0;
    for (var qs = 0u; qs < h_slots; qs = qs + 1u) {
        let src_entry = t * h_slots + qs;
        let src_off = entry_base(src_entry, d);
        let alpha_qt = Scratch[attn_weight_base + qs * h_slots + target_slot];
        var dot_v = 0.0;
        for (var vd = 0u; vd < d; vd = vd + 1u) {
            dot_v = dot_v + W_v[target_slot * d * d + vd * d + dim] * gmix_buf[src_off + vd];
        }
        v_path_acc = v_path_acc + alpha_qt * dot_v;
    }
    gcomb_buf[entry_base(entry, d) + dim] = v_path_acc;
}

@compute
@workgroup_size(16, 16, 1)
fn picard_accum_k_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let entry = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (dim >= d || entry >= n_entries) { return; }

    let t = entry / h_slots;
    let target_slot = entry % h_slots;
    let base = t * scratch_stride(d, h_slots);
    let q_base = base;
    let scale = inverseSqrt(max(1.0, f32(d)));

    var k_path_acc = 0.0;
    for (var qs = 0u; qs < h_slots; qs = qs + 1u) {
        let qs_off = qs * d;
        let src_entry = t * h_slots + qs;
        let g_score_t = gscore_buf[src_entry * h_slots + target_slot];
        for (var qd = 0u; qd < d; qd = qd + 1u) {
            k_path_acc = k_path_acc
                + W_k[qd * d + dim] * (scale * g_score_t * Scratch[q_base + qs_off + qd]);
        }
    }
    gcomb_buf[entry_base(entry, d) + dim] = k_path_acc;
}

@compute
@workgroup_size(16, 16, 1)
fn picard_accum_q_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let entry = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (dim >= d || entry >= n_entries) { return; }

    let t = entry / h_slots;
    let target_slot = entry % h_slots;

    var q_path_acc = 0.0;
    let src_entry = t * h_slots + target_slot;
    let src_off = entry_base(src_entry, d);
    for (var qd = 0u; qd < d; qd = qd + 1u) {
        q_path_acc = q_path_acc + W_q[qd * d + dim] * qgrad_buf[src_off + qd];
    }
    gcomb_buf[entry_base(entry, d) + dim] = q_path_acc;
}

// ============================================================
// Anderson Acceleration — @group(1) for Anderson pipelines only
// (Different BGL from the Picard weight group(1) above)
// ============================================================

struct AndersonParams {
    m: u32,   // ring buffer depth; effective Anderson window = m - 1
    k: u32,   // current Picard iteration index (0-indexed)
};

@group(1) @binding(0) var<uniform> anderson: AndersonParams;
@group(1) @binding(1) var<storage, read_write> anderson_hist: array<f32>;
// anderson_hist layout: [m × n_entries × d_model] flat
// slot for iteration k: (k % m) * attn_len + entry * d + dim

// Workgroup-scope scratch for anderson_mix_main (one instance per workgroup at runtime)
var<workgroup> anderson_gram: array<f32, 9>;  // up to 3×3, indexed [ri*3+rj]
var<workgroup> anderson_beta: array<f32, 3>;  // mixing coefficients

// Store current v_next into ring-buffer slot k%m
@compute @workgroup_size(256, 1, 1)
fn anderson_store_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let d = params.d_model;
    let n_entries = params.batch_size * params.seq_len * params.h_slots;
    let attn_len = n_entries * d;
    if (idx >= attn_len) { return; }
    let slot = anderson.k % anderson.m;
    anderson_hist[slot * attn_len + idx] = v_next[idx];
}

// Per-token Anderson mixing.
// Dispatch: one workgroup per token (batch_size × seq_len), WG_SIZE=64.
// Computes Gram matrix from pseudo-residuals Δ_i = hist[(k-i)%m] - hist[(k-i-1+m)%m],
// solves the constrained LS (G β = 1, normalize), overwrites v_next with mixed iterate.
@compute @workgroup_size(64, 1, 1)
fn anderson_mix_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let token = wid.x;
    let lane  = lid.x;
    let d       = params.d_model;
    let h_slots = params.h_slots;
    let n_tokens = params.batch_size * params.seq_len;
    if (token >= n_tokens) { return; }

    let n_entries = n_tokens * h_slots;
    let attn_len  = n_entries * d;
    let m = anderson.m;
    let k = anderson.k;

    // n_res = valid consecutive pseudo-residual pairs in the ring buffer.
    // A depth-m ring buffer supports at most m-1 valid consecutive pairs.
    let n_res = min(k, m - 1u);
    // Need ≥2 residuals for a non-trivial 2×2 or 3×3 solve.
    if (n_res < 2u) { return; }

    if (lane == 0u) {
        for (var i = 0u; i < 9u; i = i + 1u) { anderson_gram[i] = 0.0; }

        // Gram[ri][rj] = <Δ_ri, Δ_rj>   (Δ_ri = hist[(k-ri)%m] - hist[(k-ri-1+m)%m])
        for (var ri = 0u; ri < n_res; ri = ri + 1u) {
            for (var rj = ri; rj < n_res; rj = rj + 1u) {
                let si  = (k + m - ri      ) % m;
                let si1 = (k + m - ri - 1u ) % m;
                let sj  = (k + m - rj      ) % m;
                let sj1 = (k + m - rj - 1u ) % m;
                var dot = 0.0;
                for (var e = 0u; e < h_slots; e = e + 1u) {
                    let entry = token * h_slots + e;
                    let base  = entry * d;
                    for (var dim2 = 0u; dim2 < d; dim2 = dim2 + 1u) {
                        let da = anderson_hist[si  * attn_len + base + dim2]
                               - anderson_hist[si1 * attn_len + base + dim2];
                        let db = anderson_hist[sj  * attn_len + base + dim2]
                               - anderson_hist[sj1 * attn_len + base + dim2];
                        dot = dot + da * db;
                    }
                }
                anderson_gram[ri * 3u + rj] = dot;
                anderson_gram[rj * 3u + ri] = dot;
            }
        }

        // Solve G β = 1 via Cramer's rule, then normalize (β sums to 1).
        if (n_res == 2u) {
            var g00 = anderson_gram[0]; let g01 = anderson_gram[1];
            let g10 = anderson_gram[3]; var g11 = anderson_gram[4];
            // Tikhonov regularization to avoid ill-conditioned Gram (stabilizes coefficients).
            let reg = max(1e-8, 1e-4 * (abs(g00) + abs(g11)));
            g00 = g00 + reg;
            g11 = g11 + reg;
            let det = g00 * g11 - g01 * g10;
            let sdet = select(max(det, 1e-10), min(det, -1e-10), det < 0.0);
            anderson_beta[0] = (g11 - g01) / sdet;
            anderson_beta[1] = (g00 - g10) / sdet;
        } else {
            // n_res == 3
            var g00 = anderson_gram[0]; let g01 = anderson_gram[1]; let g02 = anderson_gram[2];
            let g10 = anderson_gram[3]; var g11 = anderson_gram[4]; let g12 = anderson_gram[5];
            let g20 = anderson_gram[6]; let g21 = anderson_gram[7]; var g22 = anderson_gram[8];
            // Tikhonov regularization on diagonal to stabilize solve.
            let reg = max(1e-8, 1e-4 * (abs(g00) + abs(g11) + abs(g22)));
            g00 = g00 + reg;
            g11 = g11 + reg;
            g22 = g22 + reg;
            let det = g00*(g11*g22 - g12*g21)
                    - g01*(g10*g22 - g12*g20)
                    + g02*(g10*g21 - g11*g20);
            let sdet = select(max(det, 1e-8), min(det, -1e-8), det < 0.0);
            anderson_beta[0] = (1.0*(g11*g22 - g12*g21) - g01*(g22 - g12) + g02*(g21 - g11)) / sdet;
            anderson_beta[1] = (g00*(g22 - g12) - 1.0*(g10*g22 - g12*g20) + g02*(g10 - g20)) / sdet;
            anderson_beta[2] = (g00*(g11 - g21) - g01*(g10 - g20) + 1.0*(g10*g21 - g11*g20)) / sdet;
        }

        // Clamp extreme values and renormalize
        var s = 0.0;
        for (var i = 0u; i < n_res; i = i + 1u) {
            anderson_beta[i] = clamp(anderson_beta[i], -2.0, 3.0);
            s = s + anderson_beta[i];
        }
        if (abs(s) < 1e-6) {
            for (var i = 0u; i < n_res; i = i + 1u) { anderson_beta[i] = 0.0; }
            anderson_beta[0] = 1.0;  // fall back: use most recent iterate
        } else {
            for (var i = 0u; i < n_res; i = i + 1u) { anderson_beta[i] = anderson_beta[i] / s; }
        }
    }
    workgroupBarrier();

    // Apply: v_next[entry, dim] = Σ_i β[i] * hist[(k-i+m)%m, entry, dim]
    for (var e = 0u; e < h_slots; e = e + 1u) {
        let entry = token * h_slots + e;
        for (var dim2 = lane; dim2 < d; dim2 = dim2 + 64u) {
            var mixed = 0.0;
            for (var i = 0u; i < n_res; i = i + 1u) {
                let hi = (k + m - i) % m;
                mixed = mixed + anderson_beta[i] * anderson_hist[hi * attn_len + entry * d + dim2];
            }
            v_next[entry * d + dim2] = mixed;
        }
    }
}
