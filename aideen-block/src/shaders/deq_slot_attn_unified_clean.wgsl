struct RunUniforms {
    batch_size: u32,
    d_model: u32,
    h_slots: u32,
    max_iters: u32,
    epsilon: f32,
    damping: f32,
    seq_len: u32,
    residual_alpha: f32,
    debug_enable: u32,
    token_start: u32,
    token_count: u32,
    diag_zero_win: u32,
    diag_one_iter: u32,
    _pad0: vec3<u32>,
}

@group(0) @binding(0) var<uniform> shape: RunUniforms;
@group(0) @binding(1) var<storage, read> S_in: array<f32>;
@group(0) @binding(2) var<storage, read> AllWeights: array<f32>;
@group(0) @binding(3) var<storage, read_write> H_curr: array<f32>;
@group(0) @binding(4) var<storage, read_write> H_next: array<f32>;
@group(0) @binding(5) var<storage, read_write> Scratch: array<f32>;
@group(0) @binding(7) var<storage, read_write> DebugLog: array<f32>;
@group(0) @binding(8) var<storage, read_write> HistCtx: array<f32>;

override SLOT_ATTN_HEAD_DIM: u32 = 32u;
override ENABLE_TOKEN_CARRY: bool = true;
override ENABLE_H_HIST: bool = false;
const WG_SIZE: u32 = 256u;
const MAX_SLOTS: u32 = 8u;
const MAX_SLOT_ATTN_HEAD_DIM: u32 = 32u;
const SLOT_HEAD_CAP: u32 = MAX_SLOTS * MAX_SLOT_ATTN_HEAD_DIM;

var<workgroup> slot_attn_weights: array<f32, MAX_SLOTS>;
var<workgroup> shared_vals: array<f32, WG_SIZE>;
var<workgroup> q_self: array<f32, MAX_SLOT_ATTN_HEAD_DIM>;
var<workgroup> k_cache: array<f32, SLOT_HEAD_CAP>;
var<workgroup> v_cache: array<f32, SLOT_HEAD_CAP>;
var<workgroup> head_mix: array<f32, MAX_SLOT_ATTN_HEAD_DIM>;
var<workgroup> max_delta_seen: f32;
var<workgroup> last_delta: f32;
var<workgroup> max_contractivity: f32;
var<workgroup> max_h_seen: f32;
var<workgroup> total_iters_seen: u32;
var<workgroup> failed_hits_seen: u32;
// Learnable raw γ per slot for h_currSSM (broadcast from tid==0, squashed before use)
var<workgroup> hhist_gamma_wg: f32;
const H_HIST_GAMMA_SCALE: f32 = 0.1;

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
fn aw_alog_base(d: u32, h: u32) -> u32 { return aw_nscale_base(d, h) - h * d; }
fn aw_hist_base(d: u32, h: u32) -> u32 { return aw_nscale_base(d, h) + d; }
fn hist_mat_len(d: u32) -> u32 { return d * d; }
fn hist_scale_base(d: u32, h_slots: u32) -> u32 { return hist_mat_len(d); }
fn hist_bias_base(d: u32, h_slots: u32) -> u32 { return hist_scale_base(d, h_slots) + h_slots * d; }
fn hist_gate_base(d: u32, h_slots: u32) -> u32 { return hist_bias_base(d, h_slots) + h_slots * d; }
fn slot_anchor_base(d: u32, h_slots: u32) -> u32 { return hist_gate_base(d, h_slots) + h_slots; }
// γ per slot: after slot_anchor(h*d) + W_delta(h*d²) + b_delta(d) + 21 flags + W_gate_hist(h*d) + W_forget(h*d) + b_forget(h)
fn hhist_gamma_base(d: u32, h: u32) -> u32 {
    return slot_anchor_base(d, h) + h * d + h * d * d + d + 21u + 2u * h * d + h;
}

@group(0) @binding(11) var<storage, read_write> H_hist: array<f32>;

override ENABLE_FPM: bool = false;
// FPM inject scale: pre += m * FPM_INJECT_SCALE, excluded from rms (same as slot_anchor).
// |m| ≤ 1 by construction (EMA of tanh), so injection ≤ 0.02 — very conservative.
const FPM_INJECT_SCALE: f32 = 0.02;
// FPM memory decay: m ← (1-FPM_ALPHA)*m + FPM_ALPHA*tanh(h*)
const FPM_ALPHA: f32 = 0.05;

fn compute_slot_attn(
    tid: u32,
    slot_idx: u32,
    d_model: u32,
    h_slots: u32,
    head_dim: u32,
    h_base: u32,
    batch_scratch_t: u32,
    wq_mat_base: u32,
    wk_bias_root: u32,
    wq_bias_base: u32,
    wo_mat_base: u32,
    attn_base: u32,
) {
    if (head_dim == 32u) {
        let half_head = head_dim / 2u;
        let active_lanes = h_slots * half_head;
        if (tid < active_lanes) {
            let ks = tid / half_head;
            let hd0 = tid % half_head;
            let hd1 = hd0 + half_head;
            let ks_off = ks * d_model;
            let wk_mat_base = aw_wk_base(d_model, h_slots) + ks_off * d_model;
            let wv_mat_base = aw_wv_base(d_model, h_slots) + ks_off * d_model;
            let wk_bias_base = wk_bias_root + ks_off;
            var k0 = AllWeights[wk_bias_base + hd0];
            var k1 = AllWeights[wk_bias_base + hd1];
            var v0 = 0.0;
            var v1 = 0.0;
            var q0 = 0.0;
            var q1 = 0.0;
            if (ks == slot_idx) {
                q0 = AllWeights[wq_bias_base + hd0];
                q1 = AllWeights[wq_bias_base + hd1];
            }
            for (var j = 0u; j < d_model; j = j + 1u) {
                let src_val = select(
                    H_curr[h_base + ks_off + j],
                    Scratch[batch_scratch_t + ks_off + j],
                    true,
                );
                let wk_row = wk_mat_base + j * d_model;
                let wv_row = wv_mat_base + j * d_model;
                k0 = k0 + AllWeights[wk_row + hd0] * src_val;
                k1 = k1 + AllWeights[wk_row + hd1] * src_val;
                v0 = v0 + AllWeights[wv_row + hd0] * src_val;
                v1 = v1 + AllWeights[wv_row + hd1] * src_val;
                if (ks == slot_idx) {
                    let wq_row = wq_mat_base + j * d_model;
                    q0 = q0 + AllWeights[wq_row + hd0] * src_val;
                    q1 = q1 + AllWeights[wq_row + hd1] * src_val;
                }
            }
            let idx0 = ks * head_dim + hd0;
            let idx1 = idx0 + half_head;
            k_cache[idx0] = k0;
            k_cache[idx1] = k1;
            v_cache[idx0] = v0;
            v_cache[idx1] = v1;
            if (ks == slot_idx) {
                q_self[hd0] = q0;
                q_self[hd1] = q1;
            }
        }
    } else {
        for (var idx = tid; idx < h_slots * head_dim; idx = idx + WG_SIZE) {
            let ks = idx / head_dim;
            let hd = idx % head_dim;
            let ks_off = ks * d_model;
            let wk_mat_base = aw_wk_base(d_model, h_slots) + ks_off * d_model;
            let wv_mat_base = aw_wv_base(d_model, h_slots) + ks_off * d_model;
            let wk_bias_base = wk_bias_root + ks_off;
            var k = AllWeights[wk_bias_base + hd];
            var v = 0.0;
            var q = 0.0;
            if (ks == slot_idx) {
                q = AllWeights[wq_bias_base + hd];
            }
            for (var j = 0u; j < d_model; j = j + 1u) {
                let src_val = select(
                    H_curr[h_base + ks_off + j],
                    Scratch[batch_scratch_t + ks_off + j],
                    true,
                );
                k = k + AllWeights[wk_mat_base + j * d_model + hd] * src_val;
                v = v + AllWeights[wv_mat_base + j * d_model + hd] * src_val;
                if (ks == slot_idx) {
                    q = q + AllWeights[wq_mat_base + j * d_model + hd] * src_val;
                }
            }
            k_cache[idx] = k;
            v_cache[idx] = v;
            if (ks == slot_idx) {
                q_self[hd] = q;
            }
        }
    }
    workgroupBarrier();

    if (tid < h_slots) {
        let ks = tid;
        var score = 0.0;
        let ks_head = ks * head_dim;
        for (var d = 0u; d < head_dim; d = d + 1u) {
            score = score + q_self[d] * k_cache[ks_head + d];
        }
        slot_attn_weights[ks] = score * inverseSqrt(max(1.0, f32(head_dim)));
    }
    workgroupBarrier();
    if (tid == 0u) {
        var max_s = -1e30;
        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
            max_s = max(max_s, slot_attn_weights[ks]);
        }
        var sum_exp = 0.0;
        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
            let w = exp(slot_attn_weights[ks] - max_s);
            slot_attn_weights[ks] = w;
            sum_exp = sum_exp + w;
        }
        sum_exp = max(sum_exp, 1e-12);
        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
            slot_attn_weights[ks] = slot_attn_weights[ks] / sum_exp;
        }
    }
    workgroupBarrier();

    for (var h = tid; h < head_dim; h = h + WG_SIZE) {
        var mix = 0.0;
        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
            mix = mix + slot_attn_weights[ks] * v_cache[ks * head_dim + h];
        }
        head_mix[h] = mix;
    }
    workgroupBarrier();

    if (d_model == WG_SIZE * 2u) {
        let d0 = tid;
        let d1 = tid + WG_SIZE;
        var attn0 = 0.0;
        var attn1 = 0.0;
        for (var h = 0u; h < head_dim; h = h + 1u) {
            let mix = head_mix[h];
            attn0 = attn0 + AllWeights[wo_mat_base + h * d_model + d0] * mix;
            attn1 = attn1 + AllWeights[wo_mat_base + h * d_model + d1] * mix;
        }
        Scratch[attn_base + d0] = attn0;
        Scratch[attn_base + d1] = attn1;
    } else {
        for (var d = tid; d < d_model; d = d + WG_SIZE) {
            var attn = 0.0;
            for (var h = 0u; h < head_dim; h = h + 1u) {
                attn = attn + AllWeights[wo_mat_base + h * d_model + d] * head_mix[h];
            }
            Scratch[attn_base + d] = attn;
        }
    }
    workgroupBarrier();
}

@compute @workgroup_size(256, 1, 1)
fn deq_slot_attn_unified_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.x;
    let slot_idx = wid.y;
    if (batch_idx >= shape.batch_size || slot_idx >= shape.h_slots) { return; }

    let d_model = shape.d_model;
    let h_slots = shape.h_slots;
    let head_dim = min(d_model, SLOT_ATTN_HEAD_DIM);
    let total_elements = h_slots * d_model;
    let slot_offset = slot_idx * d_model;
    let h_base = batch_idx * total_elements;
    let signal_span = d_model * h_slots;
    let scratch_stride = signal_span * 2u;
    let wq_mat_base = aw_wq_base(d_model, h_slots) + slot_idx * d_model * d_model;
    let wk_bias_root = aw_wk_base(d_model, h_slots) + h_slots * d_model * d_model;
    let wq_bias_base = aw_wq_base(d_model, h_slots) + h_slots * d_model * d_model + slot_idx * d_model;
    let wo_mat_base = aw_wo_base(d_model, h_slots) + slot_idx * d_model * d_model;
    let win_base = aw_win_base(d_model, h_slots) + slot_idx * d_model * d_model;
    let slot_anchor_mat_base = aw_hist_base(d_model, h_slots) + slot_anchor_base(d_model, h_slots) + slot_offset;
    let nscale_base = aw_nscale_base(d_model, h_slots);
    let inv_d_model = 1.0 / max(1.0, f32(d_model));
    let zero_win_diag = shape.diag_zero_win != 0u;
    let iter_limit = select(shape.max_iters, 1u, shape.diag_one_iter != 0u);

    if (tid == 0u) {
        max_delta_seen = 0.0;
        last_delta = 0.0;
        max_contractivity = 0.0;
        max_h_seen = 0.0;
        total_iters_seen = 0u;
        failed_hits_seen = 0u;
    }
    workgroupBarrier();

    for (var t = 0u; t < shape.token_count; t = t + 1u) {
        let global_t = shape.token_start + t;
        let batch_scratch_t = (batch_idx * shape.seq_len + global_t) * scratch_stride;
        let h_base_t = (batch_idx * shape.seq_len + global_t) * total_elements;
        let signal_base = batch_scratch_t + slot_offset;
        let attn_base = batch_scratch_t + signal_span + slot_offset;
        let s_in_base = (batch_idx * shape.seq_len + global_t) * d_model;

        if (d_model == WG_SIZE * 2u) {
            let d0 = tid;
            let d1 = tid + WG_SIZE;
            var inj0 = 0.0;
            var inj1 = 0.0;
            if (!zero_win_diag) {
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let s = S_in[s_in_base + j];
                    inj0 = inj0 + AllWeights[win_base + j * d_model + d0] * s;
                    inj1 = inj1 + AllWeights[win_base + j * d_model + d1] * s;
                }
            }
            Scratch[signal_base + d0] = inj0;
            Scratch[signal_base + d1] = inj1;
        } else {
            for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                var inj = 0.0;
                if (!zero_win_diag) {
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        inj = inj + AllWeights[win_base + j * d_model + d_out] * S_in[s_in_base + j];
                    }
                }
                Scratch[signal_base + d_out] = inj;
            }
        }
        workgroupBarrier();

        // H_hist / FPM state: loaded as stop-grad constant before Picard.
        // ENABLE_H_HIST and ENABLE_FPM are mutually exclusive — both use binding 11.
        var h_hist0 = 0.0;
        var h_hist1 = 0.0;
        if (ENABLE_H_HIST && d_model == WG_SIZE * 2u) {
            h_hist0 = H_hist[h_base + slot_offset + tid];
            h_hist1 = H_hist[h_base + slot_offset + tid + WG_SIZE];
            if (tid == 0u) {
                let gamma_raw = AllWeights[aw_hist_base(d_model, h_slots) + hhist_gamma_base(d_model, h_slots) + slot_idx];
                hhist_gamma_wg = H_HIST_GAMMA_SCALE / (1.0 + exp(-(gamma_raw - 8.0)));
            }
        } else if (ENABLE_FPM && d_model == WG_SIZE * 2u) {
            // FPM: m = bounded EMA of tanh(h*); |m| ≤ 1 by construction.
            h_hist0 = H_hist[h_base + slot_offset + tid];
            h_hist1 = H_hist[h_base + slot_offset + tid + WG_SIZE];
        }
        workgroupBarrier();

        if (global_t == 0u || !ENABLE_TOKEN_CARRY) {
            var local_sumsq0 = 0.0;
            // rms from signal only (h_hist excluded — stop-grad, same as hist_ctx)
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                let sig = Scratch[signal_base + d];
                local_sumsq0 = local_sumsq0 + sig * sig;
            }
            shared_vals[tid] = local_sumsq0;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let sig_rms = sqrt(shared_vals[0] * inv_d_model + 1e-6);
            if (ENABLE_H_HIST && d_model == WG_SIZE * 2u) {
                // signal normalized, then h_hist added as offset (after normalization)
                H_curr[h_base + slot_offset + tid] =
                    Scratch[signal_base + tid] / max(sig_rms, 1e-6) + hhist_gamma_wg * h_hist0;
                H_curr[h_base + slot_offset + tid + WG_SIZE] =
                    Scratch[signal_base + tid + WG_SIZE] / max(sig_rms, 1e-6) + hhist_gamma_wg * h_hist1;
            } else if (ENABLE_FPM && d_model == WG_SIZE * 2u) {
                // FPM: signal normalized + m injected as additive offset (stop-grad, excluded from rms)
                H_curr[h_base + slot_offset + tid] =
                    Scratch[signal_base + tid] / max(sig_rms, 1e-6) + h_hist0 * FPM_INJECT_SCALE;
                H_curr[h_base + slot_offset + tid + WG_SIZE] =
                    Scratch[signal_base + tid + WG_SIZE] / max(sig_rms, 1e-6) + h_hist1 * FPM_INJECT_SCALE;
            } else {
                for (var d = tid; d < d_model; d = d + WG_SIZE) {
                    H_curr[h_base + slot_offset + d] = Scratch[signal_base + d] / max(sig_rms, 1e-6);
                }
            }
            workgroupBarrier();
        }

        var iter = 0u;
        var converged = false;
        while (iter < iter_limit && !converged) {
            if (iter == 0u) {
                compute_slot_attn(
                    tid,
                    slot_idx,
                    d_model,
                    h_slots,
                    head_dim,
                    h_base,
                    batch_scratch_t,
                    wq_mat_base,
                    wk_bias_root,
                    wq_bias_base,
                    wo_mat_base,
                    attn_base,
                );
            }

            var local_sumsq = 0.0;
            if (ENABLE_H_HIST && d_model == WG_SIZE * 2u) {
                // h_hist is stop-grad (∂/∂h=0) → excluded from rms denominator
                // same principle as hist_ctx exclusion
                let d0 = tid;
                let d1 = tid + WG_SIZE;
                let h_prev0 = H_curr[h_base + slot_offset + d0];
                let h_prev1 = H_curr[h_base + slot_offset + d1];
                let h_dep0 = Scratch[signal_base + d0] + h_prev0 + Scratch[attn_base + d0]
                           + AllWeights[slot_anchor_mat_base + d0];
                let h_dep1 = Scratch[signal_base + d1] + h_prev1 + Scratch[attn_base + d1]
                           + AllWeights[slot_anchor_mat_base + d1];
                H_next[h_base_t + slot_offset + d0] = h_dep0 + hhist_gamma_wg * h_hist0;
                H_next[h_base_t + slot_offset + d1] = h_dep1 + hhist_gamma_wg * h_hist1;
                local_sumsq = h_dep0 * h_dep0 + h_dep1 * h_dep1; // rms from h-dep only
            } else if (ENABLE_FPM && d_model == WG_SIZE * 2u) {
                // FPM: m is stop-grad (∂/∂h=0) → excluded from rms denominator
                let d0 = tid;
                let d1 = tid + WG_SIZE;
                let h_prev0 = H_curr[h_base + slot_offset + d0];
                let h_prev1 = H_curr[h_base + slot_offset + d1];
                let h_dep0 = Scratch[signal_base + d0] + h_prev0 + Scratch[attn_base + d0]
                           + AllWeights[slot_anchor_mat_base + d0];
                let h_dep1 = Scratch[signal_base + d1] + h_prev1 + Scratch[attn_base + d1]
                           + AllWeights[slot_anchor_mat_base + d1];
                H_next[h_base_t + slot_offset + d0] = h_dep0 + h_hist0 * FPM_INJECT_SCALE;
                H_next[h_base_t + slot_offset + d1] = h_dep1 + h_hist1 * FPM_INJECT_SCALE;
                local_sumsq = h_dep0 * h_dep0 + h_dep1 * h_dep1; // rms from h-dep only
            } else {
                for (var d = tid; d < d_model; d = d + WG_SIZE) {
                    let h_prev = H_curr[h_base + slot_offset + d];
                    let pre = Scratch[signal_base + d]
                        + h_prev
                        + Scratch[attn_base + d]
                        + AllWeights[slot_anchor_mat_base + d];
                    H_next[h_base_t + slot_offset + d] = pre;
                    local_sumsq = local_sumsq + pre * pre;
                }
            }
            shared_vals[tid] = local_sumsq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let rms = sqrt(shared_vals[0] * inv_d_model + 1e-6);

            var local_max_delta = 0.0;
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                let h_prev = H_curr[h_base + slot_offset + d];
                let f_h = AllWeights[nscale_base + d] * (H_next[h_base_t + slot_offset + d] / rms);
                let val = shape.damping * f_h + (1.0 - shape.damping) * h_prev;
                local_max_delta = max(local_max_delta, abs(val - h_prev));
                H_curr[h_base + slot_offset + d] = val;
                H_next[h_base_t + slot_offset + d] = val;
            }
            shared_vals[tid] = local_max_delta;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
                }
                workgroupBarrier();
            }

            if (shared_vals[0] < shape.epsilon) {
                converged = true;
            }
            if (tid == 0u) {
                let d_curr = shared_vals[0];
                let d_prev = last_delta;
                if (iter > 0u && d_prev > 1e-12 && d_prev > shape.epsilon * 10.0) {
                    max_contractivity = max(max_contractivity, d_curr / d_prev);
                }
                last_delta = d_curr;
                max_delta_seen = max(max_delta_seen, d_curr);
            }
            iter = iter + 1u;
        }
        if (tid == 0u) {
            total_iters_seen = total_iters_seen + iter;
            if (!converged) {
                failed_hits_seen = failed_hits_seen + 1u;
            }
        }
        workgroupBarrier();

        // h_currSSM: h_ssm = a * h_ssm + (1-a) * h*
        //   a[d] = sigmoid(-A_log[d]) — per-dim time constant, zero extra params
        //   state lives directly in h* space, no projections needed
        if (ENABLE_H_HIST && d_model == WG_SIZE * 2u) {
            let d0 = tid;
            let d1 = tid + WG_SIZE;
            let alog_off = aw_alog_base(d_model, h_slots) + slot_offset;
            let a0 = 1.0 / (1.0 + exp(AllWeights[alog_off + d0]));  // sigmoid(-A_log)
            let a1 = 1.0 / (1.0 + exp(AllWeights[alog_off + d1]));
            let h_star0 = H_curr[h_base + slot_offset + d0];
            let h_star1 = H_curr[h_base + slot_offset + d1];
            H_hist[h_base + slot_offset + d0] = a0 * h_hist0 + (1.0 - a0) * h_star0;
            H_hist[h_base + slot_offset + d1] = a1 * h_hist1 + (1.0 - a1) * h_star1;
        }
        workgroupBarrier();

        // FPM outer-loop update: m ← (1-α)*m + α*tanh(h*)
        // h* is bounded only weakly; tanh clamps to [-1,1] → m stays bounded always.
        // α=0.05 means ~20 tokens to reach 63% of new steady-state — slow outer loop.
        // Uses H_hist buffer (binding 11); mutually exclusive with ENABLE_H_HIST.
        if (ENABLE_FPM && d_model == WG_SIZE * 2u) {
            let d0 = tid;
            let d1 = tid + WG_SIZE;
            let h_star0 = H_curr[h_base + slot_offset + d0];
            let h_star1 = H_curr[h_base + slot_offset + d1];
            H_hist[h_base + slot_offset + d0] = (1.0 - FPM_ALPHA) * h_hist0 + FPM_ALPHA * tanh(h_star0);
            H_hist[h_base + slot_offset + d1] = (1.0 - FPM_ALPHA) * h_hist1 + FPM_ALPHA * tanh(h_star1);
        }
        workgroupBarrier();
    }

    var local_max_h = 0.0;
    let final_t = shape.token_start + shape.token_count - 1u;
    let final_h_base_t = (batch_idx * shape.seq_len + final_t) * total_elements;
    for (var d = tid; d < d_model; d = d + WG_SIZE) {
        let h_val = H_curr[h_base + slot_offset + d];
        H_next[final_h_base_t + slot_offset + d] = h_val;
        local_max_h = max(local_max_h, abs(h_val));
    }
    shared_vals[tid] = local_max_h;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        max_h_seen = max(max_h_seen, shared_vals[0]);
        let slot_base = 32u + slot_idx * 5u;
        DebugLog[slot_base + 0u] = max_delta_seen;
        DebugLog[slot_base + 1u] = f32(failed_hits_seen);
        DebugLog[slot_base + 2u] = f32(total_iters_seen) / max(1.0, f32(shape.token_count));
        DebugLog[slot_base + 3u] = max_contractivity;
        DebugLog[slot_base + 4u] = max_h_seen;
        if (slot_idx == 0u) {
            DebugLog[8] = 901.0;
            DebugLog[10] = f32(shape.token_count);
            DebugLog[11] = f32(h_slots);
        }
    }
}
