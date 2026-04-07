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
    fpm_alpha_m: f32,
    fpm_tau: f32,
    fpm_persist_beta: f32,
}

@group(0) @binding(0) var<uniform> shape: RunUniforms;
@group(0) @binding(1) var<storage, read> S_in: array<f32>;
@group(0) @binding(2) var<storage, read> AllWeights: array<f32>;
@group(0) @binding(3) var<storage, read_write> H_curr: array<f32>;
@group(0) @binding(4) var<storage, read_write> H_next: array<f32>;
@group(0) @binding(5) var<storage, read_write> Scratch: array<f32>;
@group(0) @binding(7) var<storage, read_write> DebugLog: array<f32>;
@group(0) @binding(8) var<storage, read_write> HistCtx: array<f32>;
@group(0) @binding(9) var<storage, read_write> MState: array<f32>;

override SLOT_ATTN_HEAD_DIM: u32 = 32u;
override ENABLE_TOKEN_CARRY: bool = true;
override ENABLE_H_HIST: bool = false;
override ENABLE_HIST_CTX: bool = false;
const WG_SIZE: u32 = 256u;
const FPM_CACHE_CAP: u32 = 512u;
const MAX_SLOTS: u32 = 8u;
const MAX_SLOT_ATTN_HEAD_DIM: u32 = 32u;
const SLOT_HEAD_CAP: u32 = MAX_SLOTS * MAX_SLOT_ATTN_HEAD_DIM;

var<workgroup> slot_attn_weights: array<f32, MAX_SLOTS>;
var<workgroup> shared_vals: array<f32, WG_SIZE>;
var<workgroup> q_self: array<f32, MAX_SLOT_ATTN_HEAD_DIM>;
var<workgroup> k_cache: array<f32, SLOT_HEAD_CAP>;
var<workgroup> v_cache: array<f32, SLOT_HEAD_CAP>;
var<workgroup> head_mix: array<f32, MAX_SLOT_ATTN_HEAD_DIM>;
var<workgroup> fpm_m_cache: array<f32, FPM_CACHE_CAP>;
var<workgroup> slot_attn_prev: array<f32, MAX_SLOTS>;
var<workgroup> max_delta_seen: f32;
var<workgroup> max_m_delta_seen: f32;
var<workgroup> max_a_delta_seen: f32;
var<workgroup> sum_self_assign_seen: f32;
var<workgroup> sum_assign_entropy_seen: f32;
var<workgroup> sum_slot_move_seen: f32;
var<workgroup> max_err_h_seen: f32;
var<workgroup> max_err_m_seen: f32;
var<workgroup> max_z_seen: f32;
var<workgroup> max_update_ratio_seen: f32;
var<workgroup> max_memctx_rms_seen: f32;
var<workgroup> max_memctx_to_signal_seen: f32;
var<workgroup> rescue_count_seen: f32;
var<workgroup> rescue_recovered_seen: f32;
var<workgroup> dead_slot_seen: f32;
var<workgroup> write_saturation_seen: f32;
var<workgroup> last_delta: f32;
var<workgroup> max_contractivity: f32;
var<workgroup> max_h_seen: f32;
var<workgroup> total_iters_seen: u32;
var<workgroup> failed_hits_seen: u32;
var<workgroup> converged_flag_wg: u32;
// Learnable raw γ per slot for h_currSSM (broadcast from tid==0, squashed before use)
var<workgroup> hhist_gamma_wg: f32;
const H_HIST_GAMMA_SCALE: f32 = 0.1;
const HIST_CTX_SCALE: f32 = 0.05;

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
fn hist_delta_base(d: u32, h_slots: u32) -> u32 { return slot_anchor_base(d, h_slots) + h_slots * d; }
fn hist_delta_bias_base(d: u32, h_slots: u32) -> u32 { return hist_delta_base(d, h_slots) + h_slots * d * d; }
fn hist_gate_query_base(d: u32, h_slots: u32) -> u32 { return hist_delta_bias_base(d, h_slots) + d + 21u; }
fn w_forget_base(d: u32, h_slots: u32) -> u32 { return hist_gate_query_base(d, h_slots) + h_slots * d; }
fn b_forget_base(d: u32, h_slots: u32) -> u32 { return w_forget_base(d, h_slots) + h_slots * d; }
// γ per slot: after slot_anchor(h*d) + W_delta(h*d²) + b_delta(d) + 21 flags + W_gate_hist(h*d) + W_forget(h*d) + b_forget(h)
fn hhist_gamma_base(d: u32, h: u32) -> u32 {
    return slot_anchor_base(d, h) + h * d + h * d * d + d + 21u + 2u * h * d + h;
}
// Retain gate (low-rank): W_up (h×d×r), W_down (h×r×d), b_retain (h×d)
// r=32, placed after hhist_gamma (h floats)
const RETAIN_RANK: u32 = 32u;
fn w_retain_up_base(d: u32, h: u32) -> u32 { return hhist_gamma_base(d, h) + h; }
fn w_retain_down_base(d: u32, h: u32) -> u32 { return w_retain_up_base(d, h) + h * d * RETAIN_RANK; }
fn b_retain_base(d: u32, h: u32) -> u32 { return w_retain_down_base(d, h) + h * RETAIN_RANK * d; }
fn w_q_mem_base(d: u32, h: u32) -> u32 { return b_retain_base(d, h) + h * d; }
fn w_k_mem_base(d: u32, h: u32) -> u32 { return w_q_mem_base(d, h) + h * d * RETAIN_RANK; }

@group(0) @binding(11) var<storage, read_write> H_hist: array<f32>;

override ENABLE_FPM: bool = false;
// Joint DEQ-memory path:
//   (h, m)^(k+1) = Phi((h, m)^k; signal, slot_ctx, anchor)
// Memory participates in the same fixed-point search as the token state.
override FPM_MEM_ITERS: u32 = 1u;
const FPM_ALPHA_H: f32 = 0.2;
const FPM_RESIDUAL_SCALE: f32 = 0.1;
const FPM_RESCUE_RESIDUAL_SCALE: f32 = 0.01;
const FPM_GATE_BIAS: f32 = -1.5;
const FPM_GATE_CLAMP: f32 = 0.5;
const FPM_FATIGUE_RATE: f32 = 0.002;
const FPM_RESCUE_TAIL: u32 = 2u;
const FPM_DEAD_THRESHOLD: f32 = 0.01;
// Saturation now refers to near-max use of the slot's write budget.
// Under the old z∈[0, 0.5] clamp, 0.45 meant "almost fully saturated".
// With z reparameterized into [0, 1], the same semantics are recovered by
// checking for near-ceiling utilization instead of mid-range activity.
const FPM_SAT_THRESHOLD: f32 = 0.95;
const FPM_EPS: f32 = 1e-6;
const FPM_HOMEO_MIN_ITERS: u32 = 4u;
const FPM_HOMEO_ALPHA_ERR_SCALE: f32 = 0.15;
const FPM_HOMEO_PLATEAU_TOL: f32 = 0.10;

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
        max_m_delta_seen = 0.0;
        max_a_delta_seen = 0.0;
        sum_self_assign_seen = 0.0;
        sum_assign_entropy_seen = 0.0;
        sum_slot_move_seen = 0.0;
        max_err_h_seen = 0.0;
        max_err_m_seen = 0.0;
        max_z_seen = 0.0;
        max_update_ratio_seen = 0.0;
        max_memctx_rms_seen = 0.0;
        max_memctx_to_signal_seen = 0.0;
        rescue_count_seen = 0.0;
        rescue_recovered_seen = 0.0;
        dead_slot_seen = 0.0;
        write_saturation_seen = 0.0;
        last_delta = 0.0;
        max_contractivity = 0.0;
        max_h_seen = 0.0;
        total_iters_seen = 0u;
        failed_hits_seen = 0u;
        converged_flag_wg = 0u;
        let slot_probe_base = 520u + slot_idx * 6u;
        DebugLog[slot_probe_base + 2u] = 0.0;
        DebugLog[slot_probe_base + 3u] = 0.0;
        DebugLog[slot_probe_base + 4u] = 0.0;
        DebugLog[slot_probe_base + 5u] = 0.0;
    }
    workgroupBarrier();

    var exit_err_h_sum = 0.0;
    var exit_err_h_valid_sum = 0.0;
    var exit_iter_sum = 0.0;
    var rescue_entered_sum = 0.0;
    var pre_rescue_converged_sum = 0.0;

    // FPM intra-slot m carry: load once before the token loop.
    // fpm_m_cache evolves token-to-token within the chunk via plastic write.
    // Cross-slot memory read uses HistCtx[t-1] within the chunk and MState only
    // as first-token fallback / inter-chunk persistence.
    if (ENABLE_FPM && d_model == WG_SIZE * 2u) {
        fpm_m_cache[tid]          = MState[h_base + slot_offset + tid];
        fpm_m_cache[tid + WG_SIZE] = MState[h_base + slot_offset + tid + WG_SIZE];
        if (tid < h_slots) {
            slot_attn_weights[tid] = 1.0 / max(1.0, f32(h_slots));
            slot_attn_prev[tid]    = slot_attn_weights[tid];
        }
        workgroupBarrier();
    }

    for (var t = 0u; t < shape.token_count; t = t + 1u) {
        let global_t = shape.token_start + t;
        let batch_scratch_t = (batch_idx * shape.seq_len + global_t) * scratch_stride;
        let h_base_t = (batch_idx * shape.seq_len + global_t) * total_elements;
        let signal_base = batch_scratch_t + slot_offset;
        let attn_base = batch_scratch_t + signal_span + slot_offset;
        let s_in_base = (batch_idx * shape.seq_len + global_t) * d_model;
        let use_prev_token_mem = global_t > 0u;
        let prev_hist_base_t = select(
            0u,
            (batch_idx * shape.seq_len + (global_t - 1u)) * total_elements,
            use_prev_token_mem,
        );

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
        // fpm_base0/1 removed — fpm_m_cache carries between tokens (initialized before loop)
        var fpm_ctx0 = 0.0;
        var fpm_ctx1 = 0.0;
        var hist_ctx0 = 0.0;
        var hist_ctx1 = 0.0;
        var hist_mem0 = 0.0;
        var hist_mem1 = 0.0;
        if (ENABLE_H_HIST && d_model == WG_SIZE * 2u) {
            h_hist0 = H_hist[h_base + slot_offset + tid];
            h_hist1 = H_hist[h_base + slot_offset + tid + WG_SIZE];
            if (tid == 0u) {
                let gamma_raw = AllWeights[aw_hist_base(d_model, h_slots) + hhist_gamma_base(d_model, h_slots) + slot_idx];
                hhist_gamma_wg = H_HIST_GAMMA_SCALE / (1.0 + exp(-(gamma_raw - 8.0)));
            }
        } else if (ENABLE_FPM && d_model == WG_SIZE * 2u) {
            // FPM: h_hist loaded per-token (for persist_beta blending at end).
            // fpm_base/fpm_m_cache are NOT reloaded here — they carry from token to token.
            h_hist0 = H_hist[h_base + slot_offset + tid];
            h_hist1 = H_hist[h_base + slot_offset + tid + WG_SIZE];
        } else if (ENABLE_HIST_CTX && d_model == WG_SIZE * 2u) {
            let hist_base_t = (batch_idx * shape.seq_len + global_t) * total_elements;
            hist_ctx0 = HistCtx[hist_base_t + slot_offset + tid];
            hist_ctx1 = HistCtx[hist_base_t + slot_offset + tid + WG_SIZE];
        }
        workgroupBarrier();

        if (ENABLE_FPM && d_model == WG_SIZE * 2u) {
            // fpm_m_cache already carries from previous token — no reinit here.
            if (tid < h_slots) {
                slot_attn_prev[tid] = slot_attn_weights[tid];
                slot_attn_weights[tid] = 1.0 / max(1.0, f32(h_slots));
                slot_attn_prev[tid] = slot_attn_weights[tid];
            }
            let hist_base = aw_hist_base(d_model, h_slots);
            if (tid < h_slots * RETAIN_RANK) {
                let ms = tid / RETAIN_RANK;
                let r = tid % RETAIN_RANK;
                let mem_off = h_base + ms * d_model;
                let kmem_base = hist_base + w_k_mem_base(d_model, h_slots) + ms * d_model * RETAIN_RANK;
                let scale_base = hist_base + hist_scale_base(d_model, h_slots) + ms * d_model;
                let bias_base = hist_base + hist_bias_base(d_model, h_slots) + ms * d_model;
                var key_acc = 0.0;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let m_j = select(
                        MState[mem_off + j],
                        HistCtx[prev_hist_base_t + ms * d_model + j],
                        use_prev_token_mem,
                    );
                    let keyed_m_j =
                        m_j
                        + AllWeights[scale_base + j] * tanh(m_j)
                        + AllWeights[bias_base + j];
                    key_acc = key_acc + AllWeights[kmem_base + j * RETAIN_RANK + r] * keyed_m_j;
                }
                k_cache[tid] = key_acc;
            }
            workgroupBarrier();
        } else if (ENABLE_HIST_CTX && d_model == WG_SIZE * 2u) {
            let hist_sq = hist_ctx0 * hist_ctx0 + hist_ctx1 * hist_ctx1;
            shared_vals[tid] = hist_sq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let hist_rms = sqrt(shared_vals[0] * inv_d_model + 1e-6);
            hist_mem0 = tanh(hist_ctx0) * (HIST_CTX_SCALE / max(hist_rms, 1e-6));
            hist_mem1 = tanh(hist_ctx1) * (HIST_CTX_SCALE / max(hist_rms, 1e-6));
            workgroupBarrier();
        }

        let should_init_h = global_t == 0u || !ENABLE_TOKEN_CARRY;
        if (should_init_h) {
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
                // FPM: init stays token-local; historical memory enters below as explicit mem_ctx.
                H_curr[h_base + slot_offset + tid] =
                    Scratch[signal_base + tid] / max(sig_rms, 1e-6);
                H_curr[h_base + slot_offset + tid + WG_SIZE] =
                    Scratch[signal_base + tid + WG_SIZE] / max(sig_rms, 1e-6);
            } else if (ENABLE_HIST_CTX && d_model == WG_SIZE * 2u) {
                H_curr[h_base + slot_offset + tid] =
                    Scratch[signal_base + tid] / max(sig_rms, 1e-6);
                H_curr[h_base + slot_offset + tid + WG_SIZE] =
                    Scratch[signal_base + tid + WG_SIZE] / max(sig_rms, 1e-6);
            } else {
                for (var d = tid; d < d_model; d = d + WG_SIZE) {
                    H_curr[h_base + slot_offset + d] = Scratch[signal_base + d] / max(sig_rms, 1e-6);
                }
            }
            workgroupBarrier();
        }

        var iter = 0u;
        var converged = false;
        var token_max_m_delta = 0.0;
        var token_max_a_delta = 0.0;
        var rescue_triggered = false;
        var rescue_recovered = false;
        var converged_before_rescue = false;
        var prev_err_h = 1.0;
        var last_finite_err_h = 1.0;
        let total_iter_limit = iter_limit + FPM_RESCUE_TAIL;
        while (iter < total_iter_limit && !converged) {
            if (tid == 0u) {
                converged_flag_wg = 0u;
            }
            workgroupBarrier();
            let rescue_active = ENABLE_FPM && iter >= iter_limit;
            if (rescue_active && !rescue_triggered && tid == 0u) {
                rescue_triggered = true;
                rescue_entered_sum = rescue_entered_sum + 1.0;
                last_delta = 0.0;
            }
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

            if (ENABLE_FPM && d_model == WG_SIZE * 2u) {
                let hist_base = aw_hist_base(d_model, h_slots);
                if (tid < h_slots) {
                    slot_attn_prev[tid] = slot_attn_weights[tid];
                }
                if (tid < RETAIN_RANK) {
                    let qmem_base = hist_base + w_q_mem_base(d_model, h_slots) + slot_idx * d_model * RETAIN_RANK;
                    var q_acc = 0.0;
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        q_acc = q_acc + AllWeights[qmem_base + j * RETAIN_RANK + tid]
                            * H_curr[h_base + slot_offset + j];
                    }
                    shared_vals[tid] = q_acc;
                }
                workgroupBarrier();
                if (tid < h_slots) {
                    let ms = tid;
                    var score = 0.0;
                    for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                        score = score + shared_vals[r] * k_cache[ms * RETAIN_RANK + r];
                    }
                    score = score + AllWeights[hist_base + hist_gate_base(d_model, h_slots) + ms];
                    slot_attn_weights[ms] =
                        score * inverseSqrt(max(1.0, f32(RETAIN_RANK))) / max(shape.fpm_tau, 1e-3);
                }
                workgroupBarrier();
                if (tid == 0u) {
                    var max_s = -1e30;
                    for (var ms = 0u; ms < h_slots; ms = ms + 1u) {
                        max_s = max(max_s, slot_attn_weights[ms]);
                    }
                    var sum_exp = 0.0;
                    for (var ms = 0u; ms < h_slots; ms = ms + 1u) {
                        let w = exp(slot_attn_weights[ms] - max_s);
                        slot_attn_weights[ms] = w;
                        sum_exp = sum_exp + w;
                    }
                    sum_exp = max(sum_exp, 1e-12);
                    for (var ms = 0u; ms < h_slots; ms = ms + 1u) {
                        slot_attn_weights[ms] = slot_attn_weights[ms] / sum_exp;
                    }
                }
                workgroupBarrier();
                var mem_raw0 = 0.0;
                var mem_raw1 = 0.0;
                for (var ms = 0u; ms < h_slots; ms = ms + 1u) {
                    let mem_off = h_base + ms * d_model;
                    let hist_slot_off = prev_hist_base_t + ms * d_model;
                    let m0 = select(
                        MState[mem_off + tid],
                        HistCtx[hist_slot_off + tid],
                        use_prev_token_mem,
                    );
                    let m1 = select(
                        MState[mem_off + tid + WG_SIZE],
                        HistCtx[hist_slot_off + tid + WG_SIZE],
                        use_prev_token_mem,
                    );
                    let w = slot_attn_weights[ms];
                    let value_m0 = tanh(m0);
                    let value_m1 = tanh(m1);
                    mem_raw0 = mem_raw0 + w * value_m0;
                    mem_raw1 = mem_raw1 + w * value_m1;
                }
                let read_gate = select(
                    clamp(
                        sqrt(shape.epsilon / max(prev_err_h, shape.epsilon)),
                        0.0,
                        1.0,
                    ),
                    1.0,
                    rescue_active,
                );
                let mem_unit0 = tanh(mem_raw0);
                let mem_unit1 = tanh(mem_raw1);
                let mem_unit_sq = mem_unit0 * mem_unit0 + mem_unit1 * mem_unit1;
                shared_vals[tid] = mem_unit_sq;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let mem_unit_norm = sqrt(max(shared_vals[0], 1e-6));
                let signal_sq = Scratch[signal_base + tid] * Scratch[signal_base + tid]
                    + Scratch[signal_base + tid + WG_SIZE] * Scratch[signal_base + tid + WG_SIZE];
                shared_vals[tid] = signal_sq;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let signal_norm = sqrt(max(shared_vals[0], 1e-6));
                // Structural scale matching: memory read enters the solve with the same RMS
                // order as the token signal, instead of an unconstrained absolute scale.
                let mem_to_signal = signal_norm / max(mem_unit_norm, 1e-6);
                fpm_ctx0 = mem_unit0 * mem_to_signal * read_gate;
                fpm_ctx1 = mem_unit1 * mem_to_signal * read_gate;
                let fpm_ctx_norm = signal_norm * read_gate;
                if (tid == 0u) {
                    let memctx_rms = fpm_ctx_norm / sqrt(max(1.0, f32(d_model)));
                    let signal_rms = signal_norm / sqrt(max(1.0, f32(d_model)));
                    max_memctx_rms_seen = max(max_memctx_rms_seen, memctx_rms);
                    max_memctx_to_signal_seen = max(
                        max_memctx_to_signal_seen,
                        memctx_rms / max(signal_rms, 1e-6),
                    );
                }
                workgroupBarrier();
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
                // Joint DEQ-memory: memory context is recomputed from m_k each Picard step.
                let d0 = tid;
                let d1 = tid + WG_SIZE;
                let h_prev0 = H_curr[h_base + slot_offset + d0];
                let h_prev1 = H_curr[h_base + slot_offset + d1];
                let h_dep0 = Scratch[signal_base + d0] + h_prev0 + Scratch[attn_base + d0]
                           + AllWeights[slot_anchor_mat_base + d0] + fpm_ctx0;
                let h_dep1 = Scratch[signal_base + d1] + h_prev1 + Scratch[attn_base + d1]
                           + AllWeights[slot_anchor_mat_base + d1] + fpm_ctx1;
                H_next[h_base_t + slot_offset + d0] = h_dep0;
                H_next[h_base_t + slot_offset + d1] = h_dep1;
                local_sumsq = h_dep0 * h_dep0 + h_dep1 * h_dep1;
            } else if (ENABLE_HIST_CTX && d_model == WG_SIZE * 2u) {
                let d0 = tid;
                let d1 = tid + WG_SIZE;
                let h_prev0 = H_curr[h_base + slot_offset + d0];
                let h_prev1 = H_curr[h_base + slot_offset + d1];
                let h_dep0 = Scratch[signal_base + d0] + h_prev0 + Scratch[attn_base + d0]
                           + AllWeights[slot_anchor_mat_base + d0] + hist_mem0;
                let h_dep1 = Scratch[signal_base + d1] + h_prev1 + Scratch[attn_base + d1]
                           + AllWeights[slot_anchor_mat_base + d1] + hist_mem1;
                H_next[h_base_t + slot_offset + d0] = h_dep0;
                H_next[h_base_t + slot_offset + d1] = h_dep1;
                local_sumsq = h_dep0 * h_dep0 + h_dep1 * h_dep1;
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

            let alpha_h = select(FPM_ALPHA_H, FPM_ALPHA_H * 0.5, rescue_active);
            var local_delta_h_num = 0.0;
            var local_delta_h_den = 0.0;
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                let h_prev = H_curr[h_base + slot_offset + d];
                let f_h = AllWeights[nscale_base + d] * (H_next[h_base_t + slot_offset + d] / rms);
                let blend = select(shape.damping, alpha_h, ENABLE_FPM && d_model == WG_SIZE * 2u);
                let val = blend * f_h + (1.0 - blend) * h_prev;
                let delta_h = val - h_prev;
                local_delta_h_num = local_delta_h_num + delta_h * delta_h;
                local_delta_h_den = local_delta_h_den + h_prev * h_prev;
                H_curr[h_base + slot_offset + d] = val;
                H_next[h_base_t + slot_offset + d] = val;
            }
            shared_vals[tid] = local_delta_h_num;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let delta_h_num = shared_vals[0];
            shared_vals[tid] = local_delta_h_den;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let err_h = sqrt(delta_h_num / (shared_vals[0] + FPM_EPS));
            var local_max_delta_a = 0.0;
            if (ENABLE_FPM && tid < h_slots) {
                local_max_delta_a = abs(slot_attn_weights[tid] - slot_attn_prev[tid]);
            }
            shared_vals[tid] = local_max_delta_a;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
                }
                workgroupBarrier();
            }
            let iter_max_delta_a = shared_vals[0];
            // FPM plastic write deferred to once per token after convergence (below).
            // fpm_ctx reads the previous token's frozen memory snapshot:
            // HistCtx[t-1] within the chunk, MState only for the first absolute token.
            // That preserves Picard stability while restoring causal token-to-token memory.
            var local_max_delta_m = 0.0;
            var local_gate = 0.0;
            var local_update_ratio = 0.0;
            var local_delta_m_num = 0.0;
            var local_delta_m_den = 0.0;
            shared_vals[tid] = local_delta_m_num;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let delta_m_num = shared_vals[0];
            shared_vals[tid] = local_delta_m_den;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let err_m = sqrt(delta_m_num / (shared_vals[0] + FPM_EPS));
            if (err_h == err_h && abs(err_h) < 1e30) {
                last_finite_err_h = err_h;
            }
            shared_vals[tid] = local_max_delta_m;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
                }
                workgroupBarrier();
            }
            let iter_max_delta_m = shared_vals[0];
            shared_vals[tid] = err_h;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
                }
                workgroupBarrier();
            }

            let stop_err_h = shared_vals[0];
            let homeo_band = max(shape.epsilon, FPM_HOMEO_ALPHA_ERR_SCALE * alpha_h);
            let plateau_ratio = abs(stop_err_h - prev_err_h) / max(prev_err_h, shape.epsilon);
            let homeostatic_converged =
                iter + 1u >= FPM_HOMEO_MIN_ITERS
                && stop_err_h <= homeo_band
                && plateau_ratio <= FPM_HOMEO_PLATEAU_TOL;
            if (tid == 0u && (stop_err_h < shape.epsilon || homeostatic_converged)) {
                converged_flag_wg = 1u;
                if (!rescue_active) {
                    converged_before_rescue = true;
                }
            }
            workgroupBarrier();
            converged = converged_flag_wg != 0u;
            if (tid == 0u) {
                let d_curr = stop_err_h;
                let d_prev = last_delta;
                if (iter > 0u && d_prev > 1e-12 && d_prev > shape.epsilon * 10.0) {
                    max_contractivity = max(max_contractivity, d_curr / d_prev);
                }
                last_delta = d_curr;
                max_delta_seen = max(max_delta_seen, d_curr);
                max_m_delta_seen = max(max_m_delta_seen, iter_max_delta_m);
                max_a_delta_seen = max(max_a_delta_seen, iter_max_delta_a);
                max_err_h_seen = max(max_err_h_seen, stop_err_h);
                max_err_m_seen = max(max_err_m_seen, err_m);
                max_z_seen = max(max_z_seen, local_gate);
                max_update_ratio_seen = max(max_update_ratio_seen, local_update_ratio);
                token_max_m_delta = max(token_max_m_delta, iter_max_delta_m);
                token_max_a_delta = max(token_max_a_delta, iter_max_delta_a);
                if (rescue_active && !converged && iter == iter_limit) {
                    rescue_count_seen = rescue_count_seen + 1.0;
                }
            }
            prev_err_h = stop_err_h;
            if (converged && rescue_triggered && tid == 0u) {
                rescue_recovered = true;
            }
            iter = iter + 1u;
            workgroupBarrier();
        }
        if (tid == 0u) {
            total_iters_seen = total_iters_seen + iter;
            sum_slot_move_seen = sum_slot_move_seen + token_max_m_delta;
            exit_iter_sum = exit_iter_sum + f32(iter);
            let prev_err_h_finite = prev_err_h == prev_err_h && abs(prev_err_h) < 1e30;
            let exit_err_h = select(prev_err_h, last_finite_err_h, !prev_err_h_finite);
            if (exit_err_h == exit_err_h && abs(exit_err_h) < 1e30) {
                exit_err_h_sum = exit_err_h_sum + exit_err_h;
                exit_err_h_valid_sum = exit_err_h_valid_sum + 1.0;
            }
            var entropy = 0.0;
            for (var ms = 0u; ms < h_slots; ms = ms + 1u) {
                let w = slot_attn_weights[ms];
                entropy = entropy - w * log(max(w, 1e-12));
            }
            sum_self_assign_seen = sum_self_assign_seen + slot_attn_weights[slot_idx];
            sum_assign_entropy_seen = sum_assign_entropy_seen + entropy;
            if (slot_attn_weights[slot_idx] < FPM_DEAD_THRESHOLD) {
                dead_slot_seen = dead_slot_seen + 1.0;
            }
            if (max_z_seen > FPM_SAT_THRESHOLD) {
                write_saturation_seen = write_saturation_seen + 1.0;
            }
            if (rescue_recovered) {
                rescue_recovered_seen = rescue_recovered_seen + 1.0;
            }
            if (converged_before_rescue) {
                pre_rescue_converged_sum = pre_rescue_converged_sum + 1.0;
            }
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

        // FPM plastic write: once per token using h* (converged fixed point).
        // Identical semantics to the original per-iter update for h* because fpm_ctx
        // always read MState (global, frozen), not fpm_m_cache.
        // Saves ~(max_iters-1)/max_iters of the O(d²) W_delta matmul cost.
        if (ENABLE_FPM && d_model == WG_SIZE * 2u) {
            let d0 = tid;
            let d1 = tid + WG_SIZE;
            let h_val0 = H_curr[h_base + slot_offset + d0];
            let h_val1 = H_curr[h_base + slot_offset + d1];
            let c0 = 0.5 * (h_val0 + Scratch[signal_base + d0]);
            let c1 = 0.5 * (h_val1 + Scratch[signal_base + d1]);
            let m_prev0 = fpm_m_cache[d0];
            let m_prev1 = fpm_m_cache[d1];
            let alpha_m = clamp(shape.fpm_alpha_m, 0.01, 0.1);
            let residual_scale = FPM_RESIDUAL_SCALE;
            let hist_base = aw_hist_base(d_model, h_slots);
            let wf_base = hist_base + w_forget_base(d_model, h_slots) + slot_idx * d_model;
            var gate_partial = 0.0;
            for (var j = tid; j < d_model; j = j + WG_SIZE) {
                let c_j = 0.5 * (H_curr[h_base + slot_offset + j] + Scratch[signal_base + j]);
                gate_partial = gate_partial + AllWeights[wf_base + j] * c_j;
            }
            shared_vals[tid] = gate_partial;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let gate_bias = AllWeights[hist_base + hist_gate_base(d_model, h_slots) + slot_idx]
                + AllWeights[hist_base + b_forget_base(d_model, h_slots) + slot_idx]
                + FPM_GATE_BIAS;
            let raw_z = 1.0 / (1.0 + exp(-(shared_vals[0] * inverseSqrt(max(1.0, f32(d_model))) + gate_bias)));
            let wd_base = hist_base + hist_delta_base(d_model, h_slots) + slot_idx * d_model * d_model;
            let bd_base = hist_base + hist_delta_bias_base(d_model, h_slots);
            var delta_in0 = AllWeights[bd_base + d0];
            var delta_in1 = AllWeights[bd_base + d1];
            for (var j = 0u; j < d_model; j = j + 1u) {
                let c_j = 0.5 * (H_curr[h_base + slot_offset + j] + Scratch[signal_base + j]);
                delta_in0 = delta_in0 + AllWeights[wd_base + d0 * d_model + j] * c_j;
                delta_in1 = delta_in1 + AllWeights[wd_base + d1 * d_model + j] * c_j;
            }
            let proposal0 = tanh(delta_in0);
            let proposal1 = tanh(delta_in1);
            let proposal_sq = proposal0 * proposal0 + proposal1 * proposal1;
            shared_vals[tid] = proposal_sq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let proposal_norm = sqrt(max(shared_vals[0], 1e-6));
            let prev_sq = m_prev0 * m_prev0 + m_prev1 * m_prev1;
            shared_vals[tid] = prev_sq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let prev_norm = sqrt(max(shared_vals[0], 1e-6));
            // Structural novelty gate: write strength scales with how much the proposal differs
            // from the carried memory, rather than saturating at a fixed global clamp.
            let novelty = proposal_norm / (proposal_norm + prev_norm + 1e-6);
            let z = clamp(raw_z * novelty, 0.0, 1.0);
            // Retain gate (low-rank r=32): retain = σ(W_down · (W_up · c) + b_retain)
            // Replaces uniform fatigue decay with input-dependent selective forgetting.
            let hist_base_r = aw_hist_base(d_model, h_slots);
            let wup_base = hist_base_r + w_retain_up_base(d_model, h_slots) + slot_idx * d_model * RETAIN_RANK;
            let wdown_base = hist_base_r + w_retain_down_base(d_model, h_slots) + slot_idx * RETAIN_RANK * d_model;
            let bret_base = hist_base_r + b_retain_base(d_model, h_slots) + slot_idx * d_model;
            // Step 1: up = W_up · c  (d_model → RETAIN_RANK)
            // Threads tid < RETAIN_RANK each compute one element of up, store in shared_vals[0..32]
            if (tid < RETAIN_RANK) {
                var up_acc = 0.0;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let c_j = 0.5 * (H_curr[h_base + slot_offset + j] + Scratch[signal_base + j]);
                    up_acc = up_acc + AllWeights[wup_base + j * RETAIN_RANK + tid] * c_j;
                }
                shared_vals[tid] = up_acc;
            }
            workgroupBarrier();
            // Step 2: retain[d] = σ(W_down[d,:] · up + b_retain[d])
            var down_acc0 = AllWeights[bret_base + d0];
            var down_acc1 = AllWeights[bret_base + d1];
            for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                down_acc0 = down_acc0 + AllWeights[wdown_base + r * d_model + d0] * shared_vals[r];
                down_acc1 = down_acc1 + AllWeights[wdown_base + r * d_model + d1] * shared_vals[r];
            }
            let retain0 = 1.0 / (1.0 + exp(-down_acc0));
            let retain1 = 1.0 / (1.0 + exp(-down_acc1));
            // Structural update budget: write magnitude is coupled to the amount of forgetting.
            // If retain≈1, the slot is choosing to preserve its memory and must not inject
            // a full-strength write on top of that same retained state.
            let write_budget0 = (1.0 - retain0) * z;
            let write_budget1 = (1.0 - retain1) * z;
            let write0 = write_budget0 * (residual_scale * proposal0);
            let write1 = write_budget1 * (residual_scale * proposal1);
            let m_candidate0 = retain0 * m_prev0 + write0;
            let m_candidate1 = retain1 * m_prev1 + write1;
            let proposal_rms_p = sqrt(0.5 * (proposal0 * proposal0 + proposal1 * proposal1));
            let candidate_rms_p = sqrt(0.5 * (m_candidate0 * m_candidate0 + m_candidate1 * m_candidate1));
            let retain_avg_p = 0.5 * (retain0 + retain1);
            let retain_max_p = max(retain0, retain1);
            // Diagnostics: replicate the same reductions as the original per-iter block.
            var local_delta_m_num_p = (m_candidate0 - m_prev0) * (m_candidate0 - m_prev0)
                                    + (m_candidate1 - m_prev1) * (m_candidate1 - m_prev1);
            shared_vals[tid] = local_delta_m_num_p;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let delta_m_num_p = shared_vals[0];
            var local_delta_m_den_p = m_prev0 * m_prev0 + m_prev1 * m_prev1;
            shared_vals[tid] = local_delta_m_den_p;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let err_m_p = sqrt(delta_m_num_p / (shared_vals[0] + FPM_EPS));
            let local_max_delta_m_p = max(abs(m_candidate0 - m_prev0), abs(m_candidate1 - m_prev1));
            shared_vals[tid] = local_max_delta_m_p;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
                }
                workgroupBarrier();
            }
            let iter_max_delta_m_p = shared_vals[0];
            let upd_num_p = write0 * write0 + write1 * write1;
            let upd_den_p = m_prev0 * m_prev0 + m_prev1 * m_prev1;
            let update_ratio_p = sqrt(upd_num_p / (upd_den_p + FPM_EPS));
            if (tid == 0u) {
                max_m_delta_seen = max(max_m_delta_seen, iter_max_delta_m_p);
                max_err_m_seen = max(max_err_m_seen, err_m_p);
                max_z_seen = max(max_z_seen, z);
                max_update_ratio_seen = max(max_update_ratio_seen, update_ratio_p);
                sum_slot_move_seen = sum_slot_move_seen + iter_max_delta_m_p;
                if (z > FPM_SAT_THRESHOLD) {
                    write_saturation_seen = write_saturation_seen + 1.0;
                }
                let slot_probe_base = 520u + slot_idx * 6u;
                DebugLog[slot_probe_base + 2u] = max(DebugLog[slot_probe_base + 2u], retain_max_p);
                DebugLog[slot_probe_base + 3u] = DebugLog[slot_probe_base + 3u] + retain_avg_p;
                DebugLog[slot_probe_base + 4u] = max(DebugLog[slot_probe_base + 4u], proposal_rms_p);
                DebugLog[slot_probe_base + 5u] = max(DebugLog[slot_probe_base + 5u], candidate_rms_p);
            }
            fpm_m_cache[d0] = m_candidate0;
            fpm_m_cache[d1] = m_candidate1;
            workgroupBarrier();
        }
        workgroupBarrier();

        // Joint DEQ-memory: persist the converged memory state after the shared fixed-point search.
        if (ENABLE_FPM && d_model == WG_SIZE * 2u) {
            let working0 = fpm_m_cache[tid];
            let working1 = fpm_m_cache[tid + WG_SIZE];
            // Per-token storage for retain-gate backward: index includes token dimension.
            // encode_forward_gpu_only copies the last-token entry → mstate_buf between chunks.
            HistCtx[h_base_t + slot_offset + tid] = working0;
            HistCtx[h_base_t + slot_offset + tid + WG_SIZE] = working1;
            let beta = clamp(shape.fpm_persist_beta, 0.0, 1.0);
            let init_hist = global_t == 0u && abs(h_hist0) + abs(h_hist1) < 1e-6;
            H_hist[h_base + slot_offset + tid] = select(h_hist0 + beta * (working0 - h_hist0), working0, init_hist);
            H_hist[h_base + slot_offset + tid + WG_SIZE] = select(h_hist1 + beta * (working1 - h_hist1), working1, init_hist);
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
        DebugLog[256u + slot_idx * 2u] = max_delta_seen;
        DebugLog[256u + slot_idx * 2u + 1u] = max_m_delta_seen;
        let slot_obs_base = 320u + slot_idx * 3u;
        let token_den = max(1.0, f32(shape.token_count));
        DebugLog[slot_obs_base + 0u] = sum_self_assign_seen / token_den;
        DebugLog[slot_obs_base + 1u] = sum_assign_entropy_seen / token_den;
        DebugLog[slot_obs_base + 2u] = sum_slot_move_seen / token_den;
        DebugLog[384u + slot_idx] = max_a_delta_seen;
        let slot_diag_base = 400u + slot_idx * 12u;
        DebugLog[slot_diag_base + 0u] = max_err_h_seen;
        DebugLog[slot_diag_base + 1u] = max_err_m_seen;
        DebugLog[slot_diag_base + 2u] = max_z_seen;
        DebugLog[slot_diag_base + 3u] = rescue_count_seen;
        DebugLog[slot_diag_base + 4u] = rescue_recovered_seen;
        DebugLog[slot_diag_base + 5u] = dead_slot_seen;
        DebugLog[slot_diag_base + 6u] = max_update_ratio_seen;
        DebugLog[slot_diag_base + 7u] = write_saturation_seen;
        let exit_err_h_den = max(1.0, exit_err_h_valid_sum);
        DebugLog[slot_diag_base + 8u] = exit_err_h_sum / exit_err_h_den;
        DebugLog[slot_diag_base + 9u] = exit_iter_sum / token_den;
        DebugLog[slot_diag_base + 10u] = rescue_entered_sum;
        DebugLog[slot_diag_base + 11u] = pre_rescue_converged_sum;
        let slot_read_base = 520u + slot_idx * 6u;
        DebugLog[slot_read_base + 0u] = max_memctx_rms_seen;
        DebugLog[slot_read_base + 1u] = max_memctx_to_signal_seen;
        DebugLog[slot_read_base + 3u] = DebugLog[slot_read_base + 3u] / token_den;
        if (slot_idx == 0u) {
            DebugLog[8] = 901.0;
            DebugLog[9] = shape.epsilon;
            DebugLog[10] = f32(shape.token_count);
            DebugLog[11] = f32(h_slots);
        }
    }
}
