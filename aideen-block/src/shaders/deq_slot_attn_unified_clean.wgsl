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
    fpm_stage: u32,
    fpm_alpha_m: f32,
    fpm_tau: f32,
    fpm_read_gate_min: f32,
    segment_len: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> shape: RunUniforms;
@group(0) @binding(1) var<storage, read> S_in: array<f32>;
@group(0) @binding(3) var<storage, read_write> H_curr: array<f32>;
@group(0) @binding(4) var<storage, read_write> H_next: array<f32>;
@group(0) @binding(5) var<storage, read_write> Scratch: array<f32>;
@group(0) @binding(7) var<storage, read_write> DebugLog: array<f32>;
@group(0) @binding(8) var<storage, read_write> HistCtx: array<f32>;
@group(0) @binding(9) var<storage, read_write> MState: array<f32>;
// Previous associative source: normalized token identity signal, optionally
// shifted by the slot anchor. FPM/DEQ controls write strength separately.
@group(0) @binding(12) var<storage, read_write> PrevHStarBuf: array<f32>;

// Group 1: Associative Memory Pool (moved to satisfy Metal 16-buffer limit per group)
@group(1) @binding(0) var<storage, read_write> AssocBuf: array<f32>;
@group(1) @binding(1) var<storage, read_write> AssocPersistentBuf: array<f32>;
@group(1) @binding(2) var<storage, read_write> AssocHist: array<f32>;
@group(1) @binding(3) var<storage, read_write> AssocReadBuf: array<f32>;
@group(1) @binding(4) var<storage, read_write> AllWeights: array<f32>;

override SLOT_ATTN_HEAD_DIM: u32 = 32u;
override ENABLE_TOKEN_CARRY: bool = true;
override ENABLE_H_HIST: bool = false;
override ENABLE_HIST_CTX: bool = false;
override ENABLE_ASSOC_TRANSITION_GATE: bool = false;
override ENABLE_ASSOC_SLOT_ANCHOR: bool = false;
override ENABLE_ASSOC_REUSE_MATCH: bool = false;
override ENABLE_ASSOC_SLOT_STRIPE: bool = false;
override ENABLE_ASSOC_SLOT_OWNER: bool = false;
override ENABLE_ASSOC_READ_SLOT_PRIOR: bool = true;
override ENABLE_ASSOC_TIE_QK: bool = false;
override ENABLE_ASSOC_CONF_READ: bool = false;
override ENABLE_ASSOC_EVENT_GATE: bool = false;
override ENABLE_ASSOC_HARD_READ: bool = false;
override ENABLE_ASSOC_LINEAR_WRITE: bool = false;
override ENABLE_ASSOC_READ: bool = true;
override ENABLE_ASSOC_POST_HSTAR: bool = false;
override ENABLE_SEGMENT_MEMORY_TOKEN: bool = false;
override ENABLE_ASSOC_PERSISTENT: bool = false;
const SLOT_COORD_USE_BIAS: bool = true;
const SLOT_COORD_LOGIT_GAIN: f32 = 2.0;
const WG_SIZE: u32 = 256u;
const FPM_CACHE_CAP: u32 = 512u;
const MAX_SLOTS: u32 = 8u;
const MAX_SLOT_ATTN_HEAD_DIM: u32 = 32u;
const SLOT_HEAD_CAP: u32 = MAX_SLOTS * MAX_SLOT_ATTN_HEAD_DIM;

const ASSOC_HIST_META: u32 = 4u;
override ASSOC_BANKS: u32 = 1u;
const ASSOC_WRITE_CAP: f32 = 0.95;
const ASSOC_ALLOC_THRESHOLD: f32 = 0.05;
const ASSOC_OCCUPIED_THRESHOLD: f32 = 1.0e-4;
const ASSOC_USAGE_DECAY: f32 = 0.999;
const ASSOC_TO_FPM_SCALE: f32 = 0.02;
const ASSOC_REUSE_THRESHOLD: f32 = 0.80;
const ASSOC_ALLOC_NOVELTY_THRESHOLD: f32 = 0.20;
// Experimental multi-bank addressing strength. With ASSOC_BANKS=1 this is
// behaviorally inactive because the read softmax has a single candidate.
const ASSOC_READ_BETA: f32 = 4.0;
const ASSOC_READ_SLOT_PRIOR_BETA: f32 = 8.0;
const H_SELF_FEEDBACK: f32 = 1.0;

var<workgroup> slot_coord_weights: array<f32, MAX_SLOTS>;
// Shared reduction scratch must cover the full workgroup width. The assoc
// path only needs the first ~100 lanes, but the DEQ/FPM reductions use `tid`
// across all WG_SIZE lanes, so allocating less than WG_SIZE corrupts the
// entire token solve.
var<workgroup> shared_vals: array<f32, WG_SIZE>;
var<workgroup> q_self: array<f32, MAX_SLOT_ATTN_HEAD_DIM>;
var<workgroup> k_cache: array<f32, SLOT_HEAD_CAP>;
var<workgroup> v_cache: array<f32, SLOT_HEAD_CAP>;
var<workgroup> head_mix: array<f32, MAX_SLOT_ATTN_HEAD_DIM>;
var<workgroup> fpm_m_cache: array<f32, FPM_CACHE_CAP>;
var<workgroup> slot_coord_prev: array<f32, MAX_SLOTS>;
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

// ── Layout Functions (Clean Layout) ─────────────────────────────────────────
const RETAIN_RANK: u32 = 32u;
const ASSOC_RANK: u32 = 32u;

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

fn scratch_stride(d: u32, h: u32) -> u32 {
    return d * (h * 8u) + h * h + h;
}

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
const FPM_JOINT_POLICY_STAGE: u32 = 6u;
const FPM_DEAD_THRESHOLD: f32 = 0.01;
const FPM_SAT_THRESHOLD: f32 = 0.95;
const FPM_EPS: f32 = 1e-6;
const FPM_HOMEO_MIN_ITERS: u32 = 4u;
const FPM_HOMEO_ALPHA_ERR_SCALE: f32 = 0.15;
const FPM_HOMEO_PLATEAU_TOL: f32 = 0.10;

fn compute_slot_coord(
    tid: u32,
    slot_idx: u32,
    d_model: u32,
    h_slots: u32,
    head_dim: u32,
    signal_token_base: u32,
    alpha_base: u32,
    wq_mat_base: u32,
    wk_bias_root: u32,
    wq_bias_base: u32,
    wo_mat_base: u32,
    attn_base: u32,
) {
    let hist_base = aw_hist_base(d_model, h_slots);
    let slot_anchor_root = hist_base + slot_anchor_base(d_model, h_slots);
    if (tid < h_slots) {
        let ks = tid;
        let ks_off = ks * d_model;
        var sumsq = 0.0;
        for (var j = 0u; j < d_model; j = j + 1u) {
            let src = Scratch[signal_token_base + ks_off + j] + AllWeights[slot_anchor_root + ks_off + j];
            sumsq = sumsq + src * src;
        }
        slot_coord_prev[ks] = sqrt(sumsq / max(1.0, f32(d_model)) + 1.0e-6);
    }
    workgroupBarrier();
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
            var k0 = select(0.0, AllWeights[wk_bias_base + hd0], SLOT_COORD_USE_BIAS);
            var k1 = select(0.0, AllWeights[wk_bias_base + hd1], SLOT_COORD_USE_BIAS);
            var v0 = 0.0;
            var v1 = 0.0;
            var q0 = 0.0;
            var q1 = 0.0;
            if (ks == slot_idx) {
                q0 = select(0.0, AllWeights[wq_bias_base + hd0], SLOT_COORD_USE_BIAS);
                q1 = select(0.0, AllWeights[wq_bias_base + hd1], SLOT_COORD_USE_BIAS);
            }
            for (var j = 0u; j < d_model; j = j + 1u) {
                let src_val =
                    Scratch[signal_token_base + ks_off + j]
                    + AllWeights[slot_anchor_root + ks_off + j];
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
            var k = select(0.0, AllWeights[wk_bias_base + hd], SLOT_COORD_USE_BIAS);
            var v = 0.0;
            var q = 0.0;
            if (ks == slot_idx) {
                q = select(0.0, AllWeights[wq_bias_base + hd], SLOT_COORD_USE_BIAS);
            }
            for (var j = 0u; j < d_model; j = j + 1u) {
                let src_val =
                    Scratch[signal_token_base + ks_off + j]
                    + AllWeights[slot_anchor_root + ks_off + j];
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
        // Slot coordination already has identity through signal/residual/slot_anchor.
        // The relational branch should carry cross-slot information, not learn a redundant
        // self-loop that can dominate alpha and destabilize the solve.
        if (h_slots > 1u && ks == slot_idx) {
            slot_coord_weights[ks] = -1.0e30;
        } else {
            slot_coord_weights[ks] =
                score * inverseSqrt(max(1.0, f32(head_dim))) * SLOT_COORD_LOGIT_GAIN;
        }
    }
    workgroupBarrier();
    if (tid == 0u) {
        var max_s = -1e30;
        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
            max_s = max(max_s, slot_coord_weights[ks]);
        }
        var sum_exp = 0.0;
        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
            let w = exp(slot_coord_weights[ks] - max_s);
            slot_coord_weights[ks] = w;
            sum_exp = sum_exp + w;
        }
        sum_exp = max(sum_exp, 1e-12);
        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
            slot_coord_weights[ks] = slot_coord_weights[ks] / sum_exp;
        }
        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
            Scratch[alpha_base + slot_idx * h_slots + ks] = slot_coord_weights[ks];
        }
    }
    workgroupBarrier();

    for (var h = tid; h < head_dim; h = h + WG_SIZE) {
        var mix = 0.0;
        var mean_v = 0.0;
        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
            let v = v_cache[ks * head_dim + h];
            mix = mix + slot_coord_weights[ks] * v;
            mean_v = mean_v + v;
        }
        head_mix[h] = mix - mean_v / max(1.0, f32(h_slots));
    }
    workgroupBarrier();
    var head_mix_sumsq = 0.0;
    for (var h = tid; h < head_dim; h = h + WG_SIZE) {
        let mix = head_mix[h];
        head_mix_sumsq = head_mix_sumsq + mix * mix;
    }
    shared_vals[tid] = head_mix_sumsq;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
        }
        workgroupBarrier();
    }
    let head_mix_rms = sqrt(shared_vals[0] / max(1.0, f32(head_dim)) + 1.0e-6);

    if (d_model == WG_SIZE * 2u) {
        let d0 = tid;
        let d1 = tid + WG_SIZE;
        var attn0 = 0.0;
        var attn1 = 0.0;
        for (var h = 0u; h < head_dim; h = h + 1u) {
            let mix = head_mix[h] / head_mix_rms;
            attn0 = attn0 + AllWeights[wo_mat_base + h * d_model + d0] * mix;
            attn1 = attn1 + AllWeights[wo_mat_base + h * d_model + d1] * mix;
        }
        Scratch[attn_base + d0] = attn0;
        Scratch[attn_base + d1] = attn1;
    } else {
        for (var d = tid; d < d_model; d = d + WG_SIZE) {
            var attn = 0.0;
            for (var h = 0u; h < head_dim; h = h + 1u) {
                attn = attn + AllWeights[wo_mat_base + h * d_model + d] * (head_mix[h] / head_mix_rms);
            }
            Scratch[attn_base + d] = attn;
        }
    }
    workgroupBarrier();
    var attn_sumsq = 0.0;
    for (var d = tid; d < d_model; d = d + WG_SIZE) {
        let attn = Scratch[attn_base + d];
        attn_sumsq = attn_sumsq + attn * attn;
    }
    shared_vals[tid] = attn_sumsq;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let attn_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1.0e-6);
        let src_rms = max(slot_coord_prev[slot_idx], 1.0e-3);
        max_h_seen = src_rms / max(attn_rms, 1.0e-6);
    }
    workgroupBarrier();
}

@compute @workgroup_size(256, 1, 1)
fn deq_slot_coord_unified_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.x;
    let slot_idx = wid.y;
    if (batch_idx >= shape.batch_size || slot_idx >= shape.h_slots) { return; }
    let debug_on = true; // FORCE DEBUG FOR AUDIT
    if (debug_on && slot_idx == 0u && tid == 0u) {
        // Forward debug progress markers:
        // 101 = entered kernel, 201 = finished token loop, 901 = finalized snapshot.
        DebugLog[8] = 101.0;
        DebugLog[9] = shape.epsilon;
        DebugLog[10] = f32(shape.token_count);
        DebugLog[11] = f32(shape.h_slots);
    }

    let d_model = shape.d_model;
    let h_slots = shape.h_slots;
    let segment_memory_slot = h_slots - 1u;
    let is_segment_memory_slot = ENABLE_SEGMENT_MEMORY_TOKEN && slot_idx == segment_memory_slot;
    let head_dim = min(d_model, SLOT_ATTN_HEAD_DIM);
    let total_elements = h_slots * d_model;
    let slot_offset = slot_idx * d_model;
    let h_base = batch_idx * total_elements;
    let prev_hstar_base = h_base + slot_offset;
    let assoc_bank_stride = ASSOC_RANK + d_model + 1u;
    let assoc_slot_stride = ASSOC_BANKS * assoc_bank_stride;
    let assoc_hist_slot_stride = assoc_slot_stride + ASSOC_HIST_META;
    let assoc_slot_base = (batch_idx * h_slots + slot_idx) * assoc_slot_stride;
    let signal_span = d_model * h_slots;
    let scratch_stride = signal_span * 3u + h_slots * h_slots;
    let wq_mat_base = aw_wq_base(d_model, h_slots) + slot_idx * d_model * d_model;
    let wk_bias_root = aw_wk_base(d_model, h_slots) + h_slots * d_model * d_model;
    let wq_bias_base = aw_wq_base(d_model, h_slots) + h_slots * d_model * d_model + slot_idx * d_model;
    let wo_mat_base = aw_wo_base(d_model, h_slots) + slot_idx * d_model * d_model;
    let win_base = aw_win_base(d_model, h_slots) + slot_idx * d_model * d_model;
    let slot_anchor_mat_base = aw_hist_base(d_model, h_slots) + slot_anchor_base(d_model, h_slots) + slot_offset;
    let nscale_base = aw_nscale_base(d_model, h_slots);
    let inv_d_model = 1.0 / max(1.0, f32(d_model));
    let inv_sqrt_d_model = sqrt(inv_d_model);
    let zero_win_diag = shape.diag_zero_win != 0u;
    let iter_limit = select(shape.max_iters, 1u, shape.diag_one_iter != 0u);
    let enable_slot_coord = shape.residual_alpha > -1.5;
    // Model A keeps the H-only solve semantics. The joint H/M policy is reserved for the
    // future Model B rollout, so current FPM stages do not override the base DEQ policy.
    let fpm_policy_enabled = ENABLE_FPM && shape.fpm_stage >= FPM_JOINT_POLICY_STAGE;
    let fpm_read_enabled = ENABLE_FPM && shape.fpm_stage >= 2u;
    let fpm_inject_enabled = ENABLE_FPM && shape.fpm_stage >= 3u;
    let fpm_write_enabled = ENABLE_FPM && shape.fpm_stage >= 4u;
    let recurrent_slot_attn_enabled = fpm_policy_enabled;
    let model_a_memory_bootstrap = fpm_inject_enabled && !fpm_policy_enabled;
    // Associative memory needs write support before read can carry information.
    // Reading at stage 3 while writes are still off leaves B identically zero and
    // makes the branch structurally dead.
    let assoc_write_enabled = fpm_write_enabled;
    let assoc_read_enabled = assoc_write_enabled && ENABLE_ASSOC_READ;

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
        max_h_seen = 1.0;
        total_iters_seen = 0u;
        failed_hits_seen = 0u;
        converged_flag_wg = 0u;
        let slot_probe_base = 520u + slot_idx * 14u;
        let slot_write_base = 640u + slot_idx * 6u;
        DebugLog[slot_probe_base + 2u] = 0.0;
        DebugLog[slot_probe_base + 3u] = 0.0;
        DebugLog[slot_probe_base + 4u] = 0.0;
        DebugLog[slot_probe_base + 5u] = 0.0;
        DebugLog[slot_probe_base + 6u] = 0.0;
        DebugLog[slot_probe_base + 7u] = 0.0;
        DebugLog[slot_probe_base + 8u] = 0.0;
        DebugLog[slot_probe_base + 9u] = 0.0;
        DebugLog[slot_probe_base + 10u] = 0.0;
        DebugLog[slot_probe_base + 11u] = 0.0;
        DebugLog[slot_probe_base + 12u] = 0.0;
        DebugLog[slot_probe_base + 13u] = 0.0;
        DebugLog[slot_write_base + 0u] = 0.0;
        DebugLog[slot_write_base + 1u] = 0.0;
        DebugLog[slot_write_base + 2u] = 0.0;
        DebugLog[slot_write_base + 3u] = 0.0;
        DebugLog[slot_write_base + 4u] = 0.0;
        DebugLog[slot_write_base + 5u] = 0.0;
        // TEMPORARY ASSOCIATIVE DIAGNOSTIC: remove DebugLog[760..] after root-cause closure.
        let assoc_diag_base = 760u + slot_idx * 10u;
        DebugLog[assoc_diag_base + 0u] = 0.0;
        DebugLog[assoc_diag_base + 1u] = 0.0;
        DebugLog[assoc_diag_base + 2u] = 0.0;
        DebugLog[assoc_diag_base + 3u] = 0.0;
        DebugLog[assoc_diag_base + 4u] = 0.0;
        DebugLog[assoc_diag_base + 5u] = 0.0;
        DebugLog[assoc_diag_base + 6u] = -1.0e30;
        DebugLog[assoc_diag_base + 7u] = 1.0e30;
        DebugLog[assoc_diag_base + 8u] = 0.0;
        DebugLog[assoc_diag_base + 9u] = 0.0;
        // TEMPORARY ASSOCIATIVE DIAGNOSTIC: final-token/query-only read telemetry.
        let assoc_query_diag_base = 820u + slot_idx * 8u;
        DebugLog[assoc_query_diag_base + 0u] = 0.0;
        DebugLog[assoc_query_diag_base + 1u] = 0.0;
        DebugLog[assoc_query_diag_base + 2u] = 0.0;
        DebugLog[assoc_query_diag_base + 3u] = 0.0;
        DebugLog[assoc_query_diag_base + 4u] = 0.0;
        DebugLog[assoc_query_diag_base + 5u] = 0.0;
        DebugLog[assoc_query_diag_base + 6u] = 0.0;
        DebugLog[assoc_query_diag_base + 7u] = 0.0;
        let assoc_write_probe_base = 900u + slot_idx * 6u;
        DebugLog[assoc_write_probe_base + 0u] = 0.0;
        DebugLog[assoc_write_probe_base + 1u] = 0.0;
        DebugLog[assoc_write_probe_base + 2u] = 0.0;
        DebugLog[assoc_write_probe_base + 3u] = 0.0;
        DebugLog[assoc_write_probe_base + 4u] = 0.0;
        DebugLog[assoc_write_probe_base + 5u] = 0.0;
        // TEMPORARY ASSOCIATIVE DIAGNOSTIC: h*(t-1) vs h*(t) separability before key projection.
    }
    workgroupBarrier();

    var exit_err_h_sum = 0.0;
    var exit_err_h_valid_sum = 0.0;
    var exit_iter_sum = 0.0;
    var rescue_entered_sum = 0.0;
    var pre_rescue_converged_sum = 0.0;
    // TEMPORARY SOLVE DIAGNOSTIC: split strict epsilon exits from homeostatic
    // plateau exits while auditing H convergence. Remove after root-cause closure.
    var strict_converged_sum = 0.0;
    var homeostatic_converged_sum = 0.0;
    var failed_converged_sum = 0.0;
    var max_homeo_band_seen = 0.0;
    var solve_signal_rms_seen = 0.0;
    var solve_pre_rms_seen = 0.0;
    var solve_fh_rms_seen = 0.0;
    var solve_hprev_rms_seen = 0.0;
    var solve_nscale_abs_seen = 0.0;
    var solve_pre_to_hprev_seen = 0.0;
    var solve_fh_to_hprev_seen = 0.0;
    var solve_attn_rms_seen = 0.0;
    var solve_attn_to_signal_seen = 0.0;
    var solve_attn_scale_seen = 0.0;
    var iter0_err_h_seen = 0.0;
    var iter1_err_h_seen = 0.0;
    var iter0_attn_to_signal_seen = 0.0;
    var iter1_attn_to_signal_seen = 0.0;
    var iter0_attn_scale_seen = 0.0;
    var iter1_attn_scale_seen = 0.0;
    var iter_of_max_err_h_seen = 0.0;
    var iter_of_max_attn_ratio_seen = 0.0;
    var max_err_h_marker_seen = 0.0;
    var max_attn_ratio_marker_seen = 0.0;
    var token_of_max_err_h_seen = 0.0;
    var token_of_max_attn_ratio_seen = 0.0;

    // FPM intra-slot m carry: load once before the token loop.
    // fpm_m_cache evolves token-to-token within the chunk via plastic write.
    // Cross-slot memory read uses HistCtx[t-1] within the chunk and MState only
    // as first-token fallback / inter-chunk persistence.
    if (fpm_read_enabled && d_model == WG_SIZE * 2u) {
        fpm_m_cache[tid]          = MState[h_base + slot_offset + tid];
        fpm_m_cache[tid + WG_SIZE] = MState[h_base + slot_offset + tid + WG_SIZE];
        if (tid < h_slots) {
            slot_coord_weights[tid] = 1.0 / max(1.0, f32(h_slots));
            slot_coord_prev[tid]    = slot_coord_weights[tid];
        }
        workgroupBarrier();
    }

    for (var t = 0u; t < shape.token_count; t = t + 1u) {
        if (debug_on && slot_idx == 0u && tid == 0u && t == 1u) {
            DebugLog[8] = 186.0;
        }
        let global_t = shape.token_start + t;
        let batch_scratch_t = (batch_idx * shape.seq_len + global_t) * scratch_stride;
        let h_base_t = (batch_idx * shape.seq_len + global_t) * total_elements;
        let signal_base = batch_scratch_t + slot_offset;
        let attn_base = batch_scratch_t + signal_span + slot_offset;
        let alpha_base = batch_scratch_t + signal_span * 2u;
        let s_in_base = (batch_idx * shape.seq_len + global_t) * d_model;
        let use_prev_token_mem = t > 0u;
        var prev_hist_base_t = 0u;
        if (use_prev_token_mem) {
            prev_hist_base_t = (batch_idx * shape.seq_len + (global_t - 1u)) * total_elements;
        }

        if (d_model == WG_SIZE * 2u) {
            let d0 = tid;
            let d1 = tid + WG_SIZE;
            var inj0 = 0.0;
            var inj1 = 0.0;
            let hist_base = aw_hist_base(d_model, h_slots);
            let signal_zero = AllWeights[hist_base + signal_zero_base(d_model, h_slots)] > 0.5;
            let signal_scale = AllWeights[hist_base + signal_scale_base(d_model, h_slots)];
            if (!zero_win_diag) {
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let s = S_in[s_in_base + j];
                    inj0 = inj0 + AllWeights[win_base + j * d_model + d0] * s;
                    inj1 = inj1 + AllWeights[win_base + j * d_model + d1] * s;
                }
            }
            if (zero_win_diag || signal_zero) {
                inj0 = 0.0;
                inj1 = 0.0;
            } else {
                inj0 = inj0 * signal_scale;
                inj1 = inj1 * signal_scale;
            }
            Scratch[signal_base + d0] = inj0;
            Scratch[signal_base + d1] = inj1;
        } else {
            for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                var inj = 0.0;
                let hist_base = aw_hist_base(d_model, h_slots);
                let signal_zero = AllWeights[hist_base + signal_zero_base(d_model, h_slots)] > 0.5;
                let signal_scale = AllWeights[hist_base + signal_scale_base(d_model, h_slots)];
                if (!zero_win_diag) {
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        inj = inj + AllWeights[win_base + j * d_model + d_out] * S_in[s_in_base + j];
                    }
                }
                if (zero_win_diag || signal_zero) {
                    inj = 0.0;
                } else {
                    inj = inj * signal_scale;
                }
                Scratch[signal_base + d_out] = inj;
            }
        }
        workgroupBarrier();
        if (is_segment_memory_slot) {
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                Scratch[signal_base + d] = 0.0;
            }
        }
        workgroupBarrier();
        if (debug_on && slot_idx == 0u && tid == 0u && t == 0u) {
            DebugLog[8] = 121.0;
        }

        // Historical state sources are loaded as stop-grad constants before Picard.
        // ENABLE_H_HIST and ENABLE_FPM remain mutually exclusive because both use
        // binding 11, while ENABLE_HIST_CTX is an alternate replay/context path.
        var h_hist0 = 0.0;
        var h_hist1 = 0.0;
        // fpm_base0/1 removed — fpm_m_cache carries between tokens (initialized before loop)
        var fpm_ctx0 = 0.0;
        var fpm_ctx1 = 0.0;
        var hist_ctx0 = 0.0;
        var hist_ctx1 = 0.0;
        var hist_mem0 = 0.0;
        var hist_mem1 = 0.0;
        var assoc_post0 = 0.0;
        var assoc_post1 = 0.0;
        if (ENABLE_H_HIST && !is_segment_memory_slot && d_model == WG_SIZE * 2u) {
            h_hist0 = H_hist[h_base + slot_offset + tid];
            h_hist1 = H_hist[h_base + slot_offset + tid + WG_SIZE];
            if (tid == 0u) {
                let gamma_raw = AllWeights[aw_hist_base(d_model, h_slots) + hhist_gamma_base(d_model, h_slots) + slot_idx];
                hhist_gamma_wg = H_HIST_GAMMA_SCALE / (1.0 + exp(-(gamma_raw - 8.0)));
            }
        } else if (ENABLE_HIST_CTX && d_model == WG_SIZE * 2u) {
            let hist_base_t = (batch_idx * shape.seq_len + global_t) * total_elements;
            hist_ctx0 = HistCtx[hist_base_t + slot_offset + tid];
            hist_ctx1 = HistCtx[hist_base_t + slot_offset + tid + WG_SIZE];
        }
        workgroupBarrier();

        if (fpm_inject_enabled && d_model == WG_SIZE * 2u) {
            // fpm_m_cache already carries from previous token — no reinit here.
            if (tid < h_slots) {
                slot_coord_weights[tid] = 1.0 / max(1.0, f32(h_slots));
                slot_coord_prev[tid] = slot_coord_weights[tid];
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
                    var m_j = MState[mem_off + j];
                    if (use_prev_token_mem) {
                        m_j = HistCtx[prev_hist_base_t + ms * d_model + j];
                    }
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

        let should_init_h = t == 0u || !ENABLE_TOKEN_CARRY;
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
            if (tid == 0u) {
                solve_signal_rms_seen = max(solve_signal_rms_seen, sig_rms);
            }
            if (ENABLE_H_HIST && d_model == WG_SIZE * 2u) {
                // signal normalized, then h_hist added as offset (after normalization)
                H_curr[h_base + slot_offset + tid] =
                    Scratch[signal_base + tid] / max(sig_rms, 1e-6) + hhist_gamma_wg * h_hist0;
                H_curr[h_base + slot_offset + tid + WG_SIZE] =
                    Scratch[signal_base + tid + WG_SIZE] / max(sig_rms, 1e-6) + hhist_gamma_wg * h_hist1;
            } else if (fpm_inject_enabled && d_model == WG_SIZE * 2u) {
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
        if (debug_on && slot_idx == 0u && tid == 0u && t == 0u) {
            DebugLog[8] = 151.0;
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
        var token_strict_converged = false;
        var token_homeostatic_converged = false;
        let total_iter_limit = iter_limit + select(0u, FPM_RESCUE_TAIL, fpm_policy_enabled);
        while (iter < total_iter_limit && !converged) {
            if (tid == 0u) {
                converged_flag_wg = 0u;
            }
            workgroupBarrier();
            let rescue_active = fpm_policy_enabled && iter >= iter_limit;
            if (rescue_active && !rescue_triggered && tid == 0u) {
                rescue_triggered = true;
                rescue_entered_sum = rescue_entered_sum + 1.0;
                last_delta = 0.0;
            }
            let refresh_slot_coord = enable_slot_coord && iter == 0u;
            if (refresh_slot_coord) {
                compute_slot_coord(
                    tid,
                    slot_idx,
                    d_model,
                    h_slots,
                    head_dim,
                    signal_base - slot_offset,
                    alpha_base,
                    wq_mat_base,
                    wk_bias_root,
                    wq_bias_base,
                    wo_mat_base,
                    attn_base,
                );
            } else if (!enable_slot_coord) {
                if (d_model == WG_SIZE * 2u) {
                    Scratch[attn_base + tid] = 0.0;
                    Scratch[attn_base + tid + WG_SIZE] = 0.0;
                } else {
                    for (var d = tid; d < d_model; d = d + WG_SIZE) {
                        Scratch[attn_base + d] = 0.0;
                    }
                }
                if (tid < h_slots) {
                    slot_coord_weights[tid] = 1.0 / max(1.0, f32(h_slots));
                    slot_coord_prev[tid] = slot_coord_weights[tid];
                }
            }
            workgroupBarrier();
            let slot_attn_scale = select(1.0, max_h_seen, enable_slot_coord);
            // compute_slot_coord(iter=0) overwrites k_cache with signal-K values.
            // Model A freezes the memory read after the token has gone through one H-step,
            // so the read is built at iter==1 from the first structured token state.
            if (fpm_inject_enabled && d_model == WG_SIZE * 2u && refresh_slot_coord) {
                let hist_base_kfix = aw_hist_base(d_model, h_slots);
                if (tid < h_slots * RETAIN_RANK) {
                    let ms = tid / RETAIN_RANK;
                    let r  = tid % RETAIN_RANK;
                    let mem_off   = h_base + ms * d_model;
                    let kmem_base = hist_base_kfix + w_k_mem_base(d_model, h_slots) + ms * d_model * RETAIN_RANK;
                    let scale_base_kfix = hist_base_kfix + hist_scale_base(d_model, h_slots) + ms * d_model;
                    let bias_base_kfix  = hist_base_kfix + hist_bias_base(d_model, h_slots)  + ms * d_model;
                    var key_acc = 0.0;
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        var m_j = MState[mem_off + j];
                        if (use_prev_token_mem) {
                            m_j = HistCtx[prev_hist_base_t + ms * d_model + j];
                        }
                        let keyed_m_j = m_j
                            + AllWeights[scale_base_kfix + j] * tanh(m_j)
                            + AllWeights[bias_base_kfix  + j];
                        key_acc = key_acc + AllWeights[kmem_base + j * RETAIN_RANK + r] * keyed_m_j;
                    }
                    k_cache[tid] = key_acc;
                }
                workgroupBarrier();
            }

            if (fpm_inject_enabled && d_model == WG_SIZE * 2u && iter == 1u) {
                let hist_base = aw_hist_base(d_model, h_slots);
                let hq0 = H_curr[h_base + slot_offset + tid];
                let hq1 = H_curr[h_base + slot_offset + tid + WG_SIZE];
                shared_vals[tid] = hq0 * hq0 + hq1 * hq1;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let h_query_rms = sqrt(max(shared_vals[0], 1e-6));
                if (tid < RETAIN_RANK) {
                    let qmem_base = hist_base + w_q_mem_base(d_model, h_slots) + slot_idx * d_model * RETAIN_RANK;
                    var q_acc = 0.0;
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        q_acc = q_acc + AllWeights[qmem_base + j * RETAIN_RANK + tid]
                            * (H_curr[h_base + slot_offset + j] / h_query_rms);
                    }
                    shared_vals[tid] = q_acc;
                }
                workgroupBarrier();
                if (tid == 0u) {
                    var q_sq = 0.0;
                    for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                        q_sq = q_sq + shared_vals[r] * shared_vals[r];
                    }
                    let slot_probe_base = 520u + slot_idx * 14u;
                    DebugLog[slot_probe_base + 6u] = sqrt(q_sq / max(1.0, f32(RETAIN_RANK)));
                    var k_rms_avg = 0.0;
                    var src_m_rms_avg = 0.0;
                    var keyed_m_rms_avg = 0.0;
                    for (var ms = 0u; ms < h_slots; ms = ms + 1u) {
                        var k_sq = 0.0;
                        var m_sq = 0.0;
                        var keyed_m_sq = 0.0;
                        let mem_off = h_base + ms * d_model;
                        let scale_base = hist_base + hist_scale_base(d_model, h_slots) + ms * d_model;
                        let bias_base = hist_base + hist_bias_base(d_model, h_slots) + ms * d_model;
                        for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                            let kval = k_cache[ms * RETAIN_RANK + r];
                            k_sq = k_sq + kval * kval;
                        }
                        for (var j = 0u; j < d_model; j = j + 1u) {
                            var m_j = MState[mem_off + j];
                            if (use_prev_token_mem) {
                                m_j = HistCtx[prev_hist_base_t + ms * d_model + j];
                            }
                            let keyed_m_j =
                                m_j
                                + AllWeights[scale_base + j] * tanh(m_j)
                                + AllWeights[bias_base + j];
                            m_sq = m_sq + m_j * m_j;
                            keyed_m_sq = keyed_m_sq + keyed_m_j * keyed_m_j;
                        }
                        k_rms_avg = k_rms_avg + sqrt(k_sq / max(1.0, f32(RETAIN_RANK)));
                        src_m_rms_avg = src_m_rms_avg + sqrt(m_sq / max(1.0, f32(d_model)));
                        keyed_m_rms_avg = keyed_m_rms_avg + sqrt(keyed_m_sq / max(1.0, f32(d_model)));
                    }
                    DebugLog[slot_probe_base + 7u] = k_rms_avg / max(1.0, f32(h_slots));
                    DebugLog[slot_probe_base + 10u] = src_m_rms_avg / max(1.0, f32(h_slots));
                    DebugLog[slot_probe_base + 11u] = keyed_m_rms_avg / max(1.0, f32(h_slots));
                }
                if (tid < h_slots) {
                    let ms = tid;
                    var score = 0.0;
                    for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                        score = score + shared_vals[r] * k_cache[ms * RETAIN_RANK + r];
                    }
                    score = score + AllWeights[hist_base + b_read_mem_base(d_model, h_slots) + ms];
                    slot_coord_prev[ms] =
                        score * inverseSqrt(max(1.0, f32(RETAIN_RANK))) / max(shape.fpm_tau, 1e-3);
                }
                workgroupBarrier();
                if (tid == 0u) {
                    var max_s = -1e30;
                    var second_s = -1e30;
                    for (var ms = 0u; ms < h_slots; ms = ms + 1u) {
                        let s = slot_coord_prev[ms];
                        if (s >= max_s) {
                            second_s = max_s;
                            max_s = s;
                        } else if (s > second_s) {
                            second_s = s;
                        }
                    }
                    var sum_exp = 0.0;
                    for (var ms = 0u; ms < h_slots; ms = ms + 1u) {
                        let w = exp(slot_coord_prev[ms] - max_s);
                        slot_coord_prev[ms] = w;
                        sum_exp = sum_exp + w;
                    }
                    sum_exp = max(sum_exp, 1e-12);
                    for (var ms = 0u; ms < h_slots; ms = ms + 1u) {
                        slot_coord_prev[ms] = slot_coord_prev[ms] / sum_exp;
                    }
                    var peak_w = 0.0;
                    for (var ms = 0u; ms < h_slots; ms = ms + 1u) {
                        peak_w = max(peak_w, slot_coord_prev[ms]);
                    }
                    let uniform_w = 1.0 / max(1.0, f32(h_slots));
                    let confidence = clamp(
                        (peak_w - uniform_w) / max(1e-6, 1.0 - uniform_w),
                        0.0,
                        1.0,
                    );
                    let slot_probe_base = 520u + slot_idx * 14u;
                    DebugLog[slot_probe_base + 8u] = max_s - second_s;
                    DebugLog[slot_probe_base + 9u] = confidence;
                    DebugLog[slot_probe_base + 12u] = peak_w;
                    DebugLog[slot_probe_base + 13u] = (max_s - second_s) * max(shape.fpm_tau, 1e-3);
                    // Model A should read more strongly when the content lookup is selective.
                    // Uniform reads are low-confidence and should not inject a full residual
                    // branch into the solve.
                    // fpm_read_gate_min: override floor (0.0 = default curriculum, 1.0 = always full gate).
                    shared_vals[0] = max(sqrt(confidence), shape.fpm_read_gate_min);
                }
                workgroupBarrier();
                var mem_raw0 = 0.0;
                var mem_raw1 = 0.0;
                for (var ms = 0u; ms < h_slots; ms = ms + 1u) {
                    let mem_off = h_base + ms * d_model;
                    let hist_slot_off = prev_hist_base_t + ms * d_model;
                    var m0 = MState[mem_off + tid];
                    var m1 = MState[mem_off + tid + WG_SIZE];
                    if (use_prev_token_mem) {
                        m0 = HistCtx[hist_slot_off + tid];
                        m1 = HistCtx[hist_slot_off + tid + WG_SIZE];
                    }
                    let w = slot_coord_prev[ms];
                    mem_raw0 = mem_raw0 + w * m0;
                    mem_raw1 = mem_raw1 + w * m1;
                }
                // Model A: memory is a token-fixed context. Its strength should depend on
                // how selective the token's content lookup is, not on the current Picard
                // error of H. The gate is frozen from the first read of the token.
                let read_gate = shared_vals[0];
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
                // Memory read is a residual branch into the solve. Model A expects this
                // coupling to start small and become useful through learned content and
                // training, not through a hidden fixed residual scale that bypasses alpha_m.
                let alpha_m = max(shape.fpm_alpha_m, 0.001);
                let read_residual_scale = select(alpha_m, min(alpha_m, FPM_RESCUE_RESIDUAL_SCALE), rescue_active);
                let mem_to_signal = signal_norm * read_residual_scale / max(mem_unit_norm, 1e-6);
                fpm_ctx0 = mem_unit0 * mem_to_signal * read_gate;
                fpm_ctx1 = mem_unit1 * mem_to_signal * read_gate;
                let fpm_ctx_norm = signal_norm * read_residual_scale * read_gate;
                if (tid == 0u) {
                    let memctx_rms = fpm_ctx_norm / sqrt(max(1.0, f32(d_model)));
                    let signal_rms = signal_norm / sqrt(max(1.0, f32(d_model)));
                    max_memctx_rms_seen = max(max_memctx_rms_seen, memctx_rms);
                    max_memctx_to_signal_seen = max(
                        max_memctx_to_signal_seen,
                        memctx_rms / max(signal_rms, 1e-6),
                    );
                }
                var assoc_ctx0 = 0.0;
                var assoc_ctx1 = 0.0;
                if (d_model == WG_SIZE * 2u) {
                    AssocReadBuf[h_base_t + slot_offset + tid] = 0.0;
                    AssocReadBuf[h_base_t + slot_offset + tid + WG_SIZE] = 0.0;
                }
                if (assoc_read_enabled) {
                    let d0 = tid;
                    let d1 = tid + WG_SIZE;
                    // Associative lookup must not deform the FPM write-key geometry.
                    // It uses its own query matrix so the binding branch can learn without
                    // destabilizing the recurrent FPM write path.
                    let wq_assoc = hist_base
                        + w_q_assoc_base(d_model, h_slots)
                        + slot_idx * d_model * ASSOC_RANK;
                    let alpha_raw = AllWeights[hist_base + alpha_assoc_base(d_model, h_slots) + slot_idx];
                    let alpha_assoc = 1.0 / (1.0 + exp(-alpha_raw));
                    // Layout per bank: [bank_key | bank_value | usage].
                    // Query the durable bank in the same explicit source manifold that wrote
                    // the bank key. At QUERY time PrevHStar already holds the preceding KEY
                    // identity; mixing in the current token here rotates the query away from
                    // the stored key geometry and makes addressing chase the wrong bank.
                    if (tid < ASSOC_RANK) {
                        var query_sig_sq = 0.0;
                        for (var j = 0u; j < d_model; j = j + 1u) {
                            let query_src_j = PrevHStarBuf[prev_hstar_base + j];
                            query_sig_sq = query_sig_sq + query_src_j * query_src_j;
                        }
                        let query_sig_rms = sqrt(query_sig_sq / max(1.0, f32(d_model)) + 1.0e-6);
                        var q_acc = 0.0;
                        for (var j = 0u; j < d_model; j = j + 1u) {
                            let query_src_j =
                                PrevHStarBuf[prev_hstar_base + j] / max(query_sig_rms, 1.0e-6);
                            q_acc = q_acc
                                + AllWeights[wq_assoc + j * ASSOC_RANK + tid]
                                * query_src_j;
                        }
                        shared_vals[tid] = tanh(q_acc);
                    }
                    workgroupBarrier();
                    if (tid == 0u) {
                        var q_norm = 0.0;
                        for (var r = 0u; r < ASSOC_RANK; r = r + 1u) {
                            q_norm = q_norm + shared_vals[r] * shared_vals[r];
                        }
                        shared_vals[3u * ASSOC_RANK + ASSOC_BANKS] = q_norm;
                        var slot_read_allowed = 1.0;
                        if (ENABLE_ASSOC_SLOT_OWNER && ENABLE_ASSOC_READ_SLOT_PRIOR) {
                            let slot_anchor_root_read = hist_base + slot_anchor_base(d_model, h_slots);
                            var owner_query_sq = 0.0;
                            for (var j = 0u; j < d_model; j = j + 1u) {
                                let query_src_j = PrevHStarBuf[prev_hstar_base + j];
                                owner_query_sq = owner_query_sq + query_src_j * query_src_j;
                            }
                            let owner_query_rms = sqrt(
                                owner_query_sq / max(1.0, f32(d_model)) + 1.0e-6
                            );
                            var best_owner_score = -1.0e30;
                            var slot_owner_score = -1.0e30;
                            for (var owner = 0u; owner < h_slots; owner = owner + 1u) {
                                let owner_off = slot_anchor_root_read + owner * d_model;
                                var anchor_sq = 0.0;
                                var dot = 0.0;
                                for (var j = 0u; j < d_model; j = j + 1u) {
                                    let query_src_j =
                                        PrevHStarBuf[prev_hstar_base + j] / max(owner_query_rms, 1.0e-6);
                                    let anchor_j = AllWeights[owner_off + j];
                                    anchor_sq = anchor_sq + anchor_j * anchor_j;
                                    dot = dot + query_src_j * anchor_j;
                                }
                                let owner_score = dot / sqrt(max(anchor_sq, 1.0e-12));
                                if (owner_score > best_owner_score) {
                                    best_owner_score = owner_score;
                                }
                                if (owner == slot_idx) {
                                    slot_owner_score = owner_score;
                                }
                            }
                            // WRITE ownership can stay exclusive to avoid duplicated bindings.
                            // READ needs softer access: useful banks may live outside the best
                            // owner slot, so use owner geometry as a prior on slot contribution
                            // instead of a hard binary gate.
                            slot_read_allowed = exp(clamp(
                                ASSOC_READ_SLOT_PRIOR_BETA * (slot_owner_score - best_owner_score),
                                -8.0,
                                0.0
                            ));
                        }
                        shared_vals[3u * ASSOC_RANK + ASSOC_BANKS + 2u] = slot_read_allowed;
                    }
                    workgroupBarrier();
                    for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                        let bank_base = assoc_slot_base + bank * assoc_bank_stride;
                        let bank_key_base = bank_base;
                    let bank_usage = AssocBuf[bank_base + ASSOC_RANK + d_model];
                        if (tid < ASSOC_RANK) {
                            shared_vals[ASSOC_RANK + tid] = AssocBuf[bank_key_base + tid] * shared_vals[tid];
                        }
                        workgroupBarrier();
                        if (tid == 0u) {
                            var dot_score = 0.0;
                            var key_norm = 0.0;
                            for (var r = 0u; r < ASSOC_RANK; r = r + 1u) {
                                let key_r = AssocBuf[bank_key_base + r];
                                dot_score = dot_score + shared_vals[ASSOC_RANK + r];
                                key_norm = key_norm + key_r * key_r;
                            }
                            let q_norm = shared_vals[3u * ASSOC_RANK + ASSOC_BANKS];
                            if (bank_usage < ASSOC_OCCUPIED_THRESHOLD) {
                                shared_vals[3u * ASSOC_RANK + bank] = -25.0;
                            } else {
                                // Once empty banks are masked out, retrieval should be
                                // purely content-addressed. A usage prior biases reads
                                // toward older/more-written banks even when a different
                                // occupied bank matches the query key better.
                                let address_score = dot_score * inverseSqrt(max(key_norm * q_norm, 1.0e-12));
                                shared_vals[3u * ASSOC_RANK + bank] =
                                    clamp(ASSOC_READ_BETA * address_score, -25.0, 25.0);
                            }
                        }
                        workgroupBarrier();
                    }
                    if (tid == 0u) {
                        let slot_read_allowed = shared_vals[3u * ASSOC_RANK + ASSOC_BANKS + 2u];
                        var max_score = -1.0e30;
                        var min_score = 1.0e30;
                        for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                            max_score = max(max_score, shared_vals[3u * ASSOC_RANK + bank]);
                            min_score = min(min_score, shared_vals[3u * ASSOC_RANK + bank]);
                        }
                        var bank_key_cos = 0.0;
                        if (ASSOC_BANKS >= 2u) {
                            let bank0_base = assoc_slot_base;
                            let bank1_base = assoc_slot_base + assoc_bank_stride;
                            var key_dot = 0.0;
                            var key0_norm = 0.0;
                            var key1_norm = 0.0;
                            for (var r = 0u; r < ASSOC_RANK; r = r + 1u) {
                                let k0 = AssocBuf[bank0_base + r];
                                let k1 = AssocBuf[bank1_base + r];
                                key_dot = key_dot + k0 * k1;
                                key0_norm = key0_norm + k0 * k0;
                                key1_norm = key1_norm + k1 * k1;
                            }
                            bank_key_cos = key_dot * inverseSqrt(max(key0_norm * key1_norm, 1.0e-12));
                        }
                        if (ENABLE_ASSOC_HARD_READ) {
                            var best_bank = 0u;
                            var best_score = -1.0e30;
                            for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                                let score = shared_vals[3u * ASSOC_RANK + bank];
                                if (score > best_score) {
                                    best_score = score;
                                    best_bank = bank;
                                }
                            }
                            for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                                shared_vals[3u * ASSOC_RANK + bank] = select(0.0, 1.0, bank == best_bank);
                            }
                        } else {
                            var denom = 0.0;
                            for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                                let e = exp(shared_vals[3u * ASSOC_RANK + bank] - max_score);
                                shared_vals[3u * ASSOC_RANK + bank] = e;
                                denom = denom + e;
                            }
                            for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                                shared_vals[3u * ASSOC_RANK + bank] =
                                    shared_vals[3u * ASSOC_RANK + bank] / max(denom, 1.0e-6);
                            }
                        }
                        // TEMPORARY ASSOCIATIVE DIAGNOSTIC: read competition/occupancy telemetry.
                        var read_entropy = 0.0;
                        var read_max_prob = 0.0;
                        var read_usage = 0.0;
                        for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                            let bank_base = assoc_slot_base + bank * assoc_bank_stride;
                            let p = shared_vals[3u * ASSOC_RANK + bank];
                            read_entropy = read_entropy - p * log(max(p, 1.0e-8));
                            read_max_prob = max(read_max_prob, p);
                            read_usage = read_usage + AssocBuf[bank_base + ASSOC_RANK + d_model];
                        }
                        shared_vals[3u * ASSOC_RANK + ASSOC_BANKS + 1u] = read_max_prob;
                        if (debug_on) {
                            let assoc_diag_base = 760u + slot_idx * 10u;
                            DebugLog[assoc_diag_base + 0u] =
                                DebugLog[assoc_diag_base + 0u] + read_entropy;
                            DebugLog[assoc_diag_base + 1u] = max(
                                DebugLog[assoc_diag_base + 1u],
                                read_max_prob * slot_read_allowed
                            );
                            DebugLog[assoc_diag_base + 3u] =
                                DebugLog[assoc_diag_base + 3u] + read_usage / max(1.0, f32(ASSOC_BANKS));
                            DebugLog[assoc_diag_base + 8u] = DebugLog[assoc_diag_base + 8u] + 1.0;
                            if (t + 1u == shape.token_count) {
                                let assoc_query_diag_base = 820u + slot_idx * 8u;
                                var best_bank_idx = 0u;
                                var best_bank_prob = -1.0;
                                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                                    let p = shared_vals[3u * ASSOC_RANK + bank];
                                    if (p > best_bank_prob) {
                                        best_bank_prob = p;
                                        best_bank_idx = bank;
                                    }
                                }
                                DebugLog[assoc_query_diag_base + 0u] =
                                    DebugLog[assoc_query_diag_base + 0u] + read_entropy;
                                DebugLog[assoc_query_diag_base + 1u] = max(
                                    DebugLog[assoc_query_diag_base + 1u],
                                    read_max_prob * slot_read_allowed
                                );
                                DebugLog[assoc_query_diag_base + 3u] =
                                    DebugLog[assoc_query_diag_base + 3u] + read_usage / max(1.0, f32(ASSOC_BANKS));
                                DebugLog[assoc_query_diag_base + 4u] =
                                    max(DebugLog[assoc_query_diag_base + 4u], max_score - min_score);
                                DebugLog[assoc_query_diag_base + 5u] =
                                    max(DebugLog[assoc_query_diag_base + 5u], bank_key_cos);
                                DebugLog[assoc_query_diag_base + 6u] = DebugLog[assoc_query_diag_base + 6u] + 1.0;
                                DebugLog[assoc_query_diag_base + 7u] =
                                    DebugLog[assoc_query_diag_base + 7u]
                                    + sqrt(shared_vals[3u * ASSOC_RANK + ASSOC_BANKS] / max(1.0, f32(ASSOC_RANK)));
                                let assoc_query_pick_base = 940u + slot_idx * 2u;
                                DebugLog[assoc_query_pick_base + 0u] = f32(best_bank_idx);
                                DebugLog[assoc_query_pick_base + 1u] =
                                    best_bank_prob * slot_read_allowed;
                            }
                        }
                    }
                    workgroupBarrier();
                    if (ENABLE_ASSOC_PERSISTENT) {
                        for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                            let bank_base = assoc_slot_base + bank * assoc_bank_stride;
                            let bank_key_base = bank_base;
                            let bank_usage = AssocPersistentBuf[bank_base + ASSOC_RANK + d_model];
                            if (tid < ASSOC_RANK) {
                                shared_vals[2u * ASSOC_RANK + tid] =
                                    AssocPersistentBuf[bank_key_base + tid] * shared_vals[tid];
                            }
                            workgroupBarrier();
                            if (tid == 0u) {
                                var dot_score = 0.0;
                                var key_norm = 0.0;
                                for (var r = 0u; r < ASSOC_RANK; r = r + 1u) {
                                    let key_r = AssocPersistentBuf[bank_key_base + r];
                                    dot_score = dot_score + shared_vals[2u * ASSOC_RANK + r];
                                    key_norm = key_norm + key_r * key_r;
                                }
                                let q_norm = shared_vals[3u * ASSOC_RANK + ASSOC_BANKS];
                                if (bank_usage < ASSOC_OCCUPIED_THRESHOLD) {
                                    shared_vals[4u * ASSOC_RANK + bank] = -25.0;
                                } else {
                                    let address_score = dot_score * inverseSqrt(max(key_norm * q_norm, 1.0e-12));
                                    shared_vals[4u * ASSOC_RANK + bank] =
                                        clamp(ASSOC_READ_BETA * address_score, -25.0, 25.0);
                                }
                            }
                            workgroupBarrier();
                        }
                        if (tid == 0u) {
                            var persistent_max_score = -1.0e30;
                            if (ENABLE_ASSOC_HARD_READ) {
                                var best_bank = 0u;
                                var best_score = -1.0e30;
                                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                                    let score = shared_vals[4u * ASSOC_RANK + bank];
                                    if (score > best_score) {
                                        best_score = score;
                                        best_bank = bank;
                                    }
                                }
                                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                                    shared_vals[4u * ASSOC_RANK + bank] = select(0.0, 1.0, bank == best_bank);
                                }
                            } else {
                                var denom = 0.0;
                                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                                    persistent_max_score = max(persistent_max_score, shared_vals[4u * ASSOC_RANK + bank]);
                                }
                                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                                    let e = exp(shared_vals[4u * ASSOC_RANK + bank] - persistent_max_score);
                                    shared_vals[4u * ASSOC_RANK + bank] = e;
                                    denom = denom + e;
                                }
                                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                                    shared_vals[4u * ASSOC_RANK + bank] =
                                        shared_vals[4u * ASSOC_RANK + bank] / max(denom, 1.0e-6);
                                }
                            }
                        }
                        workgroupBarrier();
                    }
                    let slot_read_allowed = shared_vals[3u * ASSOC_RANK + ASSOC_BANKS + 2u];
                    let assoc_read_conf =
                        select(1.0, shared_vals[3u * ASSOC_RANK + ASSOC_BANKS + 1u], ENABLE_ASSOC_CONF_READ);
                    
                    // Signal Balancing: if Persistent is active, split energy 50/50 to maintain total magnitude.
                    let assoc_mix_weight = select(1.0, 0.5, ENABLE_ASSOC_PERSISTENT);

                    for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                        let bank_base = assoc_slot_base + bank * assoc_bank_stride;
                        let bank_value_base = bank_base + ASSOC_RANK;
                        let assoc_match =
                            slot_read_allowed * assoc_read_conf * shared_vals[3u * ASSOC_RANK + bank];
                        assoc_ctx0 = assoc_ctx0 + assoc_mix_weight * assoc_match * AssocBuf[bank_value_base + d0];
                        assoc_ctx1 = assoc_ctx1 + assoc_mix_weight * assoc_match * AssocBuf[bank_value_base + d1];
                    }
                    if (ENABLE_ASSOC_PERSISTENT) {
                        for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                            let bank_base = assoc_slot_base + bank * assoc_bank_stride;
                            let bank_value_base = bank_base + ASSOC_RANK;
                            let assoc_match =
                                slot_read_allowed * assoc_read_conf * shared_vals[4u * ASSOC_RANK + bank];
                            assoc_ctx0 = assoc_ctx0 + assoc_mix_weight * assoc_match * AssocPersistentBuf[bank_value_base + d0];
                            assoc_ctx1 = assoc_ctx1 + assoc_mix_weight * assoc_match * AssocPersistentBuf[bank_value_base + d1];
                        }
                    }
                    // TEMPORARY ASSOCIATIVE DIAGNOSTIC: measure injected assoc context magnitude.
                    shared_vals[tid] = assoc_ctx0 * assoc_ctx0 + assoc_ctx1 * assoc_ctx1;
                    workgroupBarrier();
                    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                        if (tid < stride) {
                            shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                        }
                        workgroupBarrier();
                    }
                    if (tid == 0u) {
                        if (debug_on) {
                            let assoc_diag_base = 760u + slot_idx * 10u;
                            DebugLog[assoc_diag_base + 2u] = max(
                                DebugLog[assoc_diag_base + 2u],
                                sqrt(shared_vals[0] / max(1.0, f32(d_model))),
                            );
                            if (t + 1u == shape.token_count) {
                                let assoc_query_diag_base = 820u + slot_idx * 8u;
                                DebugLog[assoc_query_diag_base + 2u] = max(
                                    DebugLog[assoc_query_diag_base + 2u],
                                    sqrt(shared_vals[0] / max(1.0, f32(d_model))),
                                );
                            }
                        }
                    }
                    workgroupBarrier();
                    let assoc_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)));
                    let assoc_scale = 1.0 / max(assoc_rms, 0.1); // Soft normalization floor
                    let assoc_inj0 = alpha_assoc * assoc_ctx0 * assoc_scale;
                    let assoc_inj1 = alpha_assoc * assoc_ctx1 * assoc_scale;
                    let assoc_export0 = assoc_ctx0;
                    let assoc_export1 = assoc_ctx1;
                    AssocReadBuf[h_base_t + slot_offset + d0] = assoc_export0;
                    AssocReadBuf[h_base_t + slot_offset + d1] = assoc_export1;
                    fpm_ctx0 = fpm_ctx0 + assoc_inj0;
                    fpm_ctx1 = fpm_ctx1 + assoc_inj1;
                    if (ENABLE_ASSOC_POST_HSTAR) {
                        assoc_post0 = assoc_inj0;
                        assoc_post1 = assoc_inj1;
                    }
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
                let attn0 = select(0.0, Scratch[attn_base + d0] * slot_attn_scale, recurrent_slot_attn_enabled || iter == 0u);
                let attn1 = select(0.0, Scratch[attn_base + d1] * slot_attn_scale, recurrent_slot_attn_enabled || iter == 0u);
                let h_dep0 = Scratch[signal_base + d0] + H_SELF_FEEDBACK * h_prev0 + attn0
                           + AllWeights[slot_anchor_mat_base + d0];
                let h_dep1 = Scratch[signal_base + d1] + H_SELF_FEEDBACK * h_prev1 + attn1
                           + AllWeights[slot_anchor_mat_base + d1];
                H_next[h_base_t + slot_offset + d0] = h_dep0 + hhist_gamma_wg * h_hist0;
                H_next[h_base_t + slot_offset + d1] = h_dep1 + hhist_gamma_wg * h_hist1;
                local_sumsq = h_dep0 * h_dep0 + h_dep1 * h_dep1; // rms from h-dep only
            } else if (fpm_inject_enabled && d_model == WG_SIZE * 2u) {
                // Model A: memory contributes a token-fixed context inside the H-only solve.
                // Like h_hist, it is context from the past, not a fresh H-dependent branch,
                // so it should not inflate the RMS denominator of the H operator itself.
                let d0 = tid;
                let d1 = tid + WG_SIZE;
                let h_prev0 = H_curr[h_base + slot_offset + d0];
                let h_prev1 = H_curr[h_base + slot_offset + d1];
                let attn0 = select(
                    0.0,
                    Scratch[attn_base + d0] * slot_attn_scale,
                    recurrent_slot_attn_enabled || iter == 0u,
                );
                let attn1 = select(
                    0.0,
                    Scratch[attn_base + d1] * slot_attn_scale,
                    recurrent_slot_attn_enabled || iter == 0u,
                );
                let h_dep0 = Scratch[signal_base + d0] + H_SELF_FEEDBACK * h_prev0 + attn0
                           + AllWeights[slot_anchor_mat_base + d0];
                let h_dep1 = Scratch[signal_base + d1] + H_SELF_FEEDBACK * h_prev1 + attn1
                           + AllWeights[slot_anchor_mat_base + d1];
                let pre0 = h_dep0 + fpm_ctx0;
                let pre1 = h_dep1 + fpm_ctx1;
                H_next[h_base_t + slot_offset + d0] = pre0;
                H_next[h_base_t + slot_offset + d1] = pre1;
                local_sumsq = h_dep0 * h_dep0 + h_dep1 * h_dep1;
            } else if (ENABLE_HIST_CTX && d_model == WG_SIZE * 2u) {
                let d0 = tid;
                let d1 = tid + WG_SIZE;
                let h_prev0 = H_curr[h_base + slot_offset + d0];
                let h_prev1 = H_curr[h_base + slot_offset + d1];
                let attn0 = select(0.0, Scratch[attn_base + d0] * slot_attn_scale, recurrent_slot_attn_enabled || iter == 0u);
                let attn1 = select(0.0, Scratch[attn_base + d1] * slot_attn_scale, recurrent_slot_attn_enabled || iter == 0u);
                let h_dep0 = Scratch[signal_base + d0] + H_SELF_FEEDBACK * h_prev0 + attn0
                           + AllWeights[slot_anchor_mat_base + d0] + hist_mem0;
                let h_dep1 = Scratch[signal_base + d1] + H_SELF_FEEDBACK * h_prev1 + attn1
                           + AllWeights[slot_anchor_mat_base + d1] + hist_mem1;
                H_next[h_base_t + slot_offset + d0] = h_dep0;
                H_next[h_base_t + slot_offset + d1] = h_dep1;
                local_sumsq = h_dep0 * h_dep0 + h_dep1 * h_dep1;
            } else {
                for (var d = tid; d < d_model; d = d + WG_SIZE) {
                    let h_prev = H_curr[h_base + slot_offset + d];
                    let attn = select(0.0, Scratch[attn_base + d] * slot_attn_scale, recurrent_slot_attn_enabled || iter == 0u);
                    let pre = Scratch[signal_base + d]
                        + H_SELF_FEEDBACK * h_prev
                        + attn
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
            var local_fh_sq = 0.0;
            var local_nscale_abs = 0.0;
            var local_attn_sq = 0.0;
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                let h_prev = H_curr[h_base + slot_offset + d];
                let nscale = AllWeights[nscale_base + d];
                let attn_val = Scratch[attn_base + d] * slot_attn_scale;
                let f_h = nscale * (H_next[h_base_t + slot_offset + d] / rms);
                let blend = select(shape.damping, alpha_h, fpm_policy_enabled && d_model == WG_SIZE * 2u);
                let val = blend * f_h + (1.0 - blend) * h_prev;
                let delta_h = val - h_prev;
                local_delta_h_num = local_delta_h_num + delta_h * delta_h;
                local_delta_h_den = local_delta_h_den + h_prev * h_prev;
                local_fh_sq = local_fh_sq + f_h * f_h;
                local_nscale_abs = max(local_nscale_abs, abs(nscale));
                local_attn_sq = local_attn_sq + attn_val * attn_val;
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
            let hprev_energy = shared_vals[0];
            let delta_h_rms = sqrt(delta_h_num * inv_d_model + 1e-12);
            let err_h_operator = delta_h_rms / max(rms, 1e-6);
            let err_h = err_h_operator;
            let hprev_rms = sqrt(hprev_energy * inv_d_model + 1e-6);
            shared_vals[tid] = local_fh_sq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let fh_rms = sqrt(shared_vals[0] * inv_d_model + 1e-6);
            shared_vals[tid] = local_attn_sq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let attn_rms = sqrt(shared_vals[0] * inv_d_model + 1e-6);
            shared_vals[tid] = local_nscale_abs;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
                }
                workgroupBarrier();
            }
            let nscale_abs = shared_vals[0];
            var local_max_delta_a = 0.0;
            if (fpm_inject_enabled && tid < h_slots) {
                local_max_delta_a = abs(slot_coord_prev[tid] - slot_coord_weights[tid]);
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
            let strict_converged = stop_err_h < shape.epsilon;
            let homeostatic_converged =
                iter + 1u >= FPM_HOMEO_MIN_ITERS
                && stop_err_h <= homeo_band
                && plateau_ratio <= FPM_HOMEO_PLATEAU_TOL;
            if (tid == 0u && (strict_converged || homeostatic_converged)) {
                converged_flag_wg = 1u;
                if (strict_converged) {
                    token_strict_converged = true;
                } else {
                    token_homeostatic_converged = true;
                }
                if (!rescue_active) {
                    converged_before_rescue = true;
                }
            }
            workgroupBarrier();
            converged = converged_flag_wg != 0u;
            if (tid == 0u) {
                let d_curr = stop_err_h;
                let d_prev = last_delta;
                var recurrent_iter = iter;
                if ((enable_slot_coord && !ENABLE_FPM || model_a_memory_bootstrap) && iter > 0u) {
                    recurrent_iter = iter - 1u;
                }
                if (recurrent_iter > 0u && d_prev > 1e-12 && d_prev > shape.epsilon * 10.0) {
                    max_contractivity = max(max_contractivity, d_curr / d_prev);
                }
                last_delta = d_curr;
                if (!((enable_slot_coord && !ENABLE_FPM || model_a_memory_bootstrap) && iter == 0u)) {
                    max_delta_seen = max(max_delta_seen, d_curr);
                }
                max_m_delta_seen = max(max_m_delta_seen, iter_max_delta_m);
                max_a_delta_seen = max(max_a_delta_seen, iter_max_delta_a);
                if (stop_err_h > max_err_h_marker_seen) {
                    max_err_h_marker_seen = stop_err_h;
                    iter_of_max_err_h_seen = f32(iter);
                    token_of_max_err_h_seen = f32(t);
                }
                let attn_to_signal = attn_rms / max(solve_signal_rms_seen, 1e-6);
                if (attn_to_signal > max_attn_ratio_marker_seen) {
                    max_attn_ratio_marker_seen = attn_to_signal;
                    iter_of_max_attn_ratio_seen = f32(iter);
                    token_of_max_attn_ratio_seen = f32(t);
                }
                max_err_h_seen = max(max_err_h_seen, stop_err_h);
                max_err_m_seen = max(max_err_m_seen, err_m);
                max_z_seen = max(max_z_seen, local_gate);
                max_update_ratio_seen = max(max_update_ratio_seen, local_update_ratio);
                solve_pre_rms_seen = max(solve_pre_rms_seen, rms);
                solve_fh_rms_seen = max(solve_fh_rms_seen, fh_rms);
                solve_hprev_rms_seen = max(solve_hprev_rms_seen, hprev_rms);
                solve_nscale_abs_seen = max(solve_nscale_abs_seen, nscale_abs);
                solve_pre_to_hprev_seen =
                    max(solve_pre_to_hprev_seen, rms / max(hprev_rms, 1e-6));
                solve_fh_to_hprev_seen =
                    max(solve_fh_to_hprev_seen, fh_rms / max(hprev_rms, 1e-6));
                solve_attn_rms_seen = max(solve_attn_rms_seen, attn_rms);
                solve_attn_to_signal_seen = max(solve_attn_to_signal_seen, attn_to_signal);
                solve_attn_scale_seen = max(solve_attn_scale_seen, abs(slot_attn_scale));
                max_homeo_band_seen = max(max_homeo_band_seen, homeo_band);
                if (iter == 0u) {
                    iter0_err_h_seen = max(iter0_err_h_seen, stop_err_h);
                    iter0_attn_to_signal_seen = max(
                        iter0_attn_to_signal_seen,
                        attn_rms / max(solve_signal_rms_seen, 1e-6),
                    );
                    iter0_attn_scale_seen = max(iter0_attn_scale_seen, abs(slot_attn_scale));
                } else if (iter == 1u) {
                    iter1_err_h_seen = max(iter1_err_h_seen, stop_err_h);
                    iter1_attn_to_signal_seen = max(
                        iter1_attn_to_signal_seen,
                        attn_rms / max(solve_signal_rms_seen, 1e-6),
                    );
                    iter1_attn_scale_seen = max(iter1_attn_scale_seen, abs(slot_attn_scale));
                }
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
        if (debug_on && slot_idx == 0u && tid == 0u && t == 0u) {
            DebugLog[8] = 181.0;
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
                let w = slot_coord_weights[ms];
                entropy = entropy - w * log(max(w, 1e-12));
            }
            sum_self_assign_seen = sum_self_assign_seen + slot_coord_weights[slot_idx];
            sum_assign_entropy_seen = sum_assign_entropy_seen + entropy;
            if (slot_coord_weights[slot_idx] < FPM_DEAD_THRESHOLD) {
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
            if (token_strict_converged) {
                strict_converged_sum = strict_converged_sum + 1.0;
            } else if (token_homeostatic_converged) {
                homeostatic_converged_sum = homeostatic_converged_sum + 1.0;
            } else {
                failed_converged_sum = failed_converged_sum + 1.0;
            }
            if (!converged) {
                failed_hits_seen = failed_hits_seen + 1u;
            }
        }
        if (debug_on && slot_idx == 0u && tid == 0u && t == 0u) {
            DebugLog[8] = 182.0;
        }
        workgroupBarrier();

        if (ENABLE_ASSOC_POST_HSTAR && assoc_read_enabled && d_model == WG_SIZE * 2u) {
            let d0 = tid;
            let d1 = tid + WG_SIZE;
            H_curr[h_base + slot_offset + d0] = H_curr[h_base + slot_offset + d0] + assoc_post0;
            H_curr[h_base + slot_offset + d1] = H_curr[h_base + slot_offset + d1] + assoc_post1;
            H_next[h_base_t + slot_offset + d0] = H_curr[h_base + slot_offset + d0];
            H_next[h_base_t + slot_offset + d1] = H_curr[h_base + slot_offset + d1];
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
        if (debug_on && slot_idx == 0u && tid == 0u && t == 0u) {
            DebugLog[8] = 183.0;
        }
        workgroupBarrier();

        // FPM plastic write: once per token using h* (converged fixed point).
        // Model A memory should be a temporal carrier, not just a gated proposal buffer.
        // The base state therefore follows the same slot-wise temporal dynamics as the CPU
        // reference path (a_log / w_x / w_out), while retain/z/proposal modulate the novelty
        // injected into that carrier.
        var assoc_write_gate_token = 1.0;
        if (fpm_write_enabled && !is_segment_memory_slot && d_model == WG_SIZE * 2u) {
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
            let wx_base = aw_wx_base(d_model, h_slots);
            let wout_base = aw_wout_base(d_model, h_slots);
            let alog_off = aw_alog_base(d_model, h_slots) + slot_offset;
            let h_sq = h_val0 * h_val0 + h_val1 * h_val1;
            shared_vals[tid] = h_sq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let h_rms = sqrt(max(shared_vals[0] / max(1.0, f32(d_model)), 1e-6));
            let h_unit0 = h_val0 / h_rms;
            let h_unit1 = h_val1 / h_rms;
            let wf_base = hist_base + w_write_gate_base(d_model, h_slots) + slot_idx * d_model;
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
            let gate_bias = AllWeights[hist_base + b_write_mem_base(d_model, h_slots) + slot_idx]
                + FPM_GATE_BIAS;
            let raw_z = 1.0 / (1.0 + exp(-(shared_vals[0] * inverseSqrt(max(1.0, f32(d_model))) + gate_bias)));
            // Factored k×v write proposal: bottleneck = W_k_write·c  (d→r), proposal = tanh(W_v_write·bottleneck + b)
            let wkw_base = hist_base + w_k_write_base(d_model, h_slots) + slot_idx * d_model * RETAIN_RANK;
            let wvw_base = hist_base + w_v_write_base(d_model, h_slots) + slot_idx * RETAIN_RANK * d_model;
            let bd_base  = hist_base + hist_delta_bias_base(d_model, h_slots) + slot_idx * d_model;
            // Step 1: bottleneck[r] = Σ_j W_k_write[j,r] * c_j  (RETAIN_RANK lanes)
            if (tid < RETAIN_RANK) {
                var kw_acc = 0.0;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let c_j = 0.5 * (H_curr[h_base + slot_offset + j] + Scratch[signal_base + j]);
                    kw_acc = kw_acc + AllWeights[wkw_base + j * RETAIN_RANK + tid] * c_j;
                }
                shared_vals[tid] = kw_acc;
            }
            workgroupBarrier();
            // Step 2: proposal[d] = tanh(Σ_r W_v_write[r,d] * bottleneck[r] + b_delta[d])
            var delta_in0 = AllWeights[bd_base + d0];
            var delta_in1 = AllWeights[bd_base + d1];
            for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                delta_in0 = delta_in0 + AllWeights[wvw_base + r * d_model + d0] * shared_vals[r];
                delta_in1 = delta_in1 + AllWeights[wvw_base + r * d_model + d1] * shared_vals[r];
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
            let diff0 = proposal0 - m_prev0;
            let diff1 = proposal1 - m_prev1;
            shared_vals[tid] = diff0 * diff0 + diff1 * diff1;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            // Structural novelty measures actual disagreement between the candidate write
            // and the carried memory, not just the magnitude of the candidate.
            let diff_norm = sqrt(max(shared_vals[0], 1e-6));
            let novelty = diff_norm / (diff_norm + prev_norm + 1e-6);
            let z = clamp(raw_z, 0.0, 1.0);
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
            let retain_raw0 = 1.0 / (1.0 + exp(-down_acc0));
            let retain_raw1 = 1.0 / (1.0 + exp(-down_acc1));
            // Retain is now a true preservation gate:
            // when novelty is low, preserve memory regardless of local write preference;
            // when novelty is high, the learned retain gate decides which dimensions stay fixed.
            let retain0 = 1.0 - novelty * (1.0 - retain_raw0);
            let retain1 = 1.0 - novelty * (1.0 - retain_raw1);
            let write_budget0 = (1.0 - retain0) * z;
            let write_budget1 = (1.0 - retain1) * z;
            let write0 = z * (residual_scale * proposal0);
            let write1 = z * (residual_scale * proposal1);
            let wx0 = 0.5 * tanh(AllWeights[wx_base + d0 * d_model + d0]);
            let wx1 = 0.5 * tanh(AllWeights[wx_base + d1 * d_model + d1]);
            let wx_term0 = wx0 * h_unit0;
            let wx_term1 = wx1 * h_unit1;
            let h_write0 = sqrt(max(z, 1.0e-6)) * h_unit0;
            let h_write1 = sqrt(max(z, 1.0e-6)) * h_unit1;
            let x_proj0 = h_write0 + wx_term0 + write0;
            let x_proj1 = h_write1 + wx_term1 + write1;
            let a0 = 1.0 / (1.0 + exp(AllWeights[alog_off + d0]));
            let a1 = 1.0 / (1.0 + exp(AllWeights[alog_off + d1]));
            let base_inner0 = a0 * m_prev0 + (1.0 - a0) * x_proj0;
            let base_inner1 = a1 * m_prev1 + (1.0 - a1) * x_proj1;
            let m_inner0 = retain0 * m_prev0 + (1.0 - retain0) * base_inner0;
            let m_inner1 = retain1 * m_prev1 + (1.0 - retain1) * base_inner1;
            H_next[h_base_t + slot_offset + d0] = m_inner0;
            H_next[h_base_t + slot_offset + d1] = m_inner1;
            workgroupBarrier();
            var out_acc0 = m_inner0;
            var out_acc1 = m_inner1;
            for (var j = 0u; j < d_model; j = j + 1u) {
                let m_inner_j = H_next[h_base_t + slot_offset + j];
                out_acc0 = out_acc0 + AllWeights[wout_base + d0 * d_model + j] * m_inner_j;
                out_acc1 = out_acc1 + AllWeights[wout_base + d1 * d_model + j] * m_inner_j;
            }
            // W_out remains observable as a readout/refinement term, but the recurrent
            // FPM state itself is m_inner. Recirculating (I + W_out)m_inner as memory made
            // the state semantics depend on a readout transform and became unstable when
            // FPM was given enough alpha to matter.
            let m_candidate0 = out_acc0;
            let m_candidate1 = out_acc1;
            let wout_term0 = m_candidate0 - m_inner0;
            let wout_term1 = m_candidate1 - m_inner1;
            let proposal_rms_p = sqrt(0.5 * (proposal0 * proposal0 + proposal1 * proposal1));
            let candidate_rms_p = sqrt(0.5 * (m_candidate0 * m_candidate0 + m_candidate1 * m_candidate1));
            let retain_avg_p = 0.5 * (retain0 + retain1);
            let retain_max_p = max(retain0, retain1);
            if (debug_on) {
                shared_vals[tid] = h_write0 * h_write0 + h_write1 * h_write1;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let h_write_rms_p = sqrt(shared_vals[0] * inv_d_model + 1e-6);
                shared_vals[tid] = wx_term0 * wx_term0 + wx_term1 * wx_term1;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let wx_term_rms_p = sqrt(shared_vals[0] * inv_d_model + 1e-6);
                shared_vals[tid] = write0 * write0 + write1 * write1;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let write_rms_p = sqrt(shared_vals[0] * inv_d_model + 1e-6);
                shared_vals[tid] = m_inner0 * m_inner0 + m_inner1 * m_inner1;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let m_inner_rms_p = sqrt(shared_vals[0] * inv_d_model + 1e-6);
                shared_vals[tid] = wout_term0 * wout_term0 + wout_term1 * wout_term1;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let wout_term_rms_p = sqrt(shared_vals[0] * inv_d_model + 1e-6);
                if (tid == 0u) {
                    let slot_write_base = 640u + slot_idx * 6u;
                    DebugLog[slot_write_base + 0u] = max(DebugLog[slot_write_base + 0u], h_write_rms_p);
                    DebugLog[slot_write_base + 1u] = max(DebugLog[slot_write_base + 1u], wx_term_rms_p);
                    DebugLog[slot_write_base + 2u] = max(DebugLog[slot_write_base + 2u], write_rms_p);
                    DebugLog[slot_write_base + 3u] = max(DebugLog[slot_write_base + 3u], m_inner_rms_p);
                    DebugLog[slot_write_base + 4u] = max(DebugLog[slot_write_base + 4u], wout_term_rms_p);
                    DebugLog[slot_write_base + 5u] = max(DebugLog[slot_write_base + 5u], candidate_rms_p);
                }
                workgroupBarrier();
            }
            // Diagnostics: replicate the same reductions as the original per-iter block.
            var local_delta_m_num_p = (m_inner0 - m_prev0) * (m_inner0 - m_prev0)
                                    + (m_inner1 - m_prev1) * (m_inner1 - m_prev1);
            shared_vals[tid] = local_delta_m_num_p;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let delta_m_num_p = shared_vals[0];
            var local_delta_m_prev_den_p = m_prev0 * m_prev0 + m_prev1 * m_prev1;
            shared_vals[tid] = local_delta_m_prev_den_p;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let delta_m_prev_den_p = shared_vals[0];
            let local_delta_m_cand_den_p = m_inner0 * m_inner0 + m_inner1 * m_inner1;
            shared_vals[tid] = local_delta_m_cand_den_p;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let delta_m_cand_den_p = shared_vals[0];
            let err_m_p = sqrt(delta_m_num_p / max(max(delta_m_prev_den_p, delta_m_cand_den_p), FPM_EPS));
            let local_max_delta_m_p = max(abs(m_inner0 - m_prev0), abs(m_inner1 - m_prev1));
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
            let local_assoc_write_budget = write_budget0 + write_budget1;
            shared_vals[tid] = local_assoc_write_budget;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            assoc_write_gate_token = ASSOC_WRITE_CAP * raw_z;
            if (tid == 0u) {
                max_m_delta_seen = max(max_m_delta_seen, iter_max_delta_m_p);
                max_err_m_seen = max(max_err_m_seen, err_m_p);
                max_z_seen = max(max_z_seen, z);
                max_update_ratio_seen = max(max_update_ratio_seen, update_ratio_p);
                sum_slot_move_seen = sum_slot_move_seen + iter_max_delta_m_p;
                if (z > FPM_SAT_THRESHOLD) {
                    write_saturation_seen = write_saturation_seen + 1.0;
                }
                let slot_probe_base = 520u + slot_idx * 14u;
                DebugLog[slot_probe_base + 2u] = max(DebugLog[slot_probe_base + 2u], retain_max_p);
                DebugLog[slot_probe_base + 3u] = DebugLog[slot_probe_base + 3u] + retain_avg_p;
                DebugLog[slot_probe_base + 4u] = max(DebugLog[slot_probe_base + 4u], proposal_rms_p);
                DebugLog[slot_probe_base + 5u] = max(DebugLog[slot_probe_base + 5u], candidate_rms_p);
            }
            fpm_m_cache[d0] = m_inner0;
            fpm_m_cache[d1] = m_inner1;
            workgroupBarrier();
        }
        if (debug_on && slot_idx == 0u && tid == 0u && t == 0u) {
            DebugLog[8] = 184.0;
        }
        workgroupBarrier();

        // Model A token-local history: once write is enabled, each token materializes its
        // internal FPM state into HistCtx so the next token in the same chunk can read it.
        // Inter-chunk persistence remains a separate stage handled by the bridge via MState.
        if (d_model == WG_SIZE * 2u) {
            let working0 = fpm_m_cache[tid];
            let working1 = fpm_m_cache[tid + WG_SIZE];
            // Per-token storage for the causal history snapshot and retain-gate backward.
            // The bridge Rust-side copies HistCtx[last_token] → MState after each chunk (stage>=4).
            HistCtx[h_base_t + slot_offset + tid] = working0;
            HistCtx[h_base_t + slot_offset + tid + WG_SIZE] = working1;
        }
        if (assoc_write_enabled && !is_segment_memory_slot && d_model == WG_SIZE * 2u) {
            let d0 = tid;
            let d1 = tid + WG_SIZE;
            let hist_base = aw_hist_base(d_model, h_slots);
            // Associative write keys use their own address encoder. Sharing with the
            // FPM write-key made a single unstable associative run corrupt the core
            // FPM write geometry and bifurcate training.
            let wk_assoc = hist_base + w_k_assoc_base(d_model, h_slots) + slot_idx * d_model * ASSOC_RANK;
            // W_v_assoc is reserved for the transition-gate auxiliary branch. The default
            // bank-value path stores raw token identity directly into bank_value so the
            // explicit associative memory keeps token semantics without a hidden projector.
            let wv_assoc = hist_base + w_v_assoc_base(d_model, h_slots) + slot_idx * d_model * ASSOC_RANK;
            let wevent_assoc = hist_base + w_event_assoc_base(d_model, h_slots) + slot_offset;
            let bevent_assoc = hist_base + b_event_assoc_base(d_model, h_slots) + slot_idx;
            let assoc_anchor_base = hist_base + slot_anchor_base(d_model, h_slots) + slot_offset;
            // Layout per bank: [bank_key | bank_value | usage].
            let assoc_hist_base =
                ((batch_idx * shape.token_count + t) * h_slots + slot_idx) * assoc_hist_slot_stride;
            for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                let bank_base = assoc_slot_base + bank * assoc_bank_stride;
                let hist_bank_base = assoc_hist_base + bank * assoc_bank_stride;
                if (tid < ASSOC_RANK) {
                    AssocHist[hist_bank_base + tid] = AssocBuf[bank_base + tid];
                }
                AssocHist[hist_bank_base + ASSOC_RANK + d0] =
                    AssocBuf[bank_base + ASSOC_RANK + d0];
                AssocHist[hist_bank_base + ASSOC_RANK + d1] =
                    AssocBuf[bank_base + ASSOC_RANK + d1];
                if (tid == 0u) {
                    AssocHist[hist_bank_base + ASSOC_RANK + d_model] =
                        AssocBuf[bank_base + ASSOC_RANK + d_model];
                }
            }
            workgroupBarrier();
            let assoc_raw0 =
                Scratch[signal_base + d0]
                + select(0.0, AllWeights[assoc_anchor_base + d0], ENABLE_ASSOC_SLOT_ANCHOR);
            let assoc_raw1 =
                Scratch[signal_base + d1]
                + select(0.0, AllWeights[assoc_anchor_base + d1], ENABLE_ASSOC_SLOT_ANCHOR);
            shared_vals[tid] = assoc_raw0 * assoc_raw0 + assoc_raw1 * assoc_raw1;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let assoc_signal_rms = sqrt(shared_vals[0] * inv_d_model + 1.0e-6);
            let assoc_src0 = assoc_raw0 / max(assoc_signal_rms, 1.0e-6);
            let assoc_src1 = assoc_raw1 / max(assoc_signal_rms, 1.0e-6);
            let has_prev_hstar = select(0.0, 1.0, t > 0u);
            var event_prev0 = 0.0;
            var event_prev1 = 0.0;
            var event_curr0 = 0.0;
            var event_curr1 = 0.0;
            if (t > 0u) {
                let prev_s_in_base = (batch_idx * shape.seq_len + (global_t - 1u)) * d_model;
                event_prev0 =
                    S_in[prev_s_in_base + d0]
                    + select(0.0, AllWeights[assoc_anchor_base + d0], ENABLE_ASSOC_SLOT_ANCHOR);
                event_prev1 =
                    S_in[prev_s_in_base + d1]
                    + select(0.0, AllWeights[assoc_anchor_base + d1], ENABLE_ASSOC_SLOT_ANCHOR);
            }
            event_curr0 =
                S_in[s_in_base + d0]
                + select(0.0, AllWeights[assoc_anchor_base + d0], ENABLE_ASSOC_SLOT_ANCHOR);
            event_curr1 =
                S_in[s_in_base + d1]
                + select(0.0, AllWeights[assoc_anchor_base + d1], ENABLE_ASSOC_SLOT_ANCHOR);
            shared_vals[tid] = event_prev0 * event_prev0 + event_prev1 * event_prev1;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let event_prev_rms = sqrt(shared_vals[0] * inv_d_model + 1.0e-6);
            shared_vals[tid] = event_curr0 * event_curr0 + event_curr1 * event_curr1;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let event_curr_rms = sqrt(shared_vals[0] * inv_d_model + 1.0e-6);
            // Event gate is a learned transition classifier over prev_source → curr_source.
            // The gate must react to change, not just signal magnitude; otherwise it becomes
            // an almost-constant write valve that opens for nearly every token pair.
            shared_vals[tid] =
                AllWeights[wevent_assoc + d0]
                    * (
                        (
                            event_curr0 / max(event_curr_rms, 1.0e-6)
                            - event_prev0 / max(event_prev_rms, 1.0e-6)
                        )
                        + (
                            event_curr0 / max(event_curr_rms, 1.0e-6)
                            * event_prev0 / max(event_prev_rms, 1.0e-6)
                        )
                    )
                + AllWeights[wevent_assoc + d1]
                    * (
                        (
                            event_curr1 / max(event_curr_rms, 1.0e-6)
                            - event_prev1 / max(event_prev_rms, 1.0e-6)
                        )
                        + (
                            event_curr1 / max(event_curr_rms, 1.0e-6)
                            * event_prev1 / max(event_prev_rms, 1.0e-6)
                        )
                    );
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let learned_event_gate = 1.0 / (
                1.0 + exp(-(shared_vals[0] * inv_sqrt_d_model + AllWeights[bevent_assoc]))
            );
            let event_gate = select(1.0, learned_event_gate, ENABLE_ASSOC_EVENT_GATE);
            // Structural split:
            // - the explicit associative bank writes when the transition classifier says
            //   "this pair is a binding worth storing";
            // - the FPM write budget only controls how much of that selected binding is
            //   consolidated into the slower recurrent state.
            // Using the FPM gate for both roles kept the system stable but starved binding.
            let bind_gate = event_gate * has_prev_hstar;
            // Durable binding update:
            //   bank_key   <- (1-g) bank_key   + g * W_k(source_{t-1})
            //   bank_value <- (1-g) bank_value + g * token_identity_t
            // Keys stay in the explicit associative source space so token identity remains
            // sharp. Values are written as the normalized raw current token identity: for
            // associative recall the thing we want back is the observed VAL token itself,
            // not a representation already mixed by the solve.
            var assoc_value0 = 0.0;
            var assoc_value1 = 0.0;
            if (tid < ASSOC_RANK) {
                var prev_sig_sq = 0.0;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let prev_sig_j = PrevHStarBuf[prev_hstar_base + j];
                    prev_sig_sq = prev_sig_sq + prev_sig_j * prev_sig_j;
                }
                let prev_sig_rms = sqrt(prev_sig_sq / max(1.0, f32(d_model)) + 1.0e-6);
                var k_acc = 0.0;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let prev_sig_j = PrevHStarBuf[prev_hstar_base + j] / max(prev_sig_rms, 1.0e-6); // associative source at t-1
                    k_acc = k_acc + AllWeights[wk_assoc + j * ASSOC_RANK + tid] * prev_sig_j;
                }
                // Reuse attention-local workgroup scratch in the post-solve assoc write path.
                // At this point of the token loop q_self/head_mix are dead, so they can safely
                // preserve the key/value code until allocator + bank write finish.
                q_self[tid] = tanh(k_acc);
                head_mix[tid] = 0.0;
                if (ENABLE_ASSOC_TRANSITION_GATE) {
                    var v_acc = 0.0;
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        let curr_token_j = S_in[s_in_base + j] / max(event_curr_rms, 1.0e-6);
                        v_acc = v_acc + AllWeights[wv_assoc + j * ASSOC_RANK + tid] * curr_token_j;
                    }
                    head_mix[tid] = tanh(
                        v_acc * inverseSqrt(max(1.0, f32(ASSOC_RANK)))
                    );
                }
            }
            workgroupBarrier();
            assoc_value0 = S_in[s_in_base + d0] / max(event_curr_rms, 1.0e-6);
            assoc_value1 = S_in[s_in_base + d1] / max(event_curr_rms, 1.0e-6);
            Scratch[signal_base + d0] = assoc_value0;
            Scratch[signal_base + d1] = assoc_value1;
            shared_vals[tid] = assoc_value0 * assoc_value0 + assoc_value1 * assoc_value1;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let assoc_value_rms = sqrt(shared_vals[0] * inv_d_model + 1.0e-6);
            assoc_value0 = Scratch[signal_base + d0] / max(assoc_value_rms, 1.0e-6);
            assoc_value1 = Scratch[signal_base + d1] / max(assoc_value_rms, 1.0e-6);
            if (tid == 0u && debug_on) {
                var key_sq = 0.0;
                var prev_sq = 0.0;
                var kpre_sq = 0.0;
                for (var r = 0u; r < ASSOC_RANK; r = r + 1u) {
                    let kv = q_self[r];
                    key_sq = key_sq + kv * kv;
                    let kpre = atanh(clamp(kv, -0.999999, 0.999999));
                    kpre_sq = kpre_sq + kpre * kpre;
                }
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let prev_j = PrevHStarBuf[prev_hstar_base + j];
                    prev_sq = prev_sq + prev_j * prev_j;
                }
                let assoc_write_probe_base = 900u + slot_idx * 6u;
                DebugLog[assoc_write_probe_base + 0u] =
                    DebugLog[assoc_write_probe_base + 0u] + sqrt(key_sq / max(1.0, f32(ASSOC_RANK)));
                DebugLog[assoc_write_probe_base + 1u] =
                    DebugLog[assoc_write_probe_base + 1u] + assoc_value_rms;
                DebugLog[assoc_write_probe_base + 2u] = DebugLog[assoc_write_probe_base + 2u] + 1.0;
                DebugLog[assoc_write_probe_base + 3u] =
                    DebugLog[assoc_write_probe_base + 3u] + sqrt(prev_sq / max(1.0, f32(d_model)));
                DebugLog[assoc_write_probe_base + 4u] =
                    DebugLog[assoc_write_probe_base + 4u] + sqrt(kpre_sq / max(1.0, f32(ASSOC_RANK)));
                DebugLog[assoc_write_probe_base + 5u] = DebugLog[assoc_write_probe_base + 5u];
            }
            if (tid == 0u) {
                var chosen_bank = 0u;
                var found_empty = false;
                var empty_bank = 0u;
                var best_cos = -1.0e30;
                var best_value_cos = -1.0e30;
                var best_bank = 0u;
                var min_usage = 1.0e30;
                var min_usage_bank = 0u;
                var allow_write = 0.0;
                var overwrite_bank = 0.0;
                var new_key_norm = 0.0;
                var new_value_norm = 0.0;
                for (var r = 0u; r < ASSOC_RANK; r = r + 1u) {
                    new_key_norm = new_key_norm + q_self[r] * q_self[r];
                }
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let v_j = Scratch[signal_base + j] / max(assoc_value_rms, 1.0e-6);
                    new_value_norm = new_value_norm + v_j * v_j;
                }
                var bank_scores: array<f32, 8>; 
                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                    let bank_base = assoc_slot_base + bank * assoc_bank_stride;
                    var key_norm = 0.0;
                    var score = 0.0;
                    var value_norm = 0.0;
                    var value_score = 0.0;
                    for (var r = 0u; r < ASSOC_RANK; r = r + 1u) {
                        let key_r = AssocBuf[bank_base + r];
                        key_norm = key_norm + key_r * key_r;
                        score = score + key_r * q_self[r];
                    }
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        let bank_v = AssocBuf[bank_base + ASSOC_RANK + j];
                        let curr_v = Scratch[signal_base + j] / max(assoc_value_rms, 1.0e-6);
                        value_norm = value_norm + bank_v * bank_v;
                        value_score = value_score + bank_v * curr_v;
                    }
                    let bank_usage = AssocBuf[bank_base + ASSOC_RANK + d_model];
                    if (bank_usage < min_usage) {
                        min_usage = bank_usage;
                        min_usage_bank = bank;
                    }
                    if (!found_empty && bank_usage < ASSOC_OCCUPIED_THRESHOLD) {
                        empty_bank = bank;
                        found_empty = true;
                    }
                    let cos = score / sqrt(max(key_norm * new_key_norm, 1.0e-12));
                    if (bank < 8u) { bank_scores[bank] = cos; }
                    let value_cos =
                        value_score / sqrt(max(value_norm * new_value_norm, 1.0e-12));
                    if (cos > best_cos) {
                        best_cos = cos;
                        best_value_cos = value_cos;
                        best_bank = bank;
                    }
                }

                // ── Librarian Stage 2: Competitive Slot Allocation ─────────────────────
                // All slots compete for the current token using their anchor geometry.
                // Every slot recomputes the global allocation to determine its own share.
                let slot_anchor_root_write = hist_base + slot_anchor_base(d_model, h_slots);
                var prev_sig_sq_owner = 0.0;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let prev_j = PrevHStarBuf[prev_hstar_base + j];
                    prev_sig_sq_owner = prev_sig_sq_owner + prev_j * prev_j;
                }
                let prev_sig_rms_owner = sqrt(prev_sig_sq_owner / max(1.0, f32(d_model)) + 1.0e-6);
                
                var max_slot_score = -1.0e30;
                for (var owner = 0u; owner < h_slots; owner = owner + 1u) {
                    let owner_off = slot_anchor_root_write + owner * d_model;
                    var anchor_sq = 0.0;
                    var dot = 0.0;
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        let prev_j = PrevHStarBuf[prev_hstar_base + j] / max(prev_sig_rms_owner, 1.0e-6);
                        let anchor_j = AllWeights[owner_off + j];
                        anchor_sq = anchor_sq + anchor_j * anchor_j;
                        dot = dot + prev_j * anchor_j;
                    }
                    let owner_score = dot / sqrt(max(anchor_sq, 1.0e-12));
                    shared_vals[owner] = owner_score; // Temporary store in shared_vals
                    max_slot_score = max(max_slot_score, owner_score);
                }
                var slot_denom = 0.0;
                for (var owner = 0u; owner < h_slots; owner = owner + 1u) {
                    let e = exp(4.0 * (shared_vals[owner] - max_slot_score)); // Beta=4.0 for slot selection
                    shared_vals[owner] = e;
                    slot_denom = slot_denom + e;
                }
                let p_slot_i = shared_vals[slot_idx] / max(slot_denom, 1.0e-6);

                // ── Librarian Stage 3: Competitive Bank Placement ────────────────────
                // Within the slot, banks compete for the token using cosine similarity.
                var max_bank_score = -1.0e30;
                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                    max_bank_score = max(max_bank_score, bank_scores[bank]);
                }
                var bank_denom = 0.0;
                var bank_probs: array<f32, 8>;
                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                    let e = exp(4.0 * (bank_scores[bank] - max_bank_score)); // Beta=4.0 for bank selection
                    bank_probs[bank] = e;
                    bank_denom = bank_denom + e;
                }
                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                    bank_probs[bank] = bank_probs[bank] / max(bank_denom, 1.0e-6);
                }
                
                // Hierarchical Decision: Total Write Mass = Relevance * SlotProb * BankProb
                chosen_bank = best_bank;
                allow_write = event_gate * p_slot_i * bank_probs[best_bank];
                if (best_cos < ASSOC_ALLOC_NOVELTY_THRESHOLD && found_empty) {
                    chosen_bank = empty_bank;
                    // If novel, we use the allocated slot mass but equally distributed across banks.
                    allow_write = event_gate * p_slot_i; 
                    overwrite_bank = 1.0;
                }
                shared_vals[3u * ASSOC_RANK] = f32(chosen_bank);
                shared_vals[3u * ASSOC_RANK + 1u] = allow_write;
                shared_vals[3u * ASSOC_RANK + 2u] = best_cos;
                shared_vals[3u * ASSOC_RANK + 3u] = min_usage;
                var transition_gate = 1.0;
                if (ENABLE_ASSOC_TRANSITION_GATE) {
                    var transition_score = 0.0;
                    for (var r = 0u; r < ASSOC_RANK; r = r + 1u) {
                        transition_score = transition_score + q_self[r] * head_mix[r];
                    }
                    transition_gate =
                        1.0 / (1.0 + exp(-transition_score * inverseSqrt(max(1.0, f32(ASSOC_RANK)))));
                }
                shared_vals[3u * ASSOC_RANK + 4u] = transition_gate;
                let meta_base = assoc_hist_base + assoc_slot_stride;
                AssocHist[meta_base + 0u] = f32(chosen_bank);
                AssocHist[meta_base + 1u] = allow_write;
                AssocHist[meta_base + 2u] = transition_gate;
                AssocHist[meta_base + 3u] = overwrite_bank;
            }
            workgroupBarrier();
            let chosen_bank = u32(shared_vals[3u * ASSOC_RANK]);
            let assoc_write_allowed = shared_vals[3u * ASSOC_RANK + 1u];
            let assoc_transition_gate = shared_vals[3u * ASSOC_RANK + 4u];
            let chosen_bank_base = assoc_slot_base + chosen_bank * assoc_bank_stride;
            let bank_key_base = chosen_bank_base;
            let bank_value_base = chosen_bank_base + ASSOC_RANK;
            let bank_usage_idx = chosen_bank_base + ASSOC_RANK + d_model;
            let effective_bind_gate = bind_gate * assoc_write_allowed * assoc_transition_gate;
            let effective_write_mass = select(
                effective_bind_gate * effective_bind_gate,
                effective_bind_gate,
                ENABLE_ASSOC_LINEAR_WRITE,
            );
            let overwrite_bank = AssocHist[assoc_hist_base + assoc_slot_stride + 3u];
            let key_write_mass = effective_write_mass;
            let key_keep_gate = 1.0 - effective_write_mass;
            let value_write_mass = select(0.0, effective_write_mass, overwrite_bank > 0.5);
            let value_keep_gate = select(1.0, 1.0 - value_write_mass, overwrite_bank > 0.5);
            if (tid == 0u) {
                // TEMPORARY ASSOCIATIVE DIAGNOSTIC: write allocation telemetry.
                // We use a clean snapshot of the last token in the first batch to avoid race-riddled averages.
                if (debug_on && batch_idx == 0u && t == shape.token_count - 1u) {
                    let assoc_diag_base = 760u + slot_idx * 10u;
                    DebugLog[assoc_diag_base + 0u] = 0.0; // r_ent (unused)
                    DebugLog[assoc_diag_base + 1u] = 1.0; // r_max (unused)
                    DebugLog[assoc_diag_base + 2u] = 1.0; // ctx_rms (unused)
                    DebugLog[assoc_diag_base + 3u] = min_usage;
                    DebugLog[assoc_diag_base + 4u] = assoc_write_allowed;
                    DebugLog[assoc_diag_base + 5u] = effective_bind_gate;
                    DebugLog[assoc_diag_base + 6u] = best_cos;
                    DebugLog[assoc_diag_base + 7u] = p_slot_i;
                    DebugLog[assoc_diag_base + 8u] = 1.0; // r_n marker
                    DebugLog[assoc_diag_base + 9u] = 1.0; // w_n marker
                    
                    let assoc_write_probe_base = 900u + slot_idx * 6u;
                    DebugLog[assoc_write_probe_base + 5u] = overwrite_bank;
                }
            }
            if (tid < ASSOC_RANK) {
                AssocBuf[bank_key_base + tid] =
                    key_keep_gate * AssocBuf[bank_key_base + tid] + key_write_mass * q_self[tid];
            }
            workgroupBarrier();
            AssocBuf[bank_value_base + d0] =
                value_keep_gate * AssocBuf[bank_value_base + d0] + value_write_mass * assoc_value0;
            AssocBuf[bank_value_base + d1] =
                value_keep_gate * AssocBuf[bank_value_base + d1] + value_write_mass * assoc_value1;
            if (fpm_write_enabled) {
                // Phase bridge:
                // - AssocBuf keeps the exact token-level binding (bank_value).
                // - FPM should not copy that binding verbatim, or it becomes a second,
                //   blurrier associative memory. Instead it absorbs a slow contextual trace
                //   of the token state that produced the binding. This preserves the hybrid
                //   contract: explicit memory stays precise; FPM stays contextual.
                shared_vals[tid] =
                    H_curr[h_base + slot_offset + d0] * H_curr[h_base + slot_offset + d0]
                    + H_curr[h_base + slot_offset + d1] * H_curr[h_base + slot_offset + d1];
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let fpm_trace_rms = sqrt(shared_vals[0] * inv_d_model + 1.0e-6);
                let h_trace0 = H_curr[h_base + slot_offset + d0] / max(fpm_trace_rms, 1.0e-6);
                let h_trace1 = H_curr[h_base + slot_offset + d1] / max(fpm_trace_rms, 1.0e-6);
                let consolidate =
                    ASSOC_TO_FPM_SCALE * clamp(assoc_write_gate_token, 0.0, ASSOC_WRITE_CAP) * effective_write_mass;
                fpm_m_cache[d0] =
                    (1.0 - consolidate) * fpm_m_cache[d0] + consolidate * h_trace0;
                fpm_m_cache[d1] =
                    (1.0 - consolidate) * fpm_m_cache[d1] + consolidate * h_trace1;
                HistCtx[h_base_t + slot_offset + d0] = fpm_m_cache[d0];
                HistCtx[h_base_t + slot_offset + d1] = fpm_m_cache[d1];
            }
            if (tid == 0u) {
                let prev_usage = AssocBuf[bank_usage_idx];
                // Usage should track durable committed content, not the pre-squared
                // gate. Otherwise weak filler writes mark empty banks as strongly
                // occupied even when the actual bank update mass was tiny.
                let occupied_write = effective_write_mass;
                AssocBuf[bank_usage_idx] = clamp(
                    max(ASSOC_USAGE_DECAY * prev_usage, occupied_write),
                    0.0,
                    1.0,
                );
            }
            workgroupBarrier();
            if (tid < ASSOC_RANK) {
                // keep workgroup ordering aligned before PrevHStar write
            }
            workgroupBarrier();
            // Save token-local associative source for the next query/key.
            // Identity stays explicit; FPM/DEQ coupling enters through write/read control.
            PrevHStarBuf[prev_hstar_base + d0] = assoc_src0;
            PrevHStarBuf[prev_hstar_base + d1] = assoc_src1;
        }
        if (debug_on && slot_idx == 0u && tid == 0u && t == 0u) {
            DebugLog[8] = 185.0;
        }
        workgroupBarrier();
    }
    if (debug_on && slot_idx == 0u && tid == 0u) {
        DebugLog[8] = 201.0;
    }
    if (tid == 0u) {
        max_h_seen = 0.0;
    }
    workgroupBarrier();

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
        let slot_read_base = 520u + slot_idx * 14u;
        DebugLog[slot_read_base + 0u] = max_memctx_rms_seen;
        DebugLog[slot_read_base + 1u] = max_memctx_to_signal_seen;
        DebugLog[slot_read_base + 3u] = DebugLog[slot_read_base + 3u] / token_den;
        // TEMPORARY ASSOCIATIVE DIAGNOSTIC: remove DebugLog[760..] once assoc recall is fixed.
        let assoc_diag_base = 760u + slot_idx * 10u;
        let assoc_read_den = max(1.0, DebugLog[assoc_diag_base + 8u]);
        let assoc_write_den = max(1.0, DebugLog[assoc_diag_base + 9u]);
        DebugLog[assoc_diag_base + 0u] = DebugLog[assoc_diag_base + 0u] / assoc_read_den;
        DebugLog[assoc_diag_base + 3u] = DebugLog[assoc_diag_base + 3u] / assoc_read_den;
        DebugLog[assoc_diag_base + 4u] = DebugLog[assoc_diag_base + 4u] / assoc_write_den;
        DebugLog[assoc_diag_base + 5u] = DebugLog[assoc_diag_base + 5u] / assoc_write_den;
        let assoc_write_probe_base = 900u + slot_idx * 6u;
        let assoc_write_probe_den = max(1.0, DebugLog[assoc_write_probe_base + 2u]);
        DebugLog[assoc_write_probe_base + 0u] =
            DebugLog[assoc_write_probe_base + 0u] / assoc_write_probe_den;
        DebugLog[assoc_write_probe_base + 1u] =
            DebugLog[assoc_write_probe_base + 1u] / assoc_write_probe_den;
        DebugLog[assoc_write_probe_base + 3u] =
            DebugLog[assoc_write_probe_base + 3u] / assoc_write_probe_den;
        DebugLog[assoc_write_probe_base + 4u] =
            DebugLog[assoc_write_probe_base + 4u] / assoc_write_probe_den;
        DebugLog[assoc_write_probe_base + 5u] =
            DebugLog[assoc_write_probe_base + 5u] / assoc_write_probe_den;
        let solve_exit_base = 688u + slot_idx * 4u;
        DebugLog[solve_exit_base + 0u] = strict_converged_sum;
        DebugLog[solve_exit_base + 1u] = homeostatic_converged_sum;
        DebugLog[solve_exit_base + 2u] = failed_converged_sum;
        DebugLog[solve_exit_base + 3u] = max_homeo_band_seen;
        if (debug_on && slot_idx == 0u) {
            DebugLog[8] = 901.0;
            DebugLog[9] = shape.epsilon;
            DebugLog[10] = f32(shape.token_count);
            DebugLog[11] = f32(h_slots);
            DebugLog[12] = solve_signal_rms_seen;
            DebugLog[13] = solve_pre_rms_seen;
            DebugLog[14] = solve_fh_rms_seen;
            DebugLog[15] = solve_hprev_rms_seen;
            DebugLog[16] = solve_nscale_abs_seen;
            DebugLog[17] = solve_pre_to_hprev_seen;
            DebugLog[18] = solve_fh_to_hprev_seen;
            DebugLog[19] = solve_attn_rms_seen;
            DebugLog[20] = solve_attn_to_signal_seen;
            DebugLog[21] = solve_attn_scale_seen;
            DebugLog[22] = iter0_err_h_seen;
            DebugLog[23] = iter1_err_h_seen;
            DebugLog[24] = iter0_attn_to_signal_seen;
            DebugLog[25] = iter1_attn_to_signal_seen;
            DebugLog[26] = iter0_attn_scale_seen;
            DebugLog[27] = iter1_attn_scale_seen;
            DebugLog[28] = iter_of_max_err_h_seen;
            DebugLog[29] = iter_of_max_attn_ratio_seen;
            DebugLog[30] = token_of_max_err_h_seen;
            DebugLog[31] = token_of_max_attn_ratio_seen;
        }
    }
}
