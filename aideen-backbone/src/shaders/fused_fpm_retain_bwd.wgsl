// FPM memory backward pass for the active Model-A write path.
//
// Forward (per token t, slot s):
//   c         = 0.5 * (h_star[t,s] + signal[t,s])
//   raw_z     = sigmoid(dot(W_write_gate[s], c) / sqrt(d) + b_write_mem[s] + gate_bias)
//   proposal  = tanh(W_delta[s] * c + b_delta)
//   novelty   = ||proposal - m_prev|| / (||proposal - m_prev|| + ||m_prev|| + eps)
//   z         = raw_z
//   retain    = 1 - novelty * (1 - sigmoid(W_down[s] * (W_up[s] * c) + b_retain[s]))
//   write     = z * residual_scale * proposal
//   h_unit    = h_star / rms(h_star)
//   x_proj    = sqrt(z) * h_unit + wx_diag * h_unit + write
//   a         = sigmoid(-a_log[s])
//   base      = a * m_prev + (1 - a) * x_proj
//   m_state   = retain * m_prev + (1 - retain) * base
//
// The recurrent state stored in HistCtx / fpm_m_buf is m_state. W_out is not part
// of the state recurrence; treating readout-transformed memory as the next state
// made FPM unstable when memory coupling was strengthened.
//
// This kernel updates the slot-specific write parameters that participate in the
// post-solve FPM memory recurrence:
//   - W_write_gate / b_write_mem
//   - W_delta / b_delta
//   - W_retain_up / W_retain_down / b_retain
//   - a_log
//
// Shared carrier matrix W_x is part of the new forward semantics, but
// it is shared across slots; this kernel keeps it fixed and only handles the
// slot-local parameters without introducing cross-workgroup write races.
//
// TBPTT carry semantics remain the same: tbptt_carry_buf seeds ∂L/∂M_t from the
// next chunk and receives ∂L/∂M_0 after the backward token sweep.

struct FpmBwdUniforms {
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

@group(0) @binding(0) var<uniform>             params: FpmBwdUniforms;
@group(0) @binding(1) var<storage, read>       h_star: array<f32>;   // [batch*seq*h*d]
@group(0) @binding(2) var<storage, read>       Scratch: array<f32>;  // clean layout: [signal | attn | alpha | slot_scores] per token
@group(0) @binding(3) var<storage, read>       fpm_m_buf: array<f32>; // [batch*seq*h*d]
@group(0) @binding(4) var<storage, read_write> AllGradients: array<f32>;
@group(0) @binding(5) var<storage, read_write> tbptt_carry_buf: array<f32>;
@group(0) @binding(6) var<storage, read_write> fpm_dm_buf: array<f32>;
@group(0) @binding(7) var<storage, read_write> fpm_minner_buf: array<f32>;
@group(0) @binding(8) var<storage, read_write> fpm_hunit_buf: array<f32>;
@group(0) @binding(9) var<storage, read_write> fpm_gx_buf: array<f32>;
// binding 10: adjoint v_state = ∂L/∂h* per (slot, token, dim).
// Layout: [slot * n_tokens + t_abs, dim] = slot-major, same as v_next_entry_base in adjoint.
// Used to compute the READ-path gradient: fpm_ctx_t reads M[t-1], so
// ∂L/∂M[t-1] += v_state[slot,t] * (∂fpm_ctx_t/∂M[t-1]) ≈ v_state[slot,t] * FPM_READ_GRAD_SCALE
@group(0) @binding(10) var<storage, read> v_state: array<f32>;
@group(0) @binding(11) var<storage, read> AssocState: array<f32>;
@group(0) @binding(12) var<storage, read> AssocHist: array<f32>;
// TEMPORARY ASSOCIATIVE DIAGNOSTIC: remove after backward learning path is localized.
@group(0) @binding(13) var<storage, read_write> AssocBwdDebug: array<f32>;

@group(1) @binding(0) var<storage, read_write> AllWeights: array<f32>;

override ENABLE_ASSOC_TRANSITION_GATE: bool = false;
override ENABLE_ASSOC_SLOT_ANCHOR: bool = false;
override ENABLE_ASSOC_REUSE_MATCH: bool = false;
override ENABLE_ASSOC_SLOT_STRIPE: bool = false;
override ENABLE_ASSOC_TIE_QK: bool = false;
override ENABLE_ASSOC_CONF_READ: bool = false;
override ENABLE_ASSOC_EVENT_GATE: bool = false;
override ASSOC_EVENT_L1: f32 = 0.0;

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

fn hist_gate_base(d: u32, h: u32) -> u32 { return d * d + 2u * h * d; }
fn slot_anchor_base(d: u32, h: u32) -> u32 { return hist_gate_base(d, h) + h; }
fn w_k_write_base(d: u32, h: u32) -> u32 { return slot_anchor_base(d, h) + h * d; }
fn w_v_write_base(d: u32, h: u32) -> u32 { return w_k_write_base(d, h) + h * d * RETAIN_RANK; }
fn hist_delta_bias_base(d: u32, h: u32) -> u32 { return w_v_write_base(d, h) + h * RETAIN_RANK * d; }
fn hist_gate_query_base(d: u32, h: u32) -> u32 { return hist_delta_bias_base(d, h) + d + 21u; }
fn w_write_gate_base(d: u32, h: u32) -> u32 { return hist_gate_query_base(d, h) + h * d; }
fn b_write_mem_base(d: u32, h: u32) -> u32 { return w_write_gate_base(d, h) + h * d; }
fn hhist_gamma_base(d: u32, h: u32) -> u32 { return b_write_mem_base(d, h) + h; }

const RETAIN_RANK: u32 = 32u;
fn w_retain_up_base(d: u32, h: u32) -> u32 { return hhist_gamma_base(d, h) + h; }
fn w_retain_down_base(d: u32, h: u32) -> u32 { return w_retain_up_base(d, h) + h * d * RETAIN_RANK; }
fn b_retain_base(d: u32, h: u32) -> u32 { return w_retain_down_base(d, h) + h * RETAIN_RANK * d; }
fn w_q_mem_base(d: u32, h: u32) -> u32 { return b_retain_base(d, h) + h * d; }
fn w_k_mem_base(d: u32, h: u32) -> u32 { return w_q_mem_base(d, h) + h * d * RETAIN_RANK; }
fn b_read_mem_base(d: u32, h: u32) -> u32 { return w_k_mem_base(d, h) + h * d * RETAIN_RANK; }
fn w_k_assoc_base(d: u32, h: u32) -> u32 { return b_read_mem_base(d, h) + h; }
fn w_v_assoc_base(d: u32, h: u32) -> u32 { return w_k_assoc_base(d, h) + h * d * RETAIN_RANK; }
fn w_q_assoc_base(d: u32, h: u32) -> u32 { return w_v_assoc_base(d, h) + h * d * RETAIN_RANK; }
fn alpha_assoc_base(d: u32, h: u32) -> u32 { return w_q_assoc_base(d, h) + h * d * RETAIN_RANK; }
fn w_event_assoc_base(d: u32, h: u32) -> u32 { return alpha_assoc_base(d, h) + h; }
fn b_event_assoc_base(d: u32, h: u32) -> u32 { return w_event_assoc_base(d, h) + h * d; }

fn clean_signal_off(t_abs: u32, slot: u32, d: u32, h: u32) -> u32 {
    let signal_span = h * d;
    let scratch_stride = signal_span * 3u + h * h;
    return t_abs * scratch_stride + slot * d;
}
fn hstar_off(t_abs: u32, slot: u32, d: u32, h: u32) -> u32 {
    return t_abs * h * d + slot * d;
}
fn fpm_m_off(t_abs: u32, slot: u32, d: u32, h: u32) -> u32 {
    return t_abs * h * d + slot * d;
}

const WG_SIZE: u32 = 64u;
const FPM_RESIDUAL_SCALE: f32 = 0.1;
const FPM_GATE_BIAS: f32 = -1.5;
const ASSOC_WRITE_CAP: f32 = 0.95;
override ASSOC_BANKS: u32 = 1u;
const ASSOC_ALLOC_THRESHOLD: f32 = 0.05;
// Experimental multi-bank addressing strength. With ASSOC_BANKS=1 this is
// behaviorally inactive because the read softmax has a single candidate.
const ASSOC_READ_BETA: f32 = 4.0;
// Scale for the READ-path gradient contribution (matches fpm_alpha_m default = 0.01).
// Approximates ∂fpm_ctx_t/∂M[t-1] ≈ alpha_m (ignores tanh curvature and gate).
const FPM_READ_GRAD_SCALE: f32 = 0.5;
// Amplification for direct write-matrix update from READ gradient.
// The standard write-backward path has ≈0.002 attenuation at init (retain≈0.88, z≈0.18, RESIDUAL_SCALE=0.1),
// making the READ→W_k_write/W_v_write gradient ~10⁻¹⁰ per step — essentially zero.
// This bypasses that attenuation and applies dm_new directly to write matrices.
// Calibrated so that after ~1000 sequences, W_k_write moves ~5% of init std.
const FPM_DIRECT_WRITE_SCALE: f32 = 200.0;
// Associative addressing matrices receive per-weight gradients diluted by
// d_model × rank while scalar alpha receives a reduced sum. This fan-in
// compensation is temporary until these parameters use the same normalized
// optimizer path as the rest of the model.
const ASSOC_ADDR_GRAD_SCALE: f32 = 4096.0;

var<workgroup> shared_up: array<f32, 32>;   // retain-gate up activations (W_retain_up · c)
var<workgroup> shared_kw: array<f32, 32>;   // k-write bottleneck (W_k_write · c)
var<workgroup> shared_dup: array<f32, 32>;  // ∂L/∂up or ∂L/∂bottleneck (reused)
var<workgroup> assoc_b_state: array<f32, 4096>;
var<workgroup> assoc_gb: array<f32, 4096>;
var<workgroup> assoc_q: array<f32, 32>;
var<workgroup> assoc_raw: array<f32, 32>;
var<workgroup> assoc_gprev: array<f32, 32>;
var<workgroup> shared_vec: array<f32, 512>;
var<workgroup> shared_red: array<f32, 64>;

fn reduce_sum(lane: u32, v: f32) -> f32 {
    shared_red[lane] = v;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            shared_red[lane] = shared_red[lane] + shared_red[lane + stride];
        }
        workgroupBarrier();
    }
    return shared_red[0];
}

@compute
@workgroup_size(64, 1, 1)
fn fused_fpm_retain_bwd_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let lane = lid.x;
    let slot = wid.x;
    let d = params.d_model;
    let h = params.h_slots;
    if (slot >= h) { return; }

    let dims_per_lane = d / WG_SIZE;
    let n_tokens = params.batch_size * params.seq_len;
    let n_norm = max(1.0, f32(n_tokens));
    let clip = 0.5;
    let lr_scaled = params.lr * params.grad_scale;
    let inv_sqrt_d = inverseSqrt(max(1.0, f32(d)));

    let hist_base = aw_hist_base(d, h);
    let wkw_base = hist_base + w_k_write_base(d, h) + slot * d * RETAIN_RANK;
    let wvw_base = hist_base + w_v_write_base(d, h) + slot * RETAIN_RANK * d;
    let bd_base  = hist_base + hist_delta_bias_base(d, h);
    let wf_base = hist_base + w_write_gate_base(d, h) + slot * d;
    let bwrite_idx = hist_base + b_write_mem_base(d, h) + slot;
    let alog_base = aw_alog_base(d, h) + slot * d;
    let wx_base = aw_wx_base(d, h);
    let slot_anchor_root = hist_base + slot_anchor_base(d, h) + slot * d;
    let wup_base = hist_base + w_retain_up_base(d, h) + slot * d * RETAIN_RANK;
    let wdown_base = hist_base + w_retain_down_base(d, h) + slot * RETAIN_RANK * d;
    let bret_base = hist_base + b_retain_base(d, h) + slot * d;
    // Forward uses the FPM write-key encoder as the coupled associative address
    // encoder. Backward must update the same weights; otherwise the read/write
    // path trains dead W_k_assoc/W_q_assoc matrices while inference uses W_k_write.
    let wk_assoc_base = wkw_base;
    let wv_assoc_base = hist_base + w_v_assoc_base(d, h) + slot * d * RETAIN_RANK;
    let wq_assoc_base = wkw_base;
    let alpha_assoc_idx = hist_base + alpha_assoc_base(d, h) + slot;
    let wevent_assoc_base = hist_base + w_event_assoc_base(d, h) + slot * d;
    let bevent_assoc_idx = hist_base + b_event_assoc_base(d, h) + slot;
    let assoc_bank_stride = RETAIN_RANK + d + 1u;
    let assoc_slot_stride = ASSOC_BANKS * assoc_bank_stride;
    let assoc_debug_base = slot * 16u;
    if (lane < 16u) {
        AssocBwdDebug[assoc_debug_base + lane] = 0.0;
    }
    workgroupBarrier();

    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        let assoc_slot_base = (b * h + slot) * assoc_slot_stride;
        // Load the final durable bank state: ASSOC_BANKS × [bank_key[R] | bank_value[d_model] | usage].
        for (var idx = lane; idx < assoc_slot_stride; idx = idx + WG_SIZE) {
            assoc_b_state[idx] = AssocState[assoc_slot_base + idx];
            assoc_gb[idx] = 0.0;
        }
        workgroupBarrier();
        let carry_base = (b * h + slot) * d;
        var dm_new: array<f32, 8>;
        for (var k = 0u; k < 8u; k = k + 1u) { dm_new[k] = 0.0; }
        for (var k = 0u; k < dims_per_lane; k = k + 1u) {
            dm_new[k] = tbptt_carry_buf[carry_base + lane * dims_per_lane + k];
        }

        for (var t_rev = 0u; t_rev < params.seq_len; t_rev = t_rev + 1u) {
            let t = params.seq_len - 1u - t_rev;
            let t_abs = b * params.seq_len + t;
            let sig_base = clean_signal_off(t_abs, slot, d, h);
            let h_base = hstar_off(t_abs, slot, d, h);

            var c: array<f32, 8>;
            var proposal: array<f32, 8>;
            var proposal_pre: array<f32, 8>;
            var retain: array<f32, 8>;
            var m_prev: array<f32, 8>;
            var a: array<f32, 8>;
            var x_proj: array<f32, 8>;
            var h_unit: array<f32, 8>;
            var m_inner: array<f32, 8>;
            var g_pre: array<f32, 8>;
            var g_prev: array<f32, 8>;
            var g_m_inner: array<f32, 8>;
            for (var k = 0u; k < 8u; k = k + 1u) {
                c[k] = 0.0;
                proposal[k] = 0.0;
                proposal_pre[k] = 0.0;
                retain[k] = 0.0;
                m_prev[k] = 0.0;
                a[k] = 0.0;
                x_proj[k] = 0.0;
                h_unit[k] = 0.0;
                m_inner[k] = 0.0;
                g_pre[k] = 0.0;
                g_prev[k] = 0.0;
                g_m_inner[k] = 0.0;
            }

            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                c[k] = 0.5 * (h_star[h_base + dim] + Scratch[sig_base + dim]);
                if (t > 0u) {
                    m_prev[k] = fpm_m_buf[fpm_m_off(t_abs - 1u, slot, d, h) + dim];
                }
            }

            // Recompute the state-side write budget used to modulate associative binding.
            // This scalar depends on FPM state dynamics, not on associative weights.
            var assoc_write_gate_bwd = 1.0;
            var assoc_raw_z_bwd = 1.0;
            if (t > 0u) {
                if (lane < RETAIN_RANK) {
                    shared_up[lane] = 0.0;
                    for (var j = 0u; j < d; j = j + 1u) {
                        let c_j = 0.5 * (h_star[h_base + j] + Scratch[sig_base + j]);
                        shared_up[lane] = shared_up[lane] + AllWeights[wup_base + j * RETAIN_RANK + lane] * c_j;
                    }
                }
                workgroupBarrier();
                var gate_partial_assoc = 0.0;
                for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                    let dim = lane * dims_per_lane + k;
                    gate_partial_assoc = gate_partial_assoc + AllWeights[wf_base + dim] * c[k];
                    var pre_assoc = AllWeights[bret_base + dim];
                    for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                        pre_assoc = pre_assoc + AllWeights[wdown_base + r * d + dim] * shared_up[r];
                    }
                    retain[k] = 1.0 / (1.0 + exp(-pre_assoc));
                }
                if (lane < RETAIN_RANK) {
                    var kw_acc_assoc = 0.0;
                    for (var j = 0u; j < d; j = j + 1u) {
                        let c_j = 0.5 * (h_star[h_base + j] + Scratch[sig_base + j]);
                        kw_acc_assoc = kw_acc_assoc + AllWeights[wkw_base + j * RETAIN_RANK + lane] * c_j;
                    }
                    shared_kw[lane] = kw_acc_assoc;
                }
                workgroupBarrier();
                var local_prop_sq_assoc = 0.0;
                var local_prev_sq_assoc = 0.0;
                var local_write_budget_assoc = 0.0;
                for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                    let dim = lane * dims_per_lane + k;
                    var delta_assoc = AllWeights[bd_base + dim];
                    for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                        delta_assoc = delta_assoc + AllWeights[wvw_base + r * d + dim] * shared_kw[r];
                    }
                    proposal[k] = tanh(delta_assoc);
                    local_prop_sq_assoc = local_prop_sq_assoc + proposal[k] * proposal[k];
                    local_prev_sq_assoc = local_prev_sq_assoc + m_prev[k] * m_prev[k];
                }
                let gate_dot_assoc = reduce_sum(lane, gate_partial_assoc) * inv_sqrt_d
                    + AllWeights[bwrite_idx] + FPM_GATE_BIAS;
                let raw_z_assoc = 1.0 / (1.0 + exp(-gate_dot_assoc));
                assoc_raw_z_bwd = raw_z_assoc;
                let proposal_norm_assoc = sqrt(max(reduce_sum(lane, local_prop_sq_assoc), 1.0e-6));
                let prev_norm_assoc = sqrt(max(reduce_sum(lane, local_prev_sq_assoc), 1.0e-6));
                let novelty_assoc = proposal_norm_assoc / (proposal_norm_assoc + prev_norm_assoc + 1.0e-6);
                let z_assoc = clamp(raw_z_assoc * novelty_assoc, 0.0, 1.0);
                for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                    local_write_budget_assoc = local_write_budget_assoc + (1.0 - retain[k]) * z_assoc;
                }
                assoc_write_gate_bwd = ASSOC_WRITE_CAP * raw_z_assoc;
                workgroupBarrier();
            }

            // ── Durable associative memory backward (FPM-coupled binding) ───────────────────
            // Forward per token:
            //   source_t = norm(signal_t + optional slot_anchor)
            //   q_t = tanh(W_q_assoc * source_{t-1})
            //   k_t = tanh(W_k_assoc * source_{t-1})
            //   v_t = tanh(W_v_assoc * source_t)
            //   ctx_t = Σ_b match(bank_key_b, q_t) * bank_value_b
            //   bank_b* <- EMA(bank_b*, [k_t | v_t])
            //
            // AssocHist stores the exact pre-write bank for each token, so BPTT never
            // reconstructs the EMA inverse. The backward must match the rank-space bank
            // layout used by forward: [key[R] | value[d_model] | usage] per bank.
            let alpha_assoc = AllWeights[alpha_assoc_idx];
            var local_alpha_grad = 0.0;
            let vs_base = (slot * n_tokens + t_abs) * d;
            // TEMPORARY ASSOCIATIVE DIAGNOSTIC: v_state magnitude entering assoc read backward.
            var local_v_sq_assoc_dbg = 0.0;
            for (var k_dbg = 0u; k_dbg < dims_per_lane; k_dbg = k_dbg + 1u) {
                let dim_dbg = lane * dims_per_lane + k_dbg;
                let gv_dbg = v_state[vs_base + dim_dbg];
                local_v_sq_assoc_dbg = local_v_sq_assoc_dbg + gv_dbg * gv_dbg;
            }
            let v_sq_assoc_dbg = reduce_sum(lane, local_v_sq_assoc_dbg);
            if (lane == 0u) {
                AssocBwdDebug[assoc_debug_base + 0u] =
                    AssocBwdDebug[assoc_debug_base + 0u] + sqrt(v_sq_assoc_dbg / max(1.0, f32(d)));
                AssocBwdDebug[assoc_debug_base + 1u] = AssocBwdDebug[assoc_debug_base + 1u] + 1.0;
            }
            workgroupBarrier();
            let has_prev_hstar = select(0.0, 1.0, t > 0u);
            var event_gate_bwd = 0.5;
            var prev_event_rms = 1.0;
            var curr_event_rms = 1.0;
            if (t > 0u) {
                let prev_sig_base_e = clean_signal_off(t_abs - 1u, slot, d, h);
                let curr_sig_base_e = clean_signal_off(t_abs, slot, d, h);
                var local_prev_sq_e = 0.0;
                var local_curr_sq_e = 0.0;
                for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                    let dim = lane * dims_per_lane + k;
                    let prev_j =
                        Scratch[prev_sig_base_e + dim]
                        + select(0.0, AllWeights[slot_anchor_root + dim], ENABLE_ASSOC_SLOT_ANCHOR);
                    let curr_j =
                        Scratch[curr_sig_base_e + dim]
                        + select(0.0, AllWeights[slot_anchor_root + dim], ENABLE_ASSOC_SLOT_ANCHOR);
                    local_prev_sq_e = local_prev_sq_e + prev_j * prev_j;
                    local_curr_sq_e = local_curr_sq_e + curr_j * curr_j;
                }
                prev_event_rms = sqrt(reduce_sum(lane, local_prev_sq_e) / max(1.0, f32(d)) + 1.0e-6);
                curr_event_rms = sqrt(reduce_sum(lane, local_curr_sq_e) / max(1.0, f32(d)) + 1.0e-6);
                var local_event_dot = 0.0;
                for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                    let dim = lane * dims_per_lane + k;
                    let prev_src_e =
                        (Scratch[prev_sig_base_e + dim]
                            + select(0.0, AllWeights[slot_anchor_root + dim], ENABLE_ASSOC_SLOT_ANCHOR))
                        / max(prev_event_rms, 1.0e-6);
                    let curr_src_e =
                        (Scratch[curr_sig_base_e + dim]
                            + select(0.0, AllWeights[slot_anchor_root + dim], ENABLE_ASSOC_SLOT_ANCHOR))
                        / max(curr_event_rms, 1.0e-6);
                    local_event_dot =
                        local_event_dot + AllWeights[wevent_assoc_base + dim] * (prev_src_e + curr_src_e);
                }
                let event_logit =
                    reduce_sum(lane, local_event_dot) * inv_sqrt_d + AllWeights[bevent_assoc_idx];
                let learned_event_gate_bwd = 1.0 / (1.0 + exp(-event_logit));
                event_gate_bwd = select(1.0, learned_event_gate_bwd, ENABLE_ASSOC_EVENT_GATE);
            }
            let bind_gate = clamp(assoc_write_gate_bwd, 0.0, ASSOC_WRITE_CAP) * event_gate_bwd * has_prev_hstar;
            let keep_gate = 1.0 - bind_gate;
            let assoc_hist_slot_base = ((b * params.seq_len + t) * h + slot) * assoc_slot_stride;
            for (var idx = lane; idx < assoc_slot_stride; idx = idx + WG_SIZE) {
                assoc_b_state[idx] = AssocHist[assoc_hist_slot_base + idx];
            }
            workgroupBarrier();

            // Recompute read-query, write-key and write-value exactly as forward did.
            if (lane < RETAIN_RANK) {
                var q_acc = 0.0;
                var k_acc = 0.0;
                var v_acc = 0.0;
                var prev_sig_sq = 0.0;
                var curr_sig_sq_rank = 0.0;
                if (t > 0u) {
                    let prev_sig_base = clean_signal_off(t_abs - 1u, slot, d, h);
                    for (var j = 0u; j < d; j = j + 1u) {
                        let prev_sig_j =
                            Scratch[prev_sig_base + j]
                            + select(0.0, AllWeights[slot_anchor_root + j], ENABLE_ASSOC_SLOT_ANCHOR);
                        prev_sig_sq = prev_sig_sq + prev_sig_j * prev_sig_j;
                    }
                }
                for (var j = 0u; j < d; j = j + 1u) {
                    let curr_sig_j =
                        Scratch[clean_signal_off(t_abs, slot, d, h) + j]
                        + select(0.0, AllWeights[slot_anchor_root + j], ENABLE_ASSOC_SLOT_ANCHOR);
                    curr_sig_sq_rank = curr_sig_sq_rank + curr_sig_j * curr_sig_j;
                }
                let prev_sig_rms = sqrt(prev_sig_sq / max(1.0, f32(d)) + 1.0e-6);
                let curr_sig_rms_rank = sqrt(curr_sig_sq_rank / max(1.0, f32(d)) + 1.0e-6);
                for (var j = 0u; j < d; j = j + 1u) {
                    var prev_src_j = 0.0;
                    if (t > 0u) {
                        let prev_sig_base = clean_signal_off(t_abs - 1u, slot, d, h);
                        prev_src_j =
                            (Scratch[prev_sig_base + j]
                                + select(0.0, AllWeights[slot_anchor_root + j], ENABLE_ASSOC_SLOT_ANCHOR))
                            / max(prev_sig_rms, 1.0e-6);
                    }
                    let curr_src_j =
                        (Scratch[clean_signal_off(t_abs, slot, d, h) + j]
                            + select(0.0, AllWeights[slot_anchor_root + j], ENABLE_ASSOC_SLOT_ANCHOR))
                        / max(curr_sig_rms_rank, 1.0e-6);
                    let wq_eff_base = select(wq_assoc_base, wk_assoc_base, ENABLE_ASSOC_TIE_QK);
                    q_acc = q_acc + AllWeights[wq_eff_base + j * RETAIN_RANK + lane] * prev_src_j;
                    k_acc = k_acc + AllWeights[wk_assoc_base + j * RETAIN_RANK + lane] * prev_src_j;
                    v_acc = v_acc + AllWeights[wv_assoc_base + j * RETAIN_RANK + lane] * curr_src_j;
                }
                assoc_q[lane] = tanh(q_acc * inv_sqrt_d);
                shared_kw[lane] = tanh(k_acc * inv_sqrt_d);
                assoc_raw[lane] = tanh(v_acc * inv_sqrt_d);
            }
            workgroupBarrier();

            if (lane == 0u) {
                var chosen_bank = 0u;
                var found_empty = false;
                var empty_bank = 0u;
                var best_cos = -1.0e30;
                var best_bank = 0u;
                var min_usage = 1.0e30;
                var min_usage_bank = 0u;
                var allow_write = 0.0;
                var new_key_norm = 0.0;
                for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                    new_key_norm = new_key_norm + shared_kw[r] * shared_kw[r];
                }
                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                    let bank_base = bank * assoc_bank_stride;
                    var key_norm = 0.0;
                    var score = 0.0;
                    for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                        let key_r = assoc_b_state[bank_base + r];
                        key_norm = key_norm + key_r * key_r;
                        score = score + key_r * shared_kw[r];
                    }
                    let bank_usage = assoc_b_state[bank_base + RETAIN_RANK + d];
                    if (bank_usage < min_usage) {
                        min_usage = bank_usage;
                        min_usage_bank = bank;
                    }
                    if (!found_empty && (key_norm < 1.0e-8 || bank_usage < 1.0e-4)) {
                        empty_bank = bank;
                        found_empty = true;
                    }
                    let cos = score / sqrt(max(key_norm * new_key_norm, 1.0e-12));
                    if (cos > best_cos) {
                        best_cos = cos;
                        best_bank = bank;
                    }
                }
                if (ENABLE_ASSOC_REUSE_MATCH && best_cos > 0.80) {
                    chosen_bank = best_bank;
                    allow_write = 1.0;
                } else if (found_empty) {
                    chosen_bank = empty_bank;
                    allow_write = 1.0;
                } else if (best_cos <= 0.80 && min_usage < ASSOC_ALLOC_THRESHOLD) {
                    chosen_bank = min_usage_bank;
                    allow_write = 1.0;
                }
                if (ENABLE_ASSOC_SLOT_STRIPE && (t % h) != slot) {
                    // Experimental multi-bank routing: mirror forward slot striping so
                    // gradients are assigned to the same slot/head that wrote the state.
                    allow_write = 0.0;
                }
                shared_vec[0] = f32(chosen_bank);
                shared_vec[1] = allow_write;
                var transition_gate = 1.0;
                if (ENABLE_ASSOC_TRANSITION_GATE) {
                    var transition_score = 0.0;
                    for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                        transition_score = transition_score + shared_kw[r] * assoc_raw[r];
                    }
                    transition_gate =
                        1.0 / (1.0 + exp(-transition_score * inverseSqrt(max(1.0, f32(RETAIN_RANK)))));
                }
                shared_vec[2] = transition_gate;
            }
            workgroupBarrier();
            let chosen_bank = u32(shared_vec[0]);
            let assoc_write_allowed = shared_vec[1];
            let assoc_transition_gate = shared_vec[2];
            var local_bind_grad = 0.0;
            if (lane < RETAIN_RANK) {
                let key_idx = chosen_bank * assoc_bank_stride + lane;
                local_bind_grad = local_bind_grad
                    + assoc_gb[key_idx] * (shared_kw[lane] - assoc_b_state[key_idx]);
            }
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                let val_idx = chosen_bank * assoc_bank_stride + RETAIN_RANK + dim;
                var curr_sig_sq = 0.0;
                for (var j = 0u; j < d; j = j + 1u) {
                    let curr_sig_j =
                        Scratch[clean_signal_off(t_abs, slot, d, h) + j]
                        + select(0.0, AllWeights[slot_anchor_root + j], ENABLE_ASSOC_SLOT_ANCHOR);
                    curr_sig_sq = curr_sig_sq + curr_sig_j * curr_sig_j;
                }
                let curr_sig_rms = sqrt(curr_sig_sq / max(1.0, f32(d)) + 1.0e-6);
                local_bind_grad = local_bind_grad
                    + assoc_gb[val_idx]
                    * (
                        (
                            Scratch[clean_signal_off(t_abs, slot, d, h) + dim]
                            + select(0.0, AllWeights[slot_anchor_root + dim], ENABLE_ASSOC_SLOT_ANCHOR)
                        ) / max(curr_sig_rms, 1.0e-6)
                        - assoc_b_state[val_idx]
                    );
            }
            let bind_grad = reduce_sum(lane, local_bind_grad);
            let bind_gate_logit_grad =
                bind_grad * assoc_write_allowed * assoc_transition_gate
                * event_gate_bwd * ASSOC_WRITE_CAP * assoc_raw_z_bwd * (1.0 - assoc_raw_z_bwd);
            let sparsity_penalty = select(
                0.0,
                ASSOC_EVENT_L1 * event_gate_bwd * (1.0 - event_gate_bwd) * has_prev_hstar,
                ENABLE_ASSOC_EVENT_GATE,
            );

            let event_gate_logit_grad =
                bind_grad * assoc_write_allowed * assoc_transition_gate
                * assoc_write_gate_bwd * event_gate_bwd * (1.0 - event_gate_bwd) * has_prev_hstar
                + sparsity_penalty;
            var transition_gate_grad = 0.0;
            if (ENABLE_ASSOC_TRANSITION_GATE) {
                transition_gate_grad =
                    bind_grad * assoc_write_allowed * bind_gate
                    * assoc_transition_gate * (1.0 - assoc_transition_gate);
            }

            // Backward through the selected-bank EMA write:
            //   ∂L/∂x_t += g_t * ∂L/∂B_t
            //   ∂L/∂B_{t-1} from future = (1 - g_t) * ∂L/∂B_t
            if (lane < RETAIN_RANK) {
                let key_idx = chosen_bank * assoc_bank_stride + lane;
                let g_bank_key_curr = assoc_gb[key_idx];
                let effective_bind_gate = bind_gate * assoc_write_allowed * assoc_transition_gate;
                let effective_keep_gate = 1.0 - effective_bind_gate;
                assoc_gprev[lane] = effective_bind_gate * g_bank_key_curr;
                assoc_gb[key_idx] = effective_keep_gate * g_bank_key_curr;
            }
            workgroupBarrier();
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                let effective_bind_gate = bind_gate * assoc_write_allowed * assoc_transition_gate;
                let effective_keep_gate = 1.0 - effective_bind_gate;
                let val_idx = chosen_bank * assoc_bank_stride + RETAIN_RANK + dim;
                let g_bank_val_curr = assoc_gb[val_idx];
                assoc_gb[val_idx] = effective_keep_gate * g_bank_val_curr;
            }
            workgroupBarrier();

            if (t > 0u) {
                var local_wk_step_abs_dbg = 0.0;
                for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                    let dim = lane * dims_per_lane + k;
                    let step_wg_assoc = lr_scaled * bind_gate_logit_grad * c[k] * inv_sqrt_d / n_norm;
                    if (params.grad_accum_mode == 1u) {
                        AllGradients[wf_base + dim] += step_wg_assoc;
                    } else {
                        AllWeights[wf_base + dim] -= clamp(step_wg_assoc, -clip, clip);
                    }
                }
                let prev_sig_base = clean_signal_off(t_abs - 1u, slot, d, h);
                let curr_sig_base = clean_signal_off(t_abs, slot, d, h);
                var prev_sig_sq = 0.0;
                var curr_sig_sq = 0.0;
                for (var j = 0u; j < d; j = j + 1u) {
                    let prev_sig_j = Scratch[prev_sig_base + j];
                    let prev_sig_j_anchor =
                        prev_sig_j + select(0.0, AllWeights[slot_anchor_root + j], ENABLE_ASSOC_SLOT_ANCHOR);
                    prev_sig_sq = prev_sig_sq + prev_sig_j_anchor * prev_sig_j_anchor;
                    let curr_sig_j =
                        Scratch[curr_sig_base + j]
                        + select(0.0, AllWeights[slot_anchor_root + j], ENABLE_ASSOC_SLOT_ANCHOR);
                    curr_sig_sq = curr_sig_sq + curr_sig_j * curr_sig_j;
                }
                let prev_sig_rms = sqrt(prev_sig_sq / max(1.0, f32(d)) + 1.0e-6);
                let curr_sig_rms = sqrt(curr_sig_sq / max(1.0, f32(d)) + 1.0e-6);
                for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                    let dim = lane * dims_per_lane + k;
                    let prev_src =
                        (Scratch[prev_sig_base + dim]
                            + select(0.0, AllWeights[slot_anchor_root + dim], ENABLE_ASSOC_SLOT_ANCHOR))
                        / max(prev_sig_rms, 1.0e-6);
                    let curr_src =
                        (Scratch[curr_sig_base + dim]
                            + select(0.0, AllWeights[slot_anchor_root + dim], ENABLE_ASSOC_SLOT_ANCHOR))
                        / max(curr_sig_rms, 1.0e-6);
                    let event_feature = prev_src + curr_src;
                    let step_wevent_assoc =
                        ASSOC_ADDR_GRAD_SCALE * lr_scaled * event_gate_logit_grad * event_feature * inv_sqrt_d / n_norm;
                    if (params.grad_accum_mode == 1u) {
                        AllGradients[wevent_assoc_base + dim] += step_wevent_assoc;
                    } else {
                        AllWeights[wevent_assoc_base + dim] -= clamp(step_wevent_assoc, -clip, clip);
                    }
                    for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                        let transition_to_key =
                            transition_gate_grad * assoc_raw[r]
                            * inverseSqrt(max(1.0, f32(RETAIN_RANK)));
                        let transition_to_val =
                            transition_gate_grad * shared_kw[r]
                            * inverseSqrt(max(1.0, f32(RETAIN_RANK)));
                        let gk_pre =
                            (assoc_gprev[r] + transition_to_key)
                            * (1.0 - shared_kw[r] * shared_kw[r]);
                        let gv_pre =
                            transition_to_val * (1.0 - assoc_raw[r] * assoc_raw[r]);
                        let step_wk_assoc =
                            ASSOC_ADDR_GRAD_SCALE * lr_scaled * gk_pre * prev_src * inv_sqrt_d / n_norm;
                        let step_wv_assoc =
                            ASSOC_ADDR_GRAD_SCALE * lr_scaled * gv_pre * curr_src * inv_sqrt_d / n_norm;
                        local_wk_step_abs_dbg = local_wk_step_abs_dbg + abs(step_wk_assoc);
                        if (params.grad_accum_mode == 1u) {
                            AllGradients[wk_assoc_base + dim * RETAIN_RANK + r] += step_wk_assoc;
                            AllGradients[wv_assoc_base + dim * RETAIN_RANK + r] += step_wv_assoc;
                        } else {
                            AllWeights[wk_assoc_base + dim * RETAIN_RANK + r] -= clamp(step_wk_assoc, -clip, clip);
                            AllWeights[wv_assoc_base + dim * RETAIN_RANK + r] -= clamp(step_wv_assoc, -clip, clip);
                        }
                    }
                }
                if (lane == 0u) {
                    let step_bg_assoc = lr_scaled * bind_gate_logit_grad / n_norm;
                    let step_bevent_assoc = ASSOC_ADDR_GRAD_SCALE * lr_scaled * event_gate_logit_grad / n_norm;
                    if (params.grad_accum_mode == 1u) {
                        AllGradients[bwrite_idx] += step_bg_assoc;
                        AllGradients[bevent_assoc_idx] += step_bevent_assoc;
                    } else {
                        AllWeights[bwrite_idx] -= clamp(step_bg_assoc, -clip, clip);
                        AllWeights[bevent_assoc_idx] -= clamp(step_bevent_assoc, -clip, clip);
                    }
                }
                let wk_step_abs_dbg = reduce_sum(lane, local_wk_step_abs_dbg);
                if (lane == 0u) {
                    AssocBwdDebug[assoc_debug_base + 8u] =
                        AssocBwdDebug[assoc_debug_base + 8u] + wk_step_abs_dbg;
                    AssocBwdDebug[assoc_debug_base + 9u] =
                        max(AssocBwdDebug[assoc_debug_base + 9u], wk_step_abs_dbg);
                }
                workgroupBarrier();
            }
            workgroupBarrier();

            // READ backward using the recovered pre-write bank.
            // score_b = cosine(bank_key_b, q_t), matching the forward address metric.
            if (lane < RETAIN_RANK) {
                shared_kw[lane] = 0.0;
            }
            workgroupBarrier();
            if (lane == 0u) {
                var q_norm = 0.0;
                for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                    q_norm = q_norm + assoc_q[r] * assoc_q[r];
                }
                shared_vec[16u] = q_norm;
            }
            workgroupBarrier();
            for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                let bank_base = bank * assoc_bank_stride;
                if (lane < RETAIN_RANK) {
                    assoc_raw[lane] = assoc_b_state[bank_base + lane] * assoc_q[lane];
                }
                workgroupBarrier();
                if (lane == 0u) {
                    var dot_score = 0.0;
                    var key_norm = 0.0;
                    for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                        let key_r = assoc_b_state[bank_base + r];
                        dot_score = dot_score + assoc_raw[r];
                        key_norm = key_norm + key_r * key_r;
                    }
                    let bank_usage = assoc_b_state[bank_base + RETAIN_RANK + d];
                    let q_norm = shared_vec[16u];
                    let address_score = dot_score * inverseSqrt(max(key_norm * q_norm, 1.0e-12));
                    shared_dup[bank] =
                        ASSOC_READ_BETA * address_score + log(max(bank_usage, 1.0e-4));
                    shared_vec[20u + bank * 2u] = dot_score;
                    shared_vec[20u + bank * 2u + 1u] = key_norm;
                }
                workgroupBarrier();
            }
            if (lane == 0u) {
                var max_score = -1.0e30;
                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                    max_score = max(max_score, shared_dup[bank]);
                }
                var denom = 0.0;
                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                    let e = exp(shared_dup[bank] - max_score);
                    shared_dup[bank] = e;
                    denom = denom + e;
                }
                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                    shared_dup[bank] = shared_dup[bank] / max(denom, 1.0e-6);
                }
                var read_max_prob = 0.0;
                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                    read_max_prob = max(read_max_prob, shared_dup[bank]);
                }
                shared_vec[17u] = select(1.0, read_max_prob, ENABLE_ASSOC_CONF_READ);
            }
            workgroupBarrier();
            let assoc_read_conf = shared_vec[17u];

            // Read backward: ctx_dim = alpha * Σ_b match_b * bank_value_b[dim].
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                let g_ctx = v_state[vs_base + dim];
                var ctx_dim = 0.0;
                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                    let bank_base = bank * assoc_bank_stride;
                    ctx_dim = ctx_dim + shared_dup[bank] * assoc_b_state[bank_base + RETAIN_RANK + dim];
                }
                local_alpha_grad = local_alpha_grad + g_ctx * assoc_read_conf * ctx_dim;
            }

            // Gradients to every bank_value/key plus accumulated query preactivation.
            if (lane < RETAIN_RANK) {
                shared_kw[lane] = 0.0;
            }
            workgroupBarrier();
            for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                let bank_base = bank * assoc_bank_stride;
                var local_match_grad = 0.0;
                for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                    let dim = lane * dims_per_lane + k;
                    let g_ctx = v_state[vs_base + dim];
                    local_match_grad = local_match_grad
                        + alpha_assoc * assoc_read_conf * g_ctx * assoc_b_state[bank_base + RETAIN_RANK + dim];
                    assoc_gb[bank_base + RETAIN_RANK + dim] =
                        assoc_gb[bank_base + RETAIN_RANK + dim]
                        + alpha_assoc * assoc_read_conf * shared_dup[bank] * g_ctx;
                }
                let match_grad = reduce_sum(lane, local_match_grad);
                if (lane == 0u) {
                    shared_vec[bank] = match_grad;
                }
                workgroupBarrier();
            }
            if (lane == 0u) {
                var weighted_match_grad = 0.0;
                for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                    weighted_match_grad = weighted_match_grad + shared_dup[bank] * shared_vec[bank];
                }
                shared_vec[ASSOC_BANKS] = weighted_match_grad;
            }
            workgroupBarrier();
            for (var bank = 0u; bank < ASSOC_BANKS; bank = bank + 1u) {
                let bank_base = bank * assoc_bank_stride;
                if (lane < RETAIN_RANK) {
                    let score_grad =
                        shared_dup[bank]
                        * (shared_vec[bank] - shared_vec[ASSOC_BANKS]);
                    if (lane == 0u) {
                        AssocBwdDebug[assoc_debug_base + 4u] =
                            AssocBwdDebug[assoc_debug_base + 4u] + abs(score_grad);
                        AssocBwdDebug[assoc_debug_base + 5u] =
                            max(AssocBwdDebug[assoc_debug_base + 5u], abs(score_grad));
                    }
                    let dot_score = shared_vec[20u + bank * 2u];
                    let key_norm = max(shared_vec[20u + bank * 2u + 1u], 1.0e-12);
                    let q_norm = max(shared_vec[16u], 1.0e-12);
                    let inv_norm = inverseSqrt(max(key_norm * q_norm, 1.0e-12));
                    let key_val = assoc_b_state[bank_base + lane];
                    let q_val = assoc_q[lane];
                    let dscore_dkey = ASSOC_READ_BETA * inv_norm * (q_val - (dot_score / key_norm) * key_val);
                    let dscore_dq = ASSOC_READ_BETA * inv_norm * (key_val - (dot_score / q_norm) * q_val);
                    assoc_gb[bank_base + lane] =
                        assoc_gb[bank_base + lane] + dscore_dkey * score_grad;
                    shared_kw[lane] =
                        shared_kw[lane] + dscore_dq * score_grad;
                }
                workgroupBarrier();
            }
            if (lane < RETAIN_RANK) {
                shared_kw[lane] = shared_kw[lane] * (1.0 - assoc_q[lane] * assoc_q[lane]);
            }
            workgroupBarrier();

            // Parameter updates for the associative read query encoder.
            var local_wq_step_abs_dbg = 0.0;
            if (t > 0u) {
                let prev_sig_base = clean_signal_off(t_abs - 1u, slot, d, h);
                var prev_sig_sq = 0.0;
                for (var j = 0u; j < d; j = j + 1u) {
                    let prev_sig_j =
                        Scratch[prev_sig_base + j]
                        + select(0.0, AllWeights[slot_anchor_root + j], ENABLE_ASSOC_SLOT_ANCHOR);
                    prev_sig_sq = prev_sig_sq + prev_sig_j * prev_sig_j;
                }
                let prev_sig_rms = sqrt(prev_sig_sq / max(1.0, f32(d)) + 1.0e-6);
                for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                    let dim = lane * dims_per_lane + k;
                    let prev_src =
                        (Scratch[prev_sig_base + dim]
                            + select(0.0, AllWeights[slot_anchor_root + dim], ENABLE_ASSOC_SLOT_ANCHOR))
                        / max(prev_sig_rms, 1.0e-6);
                    for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                        let step_wq_assoc =
                            ASSOC_ADDR_GRAD_SCALE * lr_scaled * shared_kw[r] * prev_src * inv_sqrt_d / n_norm;
                        local_wq_step_abs_dbg = local_wq_step_abs_dbg + abs(step_wq_assoc);
                        let wq_eff_base = select(wq_assoc_base, wk_assoc_base, ENABLE_ASSOC_TIE_QK);
                        if (params.grad_accum_mode == 1u) {
                            AllGradients[wq_eff_base + dim * RETAIN_RANK + r] += step_wq_assoc;
                        } else {
                            AllWeights[wq_eff_base + dim * RETAIN_RANK + r] -= clamp(step_wq_assoc, -clip, clip);
                        }
                    }
                }
            }
            let wq_step_abs_dbg = reduce_sum(lane, local_wq_step_abs_dbg);
            if (lane == 0u) {
                AssocBwdDebug[assoc_debug_base + 6u] =
                    AssocBwdDebug[assoc_debug_base + 6u] + wq_step_abs_dbg;
                AssocBwdDebug[assoc_debug_base + 7u] =
                    max(AssocBwdDebug[assoc_debug_base + 7u], wq_step_abs_dbg);
            }
            workgroupBarrier();

            // Alpha update
            let g_alpha_total = reduce_sum(lane, local_alpha_grad);
            if (lane == 0u) {
                let step_alpha_assoc = lr_scaled * g_alpha_total / n_norm;
                AssocBwdDebug[assoc_debug_base + 10u] =
                    AssocBwdDebug[assoc_debug_base + 10u] + abs(step_alpha_assoc);
                AssocBwdDebug[assoc_debug_base + 11u] =
                    max(AssocBwdDebug[assoc_debug_base + 11u], abs(step_alpha_assoc));
                if (params.grad_accum_mode == 1u) {
                    AllGradients[alpha_assoc_idx] += step_alpha_assoc;
                } else {
                    AllWeights[alpha_assoc_idx] = clamp(
                        AllWeights[alpha_assoc_idx] - clamp(step_alpha_assoc, -clip, clip),
                        0.0,
                        1.0,
                    );
                }
            }
            workgroupBarrier();

            let local_h_sq = reduce_sum(lane, (
                select(0.0, h_star[h_base + lane * dims_per_lane + 0u] * h_star[h_base + lane * dims_per_lane + 0u], dims_per_lane > 0u)
                + select(0.0, h_star[h_base + lane * dims_per_lane + 1u] * h_star[h_base + lane * dims_per_lane + 1u], dims_per_lane > 1u)
                + select(0.0, h_star[h_base + lane * dims_per_lane + 2u] * h_star[h_base + lane * dims_per_lane + 2u], dims_per_lane > 2u)
                + select(0.0, h_star[h_base + lane * dims_per_lane + 3u] * h_star[h_base + lane * dims_per_lane + 3u], dims_per_lane > 3u)
                + select(0.0, h_star[h_base + lane * dims_per_lane + 4u] * h_star[h_base + lane * dims_per_lane + 4u], dims_per_lane > 4u)
                + select(0.0, h_star[h_base + lane * dims_per_lane + 5u] * h_star[h_base + lane * dims_per_lane + 5u], dims_per_lane > 5u)
                + select(0.0, h_star[h_base + lane * dims_per_lane + 6u] * h_star[h_base + lane * dims_per_lane + 6u], dims_per_lane > 6u)
                + select(0.0, h_star[h_base + lane * dims_per_lane + 7u] * h_star[h_base + lane * dims_per_lane + 7u], dims_per_lane > 7u)
            ));
            let h_rms = sqrt(max(local_h_sq / max(1.0, f32(d)), 1.0e-6));

            if (lane < RETAIN_RANK) {
                var up_acc = 0.0;
                for (var j = 0u; j < d; j = j + 1u) {
                    let c_j = 0.5 * (h_star[h_base + j] + Scratch[sig_base + j]);
                    up_acc = up_acc + AllWeights[wup_base + j * RETAIN_RANK + lane] * c_j;
                }
                shared_up[lane] = up_acc;
            }
            workgroupBarrier();

            // Phase 1: gate_partial + retain gate using shared_up (W_retain_up · c)
            var gate_partial = 0.0;
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                gate_partial = gate_partial + AllWeights[wf_base + dim] * c[k];
                var pre = AllWeights[bret_base + dim];
                for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                    pre = pre + AllWeights[wdown_base + r * d + dim] * shared_up[r];
                }
                retain[k] = 1.0 / (1.0 + exp(-pre));
            }
            // Phase 2: k-write bottleneck (W_k_write · c) → shared_kw
            workgroupBarrier();
            if (lane < RETAIN_RANK) {
                var kw_acc = 0.0;
                for (var j = 0u; j < d; j = j + 1u) {
                    let c_j = 0.5 * (h_star[h_base + j] + Scratch[sig_base + j]);
                    kw_acc = kw_acc + AllWeights[wkw_base + j * RETAIN_RANK + lane] * c_j;
                }
                shared_kw[lane] = kw_acc;
            }
            workgroupBarrier();
            // Phase 3: proposal = tanh(W_v_write · shared_kw + b_delta)
            var local_prop_sq = 0.0;
            var local_prev_sq = 0.0;
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                var delta = AllWeights[bd_base + dim];
                for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                    delta = delta + AllWeights[wvw_base + r * d + dim] * shared_kw[r];
                }
                proposal_pre[k] = delta;
                proposal[k] = tanh(delta);
                local_prop_sq = local_prop_sq + proposal[k] * proposal[k];
                local_prev_sq = local_prev_sq + m_prev[k] * m_prev[k];
            }
            let gate_dot = reduce_sum(lane, gate_partial) * inv_sqrt_d
                + AllWeights[bwrite_idx] + FPM_GATE_BIAS;
            let raw_z = 1.0 / (1.0 + exp(-gate_dot));
            let proposal_norm = sqrt(max(reduce_sum(lane, local_prop_sq), 1.0e-6));
            let prev_norm = sqrt(max(reduce_sum(lane, local_prev_sq), 1.0e-6));
            var local_diff_sq = 0.0;
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let diff = proposal[k] - m_prev[k];
                local_diff_sq = local_diff_sq + diff * diff;
            }
            let diff_norm = sqrt(max(reduce_sum(lane, local_diff_sq), 1.0e-6));
            let denom = diff_norm + prev_norm + 1.0e-6;
            let novelty = diff_norm / denom;
            let z = clamp(raw_z, 0.0, 1.0);
            let dnov_ddiff = prev_norm / (denom * denom);
            let dnov_dprev = -diff_norm / (denom * denom);

            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                shared_vec[dim] = dm_new[k];
            }
            workgroupBarrier();

            var local_gz = 0.0;
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                let g_inner = dm_new[k];
                g_m_inner[k] = g_inner;
                let h_val = h_star[h_base + dim] / h_rms;
                h_unit[k] = h_val;
                let wx = 0.5 * tanh(AllWeights[wx_base + dim * d + dim]);
                let retain_eff = 1.0 - novelty * (1.0 - retain[k]);
                let write = z * FPM_RESIDUAL_SCALE * proposal[k];
                x_proj[k] = h_val + wx * h_val + write;
                a[k] = 1.0 / (1.0 + exp(AllWeights[alog_base + dim]));
                let base_inner = a[k] * m_prev[k] + (1.0 - a[k]) * x_proj[k];
                m_inner[k] = retain_eff * m_prev[k] + (1.0 - retain_eff) * base_inner;
                let g_base = (1.0 - retain_eff) * g_inner;
                let g_x = (1.0 - a[k]) * g_base;
                let g_retain_eff = (m_prev[k] - base_inner) * g_inner;
                g_prev[k] = retain_eff * g_inner + (1.0 - retain_eff) * a[k] * g_inner;
                g_pre[k] = g_retain_eff * novelty * retain[k] * (1.0 - retain[k]);
                local_gz = local_gz + g_x
                    * (FPM_RESIDUAL_SCALE * proposal[k] + 0.5 * h_unit[k] / sqrt(max(z, 1.0e-6)));
            }
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                let off = fpm_m_off(t_abs, slot, d, h) + dim;
                fpm_dm_buf[off] = dm_new[k];
                fpm_minner_buf[off] = m_inner[k];
                fpm_hunit_buf[off] = h_unit[k];
                fpm_gx_buf[off] = (1.0 - a[k]) * g_m_inner[k];
            }
            workgroupBarrier();
            let g_z = reduce_sum(lane, local_gz);
            let g_raw_z = g_z * raw_z * (1.0 - raw_z);

            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                let retain_eff = 1.0 - novelty * (1.0 - retain[k]);
                let g_x = (1.0 - retain_eff) * (1.0 - a[k]) * g_m_inner[k];
                let diff = proposal[k] - m_prev[k];
                let diff_norm_safe = max(diff_norm, 1.0e-6);
                let prev_norm_safe = max(prev_norm, 1.0e-6);
                let g_retain_eff = g_pre[k] / max(novelty * retain[k] * (1.0 - retain[k]), 1.0e-6);
                let g_nov = g_retain_eff * (-(1.0 - retain[k]));
                let g_diff_norm = g_nov * dnov_ddiff;
                let g_prev_norm = g_nov * dnov_dprev;
                let g_prop_from_nov = g_diff_norm * (diff / diff_norm_safe);
                let g_prev_from_nov =
                    g_prev_norm * (m_prev[k] / prev_norm_safe) - g_diff_norm * (diff / diff_norm_safe);
                let g_prop = g_x * z * FPM_RESIDUAL_SCALE + g_prop_from_nov;
                let g_a = (1.0 - retain_eff) * (m_prev[k] - x_proj[k]) * g_m_inner[k];
                let g_delta = g_prop * (1.0 - proposal[k] * proposal[k]);
                g_prev[k] = g_prev[k] + g_prev_from_nov;
                let g_alog = -a[k] * (1.0 - a[k]) * g_a;
                let step_alog = lr_scaled * g_alog / n_norm;
                if (params.grad_accum_mode == 1u) {
                    AllGradients[alog_base + dim] += step_alog;
                } else {
                    AllWeights[alog_base + dim] = clamp(
                        AllWeights[alog_base + dim] - clamp(step_alog, -clip, clip),
                        -1.0,
                        9.0,
                    );
                }

                let step_bdelta = lr_scaled * g_delta / n_norm;
                if (params.grad_accum_mode == 1u) {
                    AllGradients[bd_base + dim] += step_bdelta;
                } else {
                    AllWeights[bd_base + dim] -= clamp(step_bdelta, -clip, clip);
                }
                // Store g_delta for factored W_v / W_k gradient computation below.
                // g_retain for W_down / W_up handled in the retain-gate backward block.
                let step_bret = lr_scaled * g_pre[k] / n_norm;
                if (params.grad_accum_mode == 1u) {
                    AllGradients[bret_base + dim] += step_bret;
                } else {
                    AllWeights[bret_base + dim] -= clamp(step_bret, -clip, clip);
                }
                shared_vec[dim] = g_pre[k];   // used by retain-gate W_down backward
            }
            workgroupBarrier();

            // ── Retain-gate backward (W_down, W_up) ─────────────────────────────────
            if (lane < RETAIN_RANK) {
                var dup = 0.0;
                for (var j = 0u; j < d; j = j + 1u) {
                    dup = dup + AllWeights[wdown_base + lane * d + j] * shared_vec[j];
                }
                shared_dup[lane] = dup;  // ∂L/∂up_r
            }
            workgroupBarrier();
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                    let step_wdown = lr_scaled * g_pre[k] * shared_up[r] / n_norm;
                    let step_wup = lr_scaled * shared_dup[r] * c[k] / n_norm;
                    if (params.grad_accum_mode == 1u) {
                        AllGradients[wdown_base + r * d + dim] += step_wdown;
                        AllGradients[wup_base + dim * RETAIN_RANK + r] += step_wup;
                    } else {
                        AllWeights[wdown_base + r * d + dim] -= clamp(step_wdown, -clip, clip);
                        AllWeights[wup_base + dim * RETAIN_RANK + r] -= clamp(step_wup, -clip, clip);
                    }
                }
                let step_wg = lr_scaled * g_raw_z * c[k] * inv_sqrt_d / n_norm;
                if (params.grad_accum_mode == 1u) {
                    AllGradients[wf_base + dim] += step_wg;
                } else {
                    AllWeights[wf_base + dim] -= clamp(step_wg, -clip, clip);
                }
                dm_new[k] = g_prev[k];
            }

            // ── READ-path gradient: ∂L/∂M[t-1] from fpm_ctx reading M at token t ─────
            // fpm_ctx_t ≈ tanh(M[t-1]) * alpha_m → ∂fpm_ctx_t/∂M ≈ alpha_m * I
            // v_state[slot, t_abs] = ∂L/∂h*[slot, t] ≈ ∂L/∂fpm_ctx_t (additive entry)
            // Contributes to dm_new = ∂L/∂M[t-1] (passed as carry to token t-1).
            {
                let vs_base = (slot * n_tokens + t_abs) * d;
                for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                    dm_new[k] += FPM_READ_GRAD_SCALE * v_state[vs_base + lane * dims_per_lane + k];
                }
            }

            // ── Direct W_k_write/W_v_write update from READ gradient (dm_new) ──────────
            // dm_new = ∂L/∂M[t-1] seeds from READ gradient v_state[t].
            // The write-path backward multiplies this by ≈0.002 (retain/z/RESIDUAL_SCALE attenuation),
            // making W_k_write/W_v_write updates negligible via that path.
            // Instead, directly update write matrices so proposals align with dm_new:
            //   goal: W_v_write * (W_k_write * c) → dm_new (write what the future wants to read)
            // This is the correct associative learning signal without the initialization bottleneck.
            {
                // g_delta_direct[dim] = dm_new[dim] * tanh'(proposal[dim])
                for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                    let dim = lane * dims_per_lane + k;
                    shared_vec[dim] = dm_new[k] * (1.0 - proposal[k] * proposal[k]);
                }
                workgroupBarrier();
                // g_k_direct[r] = Σ_dim W_v_write[r,dim] * g_delta_direct[dim]
                if (lane < RETAIN_RANK) {
                    var g_k = 0.0;
                    for (var j = 0u; j < d; j = j + 1u) {
                        g_k = g_k + AllWeights[wvw_base + lane * d + j] * shared_vec[j];
                    }
                    shared_dup[lane] = g_k;
                }
                workgroupBarrier();
                for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                    let dim = lane * dims_per_lane + k;
                    let gd = shared_vec[dim];
                    for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                        let step_wvw = FPM_DIRECT_WRITE_SCALE * lr_scaled * gd * shared_kw[r] / n_norm;
                        let step_wkw = FPM_DIRECT_WRITE_SCALE * lr_scaled * shared_dup[r] * c[k] / n_norm;
                        if (params.grad_accum_mode == 1u) {
                            AllGradients[wvw_base + r * d + dim] += step_wvw;
                            AllGradients[wkw_base + dim * RETAIN_RANK + r] += step_wkw;
                        } else {
                            AllWeights[wvw_base + r * d + dim] -= clamp(step_wvw, -clip, clip);
                            AllWeights[wkw_base + dim * RETAIN_RANK + r] -= clamp(step_wkw, -clip, clip);
                        }
                    }
                }
                workgroupBarrier();
            }

            // ── Factored k×v write backward (W_v_write, W_k_write) ───────────────────
            // g_delta values are in registers; store in shared_vec for bottleneck backprop.
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                let retain_eff = 1.0 - novelty * (1.0 - retain[k]);
                let diff = proposal[k] - m_prev[k];
                let diff_norm_safe = max(diff_norm, 1.0e-6);
                let prev_norm_safe = max(prev_norm, 1.0e-6);
                let g_retain_eff = g_pre[k] / max(novelty * retain[k] * (1.0 - retain[k]), 1.0e-6);
                let g_nov_local = g_retain_eff * (-(1.0 - retain[k]));
                let g_diff_norm = g_nov_local * dnov_ddiff;
                let g_prev_norm = g_nov_local * dnov_dprev;
                let g_prop_from_nov =
                    g_diff_norm * (diff / diff_norm_safe);
                let _g_prev_from_nov =
                    g_prev_norm * (m_prev[k] / prev_norm_safe) - g_diff_norm * (diff / diff_norm_safe);
                let g_prop = (1.0 - retain_eff) * (1.0 - a[k]) * g_m_inner[k] * z * FPM_RESIDUAL_SCALE
                    + g_prop_from_nov;
                let g_delta_val = g_prop * (1.0 - proposal[k] * proposal[k]);
                shared_vec[dim] = g_delta_val;
            }
            workgroupBarrier();
            // g_bottleneck[r] = Σ_dim W_v_write[r,dim] * g_delta[dim]
            if (lane < RETAIN_RANK) {
                var g_bot = 0.0;
                for (var j = 0u; j < d; j = j + 1u) {
                    g_bot = g_bot + AllWeights[wvw_base + lane * d + j] * shared_vec[j];
                }
                shared_dup[lane] = g_bot;  // reuse shared_dup: now ∂L/∂bottleneck_r
            }
            workgroupBarrier();
            // W_v_write[r,dim] += g_delta[dim] * bottleneck[r]  (outer product)
            // W_k_write[j,r]   += g_bottleneck[r] * c_j         (outer product, one c_j per lane)
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                let g_delta_val = shared_vec[dim];
                for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                    let step_wvw = lr_scaled * g_delta_val * shared_kw[r] / n_norm;
                    let step_wkw = lr_scaled * shared_dup[r] * c[k] / n_norm;
                    if (params.grad_accum_mode == 1u) {
                        AllGradients[wvw_base + r * d + dim] += step_wvw;
                        AllGradients[wkw_base + dim * RETAIN_RANK + r] += step_wkw;
                    } else {
                        AllWeights[wvw_base + r * d + dim] -= clamp(step_wvw, -clip, clip);
                        AllWeights[wkw_base + dim * RETAIN_RANK + r] -= clamp(step_wkw, -clip, clip);
                    }
                }
            }
            if (lane == 0u) {
                let step_bg = lr_scaled * g_raw_z / n_norm;
                if (params.grad_accum_mode == 1u) {
                    AllGradients[bwrite_idx] += step_bg;
                } else {
                    AllWeights[bwrite_idx] -= clamp(step_bg, -clip, clip);
                }
            }
            workgroupBarrier();
        }

        for (var k = 0u; k < dims_per_lane; k = k + 1u) {
            tbptt_carry_buf[carry_base + lane * dims_per_lane + k] = dm_new[k];
        }
        workgroupBarrier();
    }
}
