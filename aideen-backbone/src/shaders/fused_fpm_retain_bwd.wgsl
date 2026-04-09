// FPM memory backward pass for the active Model-A write path.
//
// Forward (per token t, slot s):
//   c         = 0.5 * (h_star[t,s] + signal[t,s])
//   raw_z     = sigmoid(dot(W_write_gate[s], c) / sqrt(d) + b_write_mem[s] + gate_bias)
//   proposal  = tanh(W_delta[s] * c + b_delta)
//   novelty   = ||proposal|| / (||proposal|| + ||m_prev|| + eps)
//   z         = raw_z * novelty
//   retain    = sigmoid(W_down[s] * (W_up[s] * c) + b_retain[s])
//   write     = (1 - retain) * z * residual_scale * proposal
//   h_unit    = h_star / rms(h_star)
//   x_proj    = h_unit + wx_diag * h_unit + write
//   a         = sigmoid(-a_log[s])
//   m_inner   = a * m_prev + (1 - a) * x_proj
//   m_new     = (I + W_out) * m_inner
//
// This kernel updates the slot-specific write parameters that participate in the
// post-solve FPM memory recurrence:
//   - W_write_gate / b_write_mem
//   - W_delta / b_delta
//   - W_retain_up / W_retain_down / b_retain
//   - a_log
//
// Shared carrier matrices W_x / W_out are part of the new forward semantics, but
// they are shared across slots; this kernel keeps them fixed and only handles the
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
@group(0) @binding(2) var<storage, read>       Scratch: array<f32>;  // clean layout: [batch*seq*2*h*d]
@group(0) @binding(3) var<storage, read>       fpm_m_buf: array<f32>; // [batch*seq*h*d]
@group(0) @binding(4) var<storage, read_write> AllGradients: array<f32>;
@group(0) @binding(5) var<storage, read_write> tbptt_carry_buf: array<f32>;
@group(0) @binding(6) var<storage, read_write> fpm_dm_buf: array<f32>;
@group(0) @binding(7) var<storage, read_write> fpm_minner_buf: array<f32>;
@group(0) @binding(8) var<storage, read_write> fpm_hunit_buf: array<f32>;
@group(0) @binding(9) var<storage, read_write> fpm_gx_buf: array<f32>;

@group(1) @binding(0) var<storage, read_write> AllWeights: array<f32>;

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
fn hist_delta_base(d: u32, h: u32) -> u32 { return slot_anchor_base(d, h) + h * d; }
fn hist_delta_bias_base(d: u32, h: u32) -> u32 { return hist_delta_base(d, h) + h * d * d; }
fn hist_gate_query_base(d: u32, h: u32) -> u32 { return hist_delta_bias_base(d, h) + d + 21u; }
fn w_write_gate_base(d: u32, h: u32) -> u32 { return hist_gate_query_base(d, h) + h * d; }
fn b_write_mem_base(d: u32, h: u32) -> u32 { return w_write_gate_base(d, h) + h * d; }
fn hhist_gamma_base(d: u32, h: u32) -> u32 { return b_write_mem_base(d, h) + h; }

const RETAIN_RANK: u32 = 32u;
fn w_retain_up_base(d: u32, h: u32) -> u32 { return hhist_gamma_base(d, h) + h; }
fn w_retain_down_base(d: u32, h: u32) -> u32 { return w_retain_up_base(d, h) + h * d * RETAIN_RANK; }
fn b_retain_base(d: u32, h: u32) -> u32 { return w_retain_down_base(d, h) + h * RETAIN_RANK * d; }

fn clean_signal_off(t_abs: u32, slot: u32, d: u32, h: u32) -> u32 {
    return t_abs * (2u * h * d) + slot * d;
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

var<workgroup> shared_up: array<f32, 32>;
var<workgroup> shared_dup: array<f32, 32>;
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
    let wd_base = hist_base + hist_delta_base(d, h) + slot * d * d;
    let bd_base = hist_base + hist_delta_bias_base(d, h);
    let wf_base = hist_base + w_write_gate_base(d, h) + slot * d;
    let bwrite_idx = hist_base + b_write_mem_base(d, h) + slot;
    let alog_base = aw_alog_base(d, h) + slot * d;
    let wout_base = aw_wout_base(d, h);
    let wx_base = aw_wx_base(d, h);
    let wup_base = hist_base + w_retain_up_base(d, h) + slot * d * RETAIN_RANK;
    let wdown_base = hist_base + w_retain_down_base(d, h) + slot * RETAIN_RANK * d;
    let bret_base = hist_base + b_retain_base(d, h) + slot * d;

    for (var b = 0u; b < params.batch_size; b = b + 1u) {
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

            var gate_partial = 0.0;
            var local_prop_sq = 0.0;
            var local_prev_sq = 0.0;
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                gate_partial = gate_partial + AllWeights[wf_base + dim] * c[k];
                var pre = AllWeights[bret_base + dim];
                for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                    pre = pre + AllWeights[wdown_base + r * d + dim] * shared_up[r];
                }
                retain[k] = 1.0 / (1.0 + exp(-pre));
                var delta = AllWeights[bd_base + dim];
                for (var j = 0u; j < d; j = j + 1u) {
                    let c_j = 0.5 * (h_star[h_base + j] + Scratch[sig_base + j]);
                    delta = delta + AllWeights[wd_base + dim * d + j] * c_j;
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
            let denom = proposal_norm + prev_norm + 1.0e-6;
            let novelty = proposal_norm / denom;
            let z_pre = raw_z * novelty;
            let z = clamp(z_pre, 0.0, 1.0);
            let z_grad_on = select(0.0, 1.0, z_pre > 0.0 && z_pre < 1.0);
            let dn_dp = prev_norm / (denom * denom);
            let dn_dm = -proposal_norm / (denom * denom);

            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                shared_vec[dim] = dm_new[k];
            }
            workgroupBarrier();

            var local_gz = 0.0;
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                var g_inner = dm_new[k];
                for (var row = 0u; row < d; row = row + 1u) {
                    g_inner = g_inner + AllWeights[wout_base + row * d + dim] * shared_vec[row];
                }
                g_m_inner[k] = g_inner;
                let h_val = h_star[h_base + dim] / h_rms;
                h_unit[k] = h_val;
                let wx = 0.5 * tanh(AllWeights[wx_base + dim * d + dim]);
                let write = (1.0 - retain[k]) * z * FPM_RESIDUAL_SCALE * proposal[k];
                x_proj[k] = h_val + wx * h_val + write;
                a[k] = 1.0 / (1.0 + exp(AllWeights[alog_base + dim]));
                m_inner[k] = a[k] * m_prev[k] + (1.0 - a[k]) * x_proj[k];
                let g_x = (1.0 - a[k]) * g_inner;
                g_prev[k] = a[k] * g_inner;
                local_gz = local_gz + g_x * (1.0 - retain[k]) * FPM_RESIDUAL_SCALE * proposal[k];
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
            let g_raw_z = z_grad_on * g_z * novelty * raw_z * (1.0 - raw_z);
            let g_nov = z_grad_on * g_z * raw_z;

            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let dim = lane * dims_per_lane + k;
                let g_x = (1.0 - a[k]) * g_m_inner[k];
                let prop_norm_safe = max(proposal_norm, 1.0e-6);
                let prev_norm_safe = max(prev_norm, 1.0e-6);
                let g_prop_from_nov = g_nov * dn_dp * (proposal[k] / prop_norm_safe);
                let g_prev_from_nov = g_nov * dn_dm * (m_prev[k] / prev_norm_safe);
                let g_prop = g_x * (1.0 - retain[k]) * z * FPM_RESIDUAL_SCALE + g_prop_from_nov;
                let g_retain = g_x * (-z * FPM_RESIDUAL_SCALE * proposal[k]);
                let g_a = (m_prev[k] - x_proj[k]) * g_m_inner[k];
                let g_delta = g_prop * (1.0 - proposal[k] * proposal[k]);
                g_pre[k] = g_retain * retain[k] * (1.0 - retain[k]);
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
                for (var j = 0u; j < d; j = j + 1u) {
                    let c_j = 0.5 * (h_star[h_base + j] + Scratch[sig_base + j]);
                    let step_wd = lr_scaled * g_delta * c_j / n_norm;
                    if (params.grad_accum_mode == 1u) {
                        AllGradients[wd_base + dim * d + j] += step_wd;
                    } else {
                        AllWeights[wd_base + dim * d + j] -= clamp(step_wd, -clip, clip);
                    }
                }
                let step_bret = lr_scaled * g_pre[k] / n_norm;
                if (params.grad_accum_mode == 1u) {
                    AllGradients[bret_base + dim] += step_bret;
                } else {
                    AllWeights[bret_base + dim] -= clamp(step_bret, -clip, clip);
                }
                shared_vec[dim] = g_pre[k];
            }
            workgroupBarrier();

            if (lane < RETAIN_RANK) {
                var dup = 0.0;
                for (var j = 0u; j < d; j = j + 1u) {
                    dup = dup + AllWeights[wdown_base + lane * d + j] * shared_vec[j];
                }
                shared_dup[lane] = dup;
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
