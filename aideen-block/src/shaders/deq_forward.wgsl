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

override ENABLE_DEBUG_METRICS: bool = true;
override ENABLE_SLOT_QKV_PROBE: bool = false;
override ENABLE_SLOT_ATTN_MINIMAL: bool = false;

@group(0) @binding(0) var<uniform> shape: RunUniforms;
@group(0) @binding(1) var<storage, read> S_in: array<f32>;
@group(0) @binding(2) var<storage, read> AllWeights: array<f32>;
@group(0) @binding(3) var<storage, read_write> H_curr: array<f32>;
@group(0) @binding(4) var<storage, read_write> H_next: array<f32>;
@group(0) @binding(5) var<storage, read_write> Scratch: array<f32>;
@group(0) @binding(6) var<storage, read_write> H_pooled: array<f32>;
@group(0) @binding(7) var<storage, read_write> DebugLog: array<f32>;

// AllWeights layout offsets (must match fused_deq_update.wgsl and deq_bridge.rs).
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

// Scratch layout per token (must match fused_deq_update.wgsl).
fn fw_scratch_stride(d: u32, h: u32) -> u32 {
    return d * (h * 8u) + h * h + h;
}
fn fw_token_scratch_base(t: u32, d: u32, h: u32) -> u32 {
    return t * fw_scratch_stride(d, h);
}
fn fw_token_mamba_base(t: u32, d: u32, h: u32) -> u32 {
    return fw_token_scratch_base(t, d, h) + h * 4u * d;
}
fn fw_token_signal_base(t: u32, d: u32, h: u32) -> u32 {
    return fw_token_mamba_base(t, d, h) + h * d;
}

// History params layout inside AllWeights (after aw_hist_base).
fn hist_mat_len(d: u32) -> u32 { return d * d; }
fn hist_scale_base(d: u32, h: u32) -> u32 { return hist_mat_len(d); }
fn hist_gate_base(d: u32, h: u32) -> u32 {
    // skip: mat(d*d) + scale(h*d) + bias(h*d)
    return hist_mat_len(d) + h * d + h * d;
}

const WG_SIZE: u32 = 256u;
const MAX_SLOTS: u32 = 8u;
var<workgroup> shared_vals: array<f32, WG_SIZE>;
var<workgroup> shared_aux: array<f32, WG_SIZE>;
var<workgroup> hit_count: atomic<u32>;
var<workgroup> max_delta_seen: f32;
var<workgroup> last_delta: f32;
var<workgroup> curr_contractivity: f32;
var<workgroup> max_h_seen: f32;
var<workgroup> slot_attn_weights: array<f32, MAX_SLOTS>;
// Shared scalars for history computation (written by tid==0, read by all after barrier).
var<workgroup> wg_prev_rms: f32;
var<workgroup> wg_u_rms: f32;
var<workgroup> wg_inj_rms: f32;
var<workgroup> wg_hist_scale: f32;
var<workgroup> wg_hist_alpha: f32;

@compute @workgroup_size(256, 1, 1)
fn deq_forward_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.x;
    let slot_idx = wid.y;
    if (batch_idx >= shape.batch_size) { return; }

    let d_model = shape.d_model;
    let h_slots = shape.h_slots;
    if (h_slots == 0u || h_slots > MAX_SLOTS || slot_idx >= h_slots) { return; }

    let aw_win = aw_win_base(d_model, h_slots);
    let aw_nscale = aw_nscale_base(d_model, h_slots);
    let aw_hist = aw_hist_base(d_model, h_slots);
    let total_elements = h_slots * d_model;
    let slot_off = slot_idx * d_model;
    let h_base = batch_idx * total_elements;
    let zero_win_diag = shape.diag_zero_win != 0u;
    let iter_limit = select(shape.max_iters, 1u, shape.diag_one_iter != 0u);
    let inv_d_model = 1.0 / max(1.0, f32(d_model));
    let slot_attn_rank = min(d_model, 32u);
    let slot_attn_scale = inverseSqrt(max(1.0, f32(slot_attn_rank)));
    // History mode: residual_alpha > -1.5 enables history integration.
    let hist_enabled = shape.residual_alpha > -1.5;

    if (ENABLE_DEBUG_METRICS && tid == 0u) {
        atomicStore(&hit_count, 0u);
        max_delta_seen = 0.0;
        last_delta = 0.0;
        curr_contractivity = 0.0;
        max_h_seen = 0.0;
    }

    var total_iters = 0u;
    var max_contractivity = 0.0;

    for (var t = 0u; t < shape.token_count; t = t + 1u) {
        let global_t = shape.token_start + t;
        let t_abs = batch_idx * shape.seq_len + global_t;
        let signal_base = fw_token_signal_base(t_abs, d_model, h_slots) + slot_off;
        let mamba_base = fw_token_mamba_base(t_abs, d_model, h_slots) + slot_off;
        let h_base_t = t_abs * total_elements;
        let s_in_base = t_abs * d_model;
        let win_base = slot_idx * d_model * d_model;

        // ── Step 1: Compute inj = W_in * s_t, write to signal_base ──
        if (d_model == 512u) {
            let d_out0 = tid;
            let d_out1 = tid + WG_SIZE;
            var inj0 = 0.0;
            var inj1 = 0.0;
            if (!zero_win_diag) {
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let s_val = S_in[s_in_base + j];
                    let row_base = aw_win + win_base + j * d_model;
                    inj0 = inj0 + AllWeights[row_base + d_out0] * s_val;
                    inj1 = inj1 + AllWeights[row_base + d_out1] * s_val;
                }
            }
            Scratch[signal_base + d_out0] = inj0;
            Scratch[signal_base + d_out1] = inj1;
        } else {
            for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                var inj = 0.0;
                if (!zero_win_diag) {
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        inj = inj + AllWeights[aw_win + win_base + j * d_model + d_out] * S_in[s_in_base + j];
                    }
                }
                Scratch[signal_base + d_out] = inj;
            }
        }
        workgroupBarrier();

        // ── Step 2: Compute history context c_{t,k} from M_{t-1} ──
        if (hist_enabled && global_t > 0u) {
            let prev_t_abs = t_abs - 1u;
            let prev_mamba = fw_token_mamba_base(prev_t_abs, d_model, h_slots) + slot_off;

            // 2a: RMS of M_{t-1}
            var local_prev_sq = 0.0;
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                let m = Scratch[prev_mamba + d];
                local_prev_sq = local_prev_sq + m * m;
            }
            shared_vals[tid] = local_prev_sq;
            workgroupBarrier();
            for (var s = WG_SIZE / 2u; s > 0u; s = s >> 1u) {
                if (tid < s) { shared_vals[tid] = shared_vals[tid] + shared_vals[tid + s]; }
                workgroupBarrier();
            }
            if (tid == 0u) { wg_prev_rms = sqrt(shared_vals[0] * inv_d_model + 1e-6); }
            workgroupBarrier();
            let prev_rms = wg_prev_rms;

            // 2b: u = W_hist * (M/rms) + scale * (M/rms)
            var local_u_sq = 0.0;
            var local_inj_sq = 0.0;
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                let m_normed = Scratch[prev_mamba + d] / prev_rms;
                // u = hist_scale[slot,d] * m_normed + W_hist_shared * m_normed
                var u = AllWeights[aw_hist + hist_scale_base(d_model, h_slots) + slot_off + d] * m_normed;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let m_j = Scratch[prev_mamba + j] / prev_rms;
                    u = u + AllWeights[aw_hist + d * d_model + j] * m_j;
                }
                // Store u temporarily in hist_ctx scratch region
                let hctx_base = fw_token_scratch_base(t_abs, d_model, h_slots) + 7u * h_slots * d_model + slot_off;
                Scratch[hctx_base + d] = u;
                local_u_sq = local_u_sq + u * u;
                let inj_val = Scratch[signal_base + d];
                local_inj_sq = local_inj_sq + inj_val * inj_val;
            }
            shared_vals[tid] = local_u_sq;
            shared_aux[tid] = local_inj_sq;
            workgroupBarrier();
            for (var s = WG_SIZE / 2u; s > 0u; s = s >> 1u) {
                if (tid < s) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + s];
                    shared_aux[tid] = shared_aux[tid] + shared_aux[tid + s];
                }
                workgroupBarrier();
            }
            if (tid == 0u) {
                wg_u_rms = sqrt(shared_vals[0] * inv_d_model + 1e-6);
                wg_inj_rms = sqrt(shared_aux[0] * inv_d_model + 1e-6);
                // Cap: scale = min(1, tau / rms(u)) where tau = cap_mult * rms(inj)
                let tau = max(wg_inj_rms, 0.0);
                wg_hist_scale = min(1.0, tau / max(wg_u_rms, 1e-6));
                // Gate: alpha = alpha_min + (alpha_max - alpha_min) * sigmoid(gate_logit)
                let gate_logit = AllWeights[aw_hist + hist_gate_base(d_model, h_slots) + slot_idx];
                let alpha_min = 0.070;
                let alpha_max = 0.20;
                wg_hist_alpha = alpha_min + (alpha_max - alpha_min) / (1.0 + exp(-gate_logit));
            }
            workgroupBarrier();
            let scale = wg_hist_scale;
            let alpha = wg_hist_alpha;

            // 2c: c_{t,k} = alpha * scale * u — write to hist_ctx scratch
            let hctx_base = fw_token_scratch_base(t_abs, d_model, h_slots) + 7u * h_slots * d_model + slot_off;
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                Scratch[hctx_base + d] = alpha * scale * Scratch[hctx_base + d];
            }
            workgroupBarrier();
        } else if (hist_enabled && global_t == 0u) {
            // First token: zero history context, zero M_0 initial state
            let hctx_base = fw_token_scratch_base(t_abs, d_model, h_slots) + 7u * h_slots * d_model + slot_off;
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                Scratch[hctx_base + d] = 0.0;
                Scratch[mamba_base + d] = 0.0;
            }
            workgroupBarrier();
        }

        // Initialize H_curr for first token
        if (global_t == 0u) {
            var local_sumsq = 0.0;
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                let sig = Scratch[signal_base + d];
                local_sumsq = local_sumsq + sig * sig;
            }
            shared_vals[tid] = local_sumsq;
            workgroupBarrier();
            for (var s = WG_SIZE / 2u; s > 0u; s = s >> 1u) {
                if (tid < s) { shared_vals[tid] = shared_vals[tid] + shared_vals[tid + s]; }
                workgroupBarrier();
            }
            let sig_rms = sqrt(shared_vals[0] * inv_d_model + 1e-6);
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                H_curr[h_base + slot_off + d] = Scratch[signal_base + d] / max(sig_rms, 1e-6);
            }
        }

        if (ENABLE_SLOT_QKV_PROBE) {
            let qkv_scratch = fw_token_scratch_base(t_abs, d_model, h_slots);
            for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                var q = 0.0;
                var k = 0.0;
                var v = 0.0;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let h_val = H_curr[h_base + slot_off + j];
                    let w_idx = j * d_model + d_out;
                    q = q + AllWeights[w_idx] * h_val;
                    k = k + AllWeights[aw_wk_base(d_model, h_slots) + w_idx] * h_val;
                    v = v + AllWeights[aw_wv_base(d_model, h_slots) + w_idx] * h_val;
                }
                Scratch[qkv_scratch + slot_off + d_out] = q;
                Scratch[qkv_scratch + h_slots * d_model + slot_off + d_out] = k;
                Scratch[qkv_scratch + 2u * h_slots * d_model + slot_off + d_out] = v;
            }
        }

        // ── Step 3: Picard iteration (DEQ solve) ──
        let hctx_base = fw_token_scratch_base(t_abs, d_model, h_slots) + 7u * h_slots * d_model + slot_off;
        var iter = 0u;
        var converged = false;
        while (iter < iter_limit && !converged) {
            if (ENABLE_SLOT_ATTN_MINIMAL) {
                if (tid < h_slots) {
                    let k_off = tid * d_model;
                    var score = 0.0;
                    for (var j = 0u; j < slot_attn_rank; j = j + 1u) {
                        score = score + H_curr[h_base + slot_off + j] * H_curr[h_base + k_off + j];
                    }
                    slot_attn_weights[tid] = score * slot_attn_scale;
                }
                workgroupBarrier();
                if (tid == 0u) {
                    var max_s = -1e30;
                    for (var ks = 0u; ks < h_slots; ks = ks + 1u) { max_s = max(max_s, slot_attn_weights[ks]); }
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
            }
            var local_max_delta = 0.0;
            var local_sumsq = 0.0;
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                var attn_ctx = 0.0;
                if (ENABLE_SLOT_ATTN_MINIMAL) {
                    for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
                        attn_ctx = attn_ctx + slot_attn_weights[ks] * H_curr[h_base + ks * d_model + d];
                    }
                }
                var hist_ctx = 0.0;
                if (hist_enabled) {
                    hist_ctx = Scratch[hctx_base + d];
                }
                let pre = Scratch[signal_base + d] + H_curr[h_base + slot_off + d] + 0.25 * attn_ctx + hist_ctx;
                local_sumsq = local_sumsq + pre * pre;
            }
            shared_vals[tid] = local_sumsq;
            workgroupBarrier();
            for (var s = WG_SIZE / 2u; s > 0u; s = s >> 1u) {
                if (tid < s) { shared_vals[tid] = shared_vals[tid] + shared_vals[tid + s]; }
                workgroupBarrier();
            }
            let rms = sqrt(shared_vals[0] * inv_d_model + 1e-6);

            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                let h_prev = H_curr[h_base + slot_off + d];
                var attn_ctx = 0.0;
                if (ENABLE_SLOT_ATTN_MINIMAL) {
                    for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
                        attn_ctx = attn_ctx + slot_attn_weights[ks] * H_curr[h_base + ks * d_model + d];
                    }
                }
                var hist_ctx = 0.0;
                if (hist_enabled) {
                    hist_ctx = Scratch[hctx_base + d];
                }
                let pre = Scratch[signal_base + d] + h_prev + 0.25 * attn_ctx + hist_ctx;
                let f_h = AllWeights[aw_nscale + d] * (pre / rms);
                let val = shape.damping * f_h + (1.0 - shape.damping) * h_prev;
                local_max_delta = max(local_max_delta, abs(val - h_prev));
                H_curr[h_base + slot_off + d] = val;
            }

            shared_vals[tid] = local_max_delta;
            workgroupBarrier();
            for (var s = WG_SIZE / 2u; s > 0u; s = s >> 1u) {
                if (tid < s) { shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + s]); }
                workgroupBarrier();
            }

            if (shared_vals[0] < shape.epsilon) {
                converged = true;
            }

            if (ENABLE_DEBUG_METRICS && tid == 0u) {
                let d_curr = shared_vals[0];
                let d_prev = last_delta;
                curr_contractivity = 0.0;
                if (iter > 0u && d_prev > 1e-12 && d_prev > shape.epsilon * 10.0) {
                    curr_contractivity = d_curr / d_prev;
                }
                last_delta = d_curr;
                max_contractivity = max(max_contractivity, curr_contractivity);
            }

            iter = iter + 1u;
        }

        total_iters = total_iters + iter;
        if (ENABLE_DEBUG_METRICS && tid == 0u) {
            max_delta_seen = max(max_delta_seen, last_delta);
            if (!converged) {
                atomicAdd(&hit_count, 1u);
            }
        }

        // ── Step 4: Store h* in H_next ──
        var local_final_max_h = 0.0;
        for (var d = tid; d < d_model; d = d + WG_SIZE) {
            let h_val = H_curr[h_base + slot_off + d];
            H_next[h_base_t + slot_off + d] = h_val;
            local_final_max_h = max(local_final_max_h, abs(h_val));
        }
        if (ENABLE_DEBUG_METRICS) {
            shared_vals[tid] = local_final_max_h;
            workgroupBarrier();
            for (var s = WG_SIZE / 2u; s > 0u; s = s >> 1u) {
                if (tid < s) { shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + s]); }
                workgroupBarrier();
            }
            if (tid == 0u) {
                max_h_seen = max(max_h_seen, shared_vals[0]);
            }
        }

        // ── Step 5: Temporal update M_t = SSM(M_{t-1}, h*) ──
        if (hist_enabled) {
            // 5a: RMS of h*
            var local_h_sq = 0.0;
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                let h_val = H_curr[h_base + slot_off + d];
                local_h_sq = local_h_sq + h_val * h_val;
            }
            shared_vals[tid] = local_h_sq;
            workgroupBarrier();
            for (var s = WG_SIZE / 2u; s > 0u; s = s >> 1u) {
                if (tid < s) { shared_vals[tid] = shared_vals[tid] + shared_vals[tid + s]; }
                workgroupBarrier();
            }
            let h_rms = sqrt(shared_vals[0] * inv_d_model + 1e-6);

            // 5b: Compute m_inner = a * M_{t-1} + (1-a) * x_proj, store in mamba_base temporarily
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                let h_unit = H_curr[h_base + slot_off + d] / h_rms;
                let wx = 0.5 * tanh(AllWeights[aw_wx_base(d_model, h_slots) + d * d_model + d]);
                let x_proj = h_unit + wx * h_unit;
                let a_log = AllWeights[aw_alog_base(d_model, h_slots) + slot_idx * d_model + d];
                let a = 1.0 / (1.0 + exp(a_log));
                var prev_m = 0.0;
                if (global_t > 0u) {
                    let prev_mamba = fw_token_mamba_base(t_abs - 1u, d_model, h_slots) + slot_off;
                    prev_m = Scratch[prev_mamba + d];
                }
                Scratch[mamba_base + d] = a * prev_m + (1.0 - a) * x_proj;
            }
            workgroupBarrier();

            // 5c: Residual carrier M_t = m_inner + W_out * m_inner (full matmul)
            // Use hist_ctx region as temp for the matmul result
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                var wout_sum = 0.0;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    wout_sum = wout_sum + AllWeights[aw_wout_base(d_model, h_slots) + d * d_model + j] * Scratch[mamba_base + j];
                }
                // Store m_inner + W_out*m_inner in hist_ctx temp, then copy back
                Scratch[hctx_base + d] = Scratch[mamba_base + d] + wout_sum;
            }
            workgroupBarrier();
            // Copy final M_t back to mamba_base
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                Scratch[mamba_base + d] = Scratch[hctx_base + d];
            }
            workgroupBarrier();
        }
    }

    if (ENABLE_DEBUG_METRICS && tid == 0u) {
        let tokens = max(1.0, f32(shape.token_count));
        let slot_base = 32u + slot_idx * 5u;
        DebugLog[slot_base + 0u] = max_delta_seen;
        DebugLog[slot_base + 1u] = f32(atomicLoad(&hit_count));
        DebugLog[slot_base + 2u] = f32(total_iters) / tokens;
        DebugLog[slot_base + 3u] = max_contractivity;
        DebugLog[slot_base + 4u] = max_h_seen;
        if (slot_idx == 0u) {
            DebugLog[8] = 901.0;
            DebugLog[10] = f32(shape.token_count);
            DebugLog[11] = f32(h_slots);
        }
    }
}
