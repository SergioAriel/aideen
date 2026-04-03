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
override ENABLE_HIST_V2_MINIMAL: bool = false;
override ENABLE_TOKEN_CARRY: bool = true;

@group(0) @binding(0) var<uniform> shape: RunUniforms;
@group(0) @binding(1) var<storage, read> S_in: array<f32>;
@group(0) @binding(2) var<storage, read> AllWeights: array<f32>;
@group(0) @binding(3) var<storage, read_write> H_curr: array<f32>;
@group(0) @binding(4) var<storage, read_write> H_next: array<f32>;
@group(0) @binding(5) var<storage, read_write> Scratch: array<f32>;
@group(0) @binding(6) var<storage, read_write> H_pooled: array<f32>;
@group(0) @binding(7) var<storage, read_write> DebugLog: array<f32>;
@group(0) @binding(8) var<storage, read_write> HistCtx: array<f32>;
@group(0) @binding(9) var<storage, read_write> MState: array<f32>;

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

const WG_SIZE: u32 = 256u;
const MAX_SLOTS: u32 = 8u;
var<workgroup> shared_vals: array<f32, WG_SIZE>;
var<workgroup> hit_count: atomic<u32>;
var<workgroup> max_delta_seen: f32;
var<workgroup> last_delta: f32;
var<workgroup> curr_contractivity: f32;
var<workgroup> max_h_seen: f32;

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
    let total_elements = h_slots * d_model;
    let slot_off = slot_idx * d_model;
    let h_base = batch_idx * total_elements;
    let signal_span = d_model * h_slots;
    let scratch_stride = select(signal_span, signal_span * 4u, ENABLE_SLOT_QKV_PROBE);
    let zero_win_diag = shape.diag_zero_win != 0u;
    let iter_limit = select(shape.max_iters, 1u, shape.diag_one_iter != 0u);
    let inv_d_model = 1.0 / max(1.0, f32(d_model));
    let hist_base = aw_hist_base(d_model, h_slots);
    let hist_slot_scale_base = hist_base + d_model * d_model;
    let hist_bias_base = hist_slot_scale_base + h_slots * d_model;
    let hist_gate_base = hist_bias_base + h_slots * d_model;
    let slot_anchor_base = hist_gate_base + h_slots;
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
        let batch_scratch_t = (batch_idx * shape.seq_len + global_t) * scratch_stride;
        let q_base = batch_scratch_t;
        let k_base = q_base + signal_span;
        let v_base = k_base + signal_span;
        let signal_base = select(batch_scratch_t, v_base + signal_span, ENABLE_SLOT_QKV_PROBE) + slot_off;
        let h_base_t = (batch_idx * shape.seq_len + global_t) * total_elements;
        let s_in_base = (batch_idx * shape.seq_len + global_t) * d_model;
        let win_base = slot_idx * d_model * d_model;
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
        if (global_t == 0u || !ENABLE_TOKEN_CARRY) {
            var local_sumsq = 0.0;
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                var sig = Scratch[signal_base + d];
                if (ENABLE_HIST_V2_MINIMAL) {
                    sig = sig + AllWeights[hist_base + slot_anchor_base + slot_off + d];
                }
                local_sumsq = local_sumsq + sig * sig;
            }
            shared_vals[tid] = local_sumsq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let sig_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                var sig = Scratch[signal_base + d];
                if (ENABLE_HIST_V2_MINIMAL) {
                    sig = sig + AllWeights[hist_base + slot_anchor_base + slot_off + d];
                }
                H_curr[h_base + slot_off + d] = sig / max(sig_rms, 1e-6);
            }
        }

        if (ENABLE_SLOT_QKV_PROBE) {
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
                Scratch[q_base + slot_off + d_out] = q;
                Scratch[k_base + slot_off + d_out] = k;
                Scratch[v_base + slot_off + d_out] = v;
            }
        }
        var iter = 0u;
        var converged = false;
        while (iter < iter_limit && !converged) {
            var local_max_delta = 0.0;
            var local_sumsq = 0.0;
            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                let hist_ctx = select(0.0, HistCtx[h_base_t + slot_off + d], ENABLE_HIST_V2_MINIMAL);
                let slot_bias = select(0.0, AllWeights[hist_base + slot_anchor_base + slot_off + d], ENABLE_HIST_V2_MINIMAL);
                let pre = Scratch[signal_base + d] + H_curr[h_base + slot_off + d] + hist_ctx + slot_bias;
                local_sumsq = local_sumsq + pre * pre;
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

            for (var d = tid; d < d_model; d = d + WG_SIZE) {
                let h_prev = H_curr[h_base + slot_off + d];
                let hist_ctx = select(0.0, HistCtx[h_base_t + slot_off + d], ENABLE_HIST_V2_MINIMAL);
                let slot_bias = select(0.0, AllWeights[hist_base + slot_anchor_base + slot_off + d], ENABLE_HIST_V2_MINIMAL);
                let pre = Scratch[signal_base + d] + h_prev + hist_ctx + slot_bias;
                let f_h = AllWeights[aw_nscale + d] * (pre / rms);
                let val = shape.damping * f_h + (1.0 - shape.damping) * h_prev;
                local_max_delta = max(local_max_delta, abs(val - h_prev));
                H_curr[h_base + slot_off + d] = val;
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
            let d = last_delta;
            max_delta_seen = max(max_delta_seen, d);
            if (!converged) {
                atomicAdd(&hit_count, 1u);
            }
        }

        var local_final_max_h = 0.0;
        for (var d = tid; d < d_model; d = d + WG_SIZE) {
            let h_val = H_curr[h_base + slot_off + d];
            H_next[h_base_t + slot_off + d] = h_val;
            local_final_max_h = max(local_final_max_h, abs(h_val));
        }
        if (ENABLE_DEBUG_METRICS) {
            shared_vals[tid] = local_final_max_h;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
                }
                workgroupBarrier();
            }
            if (tid == 0u) {
                max_h_seen = max(max_h_seen, shared_vals[0]);
            }
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
