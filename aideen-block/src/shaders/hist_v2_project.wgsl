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
@group(0) @binding(8) var<storage, read_write> HistCtx: array<f32>;
@group(0) @binding(9) var<storage, read_write> MState: array<f32>;

const WG_SIZE: u32 = 256u;
var<workgroup> shared_vals: array<f32, WG_SIZE>;

fn aw_wo_base(d: u32, h: u32) -> u32 { return 3u * h * d * d + 2u * h * d; }
fn aw_win_base(d: u32, h: u32) -> u32 { return aw_wo_base(d, h) + h * d * d; }
fn aw_wout_base(d: u32, h: u32) -> u32 { return aw_win_base(d, h) + h * d * d + d * d; }
fn aw_alog_base(d: u32, h: u32) -> u32 { return aw_wout_base(d, h) + d * d; }
fn aw_nscale_base(d: u32, h: u32) -> u32 { return aw_alog_base(d, h) + h * d; }
fn aw_hist_base(d: u32, h: u32) -> u32 { return aw_nscale_base(d, h) + d; }

fn hist_alpha(slot_gate_logit: f32) -> f32 {
    let sigma = 1.0 / (1.0 + exp(-slot_gate_logit));
    return 0.070 + (0.20 - 0.070) * sigma;
}

@compute @workgroup_size(256, 1, 1)
fn hist_v2_project_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.x;
    let slot_idx = wid.y;
    let token_local = wid.z;
    let global_t = shape.token_start + token_local;
    if (batch_idx >= shape.batch_size || slot_idx >= shape.h_slots || token_local >= shape.token_count || global_t >= shape.seq_len) {
        return;
    }

    let d_model = shape.d_model;
    let h_slots = shape.h_slots;
    let total_elements = h_slots * d_model;
    let slot_off = slot_idx * d_model;
    let hist_base = aw_hist_base(d_model, h_slots);
    let hist_slot_scale_base = hist_base + d_model * d_model;
    let hist_bias_base = hist_slot_scale_base + h_slots * d_model;
    let hist_gate_base = hist_bias_base + h_slots * d_model;
    let hist_out = (batch_idx * shape.seq_len + global_t) * total_elements + slot_off;
    let s_in_base = (batch_idx * shape.seq_len + global_t) * d_model;
    let prev_s_in_base = (batch_idx * shape.seq_len + max(shape.token_start, global_t) - select(0u, 1u, global_t > shape.token_start)) * d_model;
    let has_prev_token = global_t > shape.token_start;
    let m_base = batch_idx * total_elements + slot_off;

    var local_inj_sumsq = 0.0;
    for (var d = tid; d < d_model; d = d + WG_SIZE) {
        var inj = 0.0;
        for (var j = 0u; j < d_model; j = j + 1u) {
            let w_idx = aw_win_base(d_model, h_slots) + j * d_model + d;
            inj = inj + AllWeights[w_idx] * S_in[s_in_base + j];
        }
        local_inj_sumsq = local_inj_sumsq + inj * inj;
    }
    shared_vals[tid] = local_inj_sumsq;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
        }
        workgroupBarrier();
    }
    let inj_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);

    var local_prev_sumsq = 0.0;
    for (var d = tid; d < d_model; d = d + WG_SIZE) {
        let m_val = MState[m_base + d];
        local_prev_sumsq = local_prev_sumsq + m_val * m_val;
    }
    shared_vals[tid] = local_prev_sumsq;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
        }
        workgroupBarrier();
    }
    let prev_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);
    let alpha = hist_alpha(AllWeights[hist_base + hist_gate_base + slot_idx]);

    var local_prev_inj_sumsq = 0.0;
    for (var d = tid; d < d_model; d = d + WG_SIZE) {
        var prev_inj = 0.0;
        if (has_prev_token) {
            for (var j = 0u; j < d_model; j = j + 1u) {
                let w_idx = aw_win_base(d_model, h_slots) + j * d_model + d;
                prev_inj = prev_inj + AllWeights[w_idx] * S_in[prev_s_in_base + j];
            }
        }
        // Reuse HistCtx as a temporary cache for the causal local projection before final writeback.
        HistCtx[hist_out + d] = prev_inj;
        local_prev_inj_sumsq = local_prev_inj_sumsq + prev_inj * prev_inj;
    }
    shared_vals[tid] = local_prev_inj_sumsq;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
        }
        workgroupBarrier();
    }
    let prev_inj_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);

    var local_u_sumsq = 0.0;
    for (var d = tid; d < d_model; d = d + WG_SIZE) {
        let prev_m = MState[m_base + d];
        let prev_unit = prev_m / max(prev_rms, 1e-6);
        var u = 0.0;
        for (var j = 0u; j < d_model; j = j + 1u) {
            let w_idx = hist_base + j * d_model + d;
            u = u + AllWeights[w_idx] * MState[m_base + j];
        }
        let slot_scale = AllWeights[hist_base + hist_slot_scale_base + slot_off + d];
        u = u + slot_scale * prev_unit;
        if (has_prev_token) {
            let prev_inj_unit = HistCtx[hist_out + d] / max(prev_inj_rms, 1e-6);
            let local_mix = AllWeights[hist_base + hist_bias_base + slot_off + d];
            u = u + 0.15 * local_mix * prev_inj_unit;
        }
        HistCtx[hist_out + d] = u;
        local_u_sumsq = local_u_sumsq + u * u;
    }
    shared_vals[tid] = local_u_sumsq;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
        }
        workgroupBarrier();
    }
    let u_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);
    let scale = select(1.0, min(inj_rms / u_rms, 1.0), u_rms > 1e-6);

    for (var d = tid; d < d_model; d = d + WG_SIZE) {
        HistCtx[hist_out + d] = HistCtx[hist_out + d] * (alpha * scale);
    }
}
