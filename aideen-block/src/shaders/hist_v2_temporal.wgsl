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
@group(0) @binding(2) var<storage, read> AllWeights: array<f32>;
@group(0) @binding(4) var<storage, read_write> H_next: array<f32>;
@group(0) @binding(9) var<storage, read_write> MState: array<f32>;

const WG_SIZE: u32 = 256u;
var<workgroup> shared_vals: array<f32, WG_SIZE>;

fn aw_wo_base(d: u32, h: u32) -> u32 { return 3u * h * d * d + 2u * h * d; }
fn aw_win_base(d: u32, h: u32) -> u32 { return aw_wo_base(d, h) + h * d * d; }
fn aw_wx_base(d: u32, h: u32) -> u32 { return aw_win_base(d, h) + h * d * d; }
fn aw_wout_base(d: u32, h: u32) -> u32 { return aw_wx_base(d, h) + d * d; }
fn aw_alog_base(d: u32, h: u32) -> u32 { return aw_wout_base(d, h) + d * d; }

@compute @workgroup_size(256, 1, 1)
fn hist_v2_temporal_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.x;
    let slot_idx = wid.y;
    if (batch_idx >= shape.batch_size || slot_idx >= shape.h_slots) {
        return;
    }

    let d_model = shape.d_model;
    let h_slots = shape.h_slots;
    let total_elements = h_slots * d_model;
    let slot_off = slot_idx * d_model;
    let m_base = batch_idx * total_elements + slot_off;
    let aw_wx = aw_wx_base(d_model, h_slots);
    let aw_wout = aw_wout_base(d_model, h_slots);
    let aw_alog = aw_alog_base(d_model, h_slots);

    for (var t = 0u; t < shape.token_count; t = t + 1u) {
        let global_t = shape.token_start + t;
        if (global_t >= shape.seq_len) {
            continue;
        }
        let h_base_t = (batch_idx * shape.seq_len + global_t) * total_elements + slot_off;

        var local_h_sumsq = 0.0;
        for (var d = tid; d < d_model; d = d + WG_SIZE) {
            let h_val = H_next[h_base_t + d];
            local_h_sumsq = local_h_sumsq + h_val * h_val;
        }
        shared_vals[tid] = local_h_sumsq;
        workgroupBarrier();
        for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
            if (tid < stride) {
                shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
            }
            workgroupBarrier();
        }
        let h_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);

        for (var d = tid; d < d_model; d = d + WG_SIZE) {
            let m_prev = MState[m_base + d];
            let a_bar = 1.0 / (1.0 + exp(AllWeights[aw_alog + slot_off + d]));
            let b_bar = 1.0 - a_bar;
            let h_unit_d = H_next[h_base_t + d] / max(h_rms, 1e-6);
            let wx_diag = 0.5 * tanh(AllWeights[aw_wx + d * d_model + d]);
            let x_proj_d = h_unit_d + wx_diag * h_unit_d;
            let m_next_d = a_bar * m_prev + b_bar * x_proj_d;

            var out_proj = 0.0;
            for (var j = 0u; j < d_model; j = j + 1u) {
                let a_bar_j = 1.0 / (1.0 + exp(AllWeights[aw_alog + slot_off + j]));
                let b_bar_j = 1.0 - a_bar_j;
                let h_unit_j = H_next[h_base_t + j] / max(h_rms, 1e-6);
                let wx_diag_j = 0.5 * tanh(AllWeights[aw_wx + j * d_model + j]);
                let x_proj_j = h_unit_j + wx_diag_j * h_unit_j;
                let m_next_j = a_bar_j * MState[m_base + j] + b_bar_j * x_proj_j;
                out_proj = out_proj + AllWeights[aw_wout + j * d_model + d] * m_next_j;
            }
            MState[m_base + d] = m_next_d + out_proj;
        }
        workgroupBarrier();
    }
}
