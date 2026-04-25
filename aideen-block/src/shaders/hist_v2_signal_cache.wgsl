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
@group(0) @binding(2) var<storage, read> AllWeights: array<f32>;
@group(0) @binding(10) var<storage, read_write> SignalCache: array<f32>;

const WG_SIZE: u32 = 256u;

fn aw_wo_base(d: u32, h: u32) -> u32 { return 3u * h * d * d + 2u * h * d; }
fn aw_win_base(d: u32, h: u32) -> u32 { return aw_wo_base(d, h) + h * d * d; }

@compute @workgroup_size(256, 1, 1)
fn hist_v2_signal_cache_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.x;
    let token_local = wid.y;
    let global_t = shape.token_start + token_local;
    if (batch_idx >= shape.batch_size || token_local >= shape.token_count || global_t >= shape.seq_len) {
        return;
    }

    let d_model = shape.d_model;
    let h_slots = shape.h_slots;
    let s_in_base = (batch_idx * shape.seq_len + global_t) * d_model;
    let cache_base = (batch_idx * shape.seq_len + global_t) * d_model;

    for (var d = tid; d < d_model; d = d + WG_SIZE) {
        var inj = 0.0;
        for (var j = 0u; j < d_model; j = j + 1u) {
            let w_idx = aw_win_base(d_model, h_slots) + j * d_model + d;
            inj = inj + AllWeights[w_idx] * S_in[s_in_base + j];
        }
        SignalCache[cache_base + d] = inj;
    }
}
