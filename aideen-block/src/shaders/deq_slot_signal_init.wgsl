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

const WG_SIZE: u32 = 256u;
var<workgroup> shared_vals: array<f32, WG_SIZE>;

fn aw_wq_base(d: u32, h: u32) -> u32 { return 0u; }
fn aw_wk_base(d: u32, h: u32) -> u32 { return h * d * d + h * d; }
fn aw_wv_base(d: u32, h: u32) -> u32 { return aw_wk_base(d, h) + h * d * d + h * d; }
fn aw_wo_base(d: u32, h: u32) -> u32 { return aw_wv_base(d, h) + h * d * d; }
fn aw_win_base(d: u32, h: u32) -> u32 { return aw_wo_base(d, h) + h * d * d; }

@compute @workgroup_size(256, 1, 1)
fn deq_slot_signal_init_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.x;
    let slot_idx = wid.y;
    if (batch_idx >= shape.batch_size || slot_idx >= shape.h_slots) { return; }

    let d_model = shape.d_model;
    let h_slots = shape.h_slots;
    let total_elements = h_slots * d_model;
    let slot_off = slot_idx * d_model;
    let h_base = batch_idx * total_elements;
    let signal_span = d_model * h_slots;
    let scratch_stride = signal_span * 5u;
    let global_t = shape.token_start;
    let batch_scratch_t = (batch_idx * shape.seq_len + global_t) * scratch_stride;
    let signal_base = batch_scratch_t + slot_off;
    let s_in_base = (batch_idx * shape.seq_len + global_t) * d_model;
    let win_base = aw_win_base(d_model, h_slots) + slot_idx * d_model * d_model;
    let zero_win_diag = shape.diag_zero_win != 0u;

    for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
        var inj = 0.0;
        if (!zero_win_diag) {
            for (var j = 0u; j < d_model; j = j + 1u) {
                inj = inj + AllWeights[win_base + j * d_model + d_out] * S_in[s_in_base + j];
            }
        }
        Scratch[signal_base + d_out] = inj;
    }
    workgroupBarrier();

    if (global_t == 0u) {
        var local_sumsq = 0.0;
        for (var d = tid; d < d_model; d = d + WG_SIZE) {
            let sig = Scratch[signal_base + d];
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
            H_curr[h_base + slot_off + d] = Scratch[signal_base + d] / max(sig_rms, 1e-6);
        }
    }
}
