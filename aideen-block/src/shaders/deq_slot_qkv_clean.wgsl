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
@group(0) @binding(3) var<storage, read_write> H_curr: array<f32>;
@group(0) @binding(5) var<storage, read_write> Scratch: array<f32>;

override SLOT_ATTN_HEAD_DIM: u32 = 32u;
const WG_SIZE: u32 = 256u;

fn aw_wq_base(d: u32, h: u32) -> u32 { return 0u; }
fn aw_wk_base(d: u32, h: u32) -> u32 { return h * d * d + h * d; }
fn aw_wv_base(d: u32, h: u32) -> u32 { return aw_wk_base(d, h) + h * d * d + h * d; }

@compute @workgroup_size(256, 1, 1)
fn deq_slot_qkv_main(
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
    let slot_off = slot_idx * d_model;
    let h_base = batch_idx * total_elements;
    let signal_span = d_model * h_slots;
    let scratch_stride = signal_span * 5u;
    let global_t = shape.token_start;
    let batch_scratch_t = (batch_idx * shape.seq_len + global_t) * scratch_stride;
    let q_base = batch_scratch_t + signal_span + slot_off;
    let k_base = batch_scratch_t + signal_span * 2u + slot_off;
    let v_base = batch_scratch_t + signal_span * 3u + slot_off;
    let wq_mat_base = aw_wq_base(d_model, h_slots) + slot_idx * d_model * d_model;
    let wk_mat_base = aw_wk_base(d_model, h_slots) + slot_idx * d_model * d_model;
    let wv_mat_base = aw_wv_base(d_model, h_slots) + slot_idx * d_model * d_model;
    let wq_bias_base = aw_wq_base(d_model, h_slots) + h_slots * d_model * d_model + slot_idx * d_model;
    let wk_bias_base = aw_wk_base(d_model, h_slots) + h_slots * d_model * d_model + slot_idx * d_model;

    for (var d_out = tid; d_out < head_dim; d_out = d_out + WG_SIZE) {
        var q = AllWeights[wq_bias_base + d_out];
        var k = AllWeights[wk_bias_base + d_out];
        var v = 0.0;
        for (var j = 0u; j < d_model; j = j + 1u) {
            let h_val = H_curr[h_base + slot_off + j];
            q = q + AllWeights[wq_mat_base + j * d_model + d_out] * h_val;
            k = k + AllWeights[wk_mat_base + j * d_model + d_out] * h_val;
            v = v + AllWeights[wv_mat_base + j * d_model + d_out] * h_val;
        }
        Scratch[q_base + d_out] = q;
        Scratch[k_base + d_out] = k;
        Scratch[v_base + d_out] = v;
    }
}
