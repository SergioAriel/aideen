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
@group(0) @binding(4) var<storage, read_write> H_next: array<f32>;
@group(0) @binding(6) var<storage, read_write> H_pooled: array<f32>;
@group(0) @binding(16) var<storage, read_write> Assoc_read: array<f32>;
@group(0) @binding(17) var<storage, read_write> Assoc_pooled: array<f32>;

const WG_SIZE: u32 = 256u;

@compute @workgroup_size(256, 1, 1)
fn deq_forward_pool_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let batch_idx = wid.y;
    let token_local = wid.z;
    let token_idx = shape.token_start + token_local;
    let d = gid.x;
    if (batch_idx >= shape.batch_size || token_local >= shape.token_count || token_idx >= shape.seq_len || d >= shape.d_model) { return; }

    let h_slots = shape.h_slots;
    let total_elements = h_slots * shape.d_model;
    let h_base = (batch_idx * shape.seq_len + token_idx) * total_elements;

    var pooled = 0.0;
    var assoc_pooled = 0.0;
    var assoc_weight_sum = 0.0;
    for (var s = 0u; s < h_slots; s = s + 1u) {
        let slot_base = h_base + s * shape.d_model;
        pooled = pooled + H_next[h_base + s * shape.d_model + d];
        var assoc_sq = 0.0;
        for (var j = 0u; j < shape.d_model; j = j + 1u) {
            let assoc_j = Assoc_read[slot_base + j];
            assoc_sq = assoc_sq + assoc_j * assoc_j;
        }
        let slot_weight = sqrt(assoc_sq / max(1.0, f32(shape.d_model)) + 1.0e-6);
        assoc_pooled = assoc_pooled + slot_weight * Assoc_read[slot_base + d];
        assoc_weight_sum = assoc_weight_sum + slot_weight;
    }
    H_pooled[(batch_idx * shape.seq_len + token_idx) * shape.d_model + d] = pooled / max(1.0, f32(h_slots));
    Assoc_pooled[(batch_idx * shape.seq_len + token_idx) * shape.d_model + d] =
        assoc_pooled / max(assoc_weight_sum, 1.0e-6);
}
