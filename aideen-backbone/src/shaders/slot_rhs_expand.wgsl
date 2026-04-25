struct SlotRhsParams {
    d_model: u32,
    h_slots: u32,
    seq_len: u32,
    batch_size: u32,
}

@group(0) @binding(0) var<uniform> params: SlotRhsParams;
@group(0) @binding(1) var<storage, read> pooled_rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhs_slot_buf: array<f32>;
@group(0) @binding(3) var<storage, read> assoc_read: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn slot_rhs_expand_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let entry = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (dim >= d || entry >= n_entries) { return; }

    let token = entry / h_slots;
    let pooled_idx = token * d + dim;
    let slot = entry % h_slots;
    var weight_sum = 0.0;
    for (var s = 0u; s < h_slots; s = s + 1u) {
        let slot_base = (token * h_slots + s) * d;
        var assoc_sq = 0.0;
        for (var j = 0u; j < d; j = j + 1u) {
            let assoc_j = assoc_read[slot_base + j];
            assoc_sq = assoc_sq + assoc_j * assoc_j;
        }
        weight_sum = weight_sum + sqrt(assoc_sq / max(1.0, f32(d)) + 1.0e-6);
    }
    let slot_base = (token * h_slots + slot) * d;
    var slot_sq = 0.0;
    for (var j = 0u; j < d; j = j + 1u) {
        let assoc_j = assoc_read[slot_base + j];
        slot_sq = slot_sq + assoc_j * assoc_j;
    }
    let slot_scale = sqrt(slot_sq / max(1.0, f32(d)) + 1.0e-6) / max(weight_sum, 1.0e-6);
    rhs_slot_buf[entry * d + dim] = pooled_rhs[pooled_idx] * slot_scale;
}
