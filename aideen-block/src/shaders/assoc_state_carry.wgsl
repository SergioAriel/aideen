
struct CarryParams {
    d_model: u32,
    h_slots: u32,
    batch_size: u32,
    full_seq_len: u32,
    token_start: u32,
    token_count: u32,
};

@group(0) @binding(0) var<uniform> params: CarryParams;
@group(0) @binding(1) var<storage, read> HistCtx: array<f32>;
@group(0) @binding(2) var<storage, read_write> MState: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn assoc_state_carry_main(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let d_idx = global_id.x;
    let batch_idx = global_id.y;
    
    if (d_idx >= params.d_model || batch_idx >= params.batch_size) {
        return;
    }
    
    let slots = params.h_slots;
    let d_model = params.d_model;
    let full_seq_len = params.full_seq_len;
    let token_start = params.token_start;
    let token_count = params.token_count;
    
    // We want the LAST token of the current segment
    let global_t = token_start + (token_count - 1u);
    
    // Layout: [batch][global_t][slot][dim]
    let base_idx = (batch_idx * full_seq_len + global_t) * (slots * d_model);
    let target_base = batch_idx * (slots * d_model);
    
    for (var s = 0u; s < slots; s = s + 1u) {
        let src_idx = base_idx + s * d_model + d_idx;
        let dst_idx = target_base + s * d_model + d_idx;
        MState[dst_idx] = HistCtx[src_idx];
    }
}
