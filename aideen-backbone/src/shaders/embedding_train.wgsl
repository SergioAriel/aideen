struct EmbeddingParams {
    d_model: u32,
    vocab_size: u32,
    seq_len: u32,
    ctx_len: u32,
    lr_bits: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: EmbeddingParams;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> embeddings: array<f32>;
@group(0) @binding(3) var<storage, read_write> seq_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> query_out: array<f32>;
@group(0) @binding(5) var<storage, read> dl_dh: array<f32>;

var<workgroup> local_sums: array<f32, 256>;
var<workgroup> shared_inv_norm: f32;

@compute
@workgroup_size(1, 64, 1)
fn embedding_gather_seq(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tok_pos = gid.x;
    let d = gid.y * 64u + lid.y;
    if (tok_pos >= params.seq_len || d >= params.d_model) {
        return;
    }
    let tok = min(token_ids[tok_pos], params.vocab_size - 1u);
    let src = tok * params.d_model + d;
    let dst = tok_pos * params.d_model + d;
    seq_out[dst] = embeddings[src];
}

@compute
@workgroup_size(256, 1, 1)
fn embedding_build_query(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let denom = f32(max(params.ctx_len, 1u));
    var partial_sq_sum = 0.0;

    for (var d = tid; d < params.d_model; d = d + 256u) {
        var acc = 0.0;
        for (var p = 0u; p < params.seq_len; p = p + 1u) {
            let pos_weight = f32(p + 1u) / denom;
            acc = acc + seq_out[p * params.d_model + d] * pos_weight;
        }
        query_out[d] = acc;
        partial_sq_sum = partial_sq_sum + acc * acc;
    }

    local_sums[tid] = partial_sq_sum;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (tid < stride) {
            local_sums[tid] = local_sums[tid] + local_sums[tid + stride];
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride / 2u;
    }

    if (tid == 0u) {
        let norm = sqrt(max(local_sums[0], 1e-12));
        shared_inv_norm = 1.0 / norm;
    }
    workgroupBarrier();

    for (var d = tid; d < params.d_model; d = d + 256u) {
        query_out[d] = query_out[d] * shared_inv_norm;
    }
}

@compute
@workgroup_size(256, 1, 1)
fn embedding_sgd_update(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let lr = bitcast<f32>(params.lr_bits);
    let denom = f32(max(params.ctx_len, 1u));

    for (var p = 0u; p < params.ctx_len; p = p + 1u) {
        let tok = min(token_ids[p], params.vocab_size - 1u);
        let pos_weight = f32(p + 1u) / denom;
        // Escala mayor para compensar la falta de momento respecto a AdamW
        let scale = lr * 5.0 * pos_weight;
        for (var d = tid; d < params.d_model; d = d + 256u) {
            let idx = tok * params.d_model + d;
            embeddings[idx] = embeddings[idx] - scale * dl_dh[d];
        }
        workgroupBarrier();
    }
}
