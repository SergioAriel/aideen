struct EmbeddingParams {
    d_model: u32,
    vocab_size: u32,
    seq_len: u32,
    ctx_len: u32,
    lr_bits: u32,
    beta1_bits: u32,
    beta2_bits: u32,
    eps_bits: u32,
    step_t: u32,
    ternary_flag: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: EmbeddingParams;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> embeddings: array<f32>;
@group(0) @binding(3) var<storage, read_write> seq_out: array<f32>;
@group(0) @binding(4) var<storage, read_write> query_out: array<f32>;
@group(0) @binding(5) var<storage, read> dl_dh: array<f32>;
@group(0) @binding(6) var<storage, read_write> m_emb: array<f32>;
@group(0) @binding(7) var<storage, read_write> v_emb: array<f32>;

const WG_SIZE: u32 = 256u;
var<workgroup> local_sums: array<f32, WG_SIZE>;
var<workgroup> shared_inv_norm: f32;

@compute
@workgroup_size(1, 64, 1)
fn embedding_gather_seq(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tok_pos = gid.x;
    let d = gid.y;
    if (tok_pos >= params.seq_len || d >= params.d_model) {
        return;
    }
    let tok = min(token_ids[tok_pos], params.vocab_size - 1u);
    let src = tok * params.d_model + d;
    let dst = tok_pos * params.d_model + d;
    // STE forward pass: quantize on-the-fly, never overwrite latent weights.
    if (params.ternary_flag == 1u) {
        let w = embeddings[src];
        if (abs(w) > 0.05) { seq_out[dst] = sign(w); }
        else { seq_out[dst] = 0.0; }
    } else {
        seq_out[dst] = embeddings[src];
    }
}

@compute
@workgroup_size(256, 1, 1)
fn embedding_build_query(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;

    let L = min(params.seq_len, params.ctx_len);
    let denom = f32(max(params.ctx_len, 1u));
    let log_denom = log2(denom + 1.0f);

    var partial_sq_sum = 0.0;

    for (var d = tid; d < params.d_model; d = d + WG_SIZE) {
        var acc = 0.0;
        for (var p = 0u; p < L; p = p + 1u) {
            // Logarithmic positional weighting for better long-range context (Genesis Audit)
            let pos_weight = log2(f32(p) + 2.0f) / log_denom;
            acc = acc + seq_out[p * params.d_model + d] * pos_weight;
        }
        query_out[d] = acc;
        partial_sq_sum = partial_sq_sum + acc * acc;
    }

    local_sums[tid] = partial_sq_sum;
    workgroupBarrier();

    // Parallel reduction in workgroup
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            local_sums[tid] += local_sums[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let norm = sqrt(max(local_sums[0], 1e-12));
        shared_inv_norm = 1.0 / norm;
    }
    workgroupBarrier();

    for (var d = tid; d < params.d_model; d = d + WG_SIZE) {
        query_out[d] = query_out[d] * shared_inv_norm;
    }
}

// Dispatch: (tokens.len(), d_r/64, 1) workgroups, workgroup_size (1, 64, 1)
@compute
@workgroup_size(1, 64, 1)
fn embedding_adamw_update(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let p = gid.x;
    let d = gid.y;

    let L = min(params.seq_len, params.ctx_len);
    if (p >= L || d >= params.d_model) { return; }

    let v = min(token_ids[p], params.vocab_size - 1u);
    let idx = v * params.d_model + d;

    let lr    = bitcast<f32>(params.lr_bits);
    let beta1 = bitcast<f32>(params.beta1_bits);
    let beta2 = bitcast<f32>(params.beta2_bits);
    let eps   = bitcast<f32>(params.eps_bits);
    let t_f   = max(1.0, f32(params.step_t));
    let bc1   = 1.0 - pow(beta1, t_f);
    let bc2   = 1.0 - pow(beta2, t_f);

    let denom = f32(max(params.ctx_len, 1u));
    let pos_weight = log2(f32(p) + 2.0f) / log2(denom + 1.0f);
    // Normalize by seq_len (L) to prevent accumulation explosion
    let grad = (dl_dh[d] * pos_weight) / f32(max(L, 1u));

    let m_new = beta1 * m_emb[idx] + (1.0 - beta1) * grad;
    let v_new = beta2 * v_emb[idx] + (1.0 - beta2) * grad * grad;
    m_emb[idx] = m_new;
    v_emb[idx] = v_new;

    let m_hat = m_new / bc1;
    let v_hat = v_new / bc2;

    // STE: always store latent f32 weight. Quantization only in the forward pass (embedding_gather_seq).
    embeddings[idx] = embeddings[idx] - lr * m_hat / (sqrt(v_hat) + eps);
}