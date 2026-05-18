struct ExactForwardParams {
    d_model: u32,
    vocab_size: u32,
    seq_len: u32,
    vocab_start: u32,
    vocab_chunk: u32,
    _pad0: vec3<u32>,
};

@group(0) @binding(0) var<uniform> params: ExactForwardParams;
@group(0) @binding(1) var<storage, read> h_pooled: array<f32>;
@group(0) @binding(2) var<storage, read> w_lm: array<f32>;
@group(0) @binding(3) var<storage, read> b_lm: array<f32>;
@group(0) @binding(4) var<storage, read> g_lm: array<f32>;
@group(0) @binding(5) var<storage, read_write> logits_out: array<f32>;

fn logits_idx(t: u32, k: u32) -> u32 {
    return t * params.vocab_chunk + k;
}

@compute @workgroup_size(8, 8, 1)
fn lm_exact_forward_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let t = gid.y;
    if (k >= params.vocab_chunk || t >= params.seq_len) {
        return;
    }

    let vocab_row = params.vocab_start + k;
    let d_model = params.d_model;
    let h_base = t * d_model;
    let w_base = vocab_row * d_model;

    var sq_sum = 0.0;
    for (var d = 0u; d < d_model; d = d + 1u) {
        let h = h_pooled[h_base + d];
        sq_sum = sq_sum + h * h;
    }
    let rms = sqrt(sq_sum / max(1.0, f32(d_model)) + 1e-5);

    var acc = b_lm[vocab_row];
    for (var d = 0u; d < d_model; d = d + 1u) {
        let h_norm = (h_pooled[h_base + d] / rms) * g_lm[d];
        acc = acc + w_lm[w_base + d] * h_norm;
    }
    logits_out[logits_idx(t, k)] = acc;
}
