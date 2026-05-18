struct ExactTokenParams {
    d_model: u32,
    vocab_size: u32,
    token_index: u32,
    target_index: u32,
    seq_scale_bits: u32,
    _pad1: array<vec4<u32>, 4>,
};

@group(0) @binding(0) var<uniform> params: ExactTokenParams;
@group(0) @binding(1) var<storage, read> h_seq: array<f32>;
@group(0) @binding(2) var<storage, read> w_lm: array<f32>;
@group(0) @binding(3) var<storage, read> b_lm: array<f32>;
@group(0) @binding(4) var<storage, read> g_lm: array<f32>;
@group(0) @binding(5) var<storage, read_write> logits_out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn lm_exact_token_logits_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= params.vocab_size) {
        return;
    }
    let base = params.token_index * params.d_model;
    var mean_sq = 0.0;
    for (var d = 0u; d < params.d_model; d = d + 1u) {
        let h = h_seq[base + d];
        mean_sq = mean_sq + h * h;
    }
    let rms = sqrt(mean_sq / f32(params.d_model) + 1e-5);
    var logit = b_lm[v];
    for (var d = 0u; d < params.d_model; d = d + 1u) {
        let h_norm = h_seq[base + d] / rms;
        let idx = d * params.vocab_size + v;
        logit = logit + w_lm[idx] * (h_norm * g_lm[d]);
    }
    logits_out[v] = logit;
}
