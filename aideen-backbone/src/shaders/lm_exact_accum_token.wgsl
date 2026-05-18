struct ExactAccumParams {
    d_model: u32,
    vocab_size: u32,
    token_index: u32,
    _pad0: u32,
    rms_bits: u32,
    _pad1: array<vec4<u32>, 3>,
};

@group(0) @binding(0) var<uniform> params: ExactAccumParams;
@group(0) @binding(1) var<storage, read> h_seq: array<f32>;
@group(0) @binding(2) var<storage, read> w_lm: array<f32>;
@group(0) @binding(3) var<storage, read> g_lm: array<f32>;
@group(0) @binding(4) var<storage, read> dl_scaled: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_w: array<f32>;
@group(0) @binding(6) var<storage, read_write> grad_g: array<f32>;
@group(0) @binding(7) var<storage, read_write> dl_dh_rms_tmp: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn lm_exact_accum_token_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= params.d_model) {
        return;
    }
    let base = params.token_index * params.d_model;
    var mean_sq = 0.0;
    for (var i = 0u; i < params.d_model; i = i + 1u) {
        let hv = h_seq[base + i];
        mean_sq = mean_sq + hv * hv;
    }
    let rms = sqrt(mean_sq / f32(params.d_model) + 1e-5);
    let h = h_seq[base + d];
    let h_norm = h / rms;
    let h_rms = h_norm * g_lm[d];
    let vocab = params.vocab_size;
    let col_base = d * vocab;
    var accum_d = 0.0;
    for (var v = 0u; v < vocab; v = v + 1u) {
        let dlv = dl_scaled[v];
        let idx = col_base + v;
        grad_w[idx] = grad_w[idx] + dlv * h_rms;
        accum_d = accum_d + w_lm[idx] * dlv;
    }
    dl_dh_rms_tmp[d] = accum_d;
    grad_g[d] = grad_g[d] + accum_d * h_norm;
}
