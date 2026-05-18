struct ExactAccumParams {
    d_model: u32,
    vocab_size: u32,
    token_index: u32,
    _pad0: u32,
    rms_bits: u32,
    _pad1: array<vec4<u32>, 3>,
};

@group(0) @binding(0) var<uniform> params: ExactAccumParams;
@group(0) @binding(1) var<storage, read> dl_scaled: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_b: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn lm_exact_accum_bias_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= params.vocab_size) {
        return;
    }
    grad_b[v] = grad_b[v] + dl_scaled[v];
}
