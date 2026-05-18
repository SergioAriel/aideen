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
@group(0) @binding(2) var<storage, read> g_lm: array<f32>;
@group(0) @binding(3) var<storage, read> dl_dh_rms_src: array<f32>;
@group(0) @binding(4) var<storage, read_write> dl_dh_seq: array<f32>;

@compute @workgroup_size(1, 1, 1)
fn lm_exact_accum_dldh_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
        return;
    }
    let d_model = params.d_model;
    let base = params.token_index * d_model;
    var mean_sq = 0.0;
    for (var i = 0u; i < d_model; i = i + 1u) {
        let hv = h_seq[base + i];
        mean_sq = mean_sq + hv * hv;
    }
    let rms = sqrt(mean_sq / f32(d_model) + 1e-5);
    var sum_dx_h = 0.0;
    for (var d = 0u; d < d_model; d = d + 1u) {
        let h = h_seq[base + d];
        let dx = dl_dh_rms_src[d] * g_lm[d] / rms;
        sum_dx_h = sum_dx_h + dx * h;
    }
    let corr_scale = sum_dx_h / (f32(d_model) * rms * rms);
    for (var d = 0u; d < d_model; d = d + 1u) {
        let h = h_seq[base + d];
        let dx = dl_dh_rms_src[d] * g_lm[d] / rms;
        dl_dh_seq[base + d] = dx - h * corr_scale;
    }
}
