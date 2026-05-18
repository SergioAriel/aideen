struct ExactBgApplyParams {
    vocab_size: u32,
    d_model: u32,
    step_t: u32,
    groups_x: u32,
    lr_bits: u32,
    beta1_bits: u32,
    beta2_bits: u32,
    eps_bits: u32,
    grad_scale_bits: u32,
    _pad1: vec3<u32>,
};

@group(0) @binding(0) var<uniform> params: ExactBgApplyParams;
@group(0) @binding(1) var<storage, read_write> b_lm: array<f32>;
@group(0) @binding(2) var<storage, read_write> moments_b: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> g_lm: array<f32>;
@group(0) @binding(4) var<storage, read_write> moments_g: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> grad_b: array<f32>;
@group(0) @binding(6) var<storage, read> grad_g: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn lm_exact_bg_apply_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.groups_x * 256u;
    let lr = bitcast<f32>(params.lr_bits);
    let beta1 = bitcast<f32>(params.beta1_bits);
    let beta2 = bitcast<f32>(params.beta2_bits);
    let eps = bitcast<f32>(params.eps_bits);
    let grad_scale = bitcast<f32>(params.grad_scale_bits);
    let t_f = max(1.0, f32(params.step_t));
    let bc1 = 1.0 - pow(beta1, t_f);
    let bc2 = 1.0 - pow(beta2, t_f);

    if (idx < params.vocab_size) {
        let gb = grad_b[idx] * grad_scale;
        let mvb = moments_b[idx];
        let m_new = beta1 * mvb.x + (1.0 - beta1) * gb;
        let v_new = beta2 * mvb.y + (1.0 - beta2) * gb * gb;
        moments_b[idx] = vec2<f32>(m_new, v_new);
        let m_hat = m_new / bc1;
        let v_hat = v_new / bc2;
        b_lm[idx] = b_lm[idx] - lr * m_hat / (sqrt(v_hat) + eps);
    }

    if (idx < params.d_model) {
        let gg = grad_g[idx] * grad_scale;
        let mvg = moments_g[idx];
        let m_new = beta1 * mvg.x + (1.0 - beta1) * gg;
        let v_new = beta2 * mvg.y + (1.0 - beta2) * gg * gg;
        moments_g[idx] = vec2<f32>(m_new, v_new);
        let m_hat = m_new / bc1;
        let v_hat = v_new / bc2;
        g_lm[idx] = g_lm[idx] - lr * m_hat / (sqrt(v_hat) + eps);
    }
}
