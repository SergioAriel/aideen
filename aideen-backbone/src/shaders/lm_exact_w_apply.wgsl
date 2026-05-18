struct ExactWApplyParams {
    num_weights: u32,
    step_t: u32,
    groups_x: u32,
    _pad0: u32,
    lr_bits: u32,
    beta1_bits: u32,
    beta2_bits: u32,
    eps_bits: u32,
    grad_scale_bits: u32,
    _pad1: vec3<u32>,
};

@group(0) @binding(0) var<uniform> params: ExactWApplyParams;
@group(0) @binding(1) var<storage, read_write> w_lm: array<f32>;
@group(0) @binding(2) var<storage, read_write> moments_w: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> grad_w: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn lm_exact_w_apply_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * params.groups_x * 256u;
    if (idx >= params.num_weights) {
        return;
    }

    let lr = bitcast<f32>(params.lr_bits);
    let beta1 = bitcast<f32>(params.beta1_bits);
    let beta2 = bitcast<f32>(params.beta2_bits);
    let eps = bitcast<f32>(params.eps_bits);
    let grad_scale = bitcast<f32>(params.grad_scale_bits);
    let t_f = max(1.0, f32(params.step_t));
    let bc1 = 1.0 - pow(beta1, t_f);
    let bc2 = 1.0 - pow(beta2, t_f);

    let g = grad_w[idx] * grad_scale;
    let mv = moments_w[idx];
    let m_new = beta1 * mv.x + (1.0 - beta1) * g;
    let v_new = beta2 * mv.y + (1.0 - beta2) * g * g;
    moments_w[idx] = vec2<f32>(m_new, v_new);
    let m_hat = m_new / bc1;
    let v_hat = v_new / bc2;
    w_lm[idx] = w_lm[idx] - lr * m_hat / (sqrt(v_hat) + eps);
}
