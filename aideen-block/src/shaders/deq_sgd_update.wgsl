struct UpdateParams {
    mat_len: u32,
    vec_len: u32,
    _pad0: u32,
    _pad1: u32,
    lr: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
};

@group(0) @binding(0) var<uniform> params: UpdateParams;
@group(0) @binding(1) var<storage, read> grad_mat: array<f32>;
@group(0) @binding(2) var<storage, read> grad_vec: array<f32>;

@group(0) @binding(3) var<storage, read_write> W_q: array<f32>;
@group(0) @binding(4) var<storage, read_write> W_k: array<f32>;
@group(0) @binding(5) var<storage, read_write> W_v: array<f32>;
@group(0) @binding(6) var<storage, read_write> W_o: array<f32>;
@group(0) @binding(7) var<storage, read_write> W_in: array<f32>;
@group(0) @binding(8) var<storage, read_write> W_x: array<f32>;
@group(0) @binding(9) var<storage, read_write> W_out: array<f32>;
@group(0) @binding(10) var<storage, read_write> A_log: array<f32>;
@group(0) @binding(11) var<storage, read_write> NormScale: array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn deq_sgd_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx < params.mat_len) {
        let g_raw = grad_mat[idx];
        let g = select(0.0, g_raw, (g_raw == g_raw) && (abs(g_raw) < 1.0e30));
        let clamped_g = clamp(g, -0.1, 0.1);
        let delta = params.lr * clamped_g;
        
        W_q[idx] = clamp(W_q[idx] - delta, -10.0, 10.0);
        W_k[idx] = clamp(W_k[idx] - delta, -10.0, 10.0);
        W_v[idx] = clamp(W_v[idx] - delta, -10.0, 10.0);
        W_o[idx] = clamp(W_o[idx] - delta, -10.0, 10.0);
        W_in[idx] = clamp(W_in[idx] - delta, -10.0, 10.0);
        W_x[idx] = clamp(W_x[idx] - delta, -10.0, 10.0);
        W_out[idx] = clamp(W_out[idx] - delta, -10.0, 10.0);
    }

    if (idx < params.vec_len) {
        let g_raw = grad_vec[idx];
        let g = select(0.0, g_raw, (g_raw == g_raw) && (abs(g_raw) < 1.0e30));
        let clamped_g = clamp(g, -0.1, 0.1);
        let delta = params.lr * clamped_g;
        
        A_log[idx] = clamp(A_log[idx] - delta, -10.0, 10.0);
        NormScale[idx] = clamp(NormScale[idx] - delta, -10.0, 10.0);
    }
}
