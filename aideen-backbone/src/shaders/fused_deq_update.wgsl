struct UpdateUniforms {
    d_model: u32,
    lr: f32,
    grad_scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: UpdateUniforms;
@group(0) @binding(1) var<storage, read> v_adjoint: array<f32>; // Adjuntoo v from CG (D)
@group(0) @binding(2) var<storage, read> q_input: array<f32>;   // Input query from forward (D)

// Weights to be updated
@group(1) @binding(0) var<storage, read_write> W_q: array<f32>;
@group(1) @binding(1) var<storage, read_write> W_k: array<f32>;
@group(1) @binding(2) var<storage, read_write> W_v: array<f32>;
@group(1) @binding(3) var<storage, read_write> W_o: array<f32>;
@group(1) @binding(4) var<storage, read_write> W_in: array<f32>;
@group(1) @binding(5) var<storage, read_write> W_x: array<f32>;
@group(1) @binding(6) var<storage, read_write> W_out: array<f32>;
@group(1) @binding(7) var<storage, read_write> A_log: array<f32>;
@group(1) @binding(8) var<storage, read_write> NormScale: array<f32>;

@compute
@workgroup_size(16, 16, 1) // Procesar en bloques de 16x16
fn fused_deq_update_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y; // i (dimensión v)
    let col = gid.x; // j (dimensión q)
    let d = params.d_model;

    if (row >= d || col >= d) { return; }

    let v = v_adjoint[row];
    let q = q_input[col];
    
    // Gradiente de rango 1: grad = v * q^T
    // Aplicamos escala de gradiente (deq_grad_scale)
    let grad_val = v * q * params.grad_scale;
    let step = params.lr * grad_val;

    let idx = row * d + col;
    
    // Actualizar todas las matrices (con un pequeño ruido multiplicador para romper la simetría)
    W_q[idx] -= step * 1.0;
    W_k[idx] -= step * 0.9;
    W_v[idx] -= step * 1.1;
    W_o[idx] -= step * 0.8;
    W_in[idx] -= step * 1.2;
    W_x[idx] -= step * 0.85;
    W_out[idx] -= step * 1.15;

    // Actualizar vectores (solo en la primera columna/hilo para evitar colisiones)
    if (col == 0u) {
        let v_step = params.lr * v * params.grad_scale;
        A_log[row] -= v_step;
        NormScale[row] -= v_step;
    }
}
