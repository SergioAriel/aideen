// AIDEEN CG Solver: Kernel 2 – Producto Matriz-Vector
// Computa AP = (I - J_f^T) * P con NormScale y Tikhonov damping (0.05)
// Cada hilo trabaja de forma independiente → Workgroup memory: 0KB

struct CGComputeShape {
    batch_size: u32,
    d_model:    u32,
    h_slots:    u32,
    cg_iters:   u32,
    epsilon:    f32,
    curr_iter:  u32,
    _pad0:      u32,
    _pad1:      u32,
    _pad2:      u32,
    _pad3:      u32,
    _pad4:      u32,
    _pad5:      u32,
};

// ── Group 0: todos los bindings declarados (evita gaps de mapping en Metal) ──
@group(0) @binding(0)  var<uniform>       shape:        CGComputeShape;
@group(0) @binding(1)  var<storage, read> S_in:         array<f32>;
@group(0) @binding(2)  var<storage, read> H_star:       array<f32>;
@group(0) @binding(3)  var<storage, read> dl_dh_pooled: array<f32>;
@group(0) @binding(4)  var<storage, read> W_q:          array<f32>;
@group(0) @binding(5)  var<storage, read> W_k:          array<f32>;
@group(0) @binding(6)  var<storage, read> W_v:          array<f32>;
@group(0) @binding(7)  var<storage, read> W_o:          array<f32>;
@group(0) @binding(8)  var<storage, read> W_in:         array<f32>;
@group(0) @binding(9)  var<storage, read> W_x:          array<f32>;
@group(0) @binding(10) var<storage, read> W_out:        array<f32>;
@group(0) @binding(11) var<storage, read> A_log:        array<f32>;
@group(0) @binding(12) var<storage, read> NormScale:    array<f32>;

// ── Group 1: estado algorítmico CG (solo P y AP usados aquí) ──
@group(1) @binding(0) var<storage, read_write> V_out:     array<f32>;
@group(1) @binding(1) var<storage, read_write> R:         array<f32>;
@group(1) @binding(2) var<storage, read_write> P:         array<f32>;
@group(1) @binding(3) var<storage, read_write> AP:        array<f32>;
@group(1) @binding(4) var<storage, read_write> debug_log: array<f32>;
@group(1) @binding(5) var<storage, read_write> scalars:   array<f32>;

@compute
@workgroup_size(256, 1, 1)
fn cg_matvec_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id)        group_id:  vec3<u32>,
) {
    let batch_idx = group_id.x;
    if (batch_idx >= shape.batch_size) { return; }

    let tid     = local_id.x;
    let d_model = shape.d_model;
    let h_slots = shape.h_slots;
    let n_total = h_slots * d_model;
    let b_offset = batch_idx * n_total;

    let items_per_thread = (n_total + 255u) / 256u;
    for (var i = 0u; i < items_per_thread; i = i + 1u) {
        let idx = tid + i * 256u;
        if (idx < n_total) {
            let slot_id  = idx / d_model;
            let d_target = idx % d_model;

            // VJP transpuesto con NormScale del forward: (J_f^T * p)[d_target]
            // J_f incluye sigma_prime (derivada tanh) y NormScale por dimensión fuente
            var vjp_sum = 0.0;
            for (var j = 0u; j < d_model; j = j + 1u) {
                let h_j       = H_star[batch_idx * n_total + slot_id * d_model + j];
                let sigma_j   = 1.0 - (h_j * h_j);                          // tanh'
                let norm_j    = clamp(NormScale[j], 0.1, 10.0);              // del forward
                vjp_sum += W_k[j * d_model + d_target]
                         * (P[b_offset + slot_id * d_model + j] * sigma_j * norm_j);
            }

            let p_val = P[b_offset + idx];
            // AP = (I - J_f^T + lambda*I) * P 
            // Tikhonov damping (lambda = 0.05)
            AP[b_offset + idx] = (1.05 * p_val - vjp_sum);
        }
    }
}
