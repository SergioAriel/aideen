// AIDEEN CG Solver: Kernel 3 – Paso de Actualización CG
// Workgroup memory: s_dot[256] = 1KB

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

// ── Group 0: todos los bindings declarados ──
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

// ── Group 1: estado algorítmico CG ──
@group(1) @binding(0) var<storage, read_write> V_out:     array<f32>;
@group(1) @binding(1) var<storage, read_write> R:         array<f32>;
@group(1) @binding(2) var<storage, read_write> P:         array<f32>;
@group(1) @binding(3) var<storage, read_write> AP:        array<f32>;
@group(1) @binding(4) var<storage, read_write> debug_log: array<f32>;
@group(1) @binding(5) var<storage, read_write> scalars:   array<f32>;

var<workgroup> s_dot: array<f32, 256>;

fn dot_parallel(tid: u32) -> f32 {
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            s_dot[tid] += s_dot[tid + stride];
        }
        workgroupBarrier();
    }
    return s_dot[0];
}

@compute
@workgroup_size(256, 1, 1)
fn cg_update_main(
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

    let rs_old = scalars[batch_idx];

    // --- 1) p_ap = dot(P, AP) ---
    var local_p_ap = 0.0;
    let items_per_thread = (n_total + 255u) / 256u;
    for (var i = 0u; i < items_per_thread; i = i + 1u) {
        let idx = tid + i * 256u;
        if (idx < n_total) {
            local_p_ap += P[b_offset + idx] * AP[b_offset + idx];
        }
    }
    s_dot[tid] = local_p_ap;
    let p_ap  = dot_parallel(tid);
    
    // SAFEGUARD PRO: Protegemos a alpha de explotar
    var alpha = 0.0;
    if (p_ap > 1e-12) {
        alpha = rs_old / p_ap;
    }

    // Escribimos DONDE RUST ESTÁ MIRANDO - SOLO EN ITER 0 PARA TELEMETRÍA
    if (tid == 0u && shape.curr_iter == 0u) {
        debug_log[5]  = p_ap;          // trainer espera p_ap aquí
        debug_log[6]  = alpha;         // trainer espera alpha aquí
        debug_log[12] = local_p_ap;    // loc_pap
        debug_log[13] = p_ap;          // sap0
    }

    // --- 2) Actualizar x y r; calcular rs_new ---
    var local_rs_new = 0.0;
    for (var i = 0u; i < items_per_thread; i = i + 1u) {
        let idx = tid + i * 256u;
        if (idx < n_total) {
            V_out[b_offset + idx] += alpha * P[b_offset + idx];
            let r_new = R[b_offset + idx] - alpha * AP[b_offset + idx];
            R[b_offset + idx] = r_new;
            local_rs_new += r_new * r_new;
        }
    }
    s_dot[tid] = local_rs_new;
    let rs_new = dot_parallel(tid);
    let beta   = rs_new / (rs_old + 1e-10);

    if (tid == 0u && shape.curr_iter == 0u) {
        debug_log[4] = beta;    // trainer espera m1 aquí
        debug_log[7] = rs_new;  // trainer espera rs_new aquí
    }

    // --- 3) Actualizar P = R + beta * P ---
    for (var i = 0u; i < items_per_thread; i = i + 1u) {
        let idx = tid + i * 256u;
        if (idx < n_total) {
            P[b_offset + idx] = R[b_offset + idx] + beta * P[b_offset + idx];
        }
    }

    // --- 4) Guardar rs_new para la siguiente iteración ---
    if (tid == 0u) {
        scalars[batch_idx] = rs_new;
    }
}