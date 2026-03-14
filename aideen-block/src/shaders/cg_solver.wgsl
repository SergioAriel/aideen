// AIDEEN GPU Kernel: Conjugate Gradient Solver (Backward Pass)
// Resuelve (I - J_f) * v = b usando derivadas direccionales numéricas.

struct CGComputeShape {
    batch_size: u32,
    d_model: u32,
    h_slots: u32,
    cg_iters: u32,
    epsilon: f32,
    damping: f32,
    curr_iter: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
};

@group(0) @binding(0) var<uniform> shape: CGComputeShape;

// Entradas dinámicas y Estado de Equilibrio
@group(0) @binding(1) var<storage, read> S_in: array<f32>;        // [batch_size, d_model]
@group(0) @binding(2) var<storage, read> H_star: array<f32>;      // [batch_size, h_slots, d_model]
@group(0) @binding(3) var<storage, read> dl_dh_pooled: array<f32>;// [batch_size, d_model]

// Pesos del Modelo (Flattened)
@group(0) @binding(4) var<storage, read> W_q: array<f32>;
@group(0) @binding(5) var<storage, read> W_k: array<f32>;
@group(0) @binding(6) var<storage, read> W_v: array<f32>;
@group(0) @binding(7) var<storage, read> W_o: array<f32>;
@group(0) @binding(8) var<storage, read> W_in: array<f32>;
@group(0) @binding(9) var<storage, read> W_x: array<f32>;
@group(0) @binding(10) var<storage, read> W_out: array<f32>;
@group(0) @binding(11) var<storage, read> A_log: array<f32>;
@group(0) @binding(12) var<storage, read> NormScale: array<f32>;

// Buffers de Trabajo del Conjugate Gradient
// Buffers de Trabajo del Conjugate Gradient
@group(1) @binding(0) var<storage, read_write> V_out:     array<f32>;
@group(1) @binding(1) var<storage, read_write> R:         array<f32>;
@group(1) @binding(2) var<storage, read_write> P:         array<f32>;
@group(1) @binding(3) var<storage, read_write> AP:        array<f32>;
@group(1) @binding(4) var<storage, read_write> debug_log: array<f32>;
@group(1) @binding(5) var<storage, read_write> scalars:   array<f32>;

var<workgroup> s_dot: array<f32, 256>; // Para reducciones (WG_SIZE=256)

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
@workgroup_size(256, 1, 1) // 1 Workgroup por Batch
fn cg_solver_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let batch_idx = group_id.x;
    if (batch_idx >= shape.batch_size) { return; }

    let tid = local_id.x; // 0..255
    let d_model = shape.d_model;
    let h_slots = shape.h_slots;
    let n_total = h_slots * d_model;
    
    let dl_offset = batch_idx * d_model;
    let b_offset = batch_idx * n_total;

    // 1) Inicialización paralela (Mamba Chain Rule for dl/dH*)
    // Compute U = (1 - a) * (W_out^T * g)
    // We temporarily store U in the first D elements of V_out.
    for (var j = tid; j < d_model; j += 256u) {
        var v_j = 0.0;
        for (var i = 0u; i < d_model; i += 1u) {
            let g_i = dl_dh_pooled[dl_offset + i] / f32(h_slots);
            v_j += W_out[j * d_model + i] * g_i;
        }
        let a_j = 1.0 / (1.0 + exp(A_log[j]));
        V_out[b_offset + j] = (1.0 - a_j) * v_j;
    }
    storageBarrier(); 

    var local_rs_old = 0.0;
    for (var idx = tid; idx < n_total; idx += 256u) {
        let k = idx % d_model;
        
        var b_val = 0.0;
        for (var j = 0u; j < d_model; j += 1u) {
            b_val += W_x[k * d_model + j] * V_out[b_offset + j];
        }
        
        let global_idx = b_offset + idx;
        R[global_idx] = b_val;
        P[global_idx] = b_val;
        local_rs_old += b_val * b_val;
    }
    s_dot[tid] = local_rs_old;
    var rs_old = dot_parallel(tid);

    storageBarrier();

    for (var idx = tid; idx < n_total; idx += 256u) {
        let global_idx = b_offset + idx;
        V_out[global_idx] = 0.0;
    }
    workgroupBarrier();

    // 2) Bucle Conjugate Gradient
    for (var iter = 0u; iter < shape.cg_iters; iter = iter + 1u) {
        
        // --- VJP REAL Y TRANSPUESTO ---
        workgroupBarrier();
        for (var idx = tid; idx < n_total; idx += 256u) {
            let slot_id = idx / d_model;
            let d_target = idx % d_model; 
            
            var vjp_sum = 0.0;
            for (var j = 0u; j < d_model; j = j + 1u) {
                let h_j = H_star[b_offset + (slot_id * d_model + j)];
                let sigma_prime_j = 1.0 - (h_j * h_j);
                vjp_sum += W_k[j * d_model + d_target] * (P[b_offset + (slot_id * d_model + j)] * sigma_prime_j);
            }
            AP[b_offset + idx] = vjp_sum; 
        }
        workgroupBarrier();

        // 2.2) AP = (P - J_f^T * P) + Damping (0.01)
        var local_p_ap = 0.0;
        for (var idx = tid; idx < n_total; idx += 256u) {
            let global_idx = b_offset + idx;
            let p_val = P[global_idx];
            let damped_ap = (p_val - AP[global_idx]) + (p_val * 0.01);
            
            AP[global_idx] = damped_ap;
            local_p_ap += p_val * damped_ap;
        }
        
        s_dot[tid] = local_p_ap;
        let p_ap = dot_parallel(tid);
        
        // SAFEGUARD
        var alpha = 0.0;
        if (p_ap > 1e-12) { 
            alpha = rs_old / p_ap; 
        } else { 
            alpha = 1e-4; 
        }

        // 2.3) Update residuo y solución
        var local_rs_new = 0.0;
        for (var idx = tid; idx < n_total; idx += 256u) {
            let global_idx = b_offset + idx;
            V_out[global_idx] += alpha * P[global_idx];
            R[global_idx] -= alpha * AP[global_idx];
            local_rs_new += R[global_idx] * R[global_idx];
        }
        s_dot[tid] = local_rs_new;
        let rs_new = dot_parallel(tid);

        if (sqrt(rs_new) < 1e-7) { break; }
        
        let beta = rs_new / rs_old;
        for (var idx = tid; idx < n_total; idx += 256u) {
            let global_idx = b_offset + idx;
            P[global_idx] = R[global_idx] + beta * P[global_idx];
        }
        rs_old = rs_new;
    }

    if (tid == 0u) {
        debug_log[2] = rs_old;
        debug_log[4] = 987.654;
    }
}
