// AIDEEN GPU Kernel: Conjugate Gradient Solver (Backward Pass)
// Resuelve (I - J_f) * v = b usando derivadas direccionales numéricas.

struct CGComputeShape {
    batch_size: u32,
    d_model: u32,
    h_slots: u32,
    cg_iters: u32,
    epsilon:    f32,
    _pad:       vec3<u32>,
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
@group(1) @binding(0) var<storage, read_write> V_out:     array<f32>;
@group(1) @binding(1) var<storage, read_write> R:         array<f32>;
@group(1) @binding(2) var<storage, read_write> P:         array<f32>;
@group(1) @binding(3) var<storage, read_write> AP:        array<f32>;
@group(1) @binding(4) var<storage, read_write> debug_log: array<f32>;
@group(1) @binding(5) var<storage, read_write> scalars:   array<f32>;

// Memoria compartida para el solver CG (8 slots * 512 dim = 4096 elementos)
// Total: 4 arrays * 4096 * 4 bytes = 65,536 bytes (64KB)
var<workgroup> s_v: array<f32, 4096>;
var<workgroup> s_r: array<f32, 4096>;
var<workgroup> s_p: array<f32, 4096>;
var<workgroup> s_ap: array<f32, 4096>;
var<workgroup> s_dot: array<f32, 256>; // Para reducciones

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
    if (tid == 0u) {
        debug_log[15] = 777.777;
        debug_log[14] = f32(shape.batch_size);
        debug_log[8] = A_log[0];
        debug_log[9] = A_log[1];
    }
    let d_model = shape.d_model;
    let h_slots = shape.h_slots;
    let n_total = h_slots * d_model; // 8 * 512 = 4096
    
    let dl_offset = batch_idx * d_model;
    let b_offset = batch_idx * n_total;

    // 1) Inicialización paralela (Cada hilo procesa 16 elementos de los 4096)
    var local_rs_old = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let idx = tid * 16u + i;
        if (idx < n_total) {
            let slot_d = idx % d_model;
            let b_val = dl_dh_pooled[dl_offset + slot_d] / f32(h_slots);
            
            s_v[idx] = 0.0;
            s_r[idx] = b_val;
            s_p[idx] = b_val;
            local_rs_old += b_val * b_val;
        }
    }
    s_dot[tid] = local_rs_old;
    var rs_old = dot_parallel(tid);

// 2) Bucle Conjugate Gradient (Optimizado para evitar Timeouts)
    for (var iter = 0u; iter < shape.cg_iters; iter = iter + 1u) {
        
        // --- VJP REAL Y TRANSPUESTO ---
        workgroupBarrier();
        for (var i = 0u; i < 16u; i = i + 1u) {
            let idx = tid * 16u + i;
            if (idx < n_total) {
                let slot_id = idx / d_model;
                let d_target = idx % d_model; // La dimensión que estamos calculando
                
                var vjp_sum = 0.0;
                // Optimizamos: leemos sigma_prime una vez fuera del loop interno
                // Pero ojo: el VJP requiere sigma_prime de la dimensión 'j', no 'idx'
                
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let h_j = H_star[batch_idx * n_total + (slot_id * d_model + j)];
                    let sigma_prime_j = 1.0 - (h_j * h_j);
                    
                    // ACCESO TRANSPUESTO: W_k[j, d_target] -> W_k[j * d_model + d_target]
                    // Multiplicamos por la dirección p[j] pesada por su derivada
                    vjp_sum += W_k[j * d_model + d_target] * (s_p[slot_id * d_model + j] * sigma_prime_j);
                }
                s_ap[idx] = vjp_sum; 
            }
        }
        workgroupBarrier();

        // 2.2) AP = (P - J_f^T * P) + Damping (0.01)
        var local_p_ap = 0.0;
        for (var i = 0u; i < 16u; i = i + 1u) {
            let idx = tid * 16u + i;
            if (idx < n_total) {
                let p_val = s_p[idx];
                // Aplicamos (I - J_f^T) + estabilidad de Tikhonov
                let damped_ap = (p_val - s_ap[idx]) + (p_val * 0.01);
                
                s_ap[idx] = damped_ap;
                local_p_ap += p_val * damped_ap;
            }
        }
        
        s_dot[tid] = local_p_ap;
        let p_ap = dot_parallel(tid);
        
        // SAFEGUARD: Evitar división por cero o NaNs
        var alpha = 0.0;
        if (p_ap > 1e-9) { 
            alpha = rs_old / p_ap; 
        } else { 
            alpha = 0.001; // Paso pequeño de emergencia
        }
        if (tid == 0u && iter == 0u) {
            debug_log[5] = p_ap;
            debug_log[6] = rs_old; 
            debug_log[10] = f32(shape.d_model);
            debug_log[11] = f32(shape.h_slots);
            debug_log[12] = local_p_ap;
            debug_log[13] = s_ap[0];
            debug_log[6] = alpha; // Reuso el 6 para alpha si prefiero, pero rs_old ya esta en debug_log[2]
        }

        // 2.3) Update residuo y solución
        var local_rs_new = 0.0;
        for (var i = 0u; i < 16u; i = i + 1u) {
            let idx = tid * 16u + i;
            if (idx < n_total) {
                s_v[idx] += alpha * s_p[idx];
                s_r[idx] -= alpha * s_ap[idx];
                local_rs_new += s_r[idx] * s_r[idx];
            }
        }
        s_dot[tid] = local_rs_new;
        let rs_new = dot_parallel(tid);

        if (tid == 0u && iter == 0u) {
            debug_log[7] = rs_new;
        }

        if (sqrt(rs_new) < 1e-6) { break; }
        
        let beta = rs_new / rs_old;
        for (var i = 0u; i < 16u; i = i + 1u) {
            let idx = tid * 16u + i;
            if (idx < n_total) {
                s_p[idx] = s_r[idx] + beta * s_p[idx];
            }
        }
        rs_old = rs_new;
        workgroupBarrier();
    }

    // 3) Escritura final a Memoria Global
    if (tid == 0u) {
        debug_log[0] = dl_dh_pooled[dl_offset + 0]; // b[0]
        debug_log[1] = dl_dh_pooled[dl_offset + 1]; // b[1]
        debug_log[2] = rs_old;                      // residuo final
        debug_log[3] = f32(n_total);                // N
        debug_log[4] = 987.654;                     // Magic number CG
    }

    for (var i = 0u; i < 16u; i = i + 1u) {
        let idx = tid * 16u + i;
        if (idx < n_total) {
            V_out[b_offset + idx] = s_v[idx];
        }
    }
}
