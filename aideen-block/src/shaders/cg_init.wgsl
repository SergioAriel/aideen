// AIDEEN CG Solver: Kernel 1 – Inicialización
// Inicializa x=0, r=b, p=b y calcula rs_old = dot(b, b) → scalars[batch_idx]
// Workgroup memory: s_dot[256] = 1KB
//
// Debug layout (para trainer.rs CG-DEBUG / CG-MASSIVE):
//   debug_log[0]  = dl_dh_pooled[0]   (b[0])
//   debug_log[1]  = dl_dh_pooled[1]   (b[1])
//   debug_log[2]  = rs_old (= ||b||²) — sobreescrito luego por cg_update
//   debug_log[8]  = A_log[0]
//   debug_log[9]  = A_log[1]
//   debug_log[10] = f32(d_model)       → CG-MASSIVE "d"
//   debug_log[11] = f32(h_slots)       → CG-MASSIVE "h"
//   debug_log[14] = f32(batch_idx)

struct CGComputeShape {
    batch_size: u32,
    d_model:    u32,
    h_slots:    u32,
    cg_iters:   u32,
    epsilon:    f32,
    damping:    f32,
    curr_iter:  u32,
    _pad0:      u32,
    _pad1:      u32,
    _pad2:      u32,
    _pad3:      u32,
    _pad4:      u32,
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

// ── Group 1: estado algorítmico CG ──
@group(1) @binding(0) var<storage, read_write> V_out:     array<f32>;
@group(1) @binding(1) var<storage, read_write> R:         array<f32>;
@group(1) @binding(2) var<storage, read_write> P:         array<f32>;
@group(1) @binding(3) var<storage, read_write> AP:        array<f32>;
@group(1) @binding(4) var<storage, read_write> debug_log: array<f32>;
@group(1) @binding(5) var<storage, read_write> scalars:   array<f32>;

var<workgroup> s_dot: array<f32, 256>;

@compute
@workgroup_size(256, 1, 1)
fn cg_init_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id)        group_id:  vec3<u32>,
) {
    let batch_idx = group_id.x;
    if (batch_idx >= shape.batch_size) { return; }

    let tid     = local_id.x;
    let d_model = shape.d_model;
    let h_slots = shape.h_slots;
    let n_total = h_slots * d_model;

    let dl_offset = batch_idx * d_model;
    let b_offset  = batch_idx * n_total;

    var local_rs = 0.0;
    let items_per_thread = (n_total + 255u) / 256u;
    for (var i = 0u; i < items_per_thread; i = i + 1u) {
        let idx = tid + i * 256u;
        if (idx < n_total) {
            let slot_id = idx / d_model;
            // Solo slot 0 recibe gradiente del pool (head)
            var r_val = 0.0;
            if (slot_id == 0u) {
                r_val = dl_dh_pooled[batch_idx * d_model + (idx % d_model)];
            }
            R[b_offset + idx] = r_val;
            P[b_offset + idx] = r_val;
            V_out[b_offset + idx] = 0.0;
            local_rs += r_val * r_val;
        }
    }

    s_dot[tid] = local_rs;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            s_dot[tid] += s_dot[tid + stride];
        }
            workgroupBarrier();
    }

    if (tid == 0u) {
        let rs_old = s_dot[0];
        scalars[batch_idx] = rs_old;

        // ── Debug: posiciones que trainer.rs espera ──
        debug_log[0] = dl_dh_pooled[dl_offset]; 
        debug_log[1] = select(0.0, dl_dh_pooled[dl_offset + 1u], d_model >= 2u); 
        debug_log[2] = rs_old; // rs_old inicial
        debug_log[8] = A_log[0]; 
        debug_log[9] = 111.111; // "Magic Number" para saber que INIT corrió
        
        debug_log[10] = f32(d_model); 
        debug_log[11] = f32(h_slots); 
        debug_log[14] = f32(batch_idx);
    }

}
