// AIDEEN CG Solver: Kernel 2 – Producto Matriz-Vector
// Computes AP = (I - J_f^T) * P for the current DEQ forward:
//   combined = Attn(h) + input
//   f_h      = NormScale * combined / rms
//   h_next   = damping * f_h + (1-damping) * h_prev
//
// Jacobian model here is a GPU-only first-order approximation aligned with
// the active attention branch:
// - Includes cross-slot softmax weights (recomputed from H_star via W_q/W_k)
// - Includes W_o and W_v chain
// - Uses local norm/damping gain on upstream P
// - Ignores second-order terms (softmax-weight dependence on q/k in VJP)
//
// This replaces the previous obsolete Mamba/sigmoid Jacobian.

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

// ── Group 1: estado algorítmico CG (solo P y AP usados aquí) ──
@group(1) @binding(0) var<storage, read_write> V_out:     array<f32>;
@group(1) @binding(1) var<storage, read_write> R:         array<f32>;
@group(1) @binding(2) var<storage, read_write> P:         array<f32>;
@group(1) @binding(3) var<storage, read_write> AP:        array<f32>;
@group(1) @binding(4) var<storage, read_write> debug_log: array<f32>;
@group(1) @binding(5) var<storage, read_write> scalars:   array<f32>;

var<workgroup> s_h:   array<f32, 1024>;
var<workgroup> s_px:  array<f32, 1024>;
var<workgroup> s_pv:  array<f32, 1024>;
var<workgroup> s_pq:  array<f32, 1024>;
var<workgroup> s_pk:  array<f32, 1024>;
var<workgroup> s_red: array<f32, 256>;
const MAX_SLOTS: u32 = 8u;
const TIKHONOV: f32 = 0.01;

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
    let n_total = h_slots * shape.d_model;
    let b_offset = batch_idx * n_total;
    let scale = inverseSqrt(max(1.0, f32(d_model)));
    if (h_slots == 0u || h_slots > MAX_SLOTS || d_model > 1024u) { return; }

    // Iteration-0 diagnostics only (keeps overhead bounded).
    var diag_ap2_acc = 0.0;
    var diag_jtp2_acc = 0.0;
    var diag_softmax_total: u32 = 0u;
    var diag_softmax_sat: u32 = 0u;

    // Process each target slot independently, accumulating contributions from all query slots.
    for (var t = 0u; t < h_slots; t = t + 1u) {
        let t_off = t * d_model;

        for (var i = tid; i < d_model; i = i + 256u) {
            s_pk[i] = 0.0;
        }
        workgroupBarrier();

        // Sum over query slots s: dL/dh_t += a_{s,t} * W_v * (W_o^T * g_attn_s)
        for (var s = 0u; s < h_slots; s = s + 1u) {
            let s_off = s * d_model;

            // q_s (diag approximation) and g_attn_s in shared.
            for (var i = tid; i < d_model; i = i + 256u) {
                let h_si = H_star[b_offset + s_off + i];
                let wq_diag = W_q[i * d_model + i];
                var q_i = wq_diag * h_si;
                s_h[i] = q_i;
                s_pq[i] = shape.damping * NormScale[i] * P[b_offset + s_off + i];
            }
            workgroupBarrier();

            // g_mix_s = W_o^T * g_attn_s
            for (var i = tid; i < d_model; i = i + 256u) {
                var g_mix_i = 0.0;
                for (var d = 0u; d < d_model; d = d + 1u) {
                    g_mix_i = g_mix_i + W_o[i * d_model + d] * s_pq[d];
                }
                s_px[i] = g_mix_i;
            }
            workgroupBarrier();

            // scores_{s,k} = <q_s, k_k> / sqrt(d)
            // k_k uses diagonal approximation for performance.
            if (tid < h_slots) {
                let k = tid;
                let k_off = k * d_model;
                var score = 0.0;
                for (var i = 0u; i < d_model; i = i + 1u) {
                    let h_ki = H_star[b_offset + k_off + i];
                    let wk_diag = W_k[i * d_model + i];
                    let k_i = wk_diag * h_ki;
                    score = score + s_h[i] * k_i;
                }
                s_pv[tid] = clamp(score * scale, -6.0, 6.0);
            }
            workgroupBarrier();

            // softmax over keys for query s.
            if (tid == 0u) {
                var max_s = -1e30;
                for (var k = 0u; k < h_slots; k = k + 1u) {
                    max_s = max(max_s, s_pv[k]);
                }
                var sum_exp = 0.0;
                for (var k = 0u; k < h_slots; k = k + 1u) {
                    let e = exp(s_pv[k] - max_s);
                    s_pv[k] = e;
                    sum_exp = sum_exp + e;
                }
                let inv_sum = 1.0 / max(sum_exp, 1e-12);
                for (var k = 0u; k < h_slots; k = k + 1u) {
                    s_pv[k] = s_pv[k] * inv_sum;
                }
                if (shape.curr_iter == 0u) {
                    var max_a = 0.0;
                    for (var k = 0u; k < h_slots; k = k + 1u) {
                        max_a = max(max_a, s_pv[k]);
                    }
                    diag_softmax_total = diag_softmax_total + 1u;
                    if (max_a > 0.95) {
                        diag_softmax_sat = diag_softmax_sat + 1u;
                    }
                }
            }
            workgroupBarrier();

            let a_st = s_pv[t];
            // Softmax score-gradient (diag approximation):
            // grad_alpha_k = <g_mix_s, v_k_diag>, grad_z_k = a_k (grad_alpha_k - sum_j a_j grad_alpha_j)
            if (tid == 0u) {
                var sum_a_ga = 0.0;
                for (var k = 0u; k < h_slots; k = k + 1u) {
                    let k_off = k * d_model;
                    var grad_alpha = 0.0;
                    for (var d = 0u; d < d_model; d = d + 1u) {
                        let wv_diag = W_v[d * d_model + d];
                        let v_diag = wv_diag * H_star[b_offset + k_off + d];
                        grad_alpha = grad_alpha + s_px[d] * v_diag;
                    }
                    s_pq[k] = grad_alpha;
                    sum_a_ga = sum_a_ga + s_pv[k] * grad_alpha;
                }
                for (var k = 0u; k < h_slots; k = k + 1u) {
                    s_pq[k] = s_pv[k] * (s_pq[k] - sum_a_ga); // grad_z_k
                }
            }
            workgroupBarrier();
            let grad_z_t = s_pq[t];

            // Accumulate contribution into target-slot gradient buffer s_pk.
            for (var i = tid; i < d_model; i = i + 256u) {
                var acc = 0.0;
                // V path: W_v^T (a_st * g_mix_s)
                for (var j = 0u; j < d_model; j = j + 1u) {
                    acc = acc + W_v[i * d_model + j] * s_px[j];
                }
                acc = a_st * acc;

                // K path (diag approximation): grad_z_t * q_s * wk_diag
                let wk_diag_i = W_k[i * d_model + i];
                acc = acc + grad_z_t * s_h[i] * wk_diag_i * scale;

                // Q path contributes only for target query slot (s == t):
                // sum_k grad_z_k * k_k_diag  then * wq_diag
                if (s == t) {
                    let wq_diag_i = W_q[i * d_model + i];
                    var sum_gz_k = 0.0;
                    for (var k = 0u; k < h_slots; k = k + 1u) {
                        let k_off = k * d_model;
                        let wk_diag_i2 = W_k[i * d_model + i];
                        let k_i = wk_diag_i2 * H_star[b_offset + k_off + i];
                        sum_gz_k = sum_gz_k + s_pq[k] * k_i;
                    }
                    acc = acc + wq_diag_i * sum_gz_k * scale;
                }

                s_pk[i] = s_pk[i] + acc;
            }
            workgroupBarrier();
        }

        // AP_t = (I - J^T + λI) p_t with residual path (1-damp)I.
        var local_ap2 = 0.0;
        var local_jtp2 = 0.0;
        for (var i = tid; i < d_model; i = i + 256u) {
            let p_i = P[b_offset + t_off + i];
            let jt_p_i = (1.0 - shape.damping) * p_i + s_pk[i];
            let ap_i = (1.0 + TIKHONOV) * p_i - jt_p_i;
            AP[b_offset + t_off + i] = ap_i;
            if (shape.curr_iter == 0u) {
                local_ap2 = local_ap2 + ap_i * ap_i;
                local_jtp2 = local_jtp2 + jt_p_i * jt_p_i;
            }
        }
        if (shape.curr_iter == 0u) {
            s_red[tid] = local_ap2;
            workgroupBarrier();
            for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    s_red[tid] = s_red[tid] + s_red[tid + stride];
                }
                workgroupBarrier();
            }
            if (tid == 0u) {
                diag_ap2_acc = diag_ap2_acc + s_red[0];
            }

            s_red[tid] = local_jtp2;
            workgroupBarrier();
            for (var stride2 = 128u; stride2 > 0u; stride2 = stride2 >> 1u) {
                if (tid < stride2) {
                    s_red[tid] = s_red[tid] + s_red[tid + stride2];
                }
                workgroupBarrier();
            }
            if (tid == 0u) {
                diag_jtp2_acc = diag_jtp2_acc + s_red[0];
            }
        }
        workgroupBarrier();
    }

    // Debug slots (kept away from indices used by cg_init/cg_update).
    // [20]=||AP||_2, [21]=||J^T p||_2, [22]=sat_ratio_softmax, [23]=curr_iter
    if (batch_idx == 0u && tid == 0u && shape.curr_iter == 0u) {
        debug_log[20] = sqrt(max(diag_ap2_acc, 0.0));
        debug_log[21] = sqrt(max(diag_jtp2_acc, 0.0));
        let denom = max(1.0, f32(diag_softmax_total));
        debug_log[22] = f32(diag_softmax_sat) / denom;
        debug_log[23] = f32(shape.curr_iter);
    }
}
