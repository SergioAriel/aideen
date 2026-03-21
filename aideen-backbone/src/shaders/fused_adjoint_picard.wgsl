// =============================================================================
// [LEGACY — NO UTILIZADO]
// Este shader fue reemplazado por staged_adjoint_picard.wgsl.
//
// Razones por las que quedó obsoleto:
//   1. scratch_stride incorrecto: usaba d*(6h+1)+h² en lugar de d*7h+h².
//      Para d=512, h=8: 25152 vs 28736 floats/token → lecturas de attn_weights OOB.
//   2. Sin soporte para hist_gated: no conoce m_inner ni slot_anchor.
//   3. Kernel monolítico: no reutiliza gcomb_buf con apply_fused_deq_update.
//
// El path activo es: run_staged_adjoint_picard_no_readback() en gpu_deq.rs
//   → staged_adjoint_picard.wgsl (picard_init → picard_gcomb → picard_gmix
//                                  → picard_gscore → picard_accum × N)
// =============================================================================
//
// AIDEEN GPU Kernel: Exact Fused Picard Adjoint Solver
// Resuelve (I - J_f^T) v = b usando iteraciones de Picard: v_{k+1} = J_f^T v_k + b
// Matemática derivada del forward en deq_forward.wgsl (RMSNorm + Attention + Damping)

struct UpdateUniforms {
    d_model: u32,
    h_slots: u32,
    lr: f32,
    grad_scale: f32,
    ternary_flag: u32,
    weight_decay: f32,
    seq_len: u32,
    damping: f32,
    residual_alpha: f32,
};

@group(0) @binding(0) var<uniform> params: UpdateUniforms;
@group(0) @binding(1) var<storage, read> b_in: array<f32>;          // dl_dh_pooled [batch, d_model]
@group(0) @binding(2) var<storage, read> H_star: array<f32>;        // [batch, h_slots, d_model]
@group(0) @binding(3) var<storage, read_write> v_state: array<f32>; // [batch, h_slots, d_model]
@group(0) @binding(4) var<storage, read_write> v_final: array<f32>; // ping-pong/output
@group(0) @binding(5) var<storage, read> NormScale: array<f32>;
@group(0) @binding(6) var<storage, read> Scratch: array<f32>;       // Intermediates de Forward (Q,K,V,attn)

@group(1) @binding(0) var<storage, read> W_q: array<f32>;
@group(1) @binding(1) var<storage, read> W_k: array<f32>;
@group(1) @binding(2) var<storage, read> W_v: array<f32>;
@group(1) @binding(3) var<storage, read> W_o: array<f32>;

var<workgroup> v_curr: array<f32, 4096>; // hs=8 * d=512
var<workgroup> v_next: array<f32, 4096>;
var<workgroup> shared_rms: array<f32, 8>;
var<workgroup> shared_coeff: array<f32, 8>;
var<workgroup> shared_red: array<f32, 256>;

@compute @workgroup_size(256, 1, 1) // 1 Workgroup procesa 1 (Batch, Token)
fn fused_adjoint_picard_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let d = params.d_model;
    let h_slots = params.h_slots;
    let tid = lid.x;
    let t = wid.x;
    let deq_only_mode = params.residual_alpha <= -1.5;
    let scale = inverseSqrt(max(1.0, f32(d)));
    
    // 1. Inicialización: dL/dh* pooled -> per-slot.
    for (var i = tid; i < h_slots * d; i += 256u) {
        let dim = i % d;
        v_curr[i] = b_in[t * d + dim] / max(1.0, f32(h_slots));
        v_next[i] = v_curr[i];
        v_state[t * h_slots * d + i] = v_curr[i];
    }
    workgroupBarrier();

    // Contexto del Forward (offsets en Scratch)
    let scratch_stride = d * (h_slots * 6u + 1u) + h_slots * h_slots;
    let base = t * scratch_stride;
    let q_base = base;
    let k_base = q_base + h_slots * d;
    let v_forward_base = k_base + h_slots * d;
    let attn_base = v_forward_base + h_slots * d;
    let signal_base = attn_base + h_slots * d;
    let attn_weight_base = signal_base + d + h_slots * d;

    // Iteraciones de Picard: v_{k+1} = J_f^T v_k + b
    for (var iter = 0u; iter < 8u; iter++) {
        workgroupBarrier();

        // Precompute exact RMSNorm VJP scalars per slot for current v_curr.
        for (var s = 0u; s < h_slots; s = s + 1u) {
            let s_off = s * d;
            var local_sumsq = 0.0;
            var local_coeff = 0.0;
            for (var dim = tid; dim < d; dim += 256u) {
                let z = Scratch[attn_base + s_off + dim] + Scratch[signal_base + dim];
                let up = params.damping * v_curr[s_off + dim];
                local_sumsq = local_sumsq + z * z;
                local_coeff = local_coeff + up * NormScale[dim] * z;
            }

            shared_red[tid] = local_sumsq;
            workgroupBarrier();
            for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_red[tid] = shared_red[tid] + shared_red[tid + stride];
                }
                workgroupBarrier();
            }
            if (tid == 0u) {
                shared_rms[s] = sqrt(shared_red[0] / max(1.0, f32(d)) + 1e-6);
            }
            workgroupBarrier();

            shared_red[tid] = local_coeff;
            workgroupBarrier();
            for (var stride2 = 128u; stride2 > 0u; stride2 = stride2 >> 1u) {
                if (tid < stride2) {
                    shared_red[tid] = shared_red[tid] + shared_red[tid + stride2];
                }
                workgroupBarrier();
            }
            if (tid == 0u) {
                shared_coeff[s] = shared_red[0];
            }
            workgroupBarrier();
        }

        // --- ADJOINT STEP: v_next = J_f^T * v_curr + b ---
        
        for (var i = tid; i < h_slots * d; i += 256u) {
            let target_slot = i / d;
            let dim = i % d;

            // Exacto para la rama residual/damping del forward actual.
            var jt_v = (1.0 - params.damping) * v_curr[i];

            if (!deq_only_mode) {
                // 1) Build exact RMSNorm VJP output g_comb for each query slot.
                var g_h = 0.0;
                var v_path_acc = 0.0;
                var k_path_acc = 0.0;
                var q_path_acc = 0.0;
                for (var qs = 0u; qs < h_slots; qs = qs + 1u) {
                    let qs_off = qs * d;
                    let rms = shared_rms[qs];
                    let coeff = shared_coeff[qs];

                    // g_mix[vd] = Σ_dout W_o[vd,dout] * g_comb[dout]
                    // First compute g_mix for all feature dims implicitly when needed.
                    // V path contribution to target slot.
                    let alpha_qt = Scratch[attn_weight_base + qs * h_slots + target_slot];
                    for (var vd = 0u; vd < d; vd = vd + 1u) {
                        let z_vd = Scratch[attn_base + qs_off + vd] + Scratch[signal_base + vd];
                        let up_vd = params.damping * v_curr[qs_off + vd];
                        let g_comb_vd =
                            (NormScale[vd] / max(rms, 1e-6)) * up_vd
                            - z_vd * coeff / (max(1.0, f32(d)) * max(rms * rms * rms, 1e-6));

                        var g_mix_vd = 0.0;
                        for (var dout = 0u; dout < d; dout = dout + 1u) {
                            g_mix_vd = g_mix_vd + W_o[vd * d + dout] * (
                                (NormScale[dout] / max(rms, 1e-6)) * (params.damping * v_curr[qs_off + dout])
                                - (Scratch[attn_base + qs_off + dout] + Scratch[signal_base + dout]) * coeff
                                    / (max(1.0, f32(d)) * max(rms * rms * rms, 1e-6))
                            );
                        }

                        v_path_acc = v_path_acc
                            + W_v[dim * d + vd] * (alpha_qt * g_mix_vd);
                    }

                    // 2) Exact softmax VJP for this query slot.
                    var alpha_dot_g = 0.0;
                    for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
                        let ks_off = ks * d;
                        var g_alpha = 0.0;
                        for (var vd = 0u; vd < d; vd = vd + 1u) {
                            var g_mix_vd = 0.0;
                            for (var dout = 0u; dout < d; dout = dout + 1u) {
                                g_mix_vd = g_mix_vd + W_o[vd * d + dout] * (
                                    (NormScale[dout] / max(rms, 1e-6)) * (params.damping * v_curr[qs_off + dout])
                                    - (Scratch[attn_base + qs_off + dout] + Scratch[signal_base + dout]) * coeff
                                        / (max(1.0, f32(d)) * max(rms * rms * rms, 1e-6))
                                );
                            }
                            g_alpha = g_alpha + g_mix_vd * Scratch[v_forward_base + ks_off + vd];
                        }
                        alpha_dot_g = alpha_dot_g
                            + Scratch[attn_weight_base + qs * h_slots + ks] * g_alpha;
                    }

                    // k-path into target_slot
                    var g_alpha_t = 0.0;
                    let target_off = target_slot * d;
                    for (var vd = 0u; vd < d; vd = vd + 1u) {
                        var g_mix_vd = 0.0;
                        for (var dout = 0u; dout < d; dout = dout + 1u) {
                            g_mix_vd = g_mix_vd + W_o[vd * d + dout] * (
                                (NormScale[dout] / max(rms, 1e-6)) * (params.damping * v_curr[qs_off + dout])
                                - (Scratch[attn_base + qs_off + dout] + Scratch[signal_base + dout]) * coeff
                                    / (max(1.0, f32(d)) * max(rms * rms * rms, 1e-6))
                            );
                        }
                        g_alpha_t = g_alpha_t + g_mix_vd * Scratch[v_forward_base + target_off + vd];
                    }
                    let alpha_t = Scratch[attn_weight_base + qs * h_slots + target_slot];
                    let g_score_t = alpha_t * (g_alpha_t - alpha_dot_g);

                    // K path hits the current target slot.
                    for (var qd = 0u; qd < d; qd = qd + 1u) {
                        k_path_acc = k_path_acc
                            + W_k[dim * d + qd] * (scale * g_score_t * Scratch[q_base + qs_off + qd]);
                    }

                    // Q path only if current target is the query slot itself.
                    if (target_slot == qs) {
                        for (var qd = 0u; qd < d; qd = qd + 1u) {
                            var g_q_qd = 0.0;
                            for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
                                let ks_off = ks * d;
                                var g_alpha = 0.0;
                                for (var vd = 0u; vd < d; vd = vd + 1u) {
                                    var g_mix_vd = 0.0;
                                    for (var dout = 0u; dout < d; dout = dout + 1u) {
                                        g_mix_vd = g_mix_vd + W_o[vd * d + dout] * (
                                            (NormScale[dout] / max(rms, 1e-6)) * (params.damping * v_curr[qs_off + dout])
                                            - (Scratch[attn_base + qs_off + dout] + Scratch[signal_base + dout]) * coeff
                                                / (max(1.0, f32(d)) * max(rms * rms * rms, 1e-6))
                                        );
                                    }
                                    g_alpha = g_alpha + g_mix_vd * Scratch[v_forward_base + ks_off + vd];
                                }
                                let alpha_k = Scratch[attn_weight_base + qs * h_slots + ks];
                                let g_score_k = alpha_k * (g_alpha - alpha_dot_g);
                                g_q_qd = g_q_qd + scale * g_score_k * Scratch[k_base + ks_off + qd];
                            }
                            q_path_acc = q_path_acc + W_q[dim * d + qd] * g_q_qd;
                        }
                    }
                }
                g_h = v_path_acc + k_path_acc + q_path_acc;
                jt_v = jt_v + g_h;
            }

            let b = b_in[t * d + dim] / max(1.0, f32(h_slots));
            v_next[i] = jt_v + b;
        }

        workgroupBarrier();

        for (var i = tid; i < h_slots * d; i += 256u) {
            v_curr[i] = v_next[i];
        }
    }
    
    // Escribir resultado final para el pipeline fused: stage1a consume v_state.
    for (var i = tid; i < h_slots * d; i += 256u) {
        let out = v_curr[i];
        v_state[t * h_slots * d + i] = out;
        v_final[t * h_slots * d + i] = out;
    }
}
