struct RunUniforms {
    batch_size: u32,
    d_model: u32,
    h_slots: u32,
    max_iters: u32,
    epsilon: f32,
    damping: f32,
    seq_len: u32,
    residual_alpha: f32,
}

@group(0) @binding(0) var<uniform> shape: RunUniforms;
@group(0) @binding(1) var<storage, read> S_in: array<f32>;
@group(0) @binding(2) var<storage, read> W_q: array<f32>;
@group(0) @binding(3) var<storage, read> W_k: array<f32>;
@group(0) @binding(4) var<storage, read> W_v: array<f32>;
@group(0) @binding(5) var<storage, read> W_o: array<f32>;
@group(0) @binding(6) var<storage, read> W_in: array<f32>;
@group(0) @binding(7) var<storage, read> W_x: array<f32>;
@group(0) @binding(8) var<storage, read> W_out: array<f32>;
@group(0) @binding(9) var<storage, read> A_log: array<f32>;
@group(0) @binding(10) var<storage, read> NormScale: array<f32>;
@group(0) @binding(11) var<storage, read_write> H_curr: array<f32>;
@group(0) @binding(12) var<storage, read_write> H_next: array<f32>;
@group(0) @binding(13) var<storage, read_write> Scratch: array<f32>;
@group(0) @binding(14) var<storage, read_write> H_pooled: array<f32>;
@group(0) @binding(15) var<storage, read_write> DebugLog: array<f32>;
@group(0) @binding(16) var<storage, read> HistParams: array<f32>;

const WG_SIZE: u32 = 256u;
const MAX_SLOTS: u32 = 8u;
const MIX_TILE: u32 = 128u;

var<workgroup> shared_vals: array<f32, WG_SIZE>;
var<workgroup> hit_count: atomic<u32>;
var<workgroup> max_delta_seen: f32;
var<workgroup> last_delta: f32;
var<workgroup> s_delta: array<f32, WG_SIZE>;
var<workgroup> curr_contractivity: f32;
var<workgroup> max_h_seen: f32;
var<workgroup> avg_inj_rms_sum: f32;
var<workgroup> avg_hist_rms_sum: f32;
var<workgroup> avg_hist_ratio_sum: f32;

fn hist_cap_mult() -> f32 {
    return 1.0;
}

fn hist_cap_floor_mult() -> f32 {
    return 0.0;
}
var<workgroup> avg_mamba_rms_sum: f32;

fn hist_selective_a_floor() -> f32 {
    return 0.10;
}
// Attention weights [query_slot, key_slot], reused across all d_out for the iter.
var<workgroup> attn_w: array<f32, 64>;
// Tiled cache for mix vector during W_o * mix projection.
var<workgroup> mix_tile: array<f32, MIX_TILE>;

@compute @workgroup_size(256, 1, 1)
fn deq_forward_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.x;
    if (batch_idx >= shape.batch_size) { return; }

    let d_model = shape.d_model;
    let h_slots = shape.h_slots;
    if (h_slots == 0u || h_slots > MAX_SLOTS) { return; }

    let total_elements = h_slots * d_model;
    let h_base = batch_idx * total_elements;
    // Stage mode sentinels from host:
    // residual_alpha <= -1.5 => DEQ-only (no attention, no mamba)
    // residual_alpha <= -1.0 => no mamba (attention active)
    // residual_alpha <= -0.75 => init mamba (history only in h0)
    // residual_alpha <= -0.5 => hist gated per-slot context
    // residual_alpha <= -0.25 => fixed mamba bias per token (attention active)
    let deq_only_mode = shape.residual_alpha <= -1.5;
    let no_mamba_mode = shape.residual_alpha > -1.5 && shape.residual_alpha <= -1.0;
    let init_mamba_mode = shape.residual_alpha > -1.0 && shape.residual_alpha <= -0.75;
    let hist_gated_mode = shape.residual_alpha > -0.75 && shape.residual_alpha <= -0.5;
    let fixed_mamba_mode = shape.residual_alpha > -0.5 && shape.residual_alpha <= -0.25;
    let full_mamba_mode = !deq_only_mode && !no_mamba_mode && !init_mamba_mode && !hist_gated_mode && !fixed_mamba_mode;

    if (tid == 0u) {
        atomicStore(&hit_count, 0u);
        max_delta_seen = 0.0;
        last_delta = 0.0;
        curr_contractivity = 0.0;
        max_h_seen = 0.0;
        avg_inj_rms_sum = 0.0;
        avg_hist_rms_sum = 0.0;
        avg_hist_ratio_sum = 0.0;
        avg_mamba_rms_sum = 0.0;
    }
    workgroupBarrier();

    var total_iters = 0u;
    var max_contractivity = 0.0;
    let scale = inverseSqrt(max(1.0, f32(d_model)));

    for (var t = 0u; t < shape.seq_len; t = t + 1u) {
        // --- Per-token Memory Striding for BPTT ---
        let scratch_stride = d_model * (h_slots * 6u + 1u) + h_slots * h_slots;
        let batch_scratch_t = (batch_idx * shape.seq_len + t) * scratch_stride;
        let q_base = batch_scratch_t;
        let k_base = q_base + h_slots * d_model;
        let v_base = k_base + h_slots * d_model;
        let attn_base = v_base + h_slots * d_model;
        let mamba_base = attn_base + h_slots * d_model;
        let signal_base = mamba_base + h_slots * d_model;
        let m_inner_base = signal_base + d_model;
        let attn_weight_base = m_inner_base + h_slots * d_model;
        let hist_mat_len = d_model * d_model;
        let hist_scale_base = hist_mat_len;
        let hist_bias_base = hist_scale_base + h_slots * d_model;
        let hist_gate_base = hist_bias_base + h_slots * d_model;
        let slot_anchor_base = hist_gate_base + h_slots;
        let hist_delta_base = slot_anchor_base + h_slots * d_model;
        let hist_delta_bias_base = hist_delta_base + d_model * d_model;
        let hist_selective_flag_base = hist_delta_bias_base + d_model;
        
        let h_base_t = (batch_idx * shape.seq_len + t) * total_elements;

        // input_signal = W_in * s_t
        let s_in_base = batch_idx * (shape.seq_len * d_model) + t * d_model;
        for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
            var inj = 0.0;
            for (var j = 0u; j < d_model; j = j + 1u) {
                inj = inj + W_in[j * d_model + d_out] * S_in[s_in_base + j];
            }
            Scratch[signal_base + d_out] = inj;
        }
        workgroupBarrier();

        if (hist_gated_mode) {
            var local_inj_sumsq = 0.0;
            for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                let inj = Scratch[signal_base + d_out];
                local_inj_sumsq = local_inj_sumsq + inj * inj;
            }
            shared_vals[tid] = local_inj_sumsq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let inj_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);
            for (var s = 0u; s < h_slots; s = s + 1u) {
                let off = s * d_model;
                var local_prev_sumsq = 0.0;
                for (var j = tid; j < d_model; j = j + WG_SIZE) {
                    var prev_v = 0.0;
                    if (t > 0u) {
                        let prev_mamba =
                            (batch_idx * shape.seq_len + t - 1u) * scratch_stride
                            + h_slots * 4u * d_model;
                        prev_v = Scratch[prev_mamba + off + j];
                    }
                    local_prev_sumsq = local_prev_sumsq + prev_v * prev_v;
                }
                shared_vals[tid] = local_prev_sumsq;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let prev_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);
                var local_hist_sumsq = 0.0;
                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    // Hist-gated product route: history must come from M_{t-1}, not from a
                    // permanent additive bias. A non-zero bias keeps the historical channel
                    // "on" even when the temporal carrier is empty and breaks the intended
                    // semantics of the interface.
                    var u = 0.0;
                    if (t > 0u) {
                        let prev_mamba =
                            (batch_idx * shape.seq_len + t - 1u) * scratch_stride
                            + h_slots * 4u * d_model;
                        for (var j = 0u; j < d_model; j = j + 1u) {
                            let prev_v = Scratch[prev_mamba + off + j] / prev_rms;
                            u = u + HistParams[d_out * d_model + j] * prev_v;
                        }
                        u = u + HistParams[hist_scale_base + off + d_out]
                            * (Scratch[prev_mamba + off + d_out] / prev_rms);
                    }
                    Scratch[m_inner_base + off + d_out] = u;
                    local_hist_sumsq = local_hist_sumsq + u * u;
                }
                shared_vals[tid] = local_hist_sumsq;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let hist_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);
                let tau = max(
                    hist_cap_mult() * inj_rms,
                    hist_cap_floor_mult() * hist_rms,
                );
                let hist_scale = min(1.0, tau / max(hist_rms, 1e-6));
                let gate_logit = HistParams[hist_gate_base + s];
                // Keep the historical channel structurally alive. The previous
                // alpha range [0.05, 0.25] stabilized the DEQ but left the
                // interface too weak to learn on practical horizons. Raising the
                // floor and init energy here is the mathematically correct locus:
                // it changes the history-to-DEQ interface gain directly instead
                // of patching the optimizer.
                let alpha = 0.08 + 0.20 * (1.0 / (1.0 + exp(-gate_logit)));
                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    Scratch[m_inner_base + off + d_out] =
                        alpha * Scratch[m_inner_base + off + d_out] * hist_scale;
                }
                if (tid == 0u) {
                    if (s == 0u) {
                        avg_inj_rms_sum = avg_inj_rms_sum + inj_rms;
                    }
                    avg_hist_rms_sum = avg_hist_rms_sum + alpha * hist_rms * hist_scale;
                    avg_hist_ratio_sum = avg_hist_ratio_sum
                        + (alpha * hist_rms * hist_scale) / max(inj_rms, 1e-6);
                }
                workgroupBarrier();
            }
        } else if (fixed_mamba_mode || init_mamba_mode) {
            var local_hist_sumsq = 0.0;
            var local_inj_sumsq = 0.0;
            for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                var pooled_m = 0.0;
                if (t > 0u) {
                    let prev_mamba = (batch_idx * shape.seq_len + t - 1u) * scratch_stride + h_slots * 4u * d_model;
                    for (var s = 0u; s < h_slots; s = s + 1u) {
                        pooled_m = pooled_m + Scratch[prev_mamba + s * d_model + d_out];
                    }
                    pooled_m = pooled_m / max(1.0, f32(h_slots));
                }
                Scratch[mamba_base + d_out] = pooled_m;
                local_hist_sumsq = local_hist_sumsq + pooled_m * pooled_m;
                let inj = Scratch[signal_base + d_out];
                local_inj_sumsq = local_inj_sumsq + inj * inj;
            }
            shared_vals[tid] = local_hist_sumsq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let hist_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);
            shared_vals[tid] = local_inj_sumsq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let inj_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);
            if (tid == 0u) {
                avg_inj_rms_sum = avg_inj_rms_sum + inj_rms;
                avg_hist_rms_sum = avg_hist_rms_sum + hist_rms;
                avg_hist_ratio_sum = avg_hist_ratio_sum + hist_rms / max(inj_rms, 1e-6);
            }
            let hist_scale = min(1.0, inj_rms / max(hist_rms, 1e-6));
            if (fixed_mamba_mode) {
                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    Scratch[signal_base + d_out] = Scratch[signal_base + d_out]
                        + Scratch[mamba_base + d_out] * hist_scale;
                }
            } else if (init_mamba_mode) {
                for (var s = 0u; s < h_slots; s = s + 1u) {
                    let off = s * d_model;
                    for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                        H_curr[h_base + off + d_out] =
                            Scratch[mamba_base + d_out] * hist_scale;
                    }
                }
            }
        }
        workgroupBarrier();

        var iter = 0u;
        var converged = false;
        while (iter < shape.max_iters && !converged) {
            var local_max_delta = 0.0;
            var local_max_h = 0.0;

            if (!deq_only_mode) {
                // Q/K/V per slot
                for (var s = 0u; s < h_slots; s = s + 1u) {
                    let off = s * d_model;
                    for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                        var q = 0.0;
                        var k = 0.0;
                        var v = 0.0;
                        for (var j = 0u; j < d_model; j = j + 1u) {
                            let h_val = H_curr[h_base + off + j];
                            let w_idx = j * d_model + d_out;
                            q = q + W_q[w_idx] * h_val;
                            k = k + W_k[w_idx] * h_val;
                            v = v + W_v[w_idx] * h_val;
                        }
                        Scratch[q_base + off + d_out] = q;
                        Scratch[k_base + off + d_out] = k;
                        Scratch[v_base + off + d_out] = v;
                    }
                }
                workgroupBarrier();

                // Precompute score matrix once per iter (independent of d_out).
                for (var idx = tid; idx < h_slots * h_slots; idx = idx + WG_SIZE) {
                    let qs = idx / h_slots;
                    let ks = idx % h_slots;
                    let q_off = qs * d_model;
                    let k_off = ks * d_model;
                    var score = 0.0;
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        score = score + Scratch[q_base + q_off + j] * Scratch[k_base + k_off + j];
                    }
                    // Reuse shared_vals as temporary score storage for softmax.
                    // Clamp logits to avoid softmax over-sharpening that destabilizes DEQ iterations.
                    shared_vals[idx] = clamp(score * scale, -4.0, 4.0);
                }
                workgroupBarrier();

                // Softmax over key slots, per query slot.
                for (var qs = tid; qs < h_slots; qs = qs + WG_SIZE) {
                    var max_s = -1e30;
                    for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
                        max_s = max(max_s, shared_vals[qs * h_slots + ks]);
                    }
                    var sum_exp = 0.0;
                    for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
                        let e = exp(shared_vals[qs * h_slots + ks] - max_s);
                        attn_w[qs * h_slots + ks] = e;
                        sum_exp = sum_exp + e;
                    }
                    let inv_sum = 1.0 / max(sum_exp, 1e-12);
                for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
                        attn_w[qs * h_slots + ks] = attn_w[qs * h_slots + ks] * inv_sum;
                        Scratch[attn_weight_base + qs * h_slots + ks] = attn_w[qs * h_slots + ks];
                    }
                }
                workgroupBarrier();

                // Build mixed V vector per query slot once:
                // mix[qs, j] = Σ_ks attn_w[qs,ks] * V[ks,j]
                // Reuse mamba_base as temporary buffer during the DEQ loop.
                for (var qs = 0u; qs < h_slots; qs = qs + 1u) {
                    let q_off = qs * d_model;
                    for (var j = tid; j < d_model; j = j + WG_SIZE) {
                        var mix = 0.0;
                        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
                            mix = mix + attn_w[qs * h_slots + ks] * Scratch[v_base + ks * d_model + j];
                        }
                        Scratch[mamba_base + q_off + j] = mix;
                    }
                }
                workgroupBarrier();

                // cross-slot attention: attn_slot = W_o * mix
                for (var qs = 0u; qs < h_slots; qs = qs + 1u) {
                    let q_off = qs * d_model;
                    for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                        var out = 0.0;
                        for (var t0 = 0u; t0 < d_model; t0 = t0 + MIX_TILE) {
                            let tile_n = min(MIX_TILE, d_model - t0);
                            // Cooperative load of mix tile into workgroup memory.
                            for (var l = tid; l < tile_n; l = l + WG_SIZE) {
                                mix_tile[l] = Scratch[mamba_base + q_off + t0 + l];
                            }
                            workgroupBarrier();

                            // Consume tile.
                            for (var l = 0u; l < tile_n; l = l + 1u) {
                                let j = t0 + l;
                                out = out + W_o[j * d_model + d_out] * mix_tile[l];
                            }
                            workgroupBarrier();
                        }
                        Scratch[attn_base + q_off + d_out] = out;
                    }
                }
                workgroupBarrier();
            } else {
                // Stage-0 DEQ stabilization: disable attention branch entirely.
                for (var s = 0u; s < h_slots; s = s + 1u) {
                    let off = s * d_model;
                    for (var d = tid; d < d_model; d = d + WG_SIZE) {
                        Scratch[attn_base + off + d] = 0.0;
                    }
                }
                workgroupBarrier();
            }

            // v14: En el loop DEQ, combinamos atención e inyección de contexto.
            // La memoria temporal y la conexión residual interna se han eliminado.
            for (var s = 0u; s < h_slots; s = s + 1u) {
                let off = s * d_model;

                var local_sumsq = 0.0;
                for (var d = tid; d < d_model; d = d + WG_SIZE) {
                    let h_prev = H_curr[h_base + off + d];
                    let combined = Scratch[attn_base + off + d]
                        + Scratch[signal_base + d]
                        + select(0.0, Scratch[m_inner_base + off + d], hist_gated_mode)
                        + HistParams[slot_anchor_base + off + d];
                    H_next[h_base_t + off + d] = combined;
                    local_sumsq = local_sumsq + combined * combined;
                }
                shared_vals[tid] = local_sumsq;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);

                for (var d = tid; d < d_model; d = d + WG_SIZE) {
                    let h_prev = H_curr[h_base + off + d];
                    let f_h = NormScale[d] * (H_next[h_base_t + off + d] / rms);
                    let val = shape.damping * f_h + (1.0 - shape.damping) * h_prev;
                    local_max_delta = max(local_max_delta, abs(val - h_prev));
                    local_max_h = max(local_max_h, abs(val));
                    H_next[h_base_t + off + d] = val; // Store exact H^*_t
                }
                workgroupBarrier();
            }

            shared_vals[tid] = local_max_delta;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
                }
                workgroupBarrier();
            }

            if (shared_vals[0] < shape.epsilon) {
                converged = true;
            }

            if (tid == 0u) {
                let d_curr = shared_vals[0];
                let d_prev = last_delta;
                curr_contractivity = 0.0;
                if (iter > 0u && d_prev > 1e-12) {
                    curr_contractivity = d_curr / d_prev;
                }
                last_delta = d_curr;
                max_contractivity = max(max_contractivity, curr_contractivity);
            }
            workgroupBarrier();

            // Track state magnitude for debugging/oracle guardrails.
            shared_vals[tid] = local_max_h;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
                }
                workgroupBarrier();
            }
            if (tid == 0u) {
                max_h_seen = max(max_h_seen, shared_vals[0]);
            }
            workgroupBarrier();

            for (var i = tid; i < total_elements; i = i + WG_SIZE) {
                H_curr[h_base + i] = H_next[h_base_t + i];
            }
            workgroupBarrier();
            iter = iter + 1u;
        }

        total_iters = total_iters + iter;
        if (tid == 0u) {
            let d = shared_vals[0];
            max_delta_seen = max(max_delta_seen, d);
            if (!converged) {
                atomicAdd(&hit_count, 1u);
            }
        }
        workgroupBarrier();

        if (hist_gated_mode || init_mamba_mode || fixed_mamba_mode || full_mamba_mode) {
            // v14: Mamba step execution POST-CONVERGENCE (Temporal Memory Transition)
            // M_t = (I + W_out) (a * M_{t-1} + (1-a) * (I + W_x) * RMSUnit(H^*))
            // The carrier keeps an identity path in both projections so reopening W_x/W_out
            // cannot self-annihilate the temporal state.
            let hist_selective = hist_gated_mode && HistParams[hist_selective_flag_base] > 0.5;
            for (var s = 0u; s < h_slots; s = s + 1u) {
                let off = s * d_model;
                var local_h_sumsq = 0.0;
                for (var d = tid; d < d_model; d = d + WG_SIZE) {
                    let h_val = H_curr[h_base + off + d];
                    local_h_sumsq = local_h_sumsq + h_val * h_val;
                }
                shared_vals[tid] = local_h_sumsq;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let h_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);

                // 1. x_proj = (I + W_x) * RMSUnit(H^*)
                for (var d = tid; d < d_model; d = d + WG_SIZE) {
                    // Safe temporal reading: zero initialize on first token
                    var m_prev = 0.0;
                    if (t > 0u) {
                        let prev_mamba = (batch_idx * shape.seq_len + t - 1u) * scratch_stride + h_slots * 4u * d_model;
                        m_prev = Scratch[prev_mamba + off + d];
                    }
                    var x_proj = H_curr[h_base + off + d] / h_rms;
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        x_proj = x_proj + W_x[j * d_model + d]
                            * (H_curr[h_base + off + j] / h_rms);
                    }
                    var a = 1.0 / (1.0 + exp(A_log[d]));
                    if (hist_selective) {
                       var delta_pre = HistParams[hist_delta_bias_base + d];
                        for (var j = 0u; j < d_model; j = j + 1u) {
                            delta_pre = delta_pre
                                + HistParams[hist_delta_base + j * d_model + d]
                                * (H_curr[h_base + off + j] / h_rms);
                        }
                        // Center delta so that delta_pre=0 => a_core = a_base (neutral).
                        // Make delta factor default to 1.0 so a_core defaults to a_base.
                        // a_core = a_base^{delta_factor}, with delta_factor in [0.5, 1.5].
                        // Use a hard floor instead of affine mixing so neutral selectivity
                        // preserves the non-selective decay (a_t = a_base when a_base > floor).
                        let delta_factor = 1.0 + 0.5 * tanh(delta_pre);
                        let a_core = pow(max(a, 1.0e-6), delta_factor);
                        if (a_core < hist_selective_a_floor()) {
                            a = hist_selective_a_floor();
                        } else {
                            a = a_core;
                        }
                    }
                    // Keep temporal inner state separate from attention Q/K/V buffers.
                    Scratch[m_inner_base + off + d] = a * m_prev + (1.0 - a) * x_proj;
                }
                workgroupBarrier();

                // 2. M_t update. Keep the residual identity path in M-space as well.
                var local_out_sumsq = 0.0;
                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    var out = Scratch[m_inner_base + off + d_out];
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        out = out + W_out[j * d_model + d_out] * Scratch[m_inner_base + off + j];
                    }
                    // Fixed-mamba mode keeps this as an external temporal bias. Full mode writes
                    // it back into H_curr, reproducing the fully coupled path.
                    Scratch[mamba_base + off + d_out] = out;
                    local_out_sumsq = local_out_sumsq + out * out;
                    if (full_mamba_mode) {
                        H_curr[h_base + off + d_out] = out;
                    }
                }
                shared_vals[tid] = local_out_sumsq;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                if (tid == 0u) {
                    avg_mamba_rms_sum = avg_mamba_rms_sum
                        + sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);
                }
                workgroupBarrier();
            }
        }

        for (var d = tid; d < d_model; d = d + WG_SIZE) {
            var acc = 0.0;
            for (var s = 0u; s < h_slots; s = s + 1u) {
                acc = acc + H_curr[h_base + s * d_model + d];
            }
            H_pooled[batch_idx * (shape.seq_len * d_model) + t * d_model + d] = acc / f32(h_slots);
        }
        workgroupBarrier();
    }
    // (Removed outdated synced H_next readback. Backprop relies entirely on H_next[t*seq_len] exact preservation)
    workgroupBarrier();

    if (batch_idx == 0u && tid == 0u) {
        DebugLog[0] = 777.7;
        DebugLog[1] = f32(shape.batch_size);
        DebugLog[2] = f32(shape.d_model);
        DebugLog[10] = f32(shape.seq_len);
        DebugLog[11] = max_h_seen;
        DebugLog[12] = H_curr[h_base];
        DebugLog[13] = f32(total_iters) / max(1.0, f32(shape.seq_len));
        DebugLog[14] = select(0.0, 1.0, atomicLoad(&hit_count) == 0u);
        DebugLog[15] = f32(atomicLoad(&hit_count));
        DebugLog[16] = max_delta_seen;
        DebugLog[17] = last_delta;
        DebugLog[18] = 0.0;
        DebugLog[19] = f32(total_elements);
        DebugLog[20] = 999.9;
        DebugLog[21] = max_contractivity;
        let slot_norm = max(1.0, f32(shape.seq_len * shape.h_slots));
        DebugLog[22] = avg_inj_rms_sum / max(1.0, f32(shape.seq_len));
        DebugLog[23] = avg_hist_rms_sum / slot_norm;
        DebugLog[24] = avg_hist_ratio_sum / slot_norm;
        DebugLog[25] = avg_mamba_rms_sum / slot_norm;
    }
}
