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
@group(0) @binding(2) var<storage, read> AllWeights: array<f32>;
@group(0) @binding(3) var<storage, read_write> H_curr: array<f32>;
@group(0) @binding(4) var<storage, read_write> H_next: array<f32>;
@group(0) @binding(5) var<storage, read_write> Scratch: array<f32>;
@group(0) @binding(6) var<storage, read_write> H_pooled: array<f32>;
@group(0) @binding(7) var<storage, read_write> DebugLog: array<f32>;

// AllWeights layout offset functions.
// Layout: W_q | W_k | W_v | W_o | W_in | W_x | W_out | A_log | NormScale | HistParams
fn aw_wq_base(d: u32, h: u32) -> u32 { return 0u; }
fn aw_wk_base(d: u32, h: u32) -> u32 { return h*d*d + h*d; }
fn aw_wv_base(d: u32, h: u32) -> u32 { return aw_wk_base(d,h) + h*d*d + h*d; }
fn aw_wo_base(d: u32, h: u32) -> u32 { return aw_wv_base(d,h) + h*d*d; }
fn aw_win_base(d: u32, h: u32) -> u32 { return aw_wo_base(d,h) + d*d; }
fn aw_wx_base(d: u32, h: u32) -> u32 { return aw_win_base(d,h) + h*d*d; }
fn aw_wout_base(d: u32, h: u32) -> u32 { return aw_wx_base(d,h) + d*d; }
fn aw_alog_base(d: u32, h: u32) -> u32 { return aw_wout_base(d,h) + d*d; }
fn aw_nscale_base(d: u32, h: u32) -> u32 { return aw_alog_base(d,h) + h*d; }
fn aw_hist_base(d: u32, h: u32) -> u32 { return aw_nscale_base(d,h) + d; }

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
var<workgroup> wg_pool_w: array<f32, 16>; // attention-received pooling weights (max 16 slots)

fn hist_cap_mult() -> f32 {
    return 0.25;
}

fn hist_cap_floor_mult() -> f32 {
    return 0.08;
}

fn hist_cap_ratio_max() -> f32 {
    return 0.80;
}
var<workgroup> avg_mamba_rms_sum: f32;
var<workgroup> avg_q_rms_sum: f32;
var<workgroup> avg_k_rms_sum: f32;
var<workgroup> avg_v_rms_sum: f32;
var<workgroup> avg_mix_rms_sum: f32;
var<workgroup> avg_attn_out_rms_sum: f32;
var<workgroup> avg_attn_max_sum: f32;
var<workgroup> avg_attn_entropy_sum: f32;
var<workgroup> avg_combined_rms_sum: f32;
var<workgroup> combined_rms_curr: f32;
var<workgroup> v_gate_slot: array<f32, 64>;
var<workgroup> inj_rms_curr: f32;

fn hist_selective_a_floor() -> f32 {
    return 0.070;
}

fn hist_alpha_min_target() -> f32 {
    return 0.070;
}

fn hist_alpha_min_start() -> f32 {
    return 0.030;
}

fn hist_alpha_max() -> f32 {
    return 0.20;
}
// Attention weights [query_slot, key_slot], reused across all d_out for the iter.
var<workgroup> attn_w: array<f32, 64>;
// Tiled cache for mix vector during W_o * mix projection.
var<workgroup> mix_tile: array<f32, MIX_TILE>;
// Tiled cache for H_curr during fused Q/K/V matmul.
var<workgroup> h_tile: array<f32, WG_SIZE>;
// Tiled cache for prev_mamba / carry during W_hist prelude matmul.
var<workgroup> prev_tile: array<f32, WG_SIZE>;

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
    let aw_wq = aw_wq_base(d_model, h_slots);
    let aw_wk = aw_wk_base(d_model, h_slots);
    let aw_wv = aw_wv_base(d_model, h_slots);
    let aw_wo = aw_wo_base(d_model, h_slots);
    let aw_win = aw_win_base(d_model, h_slots);
    let aw_wx = aw_wx_base(d_model, h_slots);
    let aw_wout = aw_wout_base(d_model, h_slots);
    let aw_alog = aw_alog_base(d_model, h_slots);
    let aw_nscale = aw_nscale_base(d_model, h_slots);
    let aw_hist = aw_hist_base(d_model, h_slots);

    let total_elements = h_slots * d_model;
    let h_base = batch_idx * total_elements;
    // Mode sentinels from host (residual_alpha encodes the active mode):
    //
    // MODO DEFAULT (en producción): hist_gated_mode — residual_alpha = -0.5
    //
    // Modos legacy (no utilizados, mantenidos como referencia de exploración):
    //   deq_only_mode   (<= -1.5): DEQ puro sin atención ni historia. Usado para
    //                              estabilización stage-0 antes de conectar atención.
    //   no_mamba_mode   (<= -1.0): Atención activa, historia desconectada. Etapa
    //                              intermedia durante el desarrollo.
    //   init_mamba_mode (<= -0.75): Historia entra solo en h_0 (no en cada iter).
    //                               Reemplazado por hist_gated que la inyecta como
    //                               constante stop-gradient en cada iteración.
    //   fixed_mamba_mode(<= -0.25): Historia como bias aditivo al signal.
    //                               Semánticamente distinto a hist_gated.
    //   full_mamba_mode (resto):    Post-convergencia sobreescribe h_curr con M_t.
    //                               Acopla SSM y DEQ — rompe la separación de roles.
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
        avg_q_rms_sum = 0.0;
        avg_k_rms_sum = 0.0;
        avg_v_rms_sum = 0.0;
        avg_mix_rms_sum = 0.0;
        avg_attn_out_rms_sum = 0.0;
        avg_attn_max_sum = 0.0;
        avg_attn_entropy_sum = 0.0;
        avg_combined_rms_sum = 0.0;
        combined_rms_curr = 0.0;
    }
    workgroupBarrier();

    var total_iters = 0u;
    var max_contractivity = 0.0;
    let scale = inverseSqrt(max(1.0, f32(d_model)));

    for (var t = 0u; t < shape.seq_len; t = t + 1u) {
        // --- Per-token Memory Striding for BPTT ---
        let scratch_stride = d_model * (h_slots * 8u) + h_slots * h_slots + h_slots;
        let batch_scratch_t = (batch_idx * shape.seq_len + t) * scratch_stride;
        let q_base = batch_scratch_t;
        let k_base = q_base + h_slots * d_model;
        let v_base = k_base + h_slots * d_model;
        let attn_base = v_base + h_slots * d_model;
        let mamba_base = attn_base + h_slots * d_model;
        let signal_base = mamba_base + h_slots * d_model;
        let m_inner_base = signal_base + h_slots * d_model;
        let hist_ctx_base = m_inner_base + h_slots * d_model;
        let attn_weight_base = hist_ctx_base + h_slots * d_model;
        let f_gate_scratch_base = attn_weight_base + h_slots * h_slots;
        let hist_mat_len = d_model * d_model;
        let hist_scale_base = hist_mat_len;
        let hist_bias_base = hist_scale_base + h_slots * d_model;
        let hist_gate_base = hist_bias_base + h_slots * d_model;
        let slot_anchor_base = hist_gate_base + h_slots;
        let hist_delta_base = slot_anchor_base + h_slots * d_model;
        let hist_delta_bias_base = hist_delta_base + h_slots * d_model * d_model;
        let hist_selective_flag_base = hist_delta_bias_base + d_model;
        let hist_warmup_base = hist_selective_flag_base + 1u;
        let hist_rms_floor_base = hist_warmup_base + 1u;
        let hist_contr_floor_base = hist_rms_floor_base + 1u;
        let hist_inject_flag_base = hist_contr_floor_base + 1u;
        let hist_minner_zero_base = hist_inject_flag_base + 1u;
        let hist_force_nomamba_base = hist_minner_zero_base + 1u;
        let hist_prelude_skip_base = hist_force_nomamba_base + 1u;
        let hist_loop_force_nomamba_base = hist_prelude_skip_base + 1u;
        let signal_zero_base = hist_loop_force_nomamba_base + 1u;
        let attn_out_mode_base = signal_zero_base + 1u;
        let attn_uniform_base = attn_out_mode_base + 1u;
        let attn_freeze_base = attn_uniform_base + 1u;
        let v_fixed_base = attn_freeze_base + 1u;
        let v_lag_base = v_fixed_base + 1u;
        let v_scale_base = v_lag_base + 1u;
        let signal_scale_base = v_scale_base + 1u;
        let v_gate_scale_base = signal_scale_base + 1u;
        let v_gate_bias_base = v_gate_scale_base + 1u;
        let v_norm_base = v_gate_bias_base + 1u;
        let v_norm_scale_base = v_norm_base + 1u;
        // W_gate_hist: h_slots × d_model dynamic gate query matrix (follows 21 scalars)
        let hist_gate_query_base = v_norm_scale_base + 1u;
        // W_forget: h_slots × d_model forget gate query (follows W_gate_hist)
        let w_forget_base = hist_gate_query_base + h_slots * d_model;
        // b_forget: h_slots bias for forget gate (follows W_forget)
        let b_forget_base = w_forget_base + h_slots * d_model;

        let h_base_t = (batch_idx * shape.seq_len + t) * total_elements;

        // Keep a dedicated copy of the forward historical context used by the DEQ loop.
        // Backward treats this branch as stop-gradient, so retaining the exact forward
        // value restores forward/backward consistency and avoids recomputing W_hist.
        for (var s = 0u; s < h_slots; s = s + 1u) {
            let off = s * d_model;
            for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                Scratch[hist_ctx_base + off + d_out] = 0.0;
            }
        }
        workgroupBarrier();

        // input_signal_s = W_in_s * s_t  (per-slot: each slot has its own W_in matrix)
        let s_in_base = batch_idx * (shape.seq_len * d_model) + t * d_model;
        for (var s = 0u; s < h_slots; s = s + 1u) {
            let slot_sig_off = s * d_model;
            let win_base = s * d_model * d_model;
            for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                var inj = 0.0;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    inj = inj + AllWeights[aw_win +win_base + j * d_model + d_out] * S_in[s_in_base + j];
                }
                if (AllWeights[aw_hist +signal_zero_base] > 0.5) {
                    inj = 0.0;
                }
                let signal_scale = AllWeights[aw_hist +signal_scale_base];
                Scratch[signal_base + slot_sig_off + d_out] = inj * signal_scale;
            }
        }
        workgroupBarrier();

        // Cache per-token inj_rms for V normalization.
        var local_inj_sumsq = 0.0;
        for (var s = 0u; s < h_slots; s = s + 1u) {
            let off = s * d_model;
            for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                let inj = Scratch[signal_base + off + d_out];
                local_inj_sumsq = local_inj_sumsq + inj * inj;
            }
        }
        shared_vals[tid] = local_inj_sumsq;
        workgroupBarrier();
        for (var stride_inj = WG_SIZE / 2u; stride_inj > 0u; stride_inj = stride_inj >> 1u) {
            if (tid < stride_inj) {
                shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride_inj];
            }
            workgroupBarrier();
        }
        if (tid == 0u) {
            inj_rms_curr = sqrt(
                shared_vals[0] / max(1.0, f32(d_model * h_slots)) + 1e-6
            );
        }
        workgroupBarrier();

        // If we start from a zero H (first token), seed H_curr with the input signal
        // to avoid a non-contractive first Picard step in the dry-start regime.
        if (t == 0u) {
            var local_inj_sumsq = 0.0;
            for (var s = 0u; s < h_slots; s = s + 1u) {
                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    let inj = Scratch[signal_base + s * d_model + d_out];
                    local_inj_sumsq = local_inj_sumsq + inj * inj;
                }
            }
            shared_vals[tid] = local_inj_sumsq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let inj_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model) * f32(h_slots)) + 1e-6);
            for (var s = 0u; s < h_slots; s = s + 1u) {
                let off = s * d_model;
                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    H_curr[h_base + off + d_out] =
                        Scratch[signal_base + off + d_out] / max(inj_rms, 1e-6);
                }
            }
            workgroupBarrier();
        }

        if (hist_gated_mode && AllWeights[aw_hist +hist_prelude_skip_base] < 0.5) {
            var local_inj_sumsq = 0.0;
            for (var s = 0u; s < h_slots; s = s + 1u) {
                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    let inj = Scratch[signal_base + s * d_model + d_out];
                    local_inj_sumsq = local_inj_sumsq + inj * inj;
                }
            }
            shared_vals[tid] = local_inj_sumsq;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                }
                workgroupBarrier();
            }
            let inj_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model) * f32(h_slots)) + 1e-6);
            for (var s = 0u; s < h_slots; s = s + 1u) {
                let off = s * d_model;
                var local_prev_sumsq = 0.0;
                for (var j = tid; j < d_model; j = j + WG_SIZE) {
                    var prev_v: f32;
                    if (t > 0u) {
                        let prev_mamba =
                            (batch_idx * shape.seq_len + t - 1u) * scratch_stride
                            + h_slots * 4u * d_model;
                        prev_v = Scratch[prev_mamba + off + j];
                    } else {
                        // M carry lives in the second half of H_curr buffer.
                        prev_v = H_curr[shape.batch_size * total_elements + batch_idx * total_elements + off + j];
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
                    var u = 0.0;
                    if (t > 0u) {
                        let prev_mamba =
                            (batch_idx * shape.seq_len + t - 1u) * scratch_stride
                            + h_slots * 4u * d_model;
                        // Tiled matmul: load prev[tile] cooperatively, accumulate W_hist * prev
                        for (var tile = 0u; tile < d_model; tile = tile + WG_SIZE) {
                            let j_load = tile + tid;
                            if (j_load < d_model) {
                                prev_tile[tid] = Scratch[prev_mamba + off + j_load] / prev_rms;
                            } else {
                                prev_tile[tid] = 0.0;
                            }
                            workgroupBarrier();
                            let tile_lim = min(WG_SIZE, d_model - tile);
                            for (var tj = 0u; tj < tile_lim; tj = tj + 1u) {
                                u = u + AllWeights[aw_hist + d_out * d_model + tile + tj] * prev_tile[tj];
                            }
                            workgroupBarrier();
                        }
                        u = u + AllWeights[aw_hist + hist_scale_base + off + d_out]
                            * (Scratch[prev_mamba + off + d_out] / prev_rms);
                    } else {
                        let carry_base = shape.batch_size * total_elements + batch_idx * total_elements + off;
                        // Tiled matmul: load carry[tile] cooperatively, accumulate W_hist * carry
                        for (var tile = 0u; tile < d_model; tile = tile + WG_SIZE) {
                            let j_load = tile + tid;
                            if (j_load < d_model) {
                                prev_tile[tid] = H_curr[carry_base + j_load] / prev_rms;
                            } else {
                                prev_tile[tid] = 0.0;
                            }
                            workgroupBarrier();
                            let tile_lim = min(WG_SIZE, d_model - tile);
                            for (var tj = 0u; tj < tile_lim; tj = tj + 1u) {
                                u = u + AllWeights[aw_hist + d_out * d_model + tile + tj] * prev_tile[tj];
                            }
                            workgroupBarrier();
                        }
                        u = u + AllWeights[aw_hist + hist_scale_base + off + d_out]
                            * (H_curr[carry_base + d_out] / prev_rms);
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
                let gate_logit = AllWeights[aw_hist +hist_gate_base + s];
                let warmup = clamp(AllWeights[aw_hist +hist_warmup_base], 0.0, 1.0);
                let alpha_min = hist_alpha_min_start()
                    + (hist_alpha_min_target() - hist_alpha_min_start()) * warmup;
                let alpha_max = hist_alpha_max();
                let alpha = alpha_min + (alpha_max - alpha_min) * (1.0 / (1.0 + exp(-gate_logit)));
                let hist_post = alpha * hist_rms;
                let hist_scale = min(1.0, (hist_cap_mult() * inj_rms) / max(hist_post, 1e-6));
                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    Scratch[m_inner_base + off + d_out] =
                        alpha * Scratch[m_inner_base + off + d_out] * hist_scale;
                }
                // Dynamic gate: query h*_{t-1} (carry in H_curr) against W_gate_hist[s].
                // Computed once per token/slot in prelude → zero overhead in DEQ loop.
                // gate_dyn = dot(W_gate_hist[s], h_{t-1}) / sqrt(d); hist_mod = 1+tanh(gate_dyn).
                var local_wgate = 0.0;
                for (var dj = tid; dj < d_model; dj = dj + WG_SIZE) {
                    local_wgate += AllWeights[aw_hist + hist_gate_query_base + off + dj]
                                   * H_curr[h_base + off + dj];
                }
                shared_vals[tid] = local_wgate;
                workgroupBarrier();
                for (var wg_stride = WG_SIZE / 2u; wg_stride > 0u; wg_stride = wg_stride >> 1u) {
                    if (tid < wg_stride) { shared_vals[tid] += shared_vals[tid + wg_stride]; }
                    workgroupBarrier();
                }
                let gate_dyn = shared_vals[0] / sqrt(f32(d_model));
                let hist_mod = 1.0 + tanh(gate_dyn);
                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    Scratch[m_inner_base + off + d_out] *= hist_mod;
                    Scratch[hist_ctx_base + off + d_out] = Scratch[m_inner_base + off + d_out];
                }
                // Forget gate: f[s] = σ(b_f[s] + dot(W_f[s,:], H_curr[t-1,s,:]) / sqrt(d))
                // Uses H_curr (h*_{t-1}), ∂/∂h=0. Applied in post-convergence SSM recurrence.
                var local_wf = AllWeights[aw_hist + b_forget_base + s]; // bias b_f[s]
                for (var dj = tid; dj < d_model; dj = dj + WG_SIZE) {
                    local_wf += AllWeights[aw_hist + w_forget_base + off + dj]
                                * H_curr[h_base + off + dj];
                }
                shared_vals[tid] = local_wf;
                workgroupBarrier();
                for (var wf_stride = WG_SIZE / 2u; wf_stride > 0u; wf_stride = wf_stride >> 1u) {
                    if (tid < wf_stride) { shared_vals[tid] += shared_vals[tid + wf_stride]; }
                    workgroupBarrier();
                }
                let f_logit = shared_vals[0] / sqrt(f32(d_model));
                let f_gate = 1.0 / (1.0 + exp(-f_logit));
                if (tid == 0u) {
                    Scratch[f_gate_scratch_base + s] = f_gate;
                }
                workgroupBarrier();
                if (tid == 0u) {
                    if (s == 0u) {
                        avg_inj_rms_sum = avg_inj_rms_sum + inj_rms;
                    }
                    avg_hist_rms_sum = avg_hist_rms_sum + hist_post * hist_scale;
                    avg_hist_ratio_sum = avg_hist_ratio_sum
                        + (hist_post * hist_scale) / max(inj_rms, 1e-6);
                }
                workgroupBarrier();
            }
        // [LEGACY — NO UTILIZADO] Prelude para fixed_mamba e init_mamba.
        // fixed_mamba: suma M_{t-1} promediado (single vector, no per-slot) al signal.
        //              No usa W_hist ni gate_logit. Reemplazado por hist_gated.
        // init_mamba:  inicializa H_curr[h_0] con M_{t-1} escalado.
        //              Reemplazado por hist_gated que inyecta en cada iter como constante.
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
            }
            // inj_rms: average across all per-slot signals
            for (var s = 0u; s < h_slots; s = s + 1u) {
                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    let inj = Scratch[signal_base + s * d_model + d_out];
                    local_inj_sumsq = local_inj_sumsq + inj * inj;
                }
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
            let inj_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model) * f32(h_slots)) + 1e-6);
            if (tid == 0u) {
                avg_inj_rms_sum = avg_inj_rms_sum + inj_rms;
                avg_hist_rms_sum = avg_hist_rms_sum + hist_rms;
                avg_hist_ratio_sum = avg_hist_ratio_sum + hist_rms / max(inj_rms, 1e-6);
            }
            let hist_scale = min(1.0, inj_rms / max(hist_rms, 1e-6));
            if (fixed_mamba_mode) {
                for (var s = 0u; s < h_slots; s = s + 1u) {
                    let slot_sig_off = s * d_model;
                    for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                        Scratch[signal_base + slot_sig_off + d_out] =
                            Scratch[signal_base + slot_sig_off + d_out]
                            + Scratch[mamba_base + d_out] * hist_scale;
                    }
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

        // Fixed V precompute (diagnostic): uses (signal + slot_anchor), one-time per token.
        // Note: hist_ctx is not retained after the solve (m_inner is overwritten by M_t),
        // so fixed projections must avoid hist_ctx to keep backward consistent.
        let v_fixed = AllWeights[aw_hist +v_fixed_base] > 0.5;
        let v_lag = AllWeights[aw_hist +v_lag_base] > 0.5;
        if (v_fixed && !deq_only_mode) {
            for (var s = 0u; s < h_slots; s = s + 1u) {
                let off = s * d_model;
                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    var v = 0.0;
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        let slot_bias = AllWeights[aw_hist +slot_anchor_base + off + j];
                        let signal = Scratch[signal_base + off + j];
                        let src = signal + slot_bias;
                        v = v + AllWeights[aw_wv + s * d_model * d_model + j * d_model + d_out] * src;
                    }
                    Scratch[v_base + off + d_out] = v;
                }
            }
            workgroupBarrier();
        }
        if (v_lag && !v_fixed && !deq_only_mode) {
            for (var s = 0u; s < h_slots; s = s + 1u) {
                let off = s * d_model;
                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    var v = 0.0;
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        let h_val = H_curr[h_base + off + j];
                        v = v + AllWeights[aw_wv + s * d_model * d_model + j * d_model + d_out] * h_val;
                    }
                    let v_scale = AllWeights[aw_hist +v_scale_base];
                    Scratch[v_base + off + d_out] = v * v_scale;
                }
            }
            workgroupBarrier();
        }

        var iter = 0u;
        var converged = false;
        while (iter < shape.max_iters && !converged) {
            var local_max_delta = 0.0;
            var local_max_h = 0.0;

            if (!deq_only_mode) {
                let attn_freeze = AllWeights[aw_hist +attn_freeze_base] > 0.5;
                if (iter == 0u || !attn_freeze) {
                // V gating (scalar per slot). If gate_scale<=0, gate=1.0.
                let v_gate_scale = AllWeights[aw_hist +v_gate_scale_base];
                let v_gate_bias = AllWeights[aw_hist +v_gate_bias_base];
                if (tid < h_slots) {
                    if (v_gate_scale > 0.0) {
                        let off = tid * d_model;
                        var sum = 0.0;
                        for (var j = 0u; j < d_model; j = j + 1u) {
                            let hv = H_curr[h_base + off + j];
                            sum = sum + hv * hv;
                        }
                        let rms = sqrt(sum / max(1.0, f32(d_model)) + 1e-12);
                        v_gate_slot[tid] = 1.0 / (1.0 + exp(-(v_gate_scale * (rms - v_gate_bias))));
                    } else {
                        v_gate_slot[tid] = 1.0;
                    }
                }
                workgroupBarrier();

                // Q/K/V per slot — Fused Tiled MatMul
                // Loads H_curr[off + j] once per tile into h_tile (shared memory),
                // then accumulates Q, K, V from the same tile in a single pass.
                // Index layout unchanged: w_idx = s * d² + j * d + d_out.
                for (var s = 0u; s < h_slots; s = s + 1u) {
                    let off = s * d_model;
                    let w_base = s * d_model * d_model;
                    var v_sum = 0.0;
                    for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                        var q = 0.0;
                        var k = 0.0;
                        var v = 0.0;
                        // --- Fused tile loop: load H_curr once, compute Q/K/V ---
                        for (var tile = 0u; tile < d_model; tile = tile + WG_SIZE) {
                            // Cooperative load of H_curr tile into shared memory
                            let j_load = tile + tid;
                            if (j_load < d_model) {
                                h_tile[tid] = H_curr[h_base + off + j_load];
                            } else {
                                h_tile[tid] = 0.0;
                            }
                            workgroupBarrier();

                            // Accumulate Q, K, V from the cached tile
                            let tile_limit = min(WG_SIZE, d_model - tile);
                            for (var tj = 0u; tj < tile_limit; tj = tj + 1u) {
                                let j = tile + tj;
                                let h_val = h_tile[tj];
                                let w_idx = w_base + j * d_model + d_out;
                                q = q + AllWeights[aw_wq + w_idx] * h_val;
                                k = k + AllWeights[aw_wk + w_idx] * h_val;
                                if (!v_fixed && !v_lag) {
                                    v = v + AllWeights[aw_wv + w_idx] * h_val;
                                }
                            }
                            workgroupBarrier();
                        }
                        // Per-slot Q/K bias
                        q = q + AllWeights[aw_wq + h_slots * d_model * d_model + s * d_model + d_out];
                        k = k + AllWeights[aw_wk + h_slots * d_model * d_model + s * d_model + d_out];
                        Scratch[q_base + off + d_out] = q;
                        Scratch[k_base + off + d_out] = k;
                        if (!v_fixed && !v_lag) {
                            v_sum = v_sum + v * v;
                            Scratch[v_base + off + d_out] = v;
                        }
                    }
                    if (!v_fixed && !v_lag) {
                        shared_vals[tid] = v_sum;
                        workgroupBarrier();
                        for (var stride_vr = WG_SIZE / 2u; stride_vr > 0u; stride_vr = stride_vr >> 1u) {
                            if (tid < stride_vr) {
                                shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride_vr];
                            }
                            workgroupBarrier();
                        }
                        let v_rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-12);
                        let v_scale = AllWeights[aw_hist + v_scale_base];
                        let v_norm_scale = AllWeights[aw_hist + v_norm_scale_base];
                        let v_cap = max(1e-6, inj_rms_curr);
                        let v_gain = (v_scale * v_norm_scale) * (v_cap / max(v_rms, 1e-6));
                        for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                            Scratch[v_base + off + d_out] =
                                Scratch[v_base + off + d_out] * v_gain * v_gate_slot[s];
                        }
                        workgroupBarrier();
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

                // Diagnostic: force uniform attention to isolate softmax sensitivity.
                if (AllWeights[aw_hist +attn_uniform_base] > 0.5) {
                    let p = 1.0 / f32(h_slots);
                    for (var qs = tid; qs < h_slots; qs = qs + WG_SIZE) {
                        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
                            attn_w[qs * h_slots + ks] = p;
                            Scratch[attn_weight_base + qs * h_slots + ks] = p;
                        }
                    }
                    workgroupBarrier();
                }

                // Entropy y max promediados sobre TODOS los slots (no solo slot 0).
                // Thread 0 recalcula desde Scratch para evitar race conditions.
                // attn_ent=log(8)=2.079 → slots uniformes. Debe bajar con entrenamiento real.
                if (tid == 0u && iter == 0u) {
                    var total_entropy = 0.0;
                    var total_max_p = 0.0;
                    for (var s = 0u; s < h_slots; s = s + 1u) {
                        var slot_ent = 0.0;
                        var slot_max = 0.0;
                        for (var k = 0u; k < h_slots; k = k + 1u) {
                            let p = Scratch[attn_weight_base + s * h_slots + k];
                            slot_max = max(slot_max, p);
                            slot_ent = slot_ent - p * log(max(p, 1.0e-8));
                        }
                        total_entropy = total_entropy + slot_ent;
                        total_max_p = total_max_p + slot_max;
                    }
                    avg_attn_entropy_sum = avg_attn_entropy_sum + total_entropy / f32(h_slots);
                    avg_attn_max_sum = avg_attn_max_sum + total_max_p / f32(h_slots);
                }

                // Build mixed V vector per query slot once:
                // mix[qs, j] = Σ_ks attn_w[qs,ks] * V[ks,j]
                // Reuse mamba_base as temporary buffer during the DEQ loop.
                // Optimization: load V[:, j] once per j and reuse across all qs.
                for (var j = tid; j < d_model; j = j + WG_SIZE) {
                    var vks: array<f32, MAX_SLOTS>;
                    for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
                        vks[ks] = Scratch[v_base + ks * d_model + j];
                    }
                    for (var qs = 0u; qs < h_slots; qs = qs + 1u) {
                        var mix = 0.0;
                        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
                            mix = mix + attn_w[qs * h_slots + ks] * vks[ks];
                        }
                        Scratch[mamba_base + qs * d_model + j] = mix;
                    }
                }
                workgroupBarrier();
                } // end (iter==0 || !attn_freeze)

                let attn_out_mode = AllWeights[aw_hist +attn_out_mode_base];
                if (attn_out_mode > 0.5) {
                    for (var s = 0u; s < h_slots; s = s + 1u) {
                        let off = s * d_model;
                        for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                            if (attn_out_mode > 1.5) {
                                Scratch[attn_base + off + d_out] = 0.0;
                            } else {
                                Scratch[attn_base + off + d_out] =
                                    Scratch[mamba_base + off + d_out];
                            }
                        }
                    }
                    workgroupBarrier();
                }

                // V lag update: recompute V from current H after attn_out,
                // so the next iteration uses V(h^ℓ) while this iteration used V(h^{ℓ-1}).
                if (v_lag && !v_fixed) {
                    for (var s = 0u; s < h_slots; s = s + 1u) {
                        let off = s * d_model;
                        let w_base = s * d_model * d_model;
                        for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                            var v_next = 0.0;
                            // Tiled matmul to reuse H_curr tile from shared memory.
                            for (var tile = 0u; tile < d_model; tile = tile + WG_SIZE) {
                                let j_load = tile + tid;
                                if (j_load < d_model) {
                                    h_tile[tid] = H_curr[h_base + off + j_load];
                                } else {
                                    h_tile[tid] = 0.0;
                                }
                                workgroupBarrier();
                                let tile_limit = min(WG_SIZE, d_model - tile);
                                for (var tj = 0u; tj < tile_limit; tj = tj + 1u) {
                                    let j = tile + tj;
                                    let h_val = h_tile[tj];
                                    v_next = v_next + AllWeights[aw_wv + w_base + j * d_model + d_out] * h_val;
                                }
                                workgroupBarrier();
                            }
                            let v_scale = AllWeights[aw_hist +v_scale_base];
                            Scratch[v_base + off + d_out] = v_next * v_scale;
                        }
                    }
                    workgroupBarrier();
                }

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
                                out = out + AllWeights[aw_wo +j * d_model + d_out] * mix_tile[l];
                            }
                            workgroupBarrier();
                        }
                        Scratch[attn_base + q_off + d_out] = out;
                    }
                }
                workgroupBarrier();

                if (iter == 0u) {
                    let off0 = 0u;
                    var q_sumsq = 0.0;
                    var k_sumsq = 0.0;
                    var v_sumsq = 0.0;
                    for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                        let qv = Scratch[q_base + off0 + d_out];
                        let kv = Scratch[k_base + off0 + d_out];
                        let vv = Scratch[v_base + off0 + d_out];
                        q_sumsq = q_sumsq + qv * qv;
                        k_sumsq = k_sumsq + kv * kv;
                        v_sumsq = v_sumsq + vv * vv;
                    }
                    shared_vals[tid] = q_sumsq;
                    workgroupBarrier();
                    for (var stride_q = WG_SIZE / 2u; stride_q > 0u; stride_q = stride_q >> 1u) {
                        if (tid < stride_q) {
                            shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride_q];
                        }
                        workgroupBarrier();
                    }
                    if (tid == 0u) {
                        avg_q_rms_sum = avg_q_rms_sum
                            + sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);
                    }
                    workgroupBarrier();

                    shared_vals[tid] = k_sumsq;
                    workgroupBarrier();
                    for (var stride_k = WG_SIZE / 2u; stride_k > 0u; stride_k = stride_k >> 1u) {
                        if (tid < stride_k) {
                            shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride_k];
                        }
                        workgroupBarrier();
                    }
                    if (tid == 0u) {
                        avg_k_rms_sum = avg_k_rms_sum
                            + sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);
                    }
                    workgroupBarrier();

                    shared_vals[tid] = v_sumsq;
                    workgroupBarrier();
                    for (var stride_v = WG_SIZE / 2u; stride_v > 0u; stride_v = stride_v >> 1u) {
                        if (tid < stride_v) {
                            shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride_v];
                        }
                        workgroupBarrier();
                    }
                    if (tid == 0u) {
                        avg_v_rms_sum = avg_v_rms_sum
                            + sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);
                    }
                    workgroupBarrier();

                    var mix_sumsq = 0.0;
                    var out_sumsq = 0.0;
                    for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                        let mv = Scratch[mamba_base + off0 + d_out];
                        let ov = Scratch[attn_base + off0 + d_out];
                        mix_sumsq = mix_sumsq + mv * mv;
                        out_sumsq = out_sumsq + ov * ov;
                    }
                    shared_vals[tid] = mix_sumsq;
                    workgroupBarrier();
                    for (var stride_m = WG_SIZE / 2u; stride_m > 0u; stride_m = stride_m >> 1u) {
                        if (tid < stride_m) {
                            shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride_m];
                        }
                        workgroupBarrier();
                    }
                    if (tid == 0u) {
                        avg_mix_rms_sum = avg_mix_rms_sum
                            + sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);
                    }
                    workgroupBarrier();

                    shared_vals[tid] = out_sumsq;
                    workgroupBarrier();
                    for (var stride_o = WG_SIZE / 2u; stride_o > 0u; stride_o = stride_o >> 1u) {
                        if (tid < stride_o) {
                            shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride_o];
                        }
                        workgroupBarrier();
                    }
                    if (tid == 0u) {
                        avg_attn_out_rms_sum = avg_attn_out_rms_sum
                            + sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);
                    }
                    workgroupBarrier();
                }
            } else {
                // [LEGACY — NO UTILIZADO] deq_only_mode: anula attn_out para estabilizar
                // el DEQ en etapas tempranas de desarrollo sin la rama de atención activa.
                // En hist_gated (default) este else nunca ejecuta.
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
                let hist_inject = AllWeights[aw_hist +hist_inject_flag_base];
                let hist_force_nomamba = AllWeights[aw_hist +hist_force_nomamba_base];
                let hist_loop_force_nomamba = AllWeights[aw_hist +hist_loop_force_nomamba_base];
                let hist_minner_zero = AllWeights[aw_hist +hist_minner_zero_base];

                for (var d = tid; d < d_model; d = d + WG_SIZE) {
                    var hist_ctx = 0.0;
                    if (hist_gated_mode && hist_inject > 0.5 && hist_minner_zero < 0.5) {
                        hist_ctx = Scratch[hist_ctx_base + off + d];
                    }
                    let slot_bias = AllWeights[aw_hist +slot_anchor_base + off + d];
                    // attn_signal is the only h-dependent term (∂slot_bias/∂h = 0,
                    // ∂hist_ctx/∂h = 0). rms is anchored to attn_signal alone so that
                    // the Lipschitz constant is independent of slot_bias and hist_ctx
                    // magnitude. Both still enter the numerator (final_combined), so
                    // they shift h* and provide slot identity / history context.
                    let attn_signal = Scratch[attn_base + off + d]
                        + Scratch[signal_base + off + d];
                    var final_combined = attn_signal + slot_bias + hist_ctx;
                    if (hist_force_nomamba > 0.5 || hist_loop_force_nomamba > 0.5) {
                        final_combined = attn_signal + slot_bias;
                    }
                    H_next[h_base_t + off + d] = final_combined;
                    local_sumsq = local_sumsq + attn_signal * attn_signal;
                }
                shared_vals[tid] = local_sumsq;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let rms_floor = AllWeights[aw_hist +hist_rms_floor_base];
                // Smooth floor inside sqrt to keep the map differentiable and enforce
                // a minimum RMS magnitude without regime switches.
                var rms = sqrt(
                    shared_vals[0] / max(1.0, f32(d_model))
                    + rms_floor * rms_floor
                    + 1e-6
                );
                if (s == 0u && tid == 0u) {
                    combined_rms_curr = rms;
                }
                if (iter == 0u && s == 0u && tid == 0u) {
                    avg_combined_rms_sum = avg_combined_rms_sum + rms;
                }

                for (var d = tid; d < d_model; d = d + WG_SIZE) {
                    let h_prev = H_curr[h_base + off + d];
                    var f_h = AllWeights[aw_nscale +d] * (H_next[h_base_t + off + d] / rms);
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
                let contr_floor = AllWeights[aw_hist +hist_contr_floor_base];
                let d_curr = shared_vals[0];
                let d_prev = last_delta;
                curr_contractivity = 0.0;
                // Only measure contractivity when the residual is meaningfully
                // larger than the convergence threshold.  Near epsilon, d_prev
                // is dominated by numerical noise and the ratio d_curr/d_prev
                // can exceed 1.0 as a measurement artifact even when the map
                // is globally contractive.  The factor 10 gives a safe margin.
                if (iter > 0u && d_prev > 1e-12 && d_prev > shape.epsilon * 10.0) {
                    curr_contractivity = d_curr / d_prev;
                }
                if (contr_floor > 0.0 && combined_rms_curr < contr_floor) {
                    curr_contractivity = 0.0;
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
            let d = last_delta;
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
            let hist_selective = hist_gated_mode && AllWeights[aw_hist +hist_selective_flag_base] > 0.5;
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

                let warmup = clamp(AllWeights[aw_hist +hist_warmup_base], 0.0, 1.0);
                let alpha_min = hist_alpha_min_start()
                    + (hist_alpha_min_target() - hist_alpha_min_start()) * warmup;
                // 1. x_proj = (I + W_x) * RMSUnit(H^*)
                for (var d = tid; d < d_model; d = d + WG_SIZE) {
                    // Safe temporal reading: use cross-sequence carry on first token
                    var m_prev: f32;
                    if (t > 0u) {
                        let prev_mamba = (batch_idx * shape.seq_len + t - 1u) * scratch_stride + h_slots * 4u * d_model;
                        m_prev = Scratch[prev_mamba + off + d];
                    } else {
                        m_prev = H_curr[shape.batch_size * total_elements + batch_idx * total_elements + off + d];
                    }
                    var x_proj = H_curr[h_base + off + d] / h_rms;
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        x_proj = x_proj + AllWeights[aw_wx +d * d_model + j]
                            * (H_curr[h_base + off + j] / h_rms);
                    }
                    var a = 1.0 / (1.0 + exp(AllWeights[aw_alog +s * d_model + d]));
                    if (hist_selective) {
                       var delta_pre = AllWeights[aw_hist +hist_delta_bias_base + d];
                        for (var j = 0u; j < d_model; j = j + 1u) {
                            delta_pre = delta_pre
                                + AllWeights[aw_hist +hist_delta_base + s * d_model * d_model + d * d_model + j]
                                * (H_curr[h_base + off + j] / h_rms);
                        }
                        // Center delta so that delta_pre=0 => a_core = a_base (neutral).
                        // Make delta factor default to 1.0 so a_core defaults to a_base.
                        // a_core = a_base^{delta_factor}, with delta_factor in [0.5, 1.5].
                        // Use a hard floor instead of affine mixing so neutral selectivity
                        // preserves the non-selective decay (a_t = a_base when a_base > floor).
                        let delta_factor = 1.0 + 0.5 * tanh(delta_pre);
                        let a_core = pow(max(a, 1.0e-6), delta_factor);
                        if (a_core < alpha_min) {
                            a = alpha_min;
                        } else {
                            a = a_core;
                        }
                    }
                    // Keep temporal inner state separate from attention Q/K/V buffers.
                    // f_gate: scalar forget gate per-slot, computed in prelude from H_curr (∂/∂h=0).
                    let f_g = Scratch[f_gate_scratch_base + s];
                    Scratch[m_inner_base + off + d] = a * f_g * m_prev + (1.0 - a) * x_proj;
                }
                workgroupBarrier();

                // 2. M_t update. Keep the residual identity path in M-space as well.
                var local_out_sumsq = 0.0;
                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    var out = Scratch[m_inner_base + off + d_out];
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        out = out + AllWeights[aw_wout +d_out * d_model + j] * Scratch[m_inner_base + off + j];
                    }
                    Scratch[mamba_base + off + d_out] = out;
                    if (t == shape.seq_len - 1u) {
                        H_curr[shape.batch_size * total_elements + batch_idx * total_elements + off + d_out] = out;
                    }
                    local_out_sumsq = local_out_sumsq + out * out;
                    // [LEGACY — NO UTILIZADO] full_mamba_mode sobreescribe h_curr con M_t
                    // post-convergencia, acoplando el SSM directamente al estado DEQ.
                    // En hist_gated (default), M_t solo vive en Scratch y alimenta al
                    // siguiente token vía prelude — no toca H_curr.
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

        // Compute attention-received pooling weights (stop-gradient, thread 0 only).
        // 70% attention-received + 30% uniform — active slots get stronger gradient,
        // dead slots get small but non-zero gradient to allow recovery.
        if (tid == 0u) {
            let w_uniform = 1.0 / f32(h_slots);
            for (var k = 0u; k < h_slots; k = k + 1u) {
                var recv_k = 0.0;
                for (var q = 0u; q < h_slots; q = q + 1u) {
                    recv_k = recv_k + Scratch[attn_weight_base + q * h_slots + k];
                }
                // total recv = h_slots (each row sums to 1), recv_k/h_slots normalizes
                let w_attn = recv_k / f32(h_slots);
                wg_pool_w[k] = 0.7 * w_attn + 0.3 * w_uniform;
            }
        }
        workgroupBarrier();

        for (var d = tid; d < d_model; d = d + WG_SIZE) {
            var acc = 0.0;
            for (var s = 0u; s < h_slots; s = s + 1u) {
                acc = acc + wg_pool_w[s] * H_curr[h_base + s * d_model + d];
            }
            H_pooled[batch_idx * (shape.seq_len * d_model) + t * d_model + d] = acc;
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
        DebugLog[26] = avg_q_rms_sum / max(1.0, f32(shape.seq_len));
        DebugLog[27] = avg_k_rms_sum / max(1.0, f32(shape.seq_len));
        DebugLog[28] = avg_v_rms_sum / max(1.0, f32(shape.seq_len));
        DebugLog[29] = avg_mix_rms_sum / max(1.0, f32(shape.seq_len));
        DebugLog[30] = avg_attn_out_rms_sum / max(1.0, f32(shape.seq_len));
        DebugLog[31] = avg_attn_max_sum / max(1.0, f32(shape.seq_len));
        DebugLog[32] = avg_attn_entropy_sum / max(1.0, f32(shape.seq_len));
        DebugLog[33] = avg_combined_rms_sum / max(1.0, f32(shape.seq_len));
        DebugLog[100] = AllWeights[aw_hist +0];
        DebugLog[101] = AllWeights[aw_hist +1];
        DebugLog[102] = AllWeights[aw_hist +2];
        let hist_mat_len_dbg = d_model * d_model;
        let hist_scale_base_dbg = hist_mat_len_dbg;
        let hist_bias_base_dbg = hist_scale_base_dbg + h_slots * d_model;
        let hist_gate_base_dbg = hist_bias_base_dbg + h_slots * d_model;
        let slot_anchor_base_dbg = hist_gate_base_dbg + h_slots;
        let hist_delta_base_dbg = slot_anchor_base_dbg + h_slots * d_model;
        let hist_delta_bias_base_dbg = hist_delta_base_dbg + h_slots * d_model * d_model;
        let hist_selective_flag_base_dbg = hist_delta_bias_base_dbg + d_model;
        let hist_warmup_base_dbg = hist_selective_flag_base_dbg + 1u;
        let hist_rms_floor_base_dbg = hist_warmup_base_dbg + 1u;
        let hist_contr_floor_base_dbg = hist_rms_floor_base_dbg + 1u;
        let hist_inject_flag_base_dbg = hist_contr_floor_base_dbg + 1u;
        let hist_minner_zero_base_dbg = hist_inject_flag_base_dbg + 1u;
        let hist_force_nomamba_base_dbg = hist_minner_zero_base_dbg + 1u;
        let hist_prelude_skip_base_dbg = hist_force_nomamba_base_dbg + 1u;
        let hist_loop_force_nomamba_base_dbg = hist_prelude_skip_base_dbg + 1u;

        DebugLog[103] = AllWeights[aw_hist +slot_anchor_base_dbg];
        DebugLog[104] = AllWeights[aw_hist +slot_anchor_base_dbg + 1u];
        DebugLog[105] = AllWeights[aw_hist +hist_rms_floor_base_dbg];
        DebugLog[106] = AllWeights[aw_hist +hist_contr_floor_base_dbg];
        DebugLog[107] = AllWeights[aw_hist +hist_inject_flag_base_dbg];
        DebugLog[108] = AllWeights[aw_hist +hist_minner_zero_base_dbg];
        DebugLog[109] = AllWeights[aw_hist +hist_force_nomamba_base_dbg];
        DebugLog[110] = AllWeights[aw_hist +hist_prelude_skip_base_dbg];
        DebugLog[111] = AllWeights[aw_hist +hist_loop_force_nomamba_base_dbg];

        // Per-token debug (slot 0) when seq_len is small: H_rms, V_rms, attn_rms.
        if (tid == 0u && shape.seq_len <= 16u) {
            let scratch_stride = d_model * h_slots * 8u + h_slots * h_slots + h_slots;
            let base_out = 200u; // leave room for existing debug slots
            for (var t = 0u; t < shape.seq_len; t = t + 1u) {
                let h_base_t = (t * h_slots) * d_model;
                let scratch_base_t = t * scratch_stride;
                let v_base_t = scratch_base_t + 2u * h_slots * d_model;
                let attn_base_t = v_base_t + h_slots * d_model;

                var h_sum = 0.0;
                var v_sum = 0.0;
                var a_sum = 0.0;
                for (var j = 0u; j < d_model; j = j + 1u) {
                    let h_val = H_curr[h_base_t + j];
                    let v_val = Scratch[v_base_t + j];
                    let a_val = Scratch[attn_base_t + j];
                    h_sum = h_sum + h_val * h_val;
                    v_sum = v_sum + v_val * v_val;
                    a_sum = a_sum + a_val * a_val;
                }
                let inv_d = 1.0 / max(1.0, f32(d_model));
                let h_rms = sqrt(h_sum * inv_d + 1e-12);
                let v_rms = sqrt(v_sum * inv_d + 1e-12);
                let a_rms = sqrt(a_sum * inv_d + 1e-12);

                let idx = base_out + t * 3u;
                DebugLog[idx + 0u] = h_rms;
                DebugLog[idx + 1u] = v_rms;
                DebugLog[idx + 2u] = a_rms;
            }
        }
    }
}
