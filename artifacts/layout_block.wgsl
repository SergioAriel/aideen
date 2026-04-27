fn hist_scale_base(d: u32, h_slots: u32) -> u32 { return hist_mat_len(d); }
fn hist_bias_base(d: u32, h_slots: u32) -> u32 { return hist_scale_base(d, h_slots) + h_slots * d; }
fn hist_gate_base(d: u32, h_slots: u32) -> u32 { return hist_bias_base(d, h_slots) + h_slots * d; }
fn slot_anchor_base(d: u32, h_slots: u32) -> u32 { return hist_gate_base(d, h_slots) + h_slots; }
// Write path: factored k×v (replaces full-rank W_delta h*d² with 2*h*d*r)
// W_k_write[slot]: d×RETAIN_RANK  — projects c to write bottleneck
// W_v_write[slot]: RETAIN_RANK×d  — expands bottleneck to proposal
fn w_k_write_base(d: u32, h: u32) -> u32 { return slot_anchor_base(d, h) + h * d; }
fn w_v_write_base(d: u32, h: u32) -> u32 { return w_k_write_base(d, h) + h * d * RETAIN_RANK; }
fn hist_delta_bias_base(d: u32, h_slots: u32) -> u32 { return w_v_write_base(d, h_slots) + h_slots * RETAIN_RANK * d; }
fn hist_selective_flag_base(d: u32, h_slots: u32) -> u32 { return hist_delta_bias_base(d, h_slots) + h_slots * d; }
fn hist_alpha_warmup_factor_base(d: u32, h_slots: u32) -> u32 { return hist_selective_flag_base(d, h_slots) + 1u; }
fn hist_rms_floor_base(d: u32, h_slots: u32) -> u32 { return hist_alpha_warmup_factor_base(d, h_slots) + 1u; }
fn hist_contr_floor_base(d: u32, h_slots: u32) -> u32 { return hist_rms_floor_base(d, h_slots) + 1u; }
fn hist_inject_flag_base(d: u32, h_slots: u32) -> u32 { return hist_contr_floor_base(d, h_slots) + 1u; }
fn hist_minner_zero_base(d: u32, h_slots: u32) -> u32 { return hist_inject_flag_base(d, h_slots) + 1u; }
fn hist_force_nofpm_base(d: u32, h_slots: u32) -> u32 { return hist_minner_zero_base(d, h_slots) + 1u; }
fn hist_prelude_skip_base(d: u32, h_slots: u32) -> u32 { return hist_force_nofpm_base(d, h_slots) + 1u; }
fn hist_loop_force_nofpm_base(d: u32, h_slots: u32) -> u32 { return hist_prelude_skip_base(d, h_slots) + 1u; }
fn signal_zero_base(d: u32, h_slots: u32) -> u32 { return hist_loop_force_nofpm_base(d, h_slots) + 1u; }
fn attn_out_mode_base(d: u32, h_slots: u32) -> u32 { return signal_zero_base(d, h_slots) + 1u; }
fn attn_uniform_base(d: u32, h_slots: u32) -> u32 { return attn_out_mode_base(d, h_slots) + 1u; }
fn attn_freeze_base(d: u32, h_slots: u32) -> u32 { return attn_uniform_base(d, h_slots) + 1u; }
fn signal_scale_base(d: u32, h_slots: u32) -> u32 { return signal_zero_base(d, h_slots) + 7u; }
fn hist_gate_query_base(d: u32, h_slots: u32) -> u32 { return hist_delta_bias_base(d, h_slots) + h_slots * d + 21u; }
fn w_write_gate_base(d: u32, h_slots: u32) -> u32 { return hist_gate_query_base(d, h_slots) + h_slots * d; }
fn b_write_mem_base(d: u32, h_slots: u32) -> u32 { return w_write_gate_base(d, h_slots) + h_slots * d; }
// γ per slot sits after the factored write path and scalar controls:
// slot_anchor(h*d) + W_k_write(h*d*r) + W_v_write(h*r*d) + b_delta(d)
// + 21 flags + W_gate_hist(h*d) + W_write_gate(h*d) + b_write_mem(h).
fn hhist_gamma_base(d: u32, h: u32) -> u32 { return b_write_mem_base(d, h) + h; }
// Retain gate (low-rank): W_up (h×d×r), W_down (h×r×d), b_retain (h×d)
// r=32, placed after hhist_gamma (h floats)
const RETAIN_RANK: u32 = 32u;
fn w_retain_up_base(d: u32, h: u32) -> u32 { return hhist_gamma_base(d, h) + h; }
fn w_retain_down_base(d: u32, h: u32) -> u32 { return w_retain_up_base(d, h) + h * d * RETAIN_RANK; }
fn b_retain_base(d: u32, h: u32) -> u32 { return w_retain_down_base(d, h) + h * RETAIN_RANK * d; }
fn w_q_mem_base(d: u32, h: u32) -> u32 { return b_retain_base(d, h) + h * d; }
fn w_k_mem_base(d: u32, h: u32) -> u32 { return w_q_mem_base(d, h) + h * d * RETAIN_RANK; }
fn b_read_mem_base(d: u32, h: u32) -> u32 { return w_k_mem_base(d, h) + h * d * RETAIN_RANK; }
fn w_k_assoc_base(d: u32, h: u32) -> u32 { return b_read_mem_base(d, h) + h; }
fn w_v_assoc_base(d: u32, h: u32) -> u32 { return w_k_assoc_base(d, h) + h * d * ASSOC_RANK; }
fn w_q_assoc_base(d: u32, h: u32) -> u32 { return w_v_assoc_base(d, h) + h * d * ASSOC_RANK; }
fn alpha_assoc_base(d: u32, h: u32) -> u32 { return w_q_assoc_base(d, h) + h * d * ASSOC_RANK; }
fn w_event_assoc_base(d: u32, h: u32) -> u32 { return alpha_assoc_base(d, h) + h; }
fn b_event_assoc_base(d: u32, h: u32) -> u32 { return w_event_assoc_base(d, h) + h * d; }

@group(0) @binding(11) var<storage, read_write> H_hist: array<f32>;

override ENABLE_FPM: bool = false;
// Joint DEQ-memory path:
//   (h, m)^(k+1) = Phi((h, m)^k; signal, slot_ctx, anchor)
// Memory participates in the same fixed-point search as the token state.
override FPM_MEM_ITERS: u32 = 1u;
const FPM_ALPHA_H: f32 = 0.2;
const FPM_RESIDUAL_SCALE: f32 = 0.1;
const FPM_RESCUE_RESIDUAL_SCALE: f32 = 0.01;
const FPM_GATE_BIAS: f32 = -1.5;
const FPM_GATE_CLAMP: f32 = 0.5;
const FPM_FATIGUE_RATE: f32 = 0.002;
const FPM_RESCUE_TAIL: u32 = 2u;
const FPM_JOINT_POLICY_STAGE: u32 = 6u;
const FPM_DEAD_THRESHOLD: f32 = 0.01;
// Saturation now refers to near-max use of the slot's write budget.
// Under the old z∈[0, 0.5] clamp, 0.45 meant "almost fully saturated".
// With z reparameterized into [0, 1], the same semantics are recovered by
// checking for near-ceiling utilization instead of mid-range activity.
const FPM_SAT_THRESHOLD: f32 = 0.95;
const FPM_EPS: f32 = 1e-6;
const FPM_HOMEO_MIN_ITERS: u32 = 4u;
const FPM_HOMEO_ALPHA_ERR_SCALE: f32 = 0.15;
const FPM_HOMEO_PLATEAU_TOL: f32 = 0.10;

fn hstar_off(t_abs: u32, slot: u32, d: u32, h: u32) -> u32 {
    return t_abs * h * d + slot * d;
}
fn fpm_m_off(t_abs: u32, slot: u32, d: u32, h: u32) -> u32 {
    return t_abs * h * d + slot * d;
}
fn clean_signal_off(t_abs: u32, slot: u32, d: u32, h: u32) -> u32 {
    return t_abs * scratch_stride(d, h) + 0u;
}
fn routing_alpha_off(t: u32, s: u32, d: u32, h: u32) -> u32 {
    return t * scratch_stride(d, h) + d * (h * 8u) + s * h;
}
fn routing_g_alpha_off(t: u32, s: u32, d: u32, h: u32) -> u32 {
    return t * scratch_stride(d, h) + d * (h * 8u) + h * h + s;
}
fn scratch_stride(d: u32, h: u32) -> u32 {
    return d * (h * 8u) + h * h + h;
}
