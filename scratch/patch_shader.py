
import sys

header = """const ASSOC_BANKS: u32 = 1u;
const ASSOC_HIST_META: u32 = 4u;
const RETAIN_RANK: u32 = 32u;
const ASSOC_RANK: u32 = 32u;
const WG_SIZE: u32 = 64u;
const FPM_RESIDUAL_SCALE: f32 = 0.1;
const FPM_GATE_BIAS: f32 = -1.5;
const FPM_READ_GRAD_SCALE: f32 = 0.5;

// Learning Rate Multipliers (Backward path)
const ASSOC_ADDR_GRAD_SCALE: f32 = 1.0;
const ASSOC_EVENT_GRAD_SCALE: f32 = 1.0;
const FPM_DIRECT_WRITE_SCALE: f32 = 0.5;

// Associative Retrieval Constants
const ASSOC_WRITE_CAP: f32 = 0.95;
const ASSOC_READ_BETA: f32 = 4.0;
const ASSOC_OCCUPIED_THRESHOLD: f32 = 1.0e-4;
const ASSOC_READ_SLOT_PRIOR_BETA: f32 = 8.0;

struct FpmBwdUniforms {
    d_model: u32,
    h_slots: u32,
    lr: f32,
    grad_scale: f32,
    ternary_flag: u32,
    weight_decay: f32,
    seq_len: u32,
    damping: f32,
    residual_alpha: f32,
    grad_accum_mode: u32,
    n_accum: u32,
    n_total_weights: u32,
    batch_size: u32,
    apply_accum: u32,
    _pad0: u32,
    assoc_lr_mult: f32,
    assoc_event_lr_mult: f32,
    assoc_alpha_lr_mult: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform>             params: FpmBwdUniforms;
@group(0) @binding(1) var<storage, read>       h_star: array<f32>;
@group(0) @binding(2) var<storage, read>       Scratch: array<f32>;
@group(0) @binding(3) var<storage, read>       fpm_m_buf: array<f32>;
@group(0) @binding(4) var<storage, read_write> AllGradients: array<f32>;
@group(0) @binding(5) var<storage, read_write> tbptt_carry_buf: array<f32>;
@group(0) @binding(6) var<storage, read_write> fpm_dm_buf: array<f32>;
@group(0) @binding(7) var<storage, read_write> fpm_minner_buf: array<f32>;
@group(0) @binding(8) var<storage, read_write> fpm_hunit_buf: array<f32>;
@group(0) @binding(9) var<storage, read_write> fpm_gx_buf: array<f32>;
@group(0) @binding(10) var<storage, read_write> v_state: array<f32>;
@group(0) @binding(11) var<storage, read_write> AssocState: array<f32>;
@group(0) @binding(12) var<storage, read_write> AssocHist: array<f32>;
@group(0) @binding(13) var<storage, read_write> AssocBwdDebug: array<f32>;
@group(0) @binding(14) var<storage, read> S_in: array<f32>;

@group(1) @binding(0) var<storage, read_write> AllWeights: array<f32>;

override ENABLE_ASSOC_TRANSITION_GATE: bool = false;
override ENABLE_ASSOC_SLOT_OWNER: bool = false;
override ENABLE_ASSOC_SLOT_ANCHOR: bool = false;
override ENABLE_ASSOC_REUSE_MATCH: bool = false;
override ENABLE_ASSOC_SLOT_STRIPE: bool = false;
override ENABLE_ASSOC_TIE_QK: bool = false;
override ENABLE_ASSOC_CONF_READ: bool = false;
override ENABLE_ASSOC_EVENT_GATE: bool = false;
override ASSOC_EVENT_L1: f32 = 0.001;

// ── Layout Functions (Clean Layout) ─────────────────────────────────────────
fn aw_wq_base(d: u32, h: u32) -> u32 { return 0u; }
fn aw_wk_base(d: u32, h: u32) -> u32 { return h * d * d + h * d; }
fn aw_wv_base(d: u32, h: u32) -> u32 { return aw_wk_base(d, h) + h * d * d + h * d; }
fn aw_wo_base(d: u32, h: u32) -> u32 { return aw_wv_base(d, h) + h * d * d; }
fn aw_win_base(d: u32, h: u32) -> u32 { return aw_wo_base(d, h) + h * d * d; }
fn aw_wx_base(d: u32, h: u32) -> u32 { return aw_win_base(d, h) + h * d * d; }
fn aw_wout_base(d: u32, h: u32) -> u32 { return aw_wx_base(d, h) + d * d; }
fn aw_alog_base(d: u32, h: u32) -> u32 { return aw_wout_base(d, h) + d * d; }
fn aw_nscale_base(d: u32, h: u32) -> u32 { return aw_alog_base(d, h) + h * d; }
fn aw_hist_base(d: u32, h: u32) -> u32 { return aw_nscale_base(d, h) + d; }

fn hist_mat_len(d: u32) -> u32 { return d * d; }
fn hist_scale_base(d: u32, h_slots: u32) -> u32 { return hist_mat_len(d); }
fn hist_bias_base(d: u32, h_slots: u32) -> u32 { return hist_scale_base(d, h_slots) + h_slots * d; }
fn hist_gate_base(d: u32, h_slots: u32) -> u32 { return hist_bias_base(d, h_slots) + h_slots * d; }
fn slot_anchor_base(d: u32, h_slots: u32) -> u32 { return hist_gate_base(d, h_slots) + h_slots; }
fn w_k_write_base(d: u32, h: u32) -> u32 { return slot_anchor_base(d, h) + h * d; }
fn w_v_write_base(d: u32, h: u32) -> u32 { return w_k_write_base(d, h) + h * d * RETAIN_RANK; }
fn hist_delta_bias_base(d: u32, h_slots: u32) -> u32 { return w_v_write_base(d, h_slots) + h_slots * RETAIN_RANK * d; }
fn hist_selective_flag_base(d: u32, h_slots: u32) -> u32 { return hist_delta_bias_base(d, h_slots) + h_slots * d; }
fn hist_gate_query_base(d: u32, h_slots: u32) -> u32 { return hist_delta_bias_base(d, h_slots) + h_slots * d + 21u; }
fn w_write_gate_base(d: u32, h_slots: u32) -> u32 { return hist_gate_query_base(d, h_slots) + h_slots * d; }
fn b_write_mem_base(d: u32, h_slots: u32) -> u32 { return w_write_gate_base(d, h_slots) + h_slots * d; }
fn hhist_gamma_base(d: u32, h: u32) -> u32 { return b_write_mem_base(d, h) + h; }
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

// ── Scratch Offsets ────────────────────────────────────────────────────────
fn scratch_stride(d: u32, h: u32) -> u32 {
    return d * (h * 8u) + h * h + h;
}
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

// ── Shared Variables ────────────────────────────────────────────────────────
var<workgroup> shared_up: array<f32, 32>;
var<workgroup> shared_kw: array<f32, 32>;
var<workgroup> shared_dup: array<f32, 32>;
var<workgroup> assoc_b_state: array<f32, 4096>;
var<workgroup> assoc_gb: array<f32, 4096>;
var<workgroup> assoc_q: array<f32, 32>;
var<workgroup> assoc_raw: array<f32, 32>;
var<workgroup> assoc_gprev: array<f32, 32>;
var<workgroup> shared_vec: array<f32, 512>;
var<workgroup> shared_red: array<f32, 64>;

fn reduce_sum(lane: u32, v: f32) -> f32 {
    shared_red[lane] = v;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            shared_red[lane] = shared_red[lane] + shared_red[lane + stride];
        }
        workgroupBarrier();
    }
    return shared_red[0];
}
"""

with open('artifacts/fpm_retain_bwd_backup.wgsl', 'r') as f:
    lines = f.readlines()

start_of_compute = -1
for i, line in enumerate(lines):
    if '@compute' in line:
        start_of_compute = i
        break

if start_of_compute != -1:
    with open('aideen-backbone/src/shaders/fused_fpm_retain_bwd.wgsl', 'w') as f:
        f.write('// FPM memory backward pass for the active Model-A write path.\n')
        f.write(header)
        f.write('\n')
        f.writelines(lines[start_of_compute:])
    print("Shader successfully reconstructed with ALL constants.")
else:
    print("Error: Could not find @compute in backup.")
    sys.exit(1)
