struct SharedUniforms {
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
};

@group(0) @binding(0) var<uniform> params: SharedUniforms;
@group(0) @binding(1) var<storage, read> fpm_dm_buf: array<f32>;
@group(0) @binding(2) var<storage, read> fpm_minner_buf: array<f32>;
@group(0) @binding(3) var<storage, read> fpm_hunit_buf: array<f32>;
@group(0) @binding(4) var<storage, read_write> AllGradients: array<f32>;
@group(0) @binding(5) var<storage, read> fpm_gx_buf: array<f32>;

@group(1) @binding(0) var<storage, read_write> AllWeights: array<f32>;

fn aw_wq_base(d: u32, h: u32) -> u32 { return 0u; }
fn aw_wk_base(d: u32, h: u32) -> u32 { return h * d * d + h * d; }
fn aw_wv_base(d: u32, h: u32) -> u32 { return aw_wk_base(d, h) + h * d * d + h * d; }
fn aw_wo_base(d: u32, h: u32) -> u32 { return aw_wv_base(d, h) + h * d * d; }
fn aw_win_base(d: u32, h: u32) -> u32 { return aw_wo_base(d, h) + h * d * d; }
fn aw_wx_base(d: u32, h: u32) -> u32 { return aw_win_base(d, h) + h * d * d; }
fn aw_wout_base(d: u32, h: u32) -> u32 { return aw_wx_base(d, h) + d * d; }
fn aw_alog_base(d: u32, h: u32) -> u32 { return aw_wout_base(d, h) + d * d; }

fn fpm_off(t_abs: u32, slot: u32, d: u32, h: u32) -> u32 {
    return t_abs * h * d + slot * d;
}

fn wx_scale() -> f32 {
    return 0.5;
}

@compute
@workgroup_size(16, 16, 1)
fn fused_fpm_stage_wout_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    let d = params.d_model;
    let h = params.h_slots;
    if (row >= d || col >= d) { return; }

    let n_entries = max(1.0, f32(params.batch_size * params.seq_len * h));
    var grad = 0.0;
    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        for (var t = 0u; t < params.seq_len; t = t + 1u) {
            let t_abs = b * params.seq_len + t;
            for (var slot = 0u; slot < h; slot = slot + 1u) {
                let off = fpm_off(t_abs, slot, d, h);
                grad = grad + fpm_dm_buf[off + row] * fpm_minner_buf[off + col];
            }
        }
    }
    grad = grad / n_entries;

    let idx = aw_wout_base(d, h) + row * d + col;
    let clip = 0.5;
    let wd_factor = 1.0 - params.lr * params.weight_decay;
    let raw_step = params.lr * grad;
    if (params.grad_accum_mode == 1u) {
        AllGradients[idx] += raw_step;
    } else {
        let before = AllWeights[idx];
        let after = before * wd_factor - clamp(raw_step, -clip, clip);
        AllWeights[idx] = after;
    }
}

@compute
@workgroup_size(64, 1, 1)
fn fused_fpm_stage_wx_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let d = params.d_model;
    let h = params.h_slots;
    if (dim >= d) { return; }

    let n_entries = max(1.0, f32(params.batch_size * params.seq_len * h));
    var grad = 0.0;
    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        for (var t = 0u; t < params.seq_len; t = t + 1u) {
            let t_abs = b * params.seq_len + t;
            for (var slot = 0u; slot < h; slot = slot + 1u) {
                let off = fpm_off(t_abs, slot, d, h);
                grad = grad + fpm_gx_buf[off + dim] * fpm_hunit_buf[off + dim];
            }
        }
    }
    grad = grad / n_entries;

    let idx = aw_wx_base(d, h) + dim * d + dim;
    let clip = 0.5;
    let wd_factor = 1.0 - params.lr * params.weight_decay;
    let wx_raw = AllWeights[idx];
    let wx_tanh = tanh(wx_raw);
    let wx_grad = grad * wx_scale() * (1.0 - wx_tanh * wx_tanh);
    let raw_step = params.lr * wx_grad;
    if (params.grad_accum_mode == 1u) {
        AllGradients[idx] += raw_step;
    } else {
        AllWeights[idx] = wx_raw * wd_factor - clamp(raw_step, -clip, clip);
    }
}
