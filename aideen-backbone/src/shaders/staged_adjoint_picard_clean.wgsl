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
    grad_accum_mode: u32,
    n_accum: u32,
    n_total_weights: u32,
    batch_size: u32,
    apply_accum: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: UpdateUniforms;
@group(0) @binding(1) var<storage, read> v_state: array<f32>;
@group(0) @binding(3) var<storage, read> H_star: array<f32>;
@group(0) @binding(5) var<storage, read> b_in: array<f32>;
@group(0) @binding(6) var<storage, read> Scratch: array<f32>;
@group(0) @binding(8) var<storage, read_write> v_next: array<f32>;

@group(1) @binding(8) var<storage, read_write> NormScale: array<f32>;
@group(1) @binding(9) var<storage, read_write> HistParams: array<f32>;

const ENTRY_GRID_X: u32 = 65535u;
var<workgroup> shared_sumsq: array<f32, 64>;
var<workgroup> shared_coeff: array<f32, 64>;

fn entry_base(entry: u32, d: u32) -> u32 {
    return entry * d;
}

fn entry_workgroup_index(wid: vec3<u32>) -> u32 {
    return wid.y * ENTRY_GRID_X + wid.x;
}

fn slot_coord_mode() -> bool {
    return params.residual_alpha > -1.5 && params.residual_alpha <= -0.9;
}

fn scratch_stride(d: u32, h_slots: u32) -> u32 {
    let signal_span = d * h_slots;
    // slot_coord forward stores [signal | attn | alpha] plus the h_slots² attention matrix.
    // The adjoint must use the same per-token stride or it will read the next token with a
    // shifted base after t=0, silently training against the wrong scratch blocks.
    let coord_span = signal_span * 3u + h_slots * h_slots;
    return select(signal_span, coord_span, slot_coord_mode());
}

fn hist_mat_len(d: u32) -> u32 {
    return d * d;
}

fn hist_scale_base(d: u32, h_slots: u32) -> u32 {
    return hist_mat_len(d);
}

fn hist_bias_base(d: u32, h_slots: u32) -> u32 {
    return hist_scale_base(d, h_slots) + h_slots * d;
}

fn hist_gate_base(d: u32, h_slots: u32) -> u32 {
    return hist_bias_base(d, h_slots) + h_slots * d;
}

fn slot_anchor_base(d: u32, h_slots: u32) -> u32 {
    return hist_gate_base(d, h_slots) + h_slots;
}

fn token_alpha_base(token: u32, d: u32, h_slots: u32) -> u32 {
    let signal_span = d * h_slots;
    return token * scratch_stride(d, h_slots) + signal_span * 2u;
}

fn token_attn_base(token: u32, slot: u32, d: u32, h_slots: u32) -> u32 {
    let signal_span = d * h_slots;
    return token * scratch_stride(d, h_slots) + signal_span + slot * d;
}

fn alpha_at(token: u32, qs: u32, ks: u32, d: u32, h_slots: u32) -> f32 {
    return Scratch[token_alpha_base(token, d, h_slots) + qs * h_slots + ks];
}

fn slot_coord_attn_scale(token: u32, slot: u32, d: u32, h_slots: u32) -> f32 {
    var src_sumsq = 0.0;
    var attn_sumsq = 0.0;
    let signal_base = token * scratch_stride(d, h_slots) + slot * d;
    let attn_base = token_attn_base(token, slot, d, h_slots);
    let anchor_base = slot_anchor_base(d, h_slots) + slot * d;
    for (var dim = 0u; dim < d; dim = dim + 1u) {
        let src = Scratch[signal_base + dim] + HistParams[anchor_base + dim];
        let attn = Scratch[attn_base + dim];
        src_sumsq = src_sumsq + src * src;
        attn_sumsq = attn_sumsq + attn * attn;
    }
    let src_rms = sqrt(src_sumsq / max(1.0, f32(d)) + 1.0e-6);
    let attn_rms = sqrt(attn_sumsq / max(1.0, f32(d)) + 1.0e-6);
    return src_rms / max(attn_rms, 1.0e-6);
}

fn slot_coord_rhs_weight(token: u32, slot: u32, d: u32, h_slots: u32) -> f32 {
    // The token state is pooled uniformly across slots before the LM head. The adjoint RHS
    // must therefore preserve that same symmetry; otherwise we inject a feedback channel
    // that does not exist in the forward and can spuriously amplify already-active slots.
    return 1.0 / max(1.0, f32(h_slots));
}

@compute
@workgroup_size(16, 16, 1)
fn picard_clean_init_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = gid.x;
    let entry = gid.y;
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (dim >= d || entry >= n_entries) { return; }

    let token = entry / h_slots;
    let slot = entry % h_slots;
    let rhs_scale = select(
        1.0 / max(1.0, f32(h_slots)),
        slot_coord_rhs_weight(token, slot, d, h_slots),
        slot_coord_mode(),
    );
    let rhs = b_in[token * d + dim] * rhs_scale;
    v_next[entry_base(entry, d) + dim] = rhs;
}

@compute
@workgroup_size(64, 1, 1)
fn picard_clean_step_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let lane = lid.x;
    let entry = entry_workgroup_index(wid);
    let d = params.d_model;
    let h_slots = params.h_slots;
    let n_entries = params.batch_size * params.seq_len * h_slots;
    if (entry >= n_entries) { return; }

    let token = entry / h_slots;
    let slot = entry % h_slots;
    let q_off = slot * d;
    let signal_base = token * scratch_stride(d, h_slots) + q_off;
    let slot_coord_mode_active = slot_coord_mode();
    let attn_base = signal_base + select(0u, d * h_slots, slot_coord_mode_active);
    let anchor_base = slot_anchor_base(d, h_slots) + q_off;
    let attn_scale = select(1.0, slot_coord_attn_scale(token, slot, d, h_slots), slot_coord_mode_active);
    let off = entry_base(entry, d);
    let inv_d = 1.0 / max(1.0, f32(d));

    var local_sumsq = 0.0;
    var local_coeff = 0.0;
    for (var dim = lane; dim < d; dim = dim + 64u) {
        let slot_branch = select(
            Scratch[signal_base + dim],
            Scratch[signal_base + dim] + Scratch[attn_base + dim] * attn_scale + HistParams[anchor_base + dim],
            slot_coord_mode_active,
        );
        let pre = slot_branch + H_star[off + dim];
        let up = params.damping * v_state[off + dim];
        let rms_term = select(pre, slot_branch, slot_coord_mode_active);
        local_sumsq = local_sumsq + rms_term * rms_term;
        local_coeff = local_coeff + up * NormScale[dim] * pre;
    }

    shared_sumsq[lane] = local_sumsq;
    shared_coeff[lane] = local_coeff;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            shared_sumsq[lane] = shared_sumsq[lane] + shared_sumsq[lane + stride];
            shared_coeff[lane] = shared_coeff[lane] + shared_coeff[lane + stride];
        }
        workgroupBarrier();
    }
    let sumsq = shared_sumsq[0];
    let coeff = shared_coeff[0];

    let rms = sqrt(sumsq * inv_d + 1.0e-6);
    let inv_rms = 1.0 / max(rms, 1.0e-6);
    let coeff_scale = coeff / max(f32(d) * rms * rms * rms, 1.0e-6);
    let rhs_scale = select(
        1.0 / max(1.0, f32(h_slots)),
        slot_coord_rhs_weight(token, slot, d, h_slots),
        slot_coord_mode_active,
    );

    for (var dim = lane; dim < d; dim = dim + 64u) {
        let slot_branch = select(
            Scratch[signal_base + dim],
            Scratch[signal_base + dim] + Scratch[attn_base + dim] * attn_scale + HistParams[anchor_base + dim],
            slot_coord_mode_active,
        );
        let pre = slot_branch + H_star[off + dim];
        let v_prev = v_state[off + dim];
        let jac_term =
            (NormScale[dim] * inv_rms) * (params.damping * v_prev)
            - select(pre, slot_branch, slot_coord_mode_active) * coeff_scale
            + (1.0 - params.damping) * v_prev;
        let rhs = b_in[token * d + dim] * rhs_scale;
        v_next[off + dim] = rhs + jac_term;
    }
}
