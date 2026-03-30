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

const ENTRY_GRID_X: u32 = 65535u;
var<workgroup> shared_sumsq: array<f32, 64>;
var<workgroup> shared_coeff: array<f32, 64>;

fn entry_base(entry: u32, d: u32) -> u32 {
    return entry * d;
}

fn entry_workgroup_index(wid: vec3<u32>) -> u32 {
    return wid.y * ENTRY_GRID_X + wid.x;
}

fn scratch_stride(d: u32, h_slots: u32) -> u32 {
    return d * h_slots;
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
    let rhs = b_in[token * d + dim] / max(1.0, f32(h_slots));
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
    let off = entry_base(entry, d);
    let inv_d = 1.0 / max(1.0, f32(d));

    var local_sumsq = 0.0;
    var local_coeff = 0.0;
    for (var dim = lane; dim < d; dim = dim + 64u) {
        let pre = Scratch[signal_base + dim] + H_star[off + dim];
        let up = params.damping * v_state[off + dim];
        local_sumsq = local_sumsq + pre * pre;
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
    let rhs_scale = 1.0 / max(1.0, f32(h_slots));

    for (var dim = lane; dim < d; dim = dim + 64u) {
        let pre = Scratch[signal_base + dim] + H_star[off + dim];
        let v_prev = v_state[off + dim];
        let jac_term =
            (NormScale[dim] * inv_rms) * (params.damping * v_prev)
            - pre * coeff_scale
            + (1.0 - params.damping) * v_prev;
        let rhs = b_in[token * d + dim] * rhs_scale;
        v_next[off + dim] = rhs + jac_term;
    }
}
