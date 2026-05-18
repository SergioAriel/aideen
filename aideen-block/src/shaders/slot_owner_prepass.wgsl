struct RunUniforms {
    batch_size: u32,
    d_model: u32,
    h_slots: u32,
    max_iters: u32,
    epsilon: f32,
    damping: f32,
    seq_len: u32,
    residual_alpha: f32,
    debug_enable: u32,
    token_start: u32,
    token_count: u32,
    diag_zero_win: u32,
    diag_one_iter: u32,
    fpm_stage: u32,
    fpm_alpha_m: f32,
    fpm_tau: f32,
    fpm_read_gate_min: f32,
    segment_len: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> shape: RunUniforms;
@group(0) @binding(1) var<storage, read> S_in: array<f32>;
@group(0) @binding(5) var<storage, read_write> Scratch: array<f32>;
@group(0) @binding(12) var<storage, read_write> PrevHStarBuf: array<f32>;
@group(1) @binding(4) var<storage, read_write> AllWeights: array<f32>;

override ENABLE_SLOT_OWNER_PREPASS: bool = false;
override SLOT_OWNER_BETA: f32 = 16.0;
override SLOT_OWNER_HASH_MIX: f32 = 8.0;

const MAX_SLOTS: u32 = 8u;

fn aw_hist_base(d: u32, h: u32) -> u32 {
    return (h * d * d + h * d) * 4u + h * d * d + d * d * 3u + h * d + d;
}

fn slot_anchor_base(d: u32, h: u32) -> u32 { return 1u + h; }

fn finite_f32(x: f32) -> bool {
    return (x == x) && abs(x) < 3.3e38;
}

fn finite_or(x: f32, fallback: f32) -> f32 {
    if (finite_f32(x)) {
        return x;
    }
    return fallback;
}

fn scratch_stride(d: u32, h: u32) -> u32 {
    let signal_span = d * h;
    return signal_span * 3u + h * h;
}

fn owner_current_base(batch_scratch_t: u32, d: u32, h: u32) -> u32 {
    let signal_span = d * h;
    return batch_scratch_t + signal_span * 2u;
}

fn owner_prev_base(batch_scratch_t: u32, d: u32, h: u32) -> u32 {
    return owner_current_base(batch_scratch_t, d, h) + h;
}

fn raw_source_at(batch_idx: u32, global_t: u32, d: u32, j: u32) -> f32 {
    return finite_or(S_in[(batch_idx * shape.seq_len + global_t) * d + j], 0.0);
}

fn prev_source_at(batch_idx: u32, global_t: u32, d: u32, h: u32, j: u32) -> f32 {
    if (global_t > 0u) {
        return raw_source_at(batch_idx, global_t - 1u, d, j);
    }
    return finite_or(PrevHStarBuf[batch_idx * h * d + j], 0.0);
}

fn slot_hash_weight(slot: u32, dim: u32) -> f32 {
    let x = f32(((slot + 1u) * 747796405u + (dim + 17u) * 2891336453u) & 1023u);
    return (x * 0.0019569471) - 1.0;
}

fn score_source_to_anchor(batch_idx: u32, global_t: u32, d: u32, h: u32, owner: u32, use_prev: bool) -> f32 {
    let anchor_root = aw_hist_base(d, h) + slot_anchor_base(d, h);
    let owner_off = anchor_root + owner * d;
    var src_sq = 0.0;
    var anchor_sq = 0.0;
    var dot = 0.0;
    var hash_dot = 0.0;
    for (var j = 0u; j < d; j = j + 1u) {
        let src_j = select(
            raw_source_at(batch_idx, global_t, d, j),
            prev_source_at(batch_idx, global_t, d, h, j),
            use_prev,
        );
        let anchor_j = finite_or(AllWeights[owner_off + j], 0.0);
        src_sq = src_sq + src_j * src_j;
        anchor_sq = anchor_sq + anchor_j * anchor_j;
        dot = dot + src_j * anchor_j;
        hash_dot = hash_dot + src_j * slot_hash_weight(owner, j);
    }
    let anchor_score = finite_or(dot * inverseSqrt(max(src_sq * anchor_sq, 1.0e-12)), 0.0);
    let hash_score = finite_or(hash_dot * inverseSqrt(max(src_sq * f32(d) / 3.0, 1.0e-12)), 0.0);
    return anchor_score + SLOT_OWNER_HASH_MIX * hash_score;
}

fn write_softmax(
    batch_idx: u32,
    global_t: u32,
    d: u32,
    h: u32,
    base: u32,
    start_slot: u32,
    end_slot_exclusive: u32,
    use_prev: bool,
) {
    if (h == 0u) {
        return;
    }
    let end_slot = min(h, max(start_slot + 1u, end_slot_exclusive));
    let count = max(1u, end_slot - start_slot);
    var scores: array<f32, MAX_SLOTS>;
    var max_score = -1.0e30;
    for (var s = 0u; s < MAX_SLOTS; s = s + 1u) {
        scores[s] = -1.0e30;
    }
    for (var s = start_slot; s < end_slot; s = s + 1u) {
        let score = score_source_to_anchor(batch_idx, global_t, d, h, s, use_prev);
        scores[s] = score;
        max_score = max(max_score, score);
    }
    var denom = 0.0;
    for (var s = start_slot; s < end_slot; s = s + 1u) {
        let e = exp(clamp(SLOT_OWNER_BETA * (scores[s] - max_score), -20.0, 20.0));
        scores[s] = e;
        denom = denom + e;
    }
    for (var s = 0u; s < h; s = s + 1u) {
        var p = 0.0;
        if (s >= start_slot && s < end_slot) {
            p = finite_or(scores[s] / max(denom, 1.0e-6), 1.0 / f32(count));
        }
        Scratch[base + s] = p;
    }
}

@compute @workgroup_size(1, 1, 1)
fn slot_owner_prepass_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    if (!ENABLE_SLOT_OWNER_PREPASS) {
        return;
    }
    let batch_idx = gid.x;
    let t = gid.y;
    if (batch_idx >= shape.batch_size || t >= shape.token_count) {
        return;
    }
    let d = shape.d_model;
    let h = min(shape.h_slots, MAX_SLOTS);
    let global_t = shape.token_start + t;
    let batch_scratch_t = (batch_idx * shape.seq_len + global_t) * scratch_stride(d, h);
    let curr_base = owner_current_base(batch_scratch_t, d, h);
    let prev_base = owner_prev_base(batch_scratch_t, d, h);

    write_softmax(batch_idx, global_t, d, h, curr_base, 0u, h, false);
    write_softmax(batch_idx, global_t, d, h, prev_base, 0u, h, true);
}
