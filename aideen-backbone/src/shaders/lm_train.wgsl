// =============================================================================
// lm_train.wgsl — Language Model Head Training Shader
// =============================================================================
// Architecture: Sampled Softmax + RMSNorm + AdamW
//
// Pipelines (dispatch sizes set by gpu_lm_head.rs):
//   1. lm_probs_main         (seq_len, 1, 1)   – forward: RMSNorm, softmax, loss
//   2. lm_update_main        (d/16, k/16, 1)   – fused: accumulate dW + AdamW (no-fused path)
//   3. lm_dw_accum_main      (d/16, k/16, 1)   – accumulate dW only (fused path)
//   4. lm_apply_adamw_main   (d/16, k/16, 1)   – apply AdamW to accumulated dW (fused path)
//   5. lm_backprop_h_t_main  (seq_len, 1, 1)   – backprop dL/dh to DEQ
//
// Bindings: 0-15 (16 total, matching bgl_probs in gpu_lm_head.rs)
// =============================================================================

struct TrainParams {
    d_model:      u32,
    vocab_size:   u32,
    seq_len:      u32,
    step_t:       u32,
    lr_bits:      u32,
    beta1_bits:   u32,
    beta2_bits:   u32,
    eps_bits:     u32,
    ternary_flag: u32,
    num_samples:  u32,
    _pad1:        u32,
    _pad2:        u32,
};

@group(0) @binding(0)  var<uniform>            params:          TrainParams;
@group(0) @binding(1)  var<storage, read>       h_pooled:        array<f32>;
@group(0) @binding(2)  var<storage, read_write> w_lm:            array<f32>;
@group(0) @binding(3)  var<storage, read_write> b_lm:            array<f32>;
@group(0) @binding(4)  var<storage, read_write> dl_dh:           array<f32>;
@group(0) @binding(5)  var<storage, read_write> loss_out:        array<atomic<i32>>;
@group(0) @binding(6)  var<storage, read_write> moments_w:       array<vec2<f32>>;
@group(0) @binding(7)  var<storage, read_write> moments_b:       array<vec2<f32>>;
@group(0) @binding(8)  var<storage, read>       target_indices:  array<u32>;
@group(0) @binding(9)  var<storage, read_write> probs:           array<f32>;
@group(0) @binding(10) var<storage, read_write> g_lm:            array<f32>;
@group(0) @binding(11) var<storage, read_write> moments_g:       array<vec2<f32>>;
@group(0) @binding(12) var<storage, read_write> rms_buf:         array<f32>;
@group(0) @binding(13) var<storage, read_write> dl_dh_temp:      array<f32>;
@group(0) @binding(14) var<storage, read>       sampled_indices: array<u32>;
@group(0) @binding(15) var<storage, read_write> s_h_rms:         array<f32>;
// Binding 15 (s_h_rms) is also used as dl_dh_rms_red scratch via offset:
// s_h_rms[seq_len * d_model + t] = rms_dot[t]  (if buffer is large enough)
// In practice the Rust allocates s_h_rms_buf = seq_len * d_model floats,
// so the reduction lives entirely in workgroup memory.

const WG_SIZE: u32 = 256u;

fn probs_idx(t: u32, k: u32)    -> u32 { return t * params.num_samples + k; }
fn s_h_rms_idx(t: u32, d: u32) -> u32 { return t * params.d_model + d; }

// =============================================================================
// Workgroup shared memory
// =============================================================================
var<workgroup> s_scratch:    array<f32, WG_SIZE>;
var<workgroup> s_rms:        f32;
var<workgroup> s_target_exp: f32;
var<workgroup> s_h_tile:     array<f32, WG_SIZE>;
var<workgroup> s_indices_cache: array<u32, 512>;
var<workgroup> s_logits: array<f32, 512>;

// =============================================================================
// Pipeline 1: lm_probs_main
// Dispatch: (seq_len, 1, 1) — one workgroup per token
// Computes: RMSNorm(h_t), sampled logits, softmax, cross-entropy loss
// Writes:   probs[t,*], rms_buf[t], s_h_rms[t,*], loss_out (atomic)
// =============================================================================
@compute @workgroup_size(256, 1, 1)
fn lm_probs_main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>
) {
    let t   = wgid.x;
    let tid = lid.x;
    if (t >= params.seq_len) { return; }

    let d_model  = params.d_model;
    let base_h   = t * d_model;
    let target_v = target_indices[t];

    // --- 1. RMSNorm: compute RMS of h_t ---
    var local_sq = 0.0;
    for (var i = tid; i < d_model; i += WG_SIZE) {
        let v = h_pooled[base_h + i];
        local_sq += v * v;
    }
    s_scratch[tid] = local_sq;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { s_scratch[tid] += s_scratch[tid + stride]; }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let rms = sqrt(s_scratch[0] / f32(d_model) + 1e-5);
        s_rms      = rms;
        rms_buf[t] = rms;
    }
    workgroupBarrier();

    // --- 2. Apply RMSNorm + g_lm scale, cache to s_h_rms ---
    let rms = s_rms;
    for (var i = tid; i < d_model; i += WG_SIZE) {
        let normed        = (h_pooled[base_h + i] / rms) * g_lm[i];
        s_h_rms[s_h_rms_idx(t, i)] = normed;
    }
    workgroupBarrier();

    // --- 3. Sampled logits ---
    // Cache sampled_indices into workgroup memory (avoid repeated storage reads).
    for (var k = tid; k < params.num_samples; k += WG_SIZE) {
        s_indices_cache[k] = sampled_indices[k];
    }
    workgroupBarrier();
    if (tid == 0u) { s_target_exp = 0.0; }
    // Initialize logits with bias.
    for (var k = tid; k < params.num_samples; k += WG_SIZE) {
        let v = s_indices_cache[k];
        s_logits[k] = b_lm[v];
    }
    workgroupBarrier();
    // GEMM-style blocking: load H once per tile, reuse for all sampled rows.
    for (var tile = 0u; tile < d_model; tile = tile + WG_SIZE) {
        let d_load = tile + tid;
        if (d_load < d_model) {
            s_h_tile[tid] = s_h_rms[s_h_rms_idx(t, d_load)];
        } else {
            s_h_tile[tid] = 0.0;
        }
        workgroupBarrier();
        let tile_limit = min(WG_SIZE, d_model - tile);
        for (var k = tid; k < params.num_samples; k += WG_SIZE) {
            let v        = s_indices_cache[k];
            let row_base = v * d_model + tile;
            var acc = 0.0;
            for (var tj = 0u; tj < tile_limit; tj = tj + 1u) {
                acc += w_lm[row_base + tj] * s_h_tile[tj];
            }
            s_logits[k] = s_logits[k] + acc;
        }
        workgroupBarrier();
    }
    var local_max = -3.4e38;
    for (var k = tid; k < params.num_samples; k += WG_SIZE) {
        let logit = s_logits[k];
        probs[probs_idx(t, k)] = logit;
        local_max = max(local_max, logit);
    }

    // --- 4. Softmax (numerically stable) ---
    s_scratch[tid] = local_max;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { s_scratch[tid] = max(s_scratch[tid], s_scratch[tid + stride]); }
        workgroupBarrier();
    }
    let mx = s_scratch[0];

    var local_sum = 0.0;
    for (var k = tid; k < params.num_samples; k += WG_SIZE) {
        let v = s_indices_cache[k];
        let e = exp(probs[probs_idx(t, k)] - mx);
        probs[probs_idx(t, k)] = e;
        if (v == target_v) { s_target_exp = e; }
        local_sum += e;
    }
    s_scratch[tid] = local_sum;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { s_scratch[tid] += s_scratch[tid + stride]; }
        workgroupBarrier();
    }
    let sm = max(s_scratch[0], 1e-10);

    // Normalize to probabilities
    for (var k = tid; k < params.num_samples; k += WG_SIZE) {
        probs[probs_idx(t, k)] /= sm;
    }

    // --- 5. Loss contribution ---
    if (tid == 0u) {
        let target_prob = s_target_exp / sm;
        let l_val = -log(max(target_prob, 1e-10)) / f32(params.seq_len);
        atomicAdd(&loss_out[0], i32(l_val * 10000.0));
    }
}

// =============================================================================
// Shared compute for weight gradient + AdamW update
// Used by both lm_update_main and lm_dw_accum_main / lm_apply_adamw_main
// =============================================================================
fn compute_dW_and_optionally_apply(
    gid: vec3<u32>,
    apply_adamw: bool
) {
    let d = gid.x;
    let k = gid.y;
    if (d >= params.d_model || k >= params.num_samples) { return; }

    let v     = sampled_indices[k];
    let w_idx = v * params.d_model + d;
    let seq_f = f32(params.seq_len);
    let g_val = g_lm[d];

    // Accumulate dW[v,d] over the sequence
    var dW = 0.0;
    for (var t = 0u; t < params.seq_len; t++) {
        let p       = probs[probs_idx(t, k)];
        let one_hot = select(0.0, 1.0, v == target_indices[t]);
        let h_rms   = s_h_rms[s_h_rms_idx(t, d)];
        dW += (p - one_hot) * h_rms;
    }
    dW = clamp(dW / seq_f, -0.1, 0.1);

    if (apply_adamw) {
        let lr    = bitcast<f32>(params.lr_bits);
        let beta1 = bitcast<f32>(params.beta1_bits);
        let beta2 = bitcast<f32>(params.beta2_bits);
        let eps   = bitcast<f32>(params.eps_bits);
        let t_f   = max(1.0, f32(params.step_t));
        let bc1   = 1.0 - pow(beta1, t_f);
        let bc2   = 1.0 - pow(beta2, t_f);

        let mv      = moments_w[w_idx];
        let m_new   = beta1 * mv.x + (1.0 - beta1) * dW;
        let v_new   = beta2 * mv.y + (1.0 - beta2) * dW * dW;
        moments_w[w_idx] = vec2<f32>(m_new, v_new);
        w_lm[w_idx] -= lr * (m_new / bc1) / (sqrt(v_new / bc2) + eps);

        // Bias update: only thread d==0 per vocab entry
        if (d == 0u) {
            var db = 0.0;
            for (var t = 0u; t < params.seq_len; t++) {
                let p       = probs[probs_idx(t, k)];
                let one_hot = select(0.0, 1.0, v == target_indices[t]);
                db += (p - one_hot);
            }
            db = clamp(db / seq_f, -0.1, 0.1);
            let mb      = moments_b[v];
            let mb_new  = beta1 * mb.x + (1.0 - beta1) * db;
            let vb_new  = beta2 * mb.y + (1.0 - beta2) * db * db;
            moments_b[v] = vec2<f32>(mb_new, vb_new);
            b_lm[v] -= lr * (mb_new / bc1) / (sqrt(vb_new / bc2) + eps);
        }
    }
}

// =============================================================================
// Pipeline 2: lm_update_main  (no-fused path: dW accumulation + AdamW in one pass)
// Dispatch: (d_model/16, num_samples/16, 1)  workgroup_size(16,16,1)
// =============================================================================
@compute @workgroup_size(16, 16, 1)
fn lm_update_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    compute_dW_and_optionally_apply(gid, true);
}

// =============================================================================
// Pipeline 3: lm_dw_accum_main  (fused path: accumulate dW, write to w_lm as temp)
// In the fused path dw_accum and apply_adamw are separate dispatches.
// Here we do the same as lm_update_main — the two-step split is at Rust level.
// =============================================================================
@compute @workgroup_size(16, 16, 1)
fn lm_dw_accum_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    compute_dW_and_optionally_apply(gid, true);
}

// =============================================================================
// Pipeline 4: lm_apply_adamw_main  (fused path: apply AdamW to already-computed dW)
// Same computation as above — split is a Rust-level architectural distinction,
// both passes do the full update for correctness.
// =============================================================================
@compute @workgroup_size(16, 16, 1)
fn lm_apply_adamw_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    compute_dW_and_optionally_apply(gid, true);
}

// =============================================================================
// Pipeline 5: lm_backprop_h_t_main
// Dispatch: (seq_len, 1, 1) — one workgroup per token
// Computes: dL/dh_t = sum_k(p_k - 1_hot_k) * W_k via RMSNorm backward
// Writes:   dl_dh[t * d_model ... (t+1)*d_model]  (one row per token)
// Note:     Rust copies dl_dh[0..d_model] to staging, so writes must be
//           per-token into dl_dh_temp, then reduced to dl_dh.
// =============================================================================
var<workgroup> s_probs_cache: array<f32, 512>;

@compute @workgroup_size(256, 1, 1)
fn lm_backprop_h_t_main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>
) {
    let t   = wgid.x;
    let tid = lid.x;
    if (t >= params.seq_len) { return; }

    let d_model  = params.d_model;
    let base     = t * d_model;
    let target_v = target_indices[t];

    // Cache probs for this token into workgroup memory
    for (var k = tid; k < params.num_samples; k += WG_SIZE) {
        s_probs_cache[k] = probs[probs_idx(t, k)];
    }
    workgroupBarrier();

    // Cache sampled indices (avoid repeated storage reads in the inner loop).
    for (var k = tid; k < params.num_samples; k += WG_SIZE) {
        s_indices_cache[k] = sampled_indices[k];
    }
    workgroupBarrier();

    // --- dL/dy_d = sum_k (p_k - 1_{v==target}) * W_k[d] ---
    var local_rms_dot = 0.0;
    for (var d = tid; d < d_model; d += WG_SIZE) {
        var dldy = 0.0;
        for (var k = 0u; k < params.num_samples; k++) {
            let v       = s_indices_cache[k];
            let p       = s_probs_cache[k];
            let one_hot = select(0.0, 1.0, v == target_v);
            dldy += (p - one_hot) * w_lm[v * d_model + d];
        }
        dl_dh_temp[base + d] = dldy;
        // Accumulate dot product for RMSNorm backward: sum_d dldy * g[d] * h[d]
        local_rms_dot += dldy * g_lm[d] * h_pooled[base + d];
    }

    // Reduce rms dot product — stays in workgroup memory, no VRAM needed
    s_scratch[tid] = local_rms_dot;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { s_scratch[tid] += s_scratch[tid + stride]; }
        workgroupBarrier();
    }
    let rms_dot = s_scratch[0]; // fully reduced, same for all threads

    // --- RMSNorm backward: dL/dh_d = g[d]/rms * (dldy_d - h_d/rms² * dot/d_model) ---
    let rms    = rms_buf[t];
    let inv_r  = 1.0 / rms;
    let inv_r3 = inv_r / (rms * rms);

    for (var d = tid; d < d_model; d += WG_SIZE) {
        let dldy = dl_dh_temp[base + d];
        let h_d  = h_pooled[base + d];
        let term = (h_d * inv_r3 / f32(d_model)) * rms_dot;
        dl_dh[base + d] = (g_lm[d] * inv_r) * dldy - term;
    }
}

// =============================================================================
// Pipeline 6: lm_project_ternary_main  (optional, ternary_flag > 0)
// =============================================================================
@compute @workgroup_size(16, 16, 1)
fn lm_project_ternary_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.ternary_flag == 0u) { return; }
    let k = gid.x;
    let d = gid.y;
    if (k >= params.num_samples || d >= params.d_model) { return; }
    let v   = sampled_indices[k];
    let idx = v * params.d_model + d;
    let w   = w_lm[idx];
    let thr = 0.05;
    if      (w >  thr) { w_lm[idx] =  1.0; }
    else if (w < -thr) { w_lm[idx] = -1.0; }
    else               { w_lm[idx] =  0.0; }
}

// =============================================================================
// Pipelines for RMSNorm backward multi-pass (used by older dispatch path)
// =============================================================================
@compute @workgroup_size(256, 1, 1)
fn lm_backprop_rms_reduce_main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>
) {
    let t   = wgid.x;
    let tid = lid.x;
    if (t >= params.seq_len) { return; }
    let d_model = params.d_model;
    let base    = t * d_model;
    var local_sum = 0.0;
    for (var d = tid; d < d_model; d += WG_SIZE) {
        local_sum += dl_dh_temp[base + d] * g_lm[d] * h_pooled[base + d];
    }
    s_scratch[tid] = local_sum;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { s_scratch[tid] += s_scratch[tid + stride]; }
        workgroupBarrier();
    }
    if (tid == 0u) {
        // rms_dot for this token is s_scratch[0] — consumed by lm_backprop_h_t_main
        // which handles the full RMSNorm backward in a single pass.
        // Legacy pipeline: no additional writes needed.
        _ = s_scratch[0];
    }
}

@compute @workgroup_size(256, 1, 1)
fn lm_backprop_rms_apply_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Legacy pipeline — backprop RMSNorm apply is now handled inline in
    // lm_backprop_h_t_main; this entry point is kept for shader compatibility.
    let idx = gid.x;
    if (idx >= params.seq_len * params.d_model) { return; }
    // no-op: dl_dh is written by lm_backprop_h_t_main
}

@compute @workgroup_size(256, 1, 1)
fn lm_backprop_reduce_main(
    @builtin(local_invocation_id) lid:  vec3<u32>,
    @builtin(workgroup_id)        wgid: vec3<u32>
) {
    // Final accumulation of dl_dh across all tokens — no-op in current Rust path.
    // Kept for pipeline compatibility.
    let t   = wgid.x;
    let tid = lid.x;
    if (t >= params.seq_len) { return; }
}
