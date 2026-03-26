struct TrainParams {
    d_model: u32,
    vocab_size: u32,
    seq_len: u32,
    step_t: u32,
    lr_bits: u32,
    beta1_bits: u32,
    beta2_bits: u32,
    eps_bits: u32,
    ternary_flag: u32, // 0 = off, 1 = on
    num_samples: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: TrainParams;
@group(0) @binding(1) var<storage, read> h_pooled: array<f32>;          // [seq_len, d_model]
@group(0) @binding(2) var<storage, read_write> w_lm: array<f32>;        // [vocab, d_model]
@group(0) @binding(3) var<storage, read_write> b_lm: array<f32>;        // [vocab]
@group(0) @binding(4) var<storage, read_write> dl_dh: array<f32>;       // [d_model]
@group(0) @binding(5) var<storage, read_write> loss_out: array<atomic<i32>>; // [1]
@group(0) @binding(6) var<storage, read_write> moments_w: array<vec2<f32>>; // AdamW [m, v]
@group(0) @binding(7) var<storage, read_write> moments_b: array<vec2<f32>>; // AdamW [m, v]
@group(0) @binding(8) var<storage, read> target_indices: array<u32>;   // [seq_len]
@group(0) @binding(9) var<storage, read_write> probs: array<f32>;      // [seq_len, num_samples]
@group(0) @binding(10) var<storage, read_write> g_lm: array<f32>;        // [d_model] RMSNorm scale
@group(0) @binding(11) var<storage, read_write> moments_g: array<vec2<f32>>; // AdamW [m, v]
@group(0) @binding(12) var<storage, read_write> rms_buf: array<f32>;     // [seq_len]
@group(0) @binding(13) var<storage, read_write> dl_dh_temp: array<f32>;  // [seq_len, d_model]
@group(0) @binding(14) var<storage, read> sampled_indices: array<u32>;  // [num_samples]
@group(0) @binding(15) var<storage, read_write> s_h_rms: array<f32>;    // Dynamic intermediate h_rms
@group(0) @binding(16) var<storage, read_write> dl_dh_rms_red: array<f32>; // [seq_len]

const WG_SIZE: u32 = 256u;
var<workgroup> s_rms: f32;
var<workgroup> s_scratch: array<f32, WG_SIZE>;
var<workgroup> s_target_e: f32;
var<workgroup> s_h_rms_local: array<f32, 1024>;  // LIMIT: d_model <= 1024
var<workgroup> s_indices_cache: array<u32, 512>; // LIMIT: num_samples <= 512
var<workgroup> s_h_tile: array<f32, WG_SIZE>;
var<workgroup> s_logits: array<f32, 512>;        // LIMIT: num_samples <= 512

// Pipeline 1: softmax + loss per token.
// Dispatch: (seq_len, 1, 1) workgroups of size (256, 1, 1).
@compute @workgroup_size(256, 1, 1)
fn lm_probs_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let t = wgid.x;
    if (t >= params.seq_len) { return; }
    let tid = lid.x;
    let d_model = params.d_model;

    let base_h = t * d_model;

    // 0a. Init shared accumulators.
    if (tid == 0u) { s_target_e = 0.0; }

    // 0b. Parallel RMS
    var local_sq = 0.0;
    for (var i = tid; i < d_model; i += WG_SIZE) {
        let val = h_pooled[base_h + i];
        local_sq += val * val;
    }
    s_scratch[tid] = local_sq;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { s_scratch[tid] += s_scratch[tid + stride]; }
        workgroupBarrier();
    }
    if (tid == 0u) {
        s_rms = sqrt(s_scratch[0] / f32(d_model) + 1e-5);
        rms_buf[t] = s_rms;
    }
    workgroupBarrier();

    let rms = s_rms;
    let base_sh = t * d_model;
    let items_per_thread = (d_model + WG_SIZE - 1u) / WG_SIZE;
    for (var i = 0u; i < items_per_thread; i = i + 1u) {
        let d = tid + i * WG_SIZE;
        if (d < d_model) {
            let val = (h_pooled[base_h + d] / rms) * g_lm[d];
            s_h_rms_local[d] = val; // Cache locally
            s_h_rms[base_sh + d] = val; // Also save for backprop
        }
    }
    workgroupBarrier();

    // 1. Cache sampled indices to shared (avoid repeated storage reads)
    for (var k = tid; k < params.num_samples; k += WG_SIZE) {
        s_indices_cache[k] = sampled_indices[k];
    }
    workgroupBarrier();

    // 2. Calculate Logits (GEMM-style for non-ternary; fallback for ternary)
    var local_max = -3.4e38;
    if (params.ternary_flag == 1u) {
        for (var k = tid; k < params.num_samples; k += WG_SIZE) {
            let v = s_indices_cache[k];
            let row_base = v * params.d_model;
            var logit = b_lm[v];
            var abs_sum = 0.0;
            for (var d = 0u; d < params.d_model; d = d + 1u) {
                abs_sum += abs(w_lm[row_base + d]);
            }
            let gamma = max(abs_sum / f32(params.d_model), 1e-8);
            for (var d = 0u; d < params.d_model; d = d + 1u) {
                let q_w = clamp(round(w_lm[row_base + d] / gamma), -1.0, 1.0);
                logit += q_w * s_h_rms_local[d];
            }
            probs[t * params.num_samples + k] = logit;
            local_max = max(local_max, logit);
        }
    } else {
        // Initialize logits with bias.
        for (var k = tid; k < params.num_samples; k += WG_SIZE) {
            let v = s_indices_cache[k];
            s_logits[k] = b_lm[v];
        }
        workgroupBarrier();
        // GEMM-style blocking: load H once per tile, reuse for all sampled rows.
        for (var tile = 0u; tile < params.d_model; tile = tile + WG_SIZE) {
            let d_load = tile + tid;
            if (d_load < params.d_model) {
                s_h_tile[tid] = s_h_rms_local[d_load];
            } else {
                s_h_tile[tid] = 0.0;
            }
            workgroupBarrier();
            let tile_limit = min(WG_SIZE, params.d_model - tile);
            for (var k = tid; k < params.num_samples; k += WG_SIZE) {
                let v = s_indices_cache[k];
                let row_base = v * params.d_model + tile;
                var acc = 0.0;
                for (var tj = 0u; tj < tile_limit; tj = tj + 1u) {
                    acc += w_lm[row_base + tj] * s_h_tile[tj];
                }
                s_logits[k] = s_logits[k] + acc;
            }
            workgroupBarrier();
        }
        for (var k = tid; k < params.num_samples; k += WG_SIZE) {
            let logit = s_logits[k];
            probs[t * params.num_samples + k] = logit;
            local_max = max(local_max, logit);
        }
    }

    // 2. Workgroup reduction for Global Max
    s_scratch[tid] = local_max;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            s_scratch[tid] = max(s_scratch[tid], s_scratch[tid + stride]);
        }
        workgroupBarrier();
    }
    let mx = s_scratch[0];

    // 3. Compute exp(logit - mx) and sum
    var local_sum = 0.0;
    for (var k = tid; k < params.num_samples; k += WG_SIZE) {
        let v = s_indices_cache[k];
        let e = exp(probs[t * params.num_samples + k] - mx);
        probs[t * params.num_samples + k] = e;
        if (v == target_indices[t]) { s_target_e = e; }
        local_sum += e;
    }

    // 4. Workgroup reduction for Global Sum
    s_scratch[tid] = local_sum;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            s_scratch[tid] += s_scratch[tid + stride];
        }
        workgroupBarrier();
    }
    let sm = max(s_scratch[0], 1e-10);

    // 5. Final normalization and loss
    if (tid == 0u) {
        let target_prob = s_target_e / sm;
        let l_val = -log(max(target_prob, 1e-10)) / f32(params.seq_len);
        atomicAdd(&loss_out[0], i32(l_val * 10000.0));
    }

    for (var k = tid; k < params.num_samples; k += WG_SIZE) {
        probs[t * params.num_samples + k] /= sm;
    }
}

// Pipeline 2: AdamW weight update (W and b).
@compute @workgroup_size(16, 16, 1)
fn lm_update_main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let d = gid.x;
    let k = gid.y;
    if (d >= params.d_model || k >= params.num_samples) { return; }

    let v = sampled_indices[k];
    let w_idx = v * params.d_model + d;

    let lr    = bitcast<f32>(params.lr_bits);
    let beta1 = bitcast<f32>(params.beta1_bits);
    let beta2 = bitcast<f32>(params.beta2_bits);
    let eps   = bitcast<f32>(params.eps_bits);
    let t_f   = max(1.0, f32(params.step_t));
    let bc1   = 1.0 - pow(beta1, t_f);
    let bc2   = 1.0 - pow(beta2, t_f);
    let seq_f = f32(params.seq_len);
    let g_val = g_lm[d];

    // 1. Accumulate dW[v,d]
    var dW = 0.0;
    for (var t = 0u; t < params.seq_len; t = t + 1u) {
        let p_val   = probs[t * params.num_samples + k];
        let one_hot = select(0.0, 1.0, v == target_indices[t]);
        let h_rms   = (h_pooled[t * params.d_model + d] / rms_buf[t]) * g_val;
        dW += (p_val - one_hot) * h_rms;
    }
    dW = dW / seq_f;
    dW = clamp(dW, -0.1, 0.1); 

    // 2. AdamW update
    let moments_val = moments_w[w_idx];
    let m_new = beta1 * moments_val.x + (1.0 - beta1) * dW;
    let v_new = beta2 * moments_val.y + (1.0 - beta2) * dW * dW;
    moments_w[w_idx] = vec2<f32>(m_new, v_new);

    w_lm[w_idx] = w_lm[w_idx] - lr * (m_new / bc1) / (sqrt(v_new / bc2) + eps);

    if (d == 0u) {
        var db = 0.0;
        for (var t = 0u; t < params.seq_len; t = t + 1u) {
            let p       = probs[t * params.num_samples + k];
            let one_hot = select(0.0, 1.0, v == target_indices[t]);
            db += (p - one_hot);
        }
        db = db / seq_f;
        db = clamp(db, -0.1, 0.1);

        let bias_moments = moments_b[v];
        let mb_new = beta1 * bias_moments.x + (1.0 - beta1) * db;
        let vb_new = beta2 * bias_moments.y + (1.0 - beta2) * db * db;
        moments_b[v] = vec2<f32>(mb_new, vb_new);
        b_lm[v] -= lr * (mb_new / bc1) / (sqrt(vb_new / bc2) + eps);
    }
}

// Pipeline 5: Ternary Weight Projection
@compute @workgroup_size(16, 16, 1)
fn lm_project_ternary_main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (params.ternary_flag == 0u) { return; }
    let k = gid.x;
    let d = gid.y;
    if (k >= params.num_samples || d >= params.d_model) { return; }
    let v = sampled_indices[k];
    let idx = v * params.d_model + d;

    let w = w_lm[idx];
    let threshold = 0.05;
    if (w > threshold) {
        w_lm[idx] = 1.0;
    } else if (w < -threshold) {
        w_lm[idx] = -1.0;
    } else {
        w_lm[idx] = 0.0;
    }
}

// Pipeline 3: Gradient backprop to h (dl_dh)
@compute @workgroup_size(256, 1, 1)
fn lm_backprop_h_t_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let t = wgid.x;
    let d = wgid.y;
    if (t >= params.seq_len || d >= params.d_model) { return; }
    let tid = lid.x;

    let target_idx = target_indices[t];
    var local_sum = 0.0;
    let t_offset_compact = t * params.num_samples;

    for (var k_idx = tid; k_idx < params.num_samples; k_idx += WG_SIZE) {
        let v = sampled_indices[k_idx];
        let p = probs[t_offset_compact + k_idx];
        let y = select(0.0, 1.0, v == target_idx);
        local_sum += (p - y) * w_lm[v * params.d_model + d];
    }
    
    s_scratch[tid] = local_sum;
    workgroupBarrier();
    
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { s_scratch[tid] += s_scratch[tid + stride]; }
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        let val = s_scratch[0];
        dl_dh_temp[t * params.d_model + d] = val; // dL/dy (pre-RMSNorm backprop)
    }
}

// Pipeline 3.5: Reduce sum_j (g_j * x_j * dL/dy_j) per token.
@compute @workgroup_size(256, 1, 1)
fn lm_backprop_rms_reduce_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let t = wgid.x;
    if (t >= params.seq_len) { return; }
    let tid = lid.x;
    let d_model = params.d_model;
    let base = t * d_model;

    var local_sum = 0.0;
    for (var d = tid; d < d_model; d += WG_SIZE) {
        let dldy = dl_dh_temp[base + d];
        let x = h_pooled[base + d];
        let g = g_lm[d];
        local_sum += dldy * g * x;
    }
    s_scratch[tid] = local_sum;
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) { s_scratch[tid] += s_scratch[tid + stride]; }
        workgroupBarrier();
    }
    if (tid == 0u) {
        dl_dh_rms_red[t] = s_scratch[0];
    }
}

// Pipeline 3.6: Apply RMSNorm backward to get dL/dh.
@compute @workgroup_size(256, 1, 1)
fn lm_backprop_rms_apply_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let t = wgid.x;
    let d = wgid.y;
    if (t >= params.seq_len || d >= params.d_model) { return; }
    let base = t * params.d_model;
    let dldy = dl_dh_temp[base + d];
    let x = h_pooled[base + d];
    let g = g_lm[d];
    let rms = rms_buf[t];
    let inv_r = 1.0 / rms;
    let inv_r3 = inv_r / (rms * rms);
    let sum = dl_dh_rms_red[t];
    let term = (x * inv_r3 / f32(params.d_model)) * sum;
    dl_dh[base + d] = (g * inv_r) * dldy - term;
}

// Pipeline 4: Final reduction and AdamW for g.
@compute @workgroup_size(256, 1, 1)
fn lm_backprop_reduce_main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let d = gid.x;
    if (d >= params.d_model) { return; }

    let lr    = bitcast<f32>(params.lr_bits);
    let beta1 = bitcast<f32>(params.beta1_bits);
    let beta2 = bitcast<f32>(params.beta2_bits);
    let eps   = bitcast<f32>(params.eps_bits);
    let t_f   = max(1.0, f32(params.step_t));
    let bc1   = 1.0 - pow(beta1, t_f);
    let bc2   = 1.0 - pow(beta2, t_f);
    let seq_f = f32(params.seq_len);

    var dg = 0.0;
    
    for (var t = 0u; t < params.seq_len; t = t + 1u) {
        let rms = rms_buf[t];
        let dl_dh_rms_t_d = dl_dh_temp[t * params.d_model + d];
        let h_val = h_pooled[t * params.d_model + d];
        
        dg += dl_dh_rms_t_d * (h_val / rms);
        // NO REDUCIR dl_dh central aquí. El CG solver usará dl_dh_temp directo.
    }
    
    // AdamW update for g remains reduced (global weights)
    let dg_avg = dg / seq_f;
    let moments = moments_g[d];
    let mg_new = beta1 * moments.x + (1.0 - beta1) * dg_avg;
    let vg_new = beta2 * moments.y + (1.0 - beta2) * dg_avg * dg_avg;
    moments_g[d] = vec2<f32>(mg_new, vg_new);
    g_lm[d] -= lr * (mg_new / bc1) / (sqrt(vg_new / bc2) + eps);
}
