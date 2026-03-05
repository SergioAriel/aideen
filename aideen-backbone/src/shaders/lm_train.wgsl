struct TrainParams {
    d_model: u32,
    vocab_size: u32,
    seq_len: u32,
    step_t: u32,
    lr_bits: u32,
    beta1_bits: u32,
    beta2_bits: u32,
    eps_bits: u32,
};

@group(0) @binding(0) var<uniform> params: TrainParams;
@group(0) @binding(1) var<storage, read> h_pooled: array<f32>;          // [seq_len, d_model]
@group(0) @binding(2) var<storage, read_write> w_lm: array<f32>;        // [vocab, d_model]  col-major: w[v + d*vocab]
@group(0) @binding(3) var<storage, read_write> b_lm: array<f32>;        // [vocab]
@group(0) @binding(4) var<storage, read_write> dl_dh: array<f32>;       // [d_model]
@group(0) @binding(5) var<storage, read_write> loss_out: array<atomic<i32>>; // [1] scaled by 1e4; zeroed by CPU before dispatch
@group(0) @binding(6) var<storage, read_write> m_w: array<f32>;         // [vocab, d_model] AdamW 1st moment W
@group(0) @binding(7) var<storage, read_write> v_w: array<f32>;         // [vocab, d_model] AdamW 2nd moment W
@group(0) @binding(8) var<storage, read_write> m_b: array<f32>;         // [vocab]
@group(0) @binding(9) var<storage, read_write> v_b: array<f32>;         // [vocab]
@group(0) @binding(10) var<storage, read> target_indices: array<u32>;   // [seq_len]
@group(0) @binding(11) var<storage, read_write> probs: array<f32>;      // [seq_len, vocab_size]

var<workgroup> s_logits: array<f32, 256>;
var<workgroup> s_scratch: array<f32, 256>;

// Pipeline 1: softmax + loss per token.
// Dispatch: (seq_len, 1, 1) workgroups of size (256, 1, 1).
// loss_out MUST be zero-initialized from the CPU before this dispatch.
@compute @workgroup_size(256, 1, 1)
fn lm_probs_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let t = wgid.x;
    if (t >= params.seq_len) { return; }
    let vid = lid.x;

    // 1. Logit[t, vid] = dot(H[t], W[vid]) + b[vid]
    var logit = b_lm[vid];
    for (var d = 0u; d < params.d_model; d = d + 1u) {
        logit += w_lm[vid + d * params.vocab_size] * h_pooled[t * params.d_model + d];
    }
    s_logits[vid] = logit;
    workgroupBarrier();

    // 2. Softmax (numerically stable)
    s_scratch[vid] = select(-3.4e38, s_logits[vid], vid < params.vocab_size);
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (vid < stride) {
            s_scratch[vid] = max(s_scratch[vid], s_scratch[vid + stride]);
        }
        workgroupBarrier();
    }

    let mx = s_scratch[0];
    s_logits[vid] = exp(s_logits[vid] - mx);
    workgroupBarrier();

    s_scratch[vid] = select(0.0, s_logits[vid], vid < params.vocab_size);
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (vid < stride) {
            s_scratch[vid] += s_scratch[vid + stride];
        }
        workgroupBarrier();
    }
    let sm = s_scratch[0];

    if (vid == 0u) {
        // Cross-entropy contribution for this token (divided by seq_len for average)
        let target_idx = target_indices[t];
        let target_prob = s_logits[target_idx] / max(sm, 1e-10);
        let l_val = -log(max(target_prob, 1e-10)) / f32(params.seq_len);
        atomicAdd(&loss_out[0], i32(l_val * 10000.0));
    }

    // 3. Normalize and store probabilities
    probs[t * params.vocab_size + vid] = s_logits[vid] / max(sm, 1e-10);
}

// Pipeline 2: AdamW weight update + dl_dh gradient backprop.
// Dispatch: (ceil(vocab/16), ceil(d_model/16), 1) workgroups.
@compute @workgroup_size(16, 16, 1)
fn lm_update_main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let v = gid.x;
    let d = gid.y;
    if (v >= params.vocab_size || d >= params.d_model) { return; }

    let lr    = bitcast<f32>(params.lr_bits);
    let beta1 = bitcast<f32>(params.beta1_bits);
    let beta2 = bitcast<f32>(params.beta2_bits);
    let eps   = bitcast<f32>(params.eps_bits);
    let t_f   = max(1.0, f32(params.step_t));
    let bc1   = 1.0 - pow(beta1, t_f);
    let bc2   = 1.0 - pow(beta2, t_f);
    let seq_f = f32(params.seq_len);

    // 1. Weight gradient: dW[v,d] = (1/T) * sum_t (p[t,v] - y[t,v]) * h[t,d]
    var dW = 0.0;
    for (var t = 0u; t < params.seq_len; t = t + 1u) {
        let p       = probs[t * params.vocab_size + v];
        let one_hot = select(0.0, 1.0, v == target_indices[t]);
        dW += (p - one_hot) * h_pooled[t * params.d_model + d];
    }
    dW = dW / seq_f;

    // 2. AdamW update for W
    let w_idx = v + d * params.vocab_size;
    let m_new = beta1 * m_w[w_idx] + (1.0 - beta1) * dW;
    let v_new = beta2 * v_w[w_idx] + (1.0 - beta2) * dW * dW;
    m_w[w_idx] = m_new;
    v_w[w_idx] = v_new;
    w_lm[w_idx] -= lr * (m_new / bc1) / (sqrt(v_new / bc2) + eps);

    // 3. Bias gradient and AdamW update (one thread per vocab dim: d == 0)
    if (d == 0u) {
        var db = 0.0;
        for (var t = 0u; t < params.seq_len; t = t + 1u) {
            let p       = probs[t * params.vocab_size + v];
            let one_hot = select(0.0, 1.0, v == target_indices[t]);
            db += (p - one_hot);
        }
        db = db / seq_f;
        let mb_new = beta1 * m_b[v] + (1.0 - beta1) * db;
        let vb_new = beta2 * v_b[v] + (1.0 - beta2) * db * db;
        m_b[v] = mb_new;
        v_b[v] = vb_new;
        b_lm[v] -= lr * (mb_new / bc1) / (sqrt(vb_new / bc2) + eps);
    }

    // 4. dl_dh: average gradient over ALL tokens (not just last).
    //    dl_dh[d] = (1/T) * sum_t sum_vv [ (p[t,vv] - y[t,vv]) * W[vv, d] ]
    //    Thread guard: v == 0 so each dimension d is written by exactly one thread.
    if (v == 0u) {
        var acc = 0.0;
        for (var t = 0u; t < params.seq_len; t = t + 1u) {
            for (var vv = 0u; vv < params.vocab_size; vv = vv + 1u) {
                let p       = probs[t * params.vocab_size + vv];
                let one_hot = select(0.0, 1.0, vv == target_indices[t]);
                acc += (p - one_hot) * w_lm[vv + d * params.vocab_size];
            }
        }
        dl_dh[d] = acc / seq_f;
    }
}
