// Retain-gate backward pass for FPM plastic write.
//
// Forward (per token t, slot s):
//   c       = 0.5 * (h_star[t,s] + signal[t,s])
//   up[r]   = Σ_d  W_up[s, d, r] * c[d]                 (d → RETAIN_RANK)
//   pre[d]  = Σ_r  W_down[s, r, d] * up[r] + b[s, d]    (RETAIN_RANK → d)
//   retain  = sigmoid(pre)
//   m_new   = retain * m_prev + z * residual_scale * tanh(W_delta * c)
//
// Backward per token t (iterate t = T-1..0, carry dm_new):
//   dm_prev       = retain * dm_new                         (propagate to prev token)
//   dp[d]         = dm_new[d] * m_prev[d] * retain[d] * (1 - retain[d])
//   db[s,d]      += dp[d]
//   dW_down[s,r,d]+= dp[d] * up[r]
//   dup[r]        = Σ_d W_down[s,r,d] * dp[d]
//   dW_up[s,d,r] += dup[r] * c[d]
//   (dm_new ← dm_prev for next iteration)
//
// dm_new seeding: currently zero (truncated BPTT within chunk).
// TODO: add binding(5) for per-token ∂L/∂H_hist[t] and accumulate into dm_new.
//
// Layout:
//   @group(0): params, h_star, scratch(signal), fpm_m_buf, AllGradients
//   @group(1): AllWeights (same as fused_deq_update)
//
// Dispatch: (h_slots, 1, 1) — one workgroup per slot, WG_SIZE=64.

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
};

@group(0) @binding(0) var<uniform>            params:     FpmBwdUniforms;
@group(0) @binding(1) var<storage, read>      h_star:     array<f32>;  // [batch*seq*h*d]
@group(0) @binding(2) var<storage, read>      Scratch:    array<f32>;  // clean layout: [batch*seq*2*h*d]
@group(0) @binding(3) var<storage, read>      fpm_m_buf:  array<f32>;  // [batch*seq*h*d] per-token m_new
@group(0) @binding(4) var<storage, read_write> AllGradients: array<f32>;

@group(1) @binding(0) var<storage, read_write> AllWeights: array<f32>;

// --- Layout helpers (must match deq_slot_attn_unified_clean.wgsl) ---

fn aw_wq_base(d: u32, h: u32) -> u32 { return 0u; }
fn aw_wk_base(d: u32, h: u32) -> u32 { return h*d*d + h*d; }
fn aw_wv_base(d: u32, h: u32) -> u32 { return aw_wk_base(d,h) + h*d*d + h*d; }
fn aw_wo_base(d: u32, h: u32) -> u32 { return aw_wv_base(d,h) + h*d*d; }
fn aw_win_base(d: u32, h: u32) -> u32 { return aw_wo_base(d,h) + h*d*d; }
fn aw_nscale_base(d: u32, h: u32) -> u32 {
    let aw_wx = aw_win_base(d, h) + h * d * d;
    let aw_wout = aw_wx + d * d;
    let aw_alog = aw_wout + d * d;
    return aw_alog + h * d;
}
fn aw_hist_base(d: u32, h: u32) -> u32 { return aw_nscale_base(d, h) + d; }

fn hist_gate_base(d: u32, h: u32) -> u32 { return d*d + 2u*h*d; }
fn slot_anchor_base(d: u32, h: u32) -> u32 { return hist_gate_base(d, h) + h; }
fn hist_delta_bias_base(d: u32, h: u32) -> u32 {
    return slot_anchor_base(d, h) + h*d + h*d*d;
}
fn hist_gate_query_base(d: u32, h: u32) -> u32 { return hist_delta_bias_base(d, h) + d + 21u; }
fn w_forget_base(d: u32, h: u32) -> u32 { return hist_gate_query_base(d, h) + h*d; }
fn b_forget_base(d: u32, h: u32) -> u32 { return w_forget_base(d, h) + h*d; }
fn hhist_gamma_base(d: u32, h: u32) -> u32 { return b_forget_base(d, h) + h; }

const RETAIN_RANK: u32 = 32u;
fn w_retain_up_base(d: u32, h: u32) -> u32   { return hhist_gamma_base(d, h) + h; }
fn w_retain_down_base(d: u32, h: u32) -> u32 { return w_retain_up_base(d, h) + h * d * RETAIN_RANK; }
fn b_retain_base(d: u32, h: u32) -> u32      { return w_retain_down_base(d, h) + h * RETAIN_RANK * d; }

// Clean-scratch signal layout: stride = 2 * h * d; signal at slot*d within stride.
fn clean_signal_off(t_abs: u32, slot: u32, d: u32, h: u32) -> u32 {
    return t_abs * (2u * h * d) + slot * d;
}

// h_star layout: [batch * seq * h * d], slot at slot*d within each (t_abs * h * d).
fn hstar_off(t_abs: u32, slot: u32, d: u32, h: u32) -> u32 {
    return t_abs * h * d + slot * d;
}

// fpm_m_buf layout: [batch * seq * h * d], same as h_star.
fn fpm_m_off(t_abs: u32, slot: u32, d: u32, h: u32) -> u32 {
    return t_abs * h * d + slot * d;
}

// --- Workgroup shared memory ---
var<workgroup> shared_up:  array<f32, 32>;   // RETAIN_RANK values after fwd up-projection
var<workgroup> shared_dup: array<f32, 32>;   // gradient w.r.t. up (one per r, lanes 0..31 own these)
var<workgroup> shared_dp:  array<f32, 512>;  // dp[0..d-1] broadcast from all threads (d ≤ 512)

const WG_SIZE: u32 = 64u;

@compute
@workgroup_size(64, 1, 1)
fn fused_fpm_retain_bwd_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let lane  = lid.x;
    let slot  = wid.x;
    let d     = params.d_model;
    let h     = params.h_slots;
    if (slot >= h) { return; }

    let hist_base  = aw_hist_base(d, h);
    let wup_base   = hist_base + w_retain_up_base(d, h) + slot * d * RETAIN_RANK;
    let wdown_base = hist_base + w_retain_down_base(d, h) + slot * RETAIN_RANK * d;
    let bret_base  = hist_base + b_retain_base(d, h) + slot * d;

    // Number of d-dimensions this thread is responsible for.
    // d must be divisible by WG_SIZE (64). For d=512: 8 dims per thread.
    let dims_per_lane = d / WG_SIZE;  // = 8 for d=512, 4 for d=256

    let n_tokens = params.batch_size * params.seq_len;
    let n_norm   = max(1.0, f32(n_tokens));
    let clip     = 0.5;
    let lr_scaled = params.lr * params.grad_scale;

    // Loop over batch elements independently (truncated BPTT — no cross-batch gradient).
    for (var b = 0u; b < params.batch_size; b = b + 1u) {

        // dm_new[k] holds ∂L/∂m_new for k-th dimension of this thread's assigned dims.
        // Seeded at 0 — truncated BPTT, no cross-chunk gradient.
        // TODO: seed from ∂L/∂H_hist[T-1] (requires an additional binding).
        var dm_new0 = 0.0; var dm_new1 = 0.0;
        var dm_new2 = 0.0; var dm_new3 = 0.0;
        var dm_new4 = 0.0; var dm_new5 = 0.0;
        var dm_new6 = 0.0; var dm_new7 = 0.0;

        // Iterate tokens backward.
        for (var t_rev = 0u; t_rev < params.seq_len; t_rev = t_rev + 1u) {
            let t     = params.seq_len - 1u - t_rev;
            let t_abs = b * params.seq_len + t;

            // --- 1. Load c (context vector) for this thread's dims ---
            let sig_base   = clean_signal_off(t_abs, slot, d, h);
            let hstar_base = hstar_off(t_abs, slot, d, h);

            let c0 = 0.5 * (h_star[hstar_base + lane * dims_per_lane + 0u] + Scratch[sig_base + lane * dims_per_lane + 0u]);
            let c1 = 0.5 * (h_star[hstar_base + lane * dims_per_lane + 1u] + Scratch[sig_base + lane * dims_per_lane + 1u]);
            let c2 = 0.5 * (h_star[hstar_base + lane * dims_per_lane + 2u] + Scratch[sig_base + lane * dims_per_lane + 2u]);
            let c3 = 0.5 * (h_star[hstar_base + lane * dims_per_lane + 3u] + Scratch[sig_base + lane * dims_per_lane + 3u]);
            // Remaining c values; only used if dims_per_lane > 4.
            var c4 = 0.0; var c5 = 0.0; var c6 = 0.0; var c7 = 0.0;
            if (dims_per_lane > 4u) {
                c4 = 0.5 * (h_star[hstar_base + lane * dims_per_lane + 4u] + Scratch[sig_base + lane * dims_per_lane + 4u]);
                c5 = 0.5 * (h_star[hstar_base + lane * dims_per_lane + 5u] + Scratch[sig_base + lane * dims_per_lane + 5u]);
                c6 = 0.5 * (h_star[hstar_base + lane * dims_per_lane + 6u] + Scratch[sig_base + lane * dims_per_lane + 6u]);
                c7 = 0.5 * (h_star[hstar_base + lane * dims_per_lane + 7u] + Scratch[sig_base + lane * dims_per_lane + 7u]);
            }

            // --- 2. Compute up[r] = Σ_d W_up[d, r] * c[d] ---
            // Threads 0..RETAIN_RANK-1 each compute one up[r].
            if (lane < RETAIN_RANK) {
                var acc = 0.0;
                for (var j = 0u; j < d; j = j + 1u) {
                    let c_j = 0.5 * (h_star[hstar_base + j] + Scratch[sig_base + j]);
                    acc = acc + AllWeights[wup_base + j * RETAIN_RANK + lane] * c_j;
                }
                shared_up[lane] = acc;
            }
            workgroupBarrier();

            // --- 3. Recompute retain[d] and load m_prev[d] for this thread's dims ---
            var retain0 = 0.0; var retain1 = 0.0;
            var retain2 = 0.0; var retain3 = 0.0;
            var retain4 = 0.0; var retain5 = 0.0;
            var retain6 = 0.0; var retain7 = 0.0;

            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let d_out = lane * dims_per_lane + k;
                var pre = AllWeights[bret_base + d_out];
                for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                    pre = pre + AllWeights[wdown_base + r * d + d_out] * shared_up[r];
                }
                let sig_v = 1.0 / (1.0 + exp(-pre));
                switch (k) {
                    case 0u: { retain0 = sig_v; }
                    case 1u: { retain1 = sig_v; }
                    case 2u: { retain2 = sig_v; }
                    case 3u: { retain3 = sig_v; }
                    case 4u: { retain4 = sig_v; }
                    case 5u: { retain5 = sig_v; }
                    case 6u: { retain6 = sig_v; }
                    default: { retain7 = sig_v; }
                }
            }

            // m_prev[d] = fpm_m_buf at token (t-1), or 0 at t=0.
            var mp0 = 0.0; var mp1 = 0.0;
            var mp2 = 0.0; var mp3 = 0.0;
            var mp4 = 0.0; var mp5 = 0.0;
            var mp6 = 0.0; var mp7 = 0.0;
            if (t > 0u) {
                let m_prev_base = fpm_m_off(t_abs - 1u, slot, d, h);
                mp0 = fpm_m_buf[m_prev_base + lane * dims_per_lane + 0u];
                mp1 = fpm_m_buf[m_prev_base + lane * dims_per_lane + 1u];
                mp2 = fpm_m_buf[m_prev_base + lane * dims_per_lane + 2u];
                mp3 = fpm_m_buf[m_prev_base + lane * dims_per_lane + 3u];
                if (dims_per_lane > 4u) {
                    mp4 = fpm_m_buf[m_prev_base + lane * dims_per_lane + 4u];
                    mp5 = fpm_m_buf[m_prev_base + lane * dims_per_lane + 5u];
                    mp6 = fpm_m_buf[m_prev_base + lane * dims_per_lane + 6u];
                    mp7 = fpm_m_buf[m_prev_base + lane * dims_per_lane + 7u];
                }
            }

            // dp[d] = dm_new[d] * m_prev[d] * retain[d] * (1 - retain[d])
            let dp0 = dm_new0 * mp0 * retain0 * (1.0 - retain0);
            let dp1 = dm_new1 * mp1 * retain1 * (1.0 - retain1);
            let dp2 = dm_new2 * mp2 * retain2 * (1.0 - retain2);
            let dp3 = dm_new3 * mp3 * retain3 * (1.0 - retain3);
            let dp4 = dm_new4 * mp4 * retain4 * (1.0 - retain4);
            let dp5 = dm_new5 * mp5 * retain5 * (1.0 - retain5);
            let dp6 = dm_new6 * mp6 * retain6 * (1.0 - retain6);
            let dp7 = dm_new7 * mp7 * retain7 * (1.0 - retain7);

            // --- 4. Broadcast dp into shared memory for cross-thread dup reduction ---
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                var dp_k = 0.0;
                switch (k) {
                    case 0u: { dp_k = dp0; } case 1u: { dp_k = dp1; }
                    case 2u: { dp_k = dp2; } case 3u: { dp_k = dp3; }
                    case 4u: { dp_k = dp4; } case 5u: { dp_k = dp5; }
                    case 6u: { dp_k = dp6; } default: { dp_k = dp7; }
                }
                shared_dp[lane * dims_per_lane + k] = dp_k;
            }
            workgroupBarrier();

            // --- 5. Accumulate ∂b_retain[d] += dp[d] ---
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let d_out = lane * dims_per_lane + k;
                var dp_k = 0.0;
                switch (k) {
                    case 0u: { dp_k = dp0; } case 1u: { dp_k = dp1; }
                    case 2u: { dp_k = dp2; } case 3u: { dp_k = dp3; }
                    case 4u: { dp_k = dp4; } case 5u: { dp_k = dp5; }
                    case 6u: { dp_k = dp6; } default: { dp_k = dp7; }
                }
                let step = lr_scaled * dp_k / n_norm;
                if (params.grad_accum_mode == 1u) {
                    AllGradients[bret_base + d_out] += step;
                } else {
                    AllWeights[bret_base + d_out] -= clamp(step, -clip, clip);
                }
            }

            // --- 6. Accumulate ∂W_down[r, d_out] += dp[d_out] * up[r] ---
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let d_out = lane * dims_per_lane + k;
                var dp_k = 0.0;
                switch (k) {
                    case 0u: { dp_k = dp0; } case 1u: { dp_k = dp1; }
                    case 2u: { dp_k = dp2; } case 3u: { dp_k = dp3; }
                    case 4u: { dp_k = dp4; } case 5u: { dp_k = dp5; }
                    case 6u: { dp_k = dp6; } default: { dp_k = dp7; }
                }
                for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                    let step = lr_scaled * dp_k * shared_up[r] / n_norm;
                    if (params.grad_accum_mode == 1u) {
                        AllGradients[wdown_base + r * d + d_out] += step;
                    } else {
                        AllWeights[wdown_base + r * d + d_out] -= clamp(step, -clip, clip);
                    }
                }
            }

            // --- 7. Compute dup[r] = Σ_d W_down[r,d] * dp[d] ---
            // Threads 0..RETAIN_RANK-1 each own one r; loop over all d using shared_dp.
            // Threads RETAIN_RANK..WG_SIZE-1 are idle.
            if (lane < RETAIN_RANK) {
                var dup_r = 0.0;
                for (var j = 0u; j < d; j = j + 1u) {
                    dup_r = dup_r + AllWeights[wdown_base + lane * d + j] * shared_dp[j];
                }
                shared_dup[lane] = dup_r;
            }
            workgroupBarrier();

            // --- 8. Accumulate ∂W_up[d_in, r] += dup[r] * c[d_in] ---
            for (var k = 0u; k < dims_per_lane; k = k + 1u) {
                let d_in = lane * dims_per_lane + k;
                var c_k = 0.0;
                switch (k) {
                    case 0u: { c_k = c0; } case 1u: { c_k = c1; }
                    case 2u: { c_k = c2; } case 3u: { c_k = c3; }
                    case 4u: { c_k = c4; } case 5u: { c_k = c5; }
                    case 6u: { c_k = c6; } default: { c_k = c7; }
                }
                for (var r = 0u; r < RETAIN_RANK; r = r + 1u) {
                    let step = lr_scaled * shared_dup[r] * c_k / n_norm;
                    if (params.grad_accum_mode == 1u) {
                        AllGradients[wup_base + d_in * RETAIN_RANK + r] += step;
                    } else {
                        AllWeights[wup_base + d_in * RETAIN_RANK + r] -= clamp(step, -clip, clip);
                    }
                }
            }

            // --- 9. Propagate gradient backward: dm_prev = retain * dm_new ---
            dm_new0 = retain0 * dm_new0;
            dm_new1 = retain1 * dm_new1;
            dm_new2 = retain2 * dm_new2;
            dm_new3 = retain3 * dm_new3;
            dm_new4 = retain4 * dm_new4;
            dm_new5 = retain5 * dm_new5;
            dm_new6 = retain6 * dm_new6;
            dm_new7 = retain7 * dm_new7;

            workgroupBarrier();
        } // end token loop
    } // end batch loop
}
