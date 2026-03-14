struct RunUniforms {
    batch_size: u32,
    d_model: u32,
    h_slots: u32,
    max_iters: u32,
    epsilon: f32,
    damping: f32,
    seq_len: u32,
    residual_alpha: f32,
}

@group(0) @binding(0) var<uniform> shape: RunUniforms;
@group(0) @binding(1) var<storage, read> S_in: array<f32>;
@group(0) @binding(2) var<storage, read> W_q: array<f32>;
@group(0) @binding(3) var<storage, read> W_k: array<f32>;
@group(0) @binding(4) var<storage, read> W_v: array<f32>;
@group(0) @binding(5) var<storage, read> W_o: array<f32>;
@group(0) @binding(6) var<storage, read> W_in: array<f32>;
@group(0) @binding(7) var<storage, read> W_x: array<f32>;
@group(0) @binding(8) var<storage, read> W_out: array<f32>;
@group(0) @binding(9) var<storage, read> A_log: array<f32>;
@group(0) @binding(10) var<storage, read> NormScale: array<f32>;
@group(0) @binding(11) var<storage, read_write> H_curr: array<f32>;
@group(0) @binding(12) var<storage, read_write> H_next: array<f32>;
@group(0) @binding(13) var<storage, read_write> Scratch: array<f32>;
@group(0) @binding(14) var<storage, read_write> H_pooled: array<f32>;
@group(0) @binding(15) var<storage, read_write> DebugLog: array<f32>;

const WG_SIZE: u32 = 256u;
const MAX_SLOTS: u32 = 8u;

var<workgroup> shared_vals: array<f32, WG_SIZE>;
var<workgroup> hit_count: atomic<u32>;
var<workgroup> max_delta_seen: f32;
var<workgroup> last_delta: f32;
var<workgroup> s_delta: array<f32, WG_SIZE>;
var<workgroup> curr_contractivity: f32;

@compute @workgroup_size(256, 1, 1)
fn deq_forward_main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let batch_idx = wid.x;
    if (batch_idx >= shape.batch_size) { return; }

    let d_model = shape.d_model;
    let h_slots = shape.h_slots;
    if (h_slots == 0u || h_slots > MAX_SLOTS) { return; }

    let total_elements = h_slots * d_model;
    let h_base = batch_idx * total_elements;

    // Scratch layout per batch:
    // q [h*d], k [h*d], v [h*d], attn [h*d], mamba [h*d], signal [d]
    let scratch_stride = d_model * (h_slots * 5u + 1u);
    let batch_scratch = batch_idx * scratch_stride;
    let q_base = batch_scratch;
    let k_base = q_base + h_slots * d_model;
    let v_base = k_base + h_slots * d_model;
    let attn_base = v_base + h_slots * d_model;
    let mamba_base = attn_base + h_slots * d_model;
    let signal_base = mamba_base + h_slots * d_model;

    if (tid == 0u) {
        atomicStore(&hit_count, 0u);
        max_delta_seen = 0.0;
        last_delta = 0.0;
        curr_contractivity = 0.0;
    }
    workgroupBarrier();

    var total_iters = 0u;
    var max_contractivity = 0.0;
    let scale = inverseSqrt(max(1.0, f32(d_model)));

    for (var t = 0u; t < shape.seq_len; t = t + 1u) {
        // input_signal = W_in * s_t
        let s_in_base = batch_idx * (shape.seq_len * d_model) + t * d_model;
        for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
            var inj = 0.0;
            for (var j = 0u; j < d_model; j = j + 1u) {
                inj = inj + W_in[j * d_model + d_out] * S_in[s_in_base + j];
            }
            Scratch[signal_base + d_out] = inj;
        }
        workgroupBarrier();

        var iter = 0u;
        var converged = false;
        while (iter < shape.max_iters && !converged) {
            var local_max_delta = 0.0;

            // Q/K/V per slot
            for (var s = 0u; s < h_slots; s = s + 1u) {
                let off = s * d_model;
                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    var q = 0.0;
                    var k = 0.0;
                    var v = 0.0;
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        let h_val = H_curr[h_base + off + j];
                        let w_idx = j * d_model + d_out;
                        q = q + W_q[w_idx] * h_val;
                        k = k + W_k[w_idx] * h_val;
                        v = v + W_v[w_idx] * h_val;
                    }
                    Scratch[q_base + off + d_out] = q;
                    Scratch[k_base + off + d_out] = k;
                    Scratch[v_base + off + d_out] = v;
                }
            }
            workgroupBarrier();

            // cross-slot attention: attn_slot = W_o * (softmax(q·k) * v)
            for (var qs = 0u; qs < h_slots; qs = qs + 1u) {
                let q_off = qs * d_model;
                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    var scores: array<f32, 8>;
                    var max_s = -1e30;
                    for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
                        let k_off = ks * d_model;
                        var score = 0.0;
                        for (var j = 0u; j < d_model; j = j + 1u) {
                            score = score + Scratch[q_base + q_off + j] * Scratch[k_base + k_off + j];
                        }
                        score = score * scale;
                        scores[ks] = score;
                        max_s = max(max_s, score);
                    }

                    var sum_exp = 0.0;
                    for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
                        sum_exp = sum_exp + exp(scores[ks] - max_s);
                    }
                    sum_exp = max(sum_exp, 1e-12);

                    var out = 0.0;
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        var mix = 0.0;
                        for (var ks = 0u; ks < h_slots; ks = ks + 1u) {
                            let w = exp(scores[ks] - max_s) / sum_exp;
                            mix = mix + w * Scratch[v_base + ks * d_model + j];
                        }
                        out = out + W_o[j * d_model + d_out] * mix;
                    }
                    Scratch[attn_base + q_off + d_out] = out;
                }
            }
            workgroupBarrier();

            // mamba_step per slot:
            // y = a*h + (1-a)*(I + W_x)h, mamba = (I + W_out)y
            // Use q_base region as temporary y-buffer to avoid read/write races.
            for (var s = 0u; s < h_slots; s = s + 1u) {
                let off = s * d_model;
                for (var d = tid; d < d_model; d = d + WG_SIZE) {
                    var x_proj = H_curr[h_base + off + d];
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        x_proj = x_proj + W_x[j * d_model + d] * H_curr[h_base + off + j];
                    }
                    let h_prev = H_curr[h_base + off + d];
                    // Stable gate in (0,1): a = sigmoid(-a_log).
                    let a = 1.0 / (1.0 + exp(A_log[d]));
                    Scratch[q_base + off + d] = a * h_prev + (1.0 - a) * x_proj;
                }
                workgroupBarrier();

                for (var d_out = tid; d_out < d_model; d_out = d_out + WG_SIZE) {
                    var out = Scratch[q_base + off + d_out];
                    for (var j = 0u; j < d_model; j = j + 1u) {
                        out = out + W_out[j * d_model + d_out] * Scratch[q_base + off + j];
                    }
                    Scratch[mamba_base + off + d_out] = out;
                }
                workgroupBarrier();
            }

            // combine + rms_norm + damped_update per slot
            for (var s = 0u; s < h_slots; s = s + 1u) {
                let off = s * d_model;

                var local_sumsq = 0.0;
                for (var d = tid; d < d_model; d = d + WG_SIZE) {
                    let h_prev = H_curr[h_base + off + d];
                    let combined = Scratch[attn_base + off + d]
                        + Scratch[mamba_base + off + d]
                        + Scratch[signal_base + d]
                        + (shape.residual_alpha * h_prev);
                    H_next[h_base + off + d] = combined;
                    local_sumsq = local_sumsq + combined * combined;
                }
                shared_vals[tid] = local_sumsq;
                workgroupBarrier();
                for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                    if (tid < stride) {
                        shared_vals[tid] = shared_vals[tid] + shared_vals[tid + stride];
                    }
                    workgroupBarrier();
                }
                let rms = sqrt(shared_vals[0] / max(1.0, f32(d_model)) + 1e-6);

                for (var d = tid; d < d_model; d = d + WG_SIZE) {
                    let h_prev = H_curr[h_base + off + d];
                    let f_h = NormScale[d] * (H_next[h_base + off + d] / rms);
                    let val = shape.damping * f_h + (1.0 - shape.damping) * h_prev;
                    local_max_delta = max(local_max_delta, abs(val - h_prev));
                    H_next[h_base + off + d] = val;
                }
                workgroupBarrier();
            }

            shared_vals[tid] = local_max_delta;
            workgroupBarrier();
            for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
                if (tid < stride) {
                    shared_vals[tid] = max(shared_vals[tid], shared_vals[tid + stride]);
                }
                workgroupBarrier();
            }

            if (shared_vals[0] < shape.epsilon) {
                converged = true;
            }

            if (tid == 0u) {
                let d_curr = shared_vals[0];
                let d_prev = last_delta;
                curr_contractivity = 0.0;
                if (iter > 0u && d_prev > 1e-12) {
                    curr_contractivity = d_curr / d_prev;
                }
                last_delta = d_curr;
                max_contractivity = max(max_contractivity, curr_contractivity);
            }
            workgroupBarrier();

            for (var i = tid; i < total_elements; i = i + WG_SIZE) {
                H_curr[h_base + i] = H_next[h_base + i];
            }
            workgroupBarrier();
            iter = iter + 1u;
        }

        total_iters = total_iters + iter;
        if (tid == 0u) {
            let d = shared_vals[0];
            max_delta_seen = max(max_delta_seen, d);
            if (!converged) {
                atomicAdd(&hit_count, 1u);
            }
        }
        workgroupBarrier();

        for (var d = tid; d < d_model; d = d + WG_SIZE) {
            var acc = 0.0;
            for (var s = 0u; s < h_slots; s = s + 1u) {
                acc = acc + H_curr[h_base + s * d_model + d];
            }
            H_pooled[batch_idx * (shape.seq_len * d_model) + t * d_model + d] = acc / f32(h_slots);
        }
        workgroupBarrier();
    }

    // Keep H_next synchronized for readback.
    for (var i = tid; i < total_elements; i = i + WG_SIZE) {
        H_next[h_base + i] = H_curr[h_base + i];
    }
    workgroupBarrier();

    if (batch_idx == 0u && tid == 0u) {
        DebugLog[0] = 777.7;
        DebugLog[1] = f32(shape.batch_size);
        DebugLog[2] = f32(shape.d_model);
        DebugLog[10] = f32(shape.seq_len);
        DebugLog[11] = 0.0;
        DebugLog[12] = H_curr[h_base];
        DebugLog[13] = f32(total_iters) / max(1.0, f32(shape.seq_len));
        DebugLog[14] = select(0.0, 1.0, atomicLoad(&hit_count) == 0u);
        DebugLog[15] = f32(atomicLoad(&hit_count));
        DebugLog[16] = max_delta_seen;
        DebugLog[17] = last_delta;
        DebugLog[18] = 0.0;
        DebugLog[19] = f32(total_elements);
        DebugLog[20] = 999.9;
        DebugLog[21] = max_contractivity;
    }
}
