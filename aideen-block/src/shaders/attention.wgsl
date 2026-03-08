// Loxi V8 — 2D Attention (Single Token Inference)
//
// Implements the two-stream attention logic:
//   x = x + out_light(Attention(QKV_light(Norm_light(x))))
//   x = x + out_heavy(Attention(QKV_heavy(Norm_heavy(x))))
//
// For single-token inference (current stage):
//   Softmax(Q @ K^T) @ V simplifies to V since score is 1.0.
//   However, we implement the full GEMV sequence to support future KV caching.

struct AttnShape {
    d_model:   u32,
    head_dim:  u32, // light_dim or heavy_dim
    seq_len:   u32,
    padding:   u32,
};

@group(0) @binding(0) var<uniform>             shape:      AttnShape;
@group(0) @binding(1) var<storage, read>       hidden:     array<f32>; // [d_model]
@group(0) @binding(2) var<storage, read>       norm_w:     array<f32>; // [d_model]
@group(0) @binding(3) var<storage, read>       qkv_w:      array<f32>; // [3 * head_dim, d_model]
@group(0) @binding(4) var<storage, read>       out_w:      array<f32>; // [d_model, head_dim]
@group(0) @binding(5) var<storage, read_write> output:     array<f32>; // [d_model] (accumulates residual)

const WG_SIZE: u32 = 256u;
var<workgroup> shared_sq_sum: f32;
var<workgroup> local_sums: array<f32, WG_SIZE>;

@compute @workgroup_size(256, 1, 1)
fn attention_step(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let d_model = shape.d_model;
    let head_dim = shape.head_dim;
    let tid = lid.x;

    // 1. RMSNorm
    var local_sum: f32 = 0.0;
    for (var i: u32 = tid; i < d_model; i += WG_SIZE) {
        let v = hidden[i];
        local_sum += v * v;
    }
    local_sums[tid] = local_sum;
    workgroupBarrier();

    // Parallel reduction in workgroup
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            local_sums[tid] += local_sums[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        shared_sq_sum = local_sums[0];
    }
    workgroupBarrier();

    let rms_inv = 1.0 / sqrt(shared_sq_sum / f32(d_model) + 1e-8);

    // Each thread computes one element of the FINAL output of this stream.
    if (gid.x < d_model) {
        let out_idx = gid.x;
        var acc: f32 = 0.0;
        
        for (var h: u32 = 0u; h < head_dim; h++) {
            // Compute V[h] = sum_j (qkv[2*head_dim+h, j] * x_normed[j])
            var v_h: f32 = 0.0;
            let v_row_offset = (2u * head_dim + h) * d_model;
            for (var j: u32 = 0u; j < d_model; j++) {
                v_h += qkv_w[v_row_offset + j] * (hidden[j] * norm_w[j] * rms_inv);
            }
            
            // output[out_idx] += out_w[out_idx, h] * V[h]
            acc += out_w[out_idx * head_dim + h] * v_h;
        }
        output[out_idx] += acc;
    }
}
