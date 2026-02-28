// kernels/softmax.wgsl
// ─────────────────────────────────────────────────────────────────────────────
// Numerically-stable Softmax: y_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// Input:   x  [N, D]  — attention logits or any pre-softmax values
// Output:  y  [N, D]  — probability distribution (sums to 1.0 per row)
//
// Each workgroup handles one row.
// Three-pass: find max → compute exp/sum → normalize.
// The max subtraction prevents exp overflow (online softmax trick).
// ─────────────────────────────────────────────────────────────────────────────

struct SoftmaxParams {
    N: u32,   // number of rows
    D: u32,   // dimension (number of logits per row)
}

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: SoftmaxParams;

var<workgroup> partial_max: array<f32, 256>;
var<workgroup> partial_sum: array<f32, 256>;
var<workgroup> row_max:     f32;
var<workgroup> row_sum:     f32;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id)        wg_id:    vec3<u32>,
) {
    let row     = wg_id.x;
    let tid     = local_id.x;
    let wg_size = 256u;

    if (row >= params.N) { return; }

    let row_offset = row * params.D;

    // ── Pass 1: find max ────────────────────────────────────────────────────
    var local_max: f32 = -3.402823466e+38;  // -FLT_MAX

    var d = tid;
    loop {
        if (d >= params.D) { break; }
        local_max = max(local_max, x[row_offset + d]);
        d += wg_size;
    }

    partial_max[tid] = local_max;
    workgroupBarrier();

    var stride = wg_size / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            partial_max[tid] = max(partial_max[tid], partial_max[tid + stride]);
        }
        workgroupBarrier();
        stride /= 2u;
    }

    if (tid == 0u) { row_max = partial_max[0]; }
    workgroupBarrier();

    // ── Pass 2: compute exp(x - max) and sum ────────────────────────────────
    var local_sum: f32 = 0.0;

    var d2 = tid;
    loop {
        if (d2 >= params.D) { break; }
        let idx = row_offset + d2;
        let e = exp(x[idx] - row_max);
        output[idx] = e;   // store temporarily
        local_sum += e;
        d2 += wg_size;
    }

    partial_sum[tid] = local_sum;
    workgroupBarrier();

    var stride2 = wg_size / 2u;
    loop {
        if (stride2 == 0u) { break; }
        if (tid < stride2) {
            partial_sum[tid] += partial_sum[tid + stride2];
        }
        workgroupBarrier();
        stride2 /= 2u;
    }

    if (tid == 0u) { row_sum = partial_sum[0]; }
    workgroupBarrier();

    // ── Pass 3: normalize ───────────────────────────────────────────────────
    let inv_sum = 1.0 / row_sum;

    var d3 = tid;
    loop {
        if (d3 >= params.D) { break; }
        let idx = row_offset + d3;
        output[idx] = output[idx] * inv_sum;
        d3 += wg_size;
    }
}
