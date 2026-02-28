// kernels/layernorm.wgsl
// ─────────────────────────────────────────────────────────────────────────────
// Layer Normalization: y = (x - mean) / sqrt(var + ε) * gamma + beta
//
// Input:   x      [batch, seq, d_model]  — flattened as [N, D]
// Params:  gamma  [D]
//          beta   [D]
// Output:  y      [N, D]
//
// Each workgroup handles one row (one token position).
// Uses two-pass Welford algorithm for numerical stability.
// D is passed as a uniform — kernel works for any model width.
// ─────────────────────────────────────────────────────────────────────────────

struct LayerNormParams {
    N:   u32,   // number of rows (batch*seq)
    D:   u32,   // model dimension
    eps: f32,   // epsilon for numerical stability (typically 1e-5)
}

@group(0) @binding(0) var<storage, read>       x:      array<f32>;
@group(0) @binding(1) var<storage, read>       gamma:  array<f32>;
@group(0) @binding(2) var<storage, read>       beta:   array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform>             params: LayerNormParams;

// Shared reduction buffers — one slot per thread in workgroup
// Max workgroup size = 256 (safe across Metal and Vulkan)
var<workgroup> partial_sum:    array<f32, 256>;
var<workgroup> partial_sq_sum: array<f32, 256>;
var<workgroup> row_mean:       f32;
var<workgroup> row_rstd:       f32;  // 1/sqrt(var + eps)

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id)  local_id:  vec3<u32>,
    @builtin(workgroup_id)         wg_id:     vec3<u32>,
) {
    let row      = wg_id.x;         // which token/row
    let tid      = local_id.x;      // thread index within workgroup
    let wg_size  = 256u;

    if (row >= params.N) { return; }

    let row_offset = row * params.D;

    // ── Pass 1: compute partial sums ────────────────────────────────────────
    var local_sum:    f32 = 0.0;
    var local_sq_sum: f32 = 0.0;

    var d = tid;
    loop {
        if (d >= params.D) { break; }
        let val = x[row_offset + d];
        local_sum    += val;
        local_sq_sum += val * val;
        d += wg_size;
    }

    partial_sum[tid]    = local_sum;
    partial_sq_sum[tid] = local_sq_sum;
    workgroupBarrier();

    // ── Parallel reduction (log2 steps) ─────────────────────────────────────
    var stride = wg_size / 2u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            partial_sum[tid]    += partial_sum[tid + stride];
            partial_sq_sum[tid] += partial_sq_sum[tid + stride];
        }
        workgroupBarrier();
        stride /= 2u;
    }

    // ── Thread 0 computes mean and rstd ─────────────────────────────────────
    if (tid == 0u) {
        let mean = partial_sum[0] / f32(params.D);
        let var_ = partial_sq_sum[0] / f32(params.D) - mean * mean;
        row_mean  = mean;
        row_rstd  = 1.0 / sqrt(var_ + params.eps);
    }
    workgroupBarrier();

    // ── Pass 2: normalize and scale ─────────────────────────────────────────
    var d2 = tid;
    loop {
        if (d2 >= params.D) { break; }
        let idx    = row_offset + d2;
        let normed = (x[idx] - row_mean) * row_rstd;
        output[idx] = normed * gamma[d2] + beta[d2];
        d2 += wg_size;
    }
}
