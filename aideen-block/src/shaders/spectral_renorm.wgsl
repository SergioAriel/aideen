// AIDEEN Spectral Renormalization – Power Iteration on GPU
// 7 workgroups (one per matrix: W_q, W_k, W_v, W_o, W_in, W_x, W_out)
// 256 threads per workgroup, each handling one row element.
// Shared memory: s_u[256] + s_v[256] + s_dot[256] = 3KB

struct SpectralParams {
    d_model:   u32,
    n_iters:   u32,
    attn_threshold: f32,
    mamba_threshold: f32,
};

@group(0) @binding(0) var<uniform>            params: SpectralParams;
@group(0) @binding(1) var<storage, read_write> W_q:   array<f32>;
@group(0) @binding(2) var<storage, read_write> W_k:   array<f32>;
@group(0) @binding(3) var<storage, read_write> W_v:   array<f32>;
@group(0) @binding(4) var<storage, read_write> W_o:   array<f32>;
@group(0) @binding(5) var<storage, read_write> W_in:  array<f32>;
@group(0) @binding(6) var<storage, read_write> W_x:   array<f32>;
@group(0) @binding(7) var<storage, read_write> W_out: array<f32>;

// Intermediate vectors are stored in storage buffers for full dynamic scaling (removes D=1024 limit)
@group(0) @binding(8) var<storage, read_write> s_u:   array<f32>;
@group(0) @binding(9) var<storage, read_write> s_v:   array<f32>;

const WG_SIZE: u32 = 256u;
var<workgroup> s_dot: array<f32, WG_SIZE>;

// Parallel reduction: sums s_dot and returns s_dot[0].
// Leaves s_dot in an undefined state except s_dot[0].
fn reduce_sum(tid: u32) -> f32 {
    workgroupBarrier();
    for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            s_dot[tid] += s_dot[tid + stride];
        }
        workgroupBarrier();
    }
    return s_dot[0];
}

// Runtime-indexed read from one of the 7 matrices (mat_idx is uniform in workgroup).
fn read_W(mat_idx: u32, idx: u32) -> f32 {
    if (mat_idx == 0u) { return W_q[idx]; }
    if (mat_idx == 1u) { return W_k[idx]; }
    if (mat_idx == 2u) { return W_v[idx]; }
    if (mat_idx == 3u) { return W_o[idx]; }
    if (mat_idx == 4u) { return W_in[idx]; }
    if (mat_idx == 5u) { return W_x[idx]; }
    return W_out[idx]; // mat_idx == 6
}

fn write_W(mat_idx: u32, idx: u32, val: f32) {
    if (mat_idx == 0u) { W_q[idx]  = val; return; }
    if (mat_idx == 1u) { W_k[idx]  = val; return; }
    if (mat_idx == 2u) { W_v[idx]  = val; return; }
    if (mat_idx == 3u) { W_o[idx]  = val; return; }
    if (mat_idx == 4u) { W_in[idx] = val; return; }
    if (mat_idx == 5u) { W_x[idx]  = val; return; }
    W_out[idx] = val;
}

@compute
@workgroup_size(256, 1, 1)
fn spectral_renorm_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id)        group_id:  vec3<u32>,
) {
    let mat_idx = group_id.x;  // 0=W_q … 6=W_out
    let tid     = local_id.x;
    let d       = params.d_model;

    // Initialize u = 1/sqrt(d) (uniform vector), zero-pad above d.
    let items_per_thread = (d + WG_SIZE - 1u) / WG_SIZE;
    
    for (var i = 0u; i < items_per_thread; i++) {
        let idx = tid + i * WG_SIZE;
        if (idx < d) {
            s_u[idx] = 1.0 / sqrt(f32(d));
        }
    }
    workgroupBarrier();

    var sigma = 1.0;

    for (var iter = 0u; iter < params.n_iters; iter++) {

        // --- Step 1: v = W^T u,  normalize v ---
        for (var i = 0u; i < items_per_thread; i++) {
            let row_idx = tid + i * WG_SIZE;
            if (row_idx < d) {
                var vt = 0.0;
                for (var j = 0u; j < d; j++) {
                    vt += read_W(mat_idx, j * d + row_idx) * s_u[j];
                }
                s_v[row_idx] = vt;
            }
        }
        
        workgroupBarrier();

        // Cálculo de norma (usando s_dot como acumulador de reducción)
        var local_v_norm_sq = 0.0;
        for (var i = 0u; i < items_per_thread; i++) {
            let row_idx = tid + i * WG_SIZE;
            if (row_idx < d) {
                let val = s_v[row_idx];
                local_v_norm_sq += val * val;
            }
        }
        s_dot[tid] = local_v_norm_sq;
        let v_norm = sqrt(reduce_sum(tid) + 1e-12);

        for (var i = 0u; i < items_per_thread; i++) {
            let row_idx = tid + i * WG_SIZE;
            if (row_idx < d) {
                s_v[row_idx] = s_v[row_idx] / v_norm;
            }
        }
        workgroupBarrier();

        // --- Step 2: u = W v,  σ ≈ ||u_unnorm||,  normalize u ---
        var local_u_norm_sq = 0.0;
        for (var i = 0u; i < items_per_thread; i++) {
            let row_idx = tid + i * WG_SIZE;
            if (row_idx < d) {
                var ut = 0.0;
                for (var j = 0u; j < d; j++) {
                    ut += read_W(mat_idx, row_idx * d + j) * s_v[j];
                }
                s_u[row_idx] = ut;
                local_u_norm_sq += ut * ut;
            }
        }
        s_dot[tid] = local_u_norm_sq;
        sigma = sqrt(reduce_sum(tid) + 1e-12);

        for (var i = 0u; i < items_per_thread; i++) {
            let row_idx = tid + i * WG_SIZE;
            if (row_idx < d) {
                s_u[row_idx] = s_u[row_idx] / sigma;
            }
        }
        workgroupBarrier();
    }

    // --- If σ > threshold, rescale W in-place ---
    let threshold = select(params.attn_threshold, params.mamba_threshold, mat_idx >= 5u);
    if (sigma > threshold) {
        let scale = threshold / sigma;
        for (var i = 0u; i < items_per_thread; i++) {
            let row_idx = tid + i * WG_SIZE;
            if (row_idx < d) {
                for (var j = 0u; j < d; j++) {
                    let idx = row_idx * d + j;
                    write_W(mat_idx, idx, read_W(mat_idx, idx) * scale);
                }
            }
        }
    }
}
