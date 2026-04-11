// AIDEEN Spectral Renormalization – Power Iteration on GPU
// (5*h_slots + 2) workgroups:
// W_q[0..h-1], W_k[h..2h-1], W_v[2h..3h-1], W_o[3h..4h-1], W_in[4h..5h-1], W_x[5h], W_out[5h+1]
// 256 threads per workgroup, each handling one row element.

struct SpectralParams {
    d_model:   u32,
    n_iters:   u32,
    attn_threshold: f32,
    win_threshold: f32,
    fpm_threshold: f32,
    wv_threshold: f32,
    wo_threshold: f32,
    h_slots:   u32,
    mat_idx:   u32,
    _pad0:     u32,
    _pad1:     u32,
    _pad2:     u32,
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

// Runtime-indexed read from matrices.
// mat_idx:
// 0..h-1         = W_q slots
// h..2h-1        = W_k slots
// 2h..3h-1       = W_v slots
// 3h..4h-1       = W_o slots
// 4h..5h-1       = W_in slots
// 5h            = W_x
// 5h+1          = W_out
fn read_W(mat_idx: u32, idx: u32) -> f32 {
    let h_slots = params.h_slots;
    let d = params.d_model;
    if (mat_idx < h_slots) { return W_q[mat_idx * d * d + idx]; }
    if (mat_idx < 2u * h_slots) { return W_k[(mat_idx - h_slots) * d * d + idx]; }
    if (mat_idx < 3u * h_slots) { return W_v[(mat_idx - 2u * h_slots) * d * d + idx]; }
    if (mat_idx < 4u * h_slots) { return W_o[(mat_idx - 3u * h_slots) * d * d + idx]; }
    if (mat_idx < 5u * h_slots) { return W_in[(mat_idx - 4u * h_slots) * d * d + idx]; }
    if (mat_idx == 5u * h_slots) { return W_x[idx]; }
    return W_out[idx];
}

fn write_W(mat_idx: u32, idx: u32, val: f32) {
    let h_slots = params.h_slots;
    let d = params.d_model;
    if (mat_idx < h_slots) { W_q[mat_idx * d * d + idx] = val; return; }
    if (mat_idx < 2u * h_slots) { W_k[(mat_idx - h_slots) * d * d + idx] = val; return; }
    if (mat_idx < 3u * h_slots) { W_v[(mat_idx - 2u * h_slots) * d * d + idx] = val; return; }
    if (mat_idx < 4u * h_slots) { W_o[(mat_idx - 3u * h_slots) * d * d + idx] = val; return; }
    if (mat_idx < 5u * h_slots) { W_in[(mat_idx - 4u * h_slots) * d * d + idx] = val; return; }
    if (mat_idx == 5u * h_slots) { W_x[idx] = val; return; }
    W_out[idx] = val;
}

@compute
@workgroup_size(256, 1, 1)
fn spectral_renorm_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id)        group_id:  vec3<u32>,
) {
    let mat_idx = params.mat_idx;  // 0..(5*h_slots+1)
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
    // Thresholds:
    // 0..(2h-1): W_q/W_k use attn_threshold
    // 2h..(3h-1): W_v uses wv_threshold
    // 3h..(4h-1): W_o uses wo_threshold
    // 4h..(5h-1): W_in slots use win_threshold
    // 5h..: W_x/W_out use fpm_threshold
    var threshold = params.attn_threshold;
    if (mat_idx >= 2u * params.h_slots && mat_idx < 3u * params.h_slots) {
        threshold = params.wv_threshold;
    } else if (mat_idx >= 3u * params.h_slots && mat_idx < 4u * params.h_slots) {
        threshold = params.wo_threshold;
    } else if (mat_idx >= 4u * params.h_slots && mat_idx < 5u * params.h_slots) {
        threshold = params.win_threshold;
    } else if (mat_idx >= 5u * params.h_slots) {
        threshold = params.fpm_threshold;
    }
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
