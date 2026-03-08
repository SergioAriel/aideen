// Loxi V8 — Mamba Pass 1: RMSNorm + in_proj
//
// Computes: xz = RMSNorm(x) @ in_proj.weight^T
// Input:  x[D]
// Output: xz[2*d_inner]   (first half = x_ssm input, second half = gate z)
//
// Dispatch: ceil(2*d_inner / 256) workgroups × 256 threads
// Thread gid.x computes xz[gid.x]

struct MambaShape {
    D:       u32,  // input dimension
    d_inner: u32,  // D * expand
    d_state: u32,
    d_conv:  u32,
    dt_rank: u32,
};

@group(0) @binding(0) var<uniform>             shape:     MambaShape;
@group(0) @binding(1) var<storage, read>       x_in:      array<f32>; // [D]
@group(0) @binding(2) var<storage, read>       norm_w:    array<f32>; // [D]
@group(0) @binding(3) var<storage, read>       in_proj_w: array<f32>; // [2*d_inner * D] row-major
@group(0) @binding(4) var<storage, read_write> xz:        array<f32>; // [2*d_inner]

@compute @workgroup_size(256, 1, 1)
fn rms_norm_in_proj(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out = gid.x;
    if (out >= 2u * shape.d_inner) { return; }

    let D = shape.D;

    // ── RMSNorm: compute sum(x²)/D, then scale ────────────────────────────
    // Each thread independently computes the norm (redundant but correct).
    // For D≤2048 this loop takes ~microseconds — acceptable for v1.
    // TODO: precompute the norm in a preparatory pass for large D.
    var sq_sum: f32 = 0.0;
    for (var i: u32 = 0u; i < D; i++) {
        sq_sum += x_in[i] * x_in[i];
    }
    let rms_inv = inverseSqrt(sq_sum / f32(D) + 1e-8);

    // ── GEMV: xz[out] = sum_d( in_proj_w[out, d] * norm_w[d] * x_in[d] * rms_inv ) ──
    var acc: f32 = 0.0;
    let row_base = out * D;
    for (var d: u32 = 0u; d < D; d++) {
        acc += in_proj_w[row_base + d] * (norm_w[d] * x_in[d] * rms_inv);
    }
    xz[out] = acc;
}
