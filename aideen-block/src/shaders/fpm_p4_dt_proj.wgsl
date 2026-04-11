// Loxi V8 — Fpm Pass 4: dt_proj + softplus → delta
//
// Computes:
//   delta_raw = proj_bcd[0:dt_rank]
//   delta[d] = softplus( dot(dt_proj_w[d, :], delta_raw) + dt_proj_b[d] )
//
// Input:  proj_bcd[0:dt_rank]    (first dt_rank elements from Pass 3)
//         dt_proj_w[d_inner, dt_rank]
//         dt_proj_b[d_inner]
// Output: delta[d_inner]
//
// Dispatch: ceil(d_inner / 256) workgroups × 256 threads

struct FpmShape {
    D: u32, d_inner: u32, d_state: u32, d_conv: u32, dt_rank: u32,
};

@group(0) @binding(0) var<uniform>             shape:      FpmShape;
@group(0) @binding(1) var<storage, read>       proj_bcd:   array<f32>; // [dt_rank+2*d_state] — only [0:dt_rank] used
@group(0) @binding(2) var<storage, read>       dt_proj_w:  array<f32>; // [d_inner * dt_rank] row-major
@group(0) @binding(3) var<storage, read>       dt_proj_b:  array<f32>; // [d_inner]
@group(0) @binding(4) var<storage, read_write> delta:      array<f32>; // [d_inner]

fn softplus(x: f32) -> f32 {
    // log(1 + exp(x)) — numerically stable for large x
    return select(log(1.0 + exp(x)), x, x > 20.0);
}

@compute @workgroup_size(256, 1, 1)
fn dt_proj_softplus(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= shape.d_inner) { return; }

    // GEMV: raw_d = sum_r( dt_proj_w[d, r] * delta_raw[r] ) + dt_proj_b[d]
    var raw: f32 = dt_proj_b[d];
    let row = d * shape.dt_rank;
    for (var r: u32 = 0u; r < shape.dt_rank; r++) {
        raw += dt_proj_w[row + r] * proj_bcd[r];
    }
    delta[d] = softplus(raw);
}
