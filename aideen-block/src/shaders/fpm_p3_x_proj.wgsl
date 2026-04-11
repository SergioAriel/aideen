// Loxi V8 — Fpm Pass 3: x_proj GEMV
//
// Computes: proj_bcd = x_act @ x_proj_w^T
// proj_bcd layout: [delta_raw[dt_rank] | B[d_state] | C[d_state]]
//
// Input:  x_act[d_inner]
// Output: proj_bcd[dt_rank + 2*d_state]
//
// Dispatch: ceil((dt_rank + 2*d_state) / 256) workgroups × 256 threads

struct FpmShape {
    D: u32, d_inner: u32, d_state: u32, d_conv: u32, dt_rank: u32,
};

@group(0) @binding(0) var<uniform>             shape:     FpmShape;
@group(0) @binding(1) var<storage, read>       x_act:     array<f32>; // [d_inner]
@group(0) @binding(2) var<storage, read>       x_proj_w:  array<f32>; // [(dt_rank+2*d_state)*d_inner]
@group(0) @binding(3) var<storage, read_write> proj_bcd:  array<f32>; // [dt_rank+2*d_state]

@compute @workgroup_size(256, 1, 1)
fn x_proj(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out = gid.x;
    let proj_size = shape.dt_rank + 2u * shape.d_state;
    if (out >= proj_size) { return; }

    // GEMV row: proj_bcd[out] = sum_d( x_proj_w[out, d] * x_act[d] )
    var acc: f32 = 0.0;
    let row = out * shape.d_inner;
    for (var d: u32 = 0u; d < shape.d_inner; d++) {
        acc += x_proj_w[row + d] * x_act[d];
    }
    proj_bcd[out] = acc;
}
