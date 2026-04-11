// Loxi V8 — Fpm Pass 6: out_proj GEMV + residual
//
// Computes: output[i] = sum_d( out_proj_w[i, d] * y_inner[d] ) + x_in[i]
//
// The x_in residual connection completes the Fpm block:
//   output = out_proj(y_inner) + x_in
// This matches the Python reference:
//   return self.out_proj(y) + residual
//
// Input:  y_inner[d_inner], out_proj_w[D * d_inner], x_in[D]
// Output: output[D]
//
// Dispatch: ceil(D / 256) workgroups × 256 threads. Each thread computes one output[i].

struct FpmShape {
    D: u32, d_inner: u32, d_state: u32, d_conv: u32, dt_rank: u32,
};

@group(0) @binding(0) var<uniform>             shape:       FpmShape;
@group(0) @binding(1) var<storage, read>       y_inner:     array<f32>; // [d_inner]
@group(0) @binding(2) var<storage, read>       out_proj_w:  array<f32>; // [D * d_inner] row-major
@group(0) @binding(3) var<storage, read>       x_in:        array<f32>; // [D]
@group(0) @binding(4) var<storage, read_write> output:      array<f32>; // [D]

@compute @workgroup_size(256, 1, 1)
fn out_proj_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= shape.D) { return; }

    // GEMV row: out_local[i] = sum_d( out_proj_w[i, d] * y_inner[d] )
    var acc: f32 = 0.0;
    let row = i * shape.d_inner;
    for (var d: u32 = 0u; d < shape.d_inner; d++) {
        acc += out_proj_w[row + d] * y_inner[d];
    }

    // Residual connection
    output[i] = acc + x_in[i];
}
