// Loxi V8 — Inter-block Projection GEMV
//
// Projects hidden[d_from] → output[d_to] using a linear matrix multiply.
// Used at block boundaries (A→B, B→C, C→D) where the dimension changes.
//
// Computes: output[i] = sum_j( proj_w[i, j] * hidden[j] )
//
// No bias, no activation — pure linear projection (nn.Linear(..., bias=False)).
//
// Input:  hidden[d_from], proj_w[d_to * d_from]
// Output: output[d_to]
//
// Dispatch: ceil(d_to / 256) workgroups × 256 threads.

struct ProjShape {
    d_from: u32,
    d_to:   u32,
};

@group(0) @binding(0) var<uniform>             shape:   ProjShape;
@group(0) @binding(1) var<storage, read>       hidden:  array<f32>; // [d_from]
@group(0) @binding(2) var<storage, read>       proj_w:  array<f32>; // [d_to * d_from] row-major
@group(0) @binding(3) var<storage, read_write> output:  array<f32>; // [d_to]

@compute @workgroup_size(256, 1, 1)
fn linear_proj(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= shape.d_to) { return; }

    var acc: f32 = 0.0;
    let row = i * shape.d_from;
    for (var j: u32 = 0u; j < shape.d_from; j++) {
        acc += proj_w[row + j] * hidden[j];
    }
    output[i] = acc;
}
