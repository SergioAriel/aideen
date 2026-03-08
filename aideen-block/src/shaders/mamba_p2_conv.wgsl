// Loxi V8 — Mamba Pass 2: conv1d + SiLU
//
// Applies causal depthwise conv1d to x_ssm (first half of xz).
// For single-token inference with zero state:
//   conv_out[d] = x_ssm[d] * conv_w[d, d_conv-1] + conv_b[d]
//   (only the last kernel position applies because all prior state is zero)
// Then applies SiLU activation.
//
// Input:  xz[0:d_inner]       (x_ssm, first half of Pass 1 output)
// Output: x_act[d_inner]      (activated conv output, input to x_proj & out_proj)
//         xz[d_inner:2*d_inner] remains untouched — read as gate z in Pass 5
//
// Dispatch: ceil(d_inner / 256) workgroups × 256 threads

struct MambaShape {
    D: u32, d_inner: u32, d_state: u32, d_conv: u32, dt_rank: u32,
};

@group(0) @binding(0) var<uniform>             shape:    MambaShape;
@group(0) @binding(1) var<storage, read>       xz:       array<f32>; // [2*d_inner]
@group(0) @binding(2) var<storage, read>       conv1d_w: array<f32>; // [d_inner * d_conv]
@group(0) @binding(3) var<storage, read>       conv1d_b: array<f32>; // [d_inner]
@group(0) @binding(4) var<storage, read_write> x_act:    array<f32>; // [d_inner]

fn silu(v: f32) -> f32 {
    return v / (1.0 + exp(-v));
}

@compute @workgroup_size(256, 1, 1)
fn conv1d_silu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= shape.d_inner) { return; }

    // x_ssm[d] = xz[d]  (first half of xz)
    let x_ssm_d = xz[d];

    // Depthwise conv1d, zero-state single-token:
    // kernel for channel d is conv1d_w[d*d_conv .. d*d_conv+d_conv]
    // with zero state, only the last kernel position [d_conv-1] is non-zero
    let k_last = conv1d_w[d * shape.d_conv + (shape.d_conv - 1u)];
    let conv_out = x_ssm_d * k_last + conv1d_b[d];

    x_act[d] = silu(conv_out);
}
