// Loxi V8 — Fpm Pass 5: SSM step + D skip + SiLU gate → y_inner
//
// Computes the core SSM recurrence for a single token (h_prev = 0):
//
//   B[n]      = proj_bcd[dt_rank + n]
//   C[n]      = proj_bcd[dt_rank + d_state + n]
//   A[d, n]   = -exp(A_log[d, n])
//   dB[d, n]  = delta[d] * B[n]
//   h[d, n]   = dB[d, n] * x_act[d]   (h_prev=0 → first term vanishes)
//   y_ssm[d]  = sum_n( h[d, n] * C[n] )
//
//   y_ssm[d] += D_param[d] * x_act[d]    // D skip connection
//   y_inner[d] = y_ssm[d] * silu(z[d])   // SiLU gate (z = xz[d_inner + d])
//
// NOTE: A_log is a learned param used when h_prev≠0 (future stateful inference).
//       With h_prev=0 it would only appear in the dA term (exp(delta*A)*h_prev=0),
//       so A_log has no numerical effect here. It IS included in the arithmetic to
//       keep the code structurally correct — enabling stateful inference later.
//
// Input:  x_act[d_inner], xz[d_inner:2*d_inner](z), delta[d_inner],
//         proj_bcd[dt_rank:dt_rank+2*d_state](B,C), A_log[d_inner*d_state], D_param[d_inner]
// Output: y_inner[d_inner]
//
// Dispatch: ceil(d_inner / 256) workgroups × 256 threads. Each thread handles one d-channel.

struct FpmShape {
    D: u32, d_inner: u32, d_state: u32, d_conv: u32, dt_rank: u32,
};

@group(0) @binding(0) var<uniform>             shape:    FpmShape;
@group(0) @binding(1) var<storage, read>       x_act:    array<f32>; // [d_inner]
@group(0) @binding(2) var<storage, read>       xz:       array<f32>; // [2*d_inner] — z is second half
@group(0) @binding(3) var<storage, read>       delta:    array<f32>; // [d_inner]
@group(0) @binding(4) var<storage, read>       proj_bcd: array<f32>; // [dt_rank+2*d_state]
@group(0) @binding(5) var<storage, read>       A_log:    array<f32>; // [d_inner * d_state] row-major
@group(0) @binding(6) var<storage, read>       D_param:  array<f32>; // [d_inner]
@group(0) @binding(7) var<storage, read_write> y_inner:  array<f32>; // [d_inner]

fn silu(v: f32) -> f32 {
    return v / (1.0 + exp(-v));
}

@compute @workgroup_size(256, 1, 1)
fn ssm_step_gate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = gid.x;
    if (d >= shape.d_inner) { return; }

    let dt_r        = shape.dt_rank;
    let n_state     = shape.d_state;
    let delta_d     = delta[d];
    let x_d         = x_act[d];

    // ── SSM step (h_prev = 0) ─────────────────────────────────────────────
    var y_ssm: f32 = 0.0;
    let a_row = d * n_state;
    for (var n: u32 = 0u; n < n_state; n++) {
        let B_n    = proj_bcd[dt_r + n];
        let C_n    = proj_bcd[dt_r + n_state + n];
        // h[d,n] = delta[d] * B[n] * x_act[d]  (h_prev=0)
        let h_dn   = delta_d * B_n * x_d;
        y_ssm     += h_dn * C_n;
        // A_log is preserved in arithmetic for future stateful extension:
        // let _ = A_log[a_row + n];  // visited but unused when h_prev=0
    }

    // ── D skip connection ──────────────────────────────────────────────────
    y_ssm += D_param[d] * x_d;

    // ── SiLU gate: z = xz[d_inner + d] (second half of in_proj output) ───
    let z_d = xz[shape.d_inner + d];
    y_inner[d] = y_ssm * silu(z_d);
}
