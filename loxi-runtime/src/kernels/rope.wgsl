// kernels/rope.wgsl
// ─────────────────────────────────────────────────────────────────────────────
// Rotary Position Embedding (RoPE)
//
// Applied to query and key tensors before attention computation.
// Encodes position by rotating pairs of dimensions.
//
// For each token at position pos and head dimension d:
//   θ_d = pos / 10000^(2d / D_head)
//   [x_{2d}, x_{2d+1}] → [x_{2d}*cos(θ) - x_{2d+1}*sin(θ),
//                          x_{2d}*sin(θ) + x_{2d+1}*cos(θ)]
//
// Input shape:  [batch, n_heads, seq_len, d_head]  (flattened)
// Output shape: same as input
//
// The tensor is stored as [B * H * S * D_head] row-major.
// Each thread handles one (batch, head, position, dim_pair).
// ─────────────────────────────────────────────────────────────────────────────

struct RopeParams {
    B:       u32;   // batch size
    H:       u32;   // number of attention heads
    S:       u32;   // sequence length
    D_head:  u32;   // dimension per head (must be even)
    // RoPE base — 10000.0 for standard, 500000.0 for LLaMA-3 extended context
    base:    f32;
}

@group(0) @binding(0) var<storage, read>       x_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> x_out: array<f32>;
@group(0) @binding(2) var<uniform>             p:     RopeParams;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    // Thread handles one pair (2 dims) at a given [b, h, s] position
    // global_id.x = pair index across all [B, H, S, D_head/2]
    let total_pairs = p.B * p.H * p.S * (p.D_head / 2u);
    let idx = global_id.x;
    if (idx >= total_pairs) { return; }

    // Decompose flat pair index
    let half_d  = p.D_head / 2u;
    let d_pair  = idx % half_d;
    let rest    = idx / half_d;
    let s       = rest % p.S;
    let rest2   = rest / p.S;
    let h       = rest2 % p.H;
    let b       = rest2 / p.H;

    // Compute rotation angle: θ = pos / base^(2*d_pair / D_head)
    let freq = 1.0 / pow(p.base, f32(2u * d_pair) / f32(p.D_head));
    let theta = f32(s) * freq;
    let cos_t = cos(theta);
    let sin_t = sin(theta);

    // Flat offset for the two elements in the pair
    let base_offset = ((b * p.H + h) * p.S + s) * p.D_head;
    let i0 = base_offset + 2u * d_pair;
    let i1 = base_offset + 2u * d_pair + 1u;

    let x0 = x_in[i0];
    let x1 = x_in[i1];

    // Rotate the pair
    x_out[i0] = x0 * cos_t - x1 * sin_t;
    x_out[i1] = x0 * sin_t + x1 * cos_t;
}
