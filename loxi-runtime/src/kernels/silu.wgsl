// kernels/silu.wgsl
// ─────────────────────────────────────────────────────────────────────────────
// SiLU (Sigmoid Linear Unit): silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// Used in LLaMA-style FFN (replaces ReLU/GELU).
//
// Also includes other elementwise ops that share the same dispatch pattern:
//   - GELU (for BERT-style models)
//   - RoPE (Rotary Position Embedding) — more complex, handled separately
//   - Elementwise add (residual connections)
//   - Elementwise multiply
//   - Scale (multiply by scalar)
//
// All operate on flat arrays — caller is responsible for shape tracking.
// Dispatch: ceil(N / 256) workgroups of 256 threads.
// ─────────────────────────────────────────────────────────────────────────────

// ── SiLU ─────────────────────────────────────────────────────────────────────

struct ElemParams {
    N: u32,   // total number of elements
}

@group(0) @binding(0) var<storage, read>       input:  array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: ElemParams;

@compute @workgroup_size(256)
fn silu_forward(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i = global_id.x;
    if (i >= params.N) { return; }
    let x = input[i];
    output[i] = x / (1.0 + exp(-x));
}

// ── GELU (tanh approximation) ─────────────────────────────────────────────────
// gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

@compute @workgroup_size(256)
fn gelu_forward(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i = global_id.x;
    if (i >= params.N) { return; }
    let x = input[i];
    let inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    output[i] = 0.5 * x * (1.0 + tanh(inner));
}

// ── Elementwise Add (residual connections) ────────────────────────────────────
// C[i] = A[i] + B[i]

@group(0) @binding(0) var<storage, read>       A:         array<f32>;
@group(0) @binding(1) var<storage, read>       B:         array<f32>;
@group(0) @binding(2) var<storage, read_write> C_out:     array<f32>;
@group(0) @binding(3) var<uniform>             add_params: ElemParams;

@compute @workgroup_size(256)
fn add_forward(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i = global_id.x;
    if (i >= add_params.N) { return; }
    C_out[i] = A[i] + B[i];
}

// ── Elementwise Multiply ──────────────────────────────────────────────────────
// C[i] = A[i] * B[i]
// Used for: expert gate weighting, attention mask application

@compute @workgroup_size(256)
fn mul_forward(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i = global_id.x;
    if (i >= add_params.N) { return; }
    C_out[i] = A[i] * B[i];
}

// ── Scalar Scale ──────────────────────────────────────────────────────────────
// output[i] = input[i] * scale
// Used for: attention score normalization (/ sqrt(d_k)), learning rate scaling

struct ScaleParams {
    N:     u32;
    scale: f32;
}

@group(0) @binding(0) var<storage, read>       scale_in:  array<f32>;
@group(0) @binding(1) var<storage, read_write> scale_out: array<f32>;
@group(0) @binding(2) var<uniform>             sp:        ScaleParams;

@compute @workgroup_size(256)
fn scale_forward(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i = global_id.x;
    if (i >= sp.N) { return; }
    scale_out[i] = scale_in[i] * sp.scale;
}
