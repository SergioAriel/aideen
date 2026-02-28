// kernels/adamw.wgsl
// ─────────────────────────────────────────────────────────────────────────────
// AdamW Optimizer Step (Decoupled Weight Decay)
//
// One kernel call updates ALL parameters of a single weight tensor.
// Called once per parameter group per training step.
//
// Update rule:
//   m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
//   v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
//   m̂_t = m_t / (1 - β₁^t)   (bias correction)
//   v̂_t = v_t / (1 - β₂^t)   (bias correction)
//   θ_t = θ_{t-1} * (1 - lr * wd) - lr * m̂_t / (sqrt(v̂_t) + ε)
//
// Note: weight decay is applied BEFORE the gradient step (decoupled = AdamW).
// ─────────────────────────────────────────────────────────────────────────────

struct AdamWParams {
    N:           u32;   // number of parameters
    lr:          f32;   // learning rate η
    beta1:       f32;   // momentum β₁ (typically 0.9)
    beta2:       f32;   // RMS β₂ (typically 0.999)
    eps:         f32;   // numerical stability ε (typically 1e-8)
    weight_decay: f32;  // λ (typically 0.01)
    // Bias correction factors: bc1 = 1 - β₁^t, bc2 = 1 - β₂^t
    // Pre-computed on CPU to avoid repeated pow() in shader
    bc1:         f32;
    bc2:         f32;
}

// Parameter tensor (read-write — updated in place)
@group(0) @binding(0) var<storage, read_write> theta:   array<f32>;
// Gradient tensor (read-only — produced by backward pass)
@group(0) @binding(1) var<storage, read>       grad:    array<f32>;
// First moment (momentum) — m_t
@group(0) @binding(2) var<storage, read_write> moment1: array<f32>;
// Second moment (RMS) — v_t
@group(0) @binding(3) var<storage, read_write> moment2: array<f32>;
// Hyperparameters
@group(0) @binding(4) var<uniform>             p:       AdamWParams;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i = global_id.x;
    if (i >= p.N) { return; }

    let g  = grad[i];
    let m  = p.beta1 * moment1[i] + (1.0 - p.beta1) * g;
    let v  = p.beta2 * moment2[i] + (1.0 - p.beta2) * g * g;

    moment1[i] = m;
    moment2[i] = v;

    let m_hat = m / p.bc1;
    let v_hat = v / p.bc2;

    // Decoupled weight decay: θ ← θ * (1 - lr * λ)
    let decay = theta[i] * (1.0 - p.lr * p.weight_decay);
    // Adam update
    theta[i] = decay - p.lr * m_hat / (sqrt(v_hat) + p.eps);
}
