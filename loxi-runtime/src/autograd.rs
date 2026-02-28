// autograd.rs
// ─────────────────────────────────────────────────────────────────────────────
// Analytical gradients for backward pass.
//
// Loxi uses ANALYTICAL gradients (not automatic differentiation).
// This means: we write the exact mathematical derivative for each operation.
// This is:
//   - Faster than AD (no tape overhead)
//   - More predictable (no graph construction at runtime)
//   - More code (one backward per forward op)
//
// For a language model, the ops with non-trivial gradients are:
//   1. MatMul (most common)
//   2. LayerNorm
//   3. Softmax (attention weights)
//   4. SiLU/GELU
//   5. Cross-entropy loss
//
// Architecture:
//   Each backward function takes:
//     - grad_output: gradient flowing in from above (∂L/∂y)
//     - [cached activations from forward pass]
//   Returns:
//     - grad_input: gradient for the input (∂L/∂x)
//     - grad_params: gradients for learnable parameters
//
// GPU kernels for backward ops are implemented as additional WGSL shaders.
// For the initial implementation, we compute backward passes on CPU and
// use GPU only for the optimizer step. This trades speed for simplicity
// during the initial training phases.
// ─────────────────────────────────────────────────────────────────────────────

use anyhow::Result;
use crate::tensor::{GpuContext, Shape, Tensor};
use crate::dispatch::Dispatcher;

// ─── MatMul Gradients ─────────────────────────────────────────────────────────
// Forward:  C    = A × B          [M,K] × [K,N] → [M,N]
// Backward: ∂L/∂A = ∂L/∂C × Bᵀ  [M,N] × [N,K] → [M,K]
//           ∂L/∂B = Aᵀ × ∂L/∂C  [K,M] × [M,N] → [K,N]
//
// We reuse the matmul kernel with transposed operands.
// Transposing a [R,C] matrix to [C,R] is done by reshaping + kernel trick.

pub struct MatMulGrad {
    pub grad_a: Tensor,
    pub grad_b: Tensor,
}

pub async fn matmul_backward(
    grad_output: &Tensor,  // ∂L/∂C: [M, N]
    a: &Tensor,            // saved A: [M, K]
    b: &Tensor,            // saved B: [K, N]
    dispatch: &Dispatcher,
) -> Result<MatMulGrad> {
    // Compute Bᵀ [N, K] and Aᵀ [K, M] on CPU, upload, then matmul on GPU.
    // TODO: implement GPU transpose kernel for full speed.
    // For now: readback → transpose → re-upload.

    let a_cpu = a.read_to_cpu().await?;
    let b_cpu = b.read_to_cpu().await?;
    let grad_cpu = grad_output.read_to_cpu().await?;

    let (m, k) = (a.shape.dim(0)?, a.shape.dim(1)?);
    let n = b.shape.dim(1)?;

    // ∂L/∂A = ∂L/∂C × Bᵀ  → [M,N] × [N,K] → [M,K]
    let b_t = transpose_cpu(&b_cpu, k, n);
    let grad_a_data = matmul_cpu(&grad_cpu, &b_t, m, n, k);

    // ∂L/∂B = Aᵀ × ∂L/∂C  → [K,M] × [M,N] → [K,N]
    let a_t = transpose_cpu(&a_cpu, m, k);
    let grad_b_data = matmul_cpu(&a_t, &grad_cpu, k, m, n);

    let grad_a = Tensor::from_slice(&grad_a_data, Shape::new(vec![m, k]), dispatch.ctx.clone())?;
    let grad_b = Tensor::from_slice(&grad_b_data, Shape::new(vec![k, n]), dispatch.ctx.clone())?;

    Ok(MatMulGrad { grad_a, grad_b })
}

// ─── LayerNorm Gradients ───────────────────────────────────────────────────────
// Forward:  y = (x - μ) / σ * γ + β
// Backward:
//   ∂L/∂γ = sum(∂L/∂y * x̂)     over batch  [D]
//   ∂L/∂β = sum(∂L/∂y)          over batch  [D]
//   ∂L/∂x = complicated (see Ioffe & Szegedy 2015)
//
// Full derivation:
//   ∂L/∂x = (1/σ) * [∂L/∂y * γ - (1/D) * sum(∂L/∂y * γ) - (x̂/D) * sum(∂L/∂y * γ * x̂)]

pub struct LayerNormGrad {
    pub grad_x:     Tensor,
    pub grad_gamma: Tensor,
    pub grad_beta:  Tensor,
}

pub async fn layernorm_backward(
    grad_output: &Tensor,   // ∂L/∂y: [N, D]
    x:           &Tensor,   // saved input: [N, D]
    gamma:       &Tensor,   // γ: [D]
    eps:         f32,
    ctx:         GpuContext,
) -> Result<LayerNormGrad> {
    let grad_cpu = grad_output.read_to_cpu().await?;
    let x_cpu    = x.read_to_cpu().await?;
    let g_cpu    = gamma.read_to_cpu().await?;

    let n = x.shape.dim(0)?;
    let d = x.shape.dim(1)?;

    let mut grad_x     = vec![0.0f32; n * d];
    let mut grad_gamma = vec![0.0f32; d];
    let mut grad_beta  = vec![0.0f32; d];

    for row in 0..n {
        let off = row * d;
        let x_row  = &x_cpu[off..off + d];
        let dy_row = &grad_cpu[off..off + d];

        // Compute mean and variance for this row
        let mean: f32 = x_row.iter().sum::<f32>() / d as f32;
        let var:  f32 = x_row.iter().map(|xi| (xi - mean).powi(2)).sum::<f32>() / d as f32;
        let rstd: f32 = 1.0 / (var + eps).sqrt();

        // x̂ = (x - mean) / std
        let x_hat: Vec<f32> = x_row.iter().map(|xi| (xi - mean) * rstd).collect();

        // Accumulate γ and β gradients
        for j in 0..d {
            grad_gamma[j] += dy_row[j] * x_hat[j];
            grad_beta[j]  += dy_row[j];
        }

        // Compute ∂L/∂x for this row
        // sum1 = sum(dy * γ)
        // sum2 = sum(dy * γ * x̂)
        let sum1: f32 = (0..d).map(|j| dy_row[j] * g_cpu[j]).sum();
        let sum2: f32 = (0..d).map(|j| dy_row[j] * g_cpu[j] * x_hat[j]).sum();

        for j in 0..d {
            let dx_j = rstd * (dy_row[j] * g_cpu[j]
                              - sum1 / d as f32
                              - x_hat[j] * sum2 / d as f32);
            grad_x[off + j] = dx_j;
        }
    }

    Ok(LayerNormGrad {
        grad_x:     Tensor::from_slice(&grad_x,     x.shape.clone(),     ctx.clone())?,
        grad_gamma: Tensor::from_slice(&grad_gamma, gamma.shape.clone(), ctx.clone())?,
        grad_beta:  Tensor::from_slice(&grad_beta,  gamma.shape.clone(), ctx)?,
    })
}

// ─── Cross-Entropy Loss ───────────────────────────────────────────────────────
// Forward:  loss = -log(softmax(logits)[target])
//           = -logits[target] + log(sum(exp(logits)))
//
// Backward: ∂L/∂logits = softmax(logits) - one_hot(target)
//
// This is the cleanest gradient in deep learning — softmax + cross-entropy
// combine to give (probs - target), which is already computed in the forward pass.

pub struct CrossEntropyResult {
    pub loss:   f32,
    pub probs:  Vec<f32>,   // kept for backward pass
}

/// Compute cross-entropy loss for a batch of token predictions.
/// logits: [batch * seq, vocab_size] — raw unnormalized scores
/// targets: [batch * seq]           — ground truth token IDs
pub async fn cross_entropy_forward(
    logits:  &Tensor,   // [N, V] where N = batch*seq, V = vocab_size
    targets: &[u32],    // [N] target token IDs
    ctx:     GpuContext,
) -> Result<CrossEntropyResult> {
    let logits_cpu = logits.read_to_cpu().await?;
    let n = logits.shape.dim(0)?;
    let v = logits.shape.dim(1)?;

    let mut total_loss = 0.0f32;
    let mut probs = vec![0.0f32; n * v];

    for i in 0..n {
        let off = i * v;
        let row = &logits_cpu[off..off + v];
        let target = targets[i] as usize;

        // Numerically stable softmax
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();

        for j in 0..v {
            probs[off + j] = exps[j] / sum_exp;
        }

        // Cross-entropy: -log(p[target])
        total_loss += -(probs[off + target] + 1e-10).ln();
    }

    Ok(CrossEntropyResult {
        loss: total_loss / n as f32,
        probs,
    })
}

/// Gradient of cross-entropy loss w.r.t. logits.
/// ∂L/∂logits[i,j] = (probs[i,j] - 1_{j == target[i]}) / N
pub fn cross_entropy_backward(
    probs:   &[f32],   // softmax outputs from forward pass
    targets: &[u32],
    n:       usize,
    v:       usize,
    ctx:     GpuContext,
) -> Result<Tensor> {
    let mut grad = probs.to_vec();
    let scale = 1.0 / n as f32;

    for i in 0..n {
        let target = targets[i] as usize;
        grad[i * v + target] -= 1.0;
        for j in 0..v {
            grad[i * v + j] *= scale;
        }
    }

    Tensor::from_slice(&grad, Shape::new(vec![n, v]), ctx)
}

// ─── SiLU Backward ───────────────────────────────────────────────────────────
// Forward:  y = x * sigmoid(x) = x / (1 + exp(-x))
// Backward: ∂L/∂x = ∂L/∂y * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
//                 = ∂L/∂y * (silu(x)/x + sigmoid(x) * (1 - sigmoid(x)/x))
//           Simplified: ∂L/∂x = ∂L/∂y * (silu(x) + sigmoid(x) * (1 - silu(x)))

pub async fn silu_backward(
    grad_output: &Tensor,
    x: &Tensor,
    ctx: GpuContext,
) -> Result<Tensor> {
    let grad_cpu = grad_output.read_to_cpu().await?;
    let x_cpu    = x.read_to_cpu().await?;
    let n        = x.numel();

    let grad_input: Vec<f32> = (0..n).map(|i| {
        let xi  = x_cpu[i];
        let sig = 1.0 / (1.0 + (-xi).exp());
        let silu_xi = xi * sig;
        grad_cpu[i] * (silu_xi + sig * (1.0 - silu_xi))
    }).collect();

    Tensor::from_slice(&grad_input, x.shape.clone(), ctx)
}

// ─── CPU helpers (used in backward passes until GPU transpose is ready) ────────

fn transpose_cpu(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = mat[r * cols + c];
        }
    }
    out
}

// Naive CPU matmul (only used in backward pass for small matrices)
fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}
