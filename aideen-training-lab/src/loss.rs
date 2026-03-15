use nalgebra::DVector;

/// Compute softmax of a logits vector.
pub fn softmax(logits: &DVector<f32>) -> DVector<f32> {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: DVector<f32> = logits.map(|l| (l - max_l).exp());
    let sum_exp: f32 = exps.sum();
    exps / sum_exp
}

/// Cross-entropy loss for a single target token.
pub fn cross_entropy_loss(logits: &DVector<f32>, target: u32) -> f32 {
    cross_entropy(logits, target)
}

/// Cross-entropy loss for a single target token (alias).
pub fn cross_entropy(logits: &DVector<f32>, target: u32) -> f32 {
    let probs = softmax(logits);
    let p = probs[target as usize].max(1e-12);
    -p.ln()
}

/// Gradient of cross-entropy loss w.r.t. logits.
pub fn cross_entropy_backward(logits: &DVector<f32>, target: u32) -> DVector<f32> {
    cross_entropy_grad(logits, target)
}

/// Gradient of cross-entropy loss w.r.t. logits (alias).
pub fn cross_entropy_grad(logits: &DVector<f32>, target: u32) -> DVector<f32> {
    let mut probs = softmax(logits);
    probs[target as usize] -= 1.0;
    probs
}
