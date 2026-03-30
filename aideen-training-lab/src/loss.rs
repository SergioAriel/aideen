//! Cross-entropy loss and its analytic gradient.

use nalgebra::DVector;

/// Cross-entropy loss between logits and a one-hot target.
/// `logits`: raw output from LmHead [vocab_size].
/// `target`: index of the correct token.
pub fn cross_entropy(logits: &DVector<f32>, target: u32) -> f32 {
    let probs = softmax(logits);
    let p = probs[target as usize].max(1e-10); // clamp to avoid log(0)
    -p.ln()
}

/// Cross-entropy gradient with respect to logits.
/// dL/d_logits = softmax(logits) - one_hot(target)
/// This is the exact analytic formula — no autograd required.
pub fn cross_entropy_grad(logits: &DVector<f32>, target: u32) -> DVector<f32> {
    let mut grad = softmax(logits);
    grad[target as usize] -= 1.0;
    grad
}

/// Stable softmax (subtract max to avoid overflow).
fn softmax(logits: &DVector<f32>) -> DVector<f32> {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: DVector<f32> = logits.map(|l| (l - max_l).exp());
    let sum_exp: f32 = exps.sum();
    exps / sum_exp
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_entropy_zero_for_perfect_prediction() {
        // Logits con un valor muy alto en la posición correcta
        let mut logits = DVector::zeros(10);
        logits[3] = 100.0;
        let loss = cross_entropy(&logits, 3);
        assert!(
            loss < 0.001,
            "loss debería ser ~0 con predicción perfecta, got {loss}"
        );
    }

    #[test]
    fn cross_entropy_positive_for_wrong_prediction() {
        let logits = DVector::from_element(10, 0.0); // uniform
        let loss = cross_entropy(&logits, 5);
        assert!(
            loss > 1.0,
            "loss debería ser alto con predicción uniform, got {loss}"
        );
    }

    #[test]
    fn grad_sums_to_zero() {
        // Propiedad: sum(softmax - one_hot) = 1 - 1 = 0
        let logits = DVector::from_fn(8, |i, _| (i as f32) * 0.5);
        let grad = cross_entropy_grad(&logits, 3);
        let sum: f32 = grad.sum();
        assert!(sum.abs() < 1e-5, "gradiente debe sumar ~0, got {sum}");
    }

    #[test]
    fn grad_is_negative_at_target() {
        let logits = DVector::from_element(8, 0.0); // uniform
        let grad = cross_entropy_grad(&logits, 2);
        // softmax(uniform) = 1/8 = 0.125, grad[2] = 0.125 - 1.0 = -0.875
        assert!(grad[2] < 0.0, "gradiente en target debe ser negativo");
    }
}
